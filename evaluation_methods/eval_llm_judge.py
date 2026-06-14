"""
LLM-as-a-judge для оценки автоформализации.

Скрипт прогоняет выбранную модель как судью: для каждого
примера он решает, является ли `lean4_prediction` корректной (семантически
эквивалентной) формализацией `nl_statement`. На выходе — CSV с предсказаниями,
а в конце печатается, насколько судья совпал с эталонной разметкой `correct`.
"""

import argparse
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from datasets import load_dataset
from openai import OpenAI

DATASET = "AlexVav01/autoformalization-bench"

# Промпт судьи
SYSTEM_PROMPT = """\
You are a meticulous expert in Lean 4, Mathlib, and formal mathematics, acting \
as a strict grader of autoformalization.

You will be given: (1) a mathematical statement in natural language, (2) a
REFERENCE Lean 4 formalization that is known to be correct, and (3) a CANDIDATE
formalization produced by another model.

Your task: decide whether the CANDIDATE is a CORRECT formalization of the
natural-language statement, i.e. whether it is SEMANTICALLY EQUIVALENT to it.

Guidelines:
- The candidate does NOT need to be syntactically identical to the reference.
  Equivalent statements may differ by bound-variable renaming, function
  application / beta-reduction, unfolding definitions, reordering hypotheses,
  or using a different but equivalent encoding.
- Mark the candidate INCORRECT if it differs in MEANING from the NL statement.
  Watch specifically for: missing, extra, or invalid hypotheses; conditions that
  are strictly stronger or weaker than intended; wrong types or Lean defaults
  (ℕ starts at 0; `3 / 2 = 1` for naturals; integer vs. real division; Nat
  subtraction truncating at 0); instances/definitions whose meaning differs from
  the NL one; misplaced parentheses in quantified or propositional formulas that
  change the logic; trivial, vacuous, or self-contradictory statements.
- A missing proof (e.g. `sorry`) does NOT by itself make the statement
  incorrect. An empty or non-parsing candidate is INCORRECT.
- When genuinely in doubt, prefer INCORRECT.

First reason briefly, then output your verdict on the LAST line in EXACTLY this
format and nothing else:
VERDICT: CORRECT
or
VERDICT: INCORRECT
"""

USER_TEMPLATE = """\
# Natural-language statement
{nl}
{header_block}
# Reference Lean 4 formalization (known correct)
```lean
{ref}
```
 
# Candidate Lean 4 formalization (to be judged)
```lean
{cand}
```
 
Is the candidate a correct (semantically equivalent) formalization of the
natural-language statement? Reason briefly, then end with the VERDICT line."""


# --- Вспомогательные функции ----------------------------------------------- #
def to_bool(value):
    """Привести значение к bool (или None, если непонятно)."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    s = str(value).strip().lower()
    if s in {"true", "1", "yes", "correct"}:
        return True
    if s in {"false", "0", "no", "incorrect"}:
        return False
    return None


def parse_verdict(text):
    """Вытащить вердикт из ответа судьи: True / False / None."""
    m = re.findall(r"verdict\s*[:\-]?\s*\**\s*(correct|incorrect)", text, re.I)
    if m:
        return m[-1].lower() == "correct"
    return None


def vote(verdicts, mode):
    """Свести несколько вердиктов в один."""
    verdicts = [v for v in verdicts if v is not None]
    if not verdicts:
        return None
    if mode == "unanimous":
        return all(verdicts)
    # majority: при ничьей — консервативно INCORRECT
    return sum(verdicts) * 2 > len(verdicts)


def judge_one(client, model, row, samples, temperature, voting):
    """Опросить судью `samples` раз про один пример.

    Возвращает кортеж (вердикт, текст_промпта, полная_генерация):
      - вердикт: True / False / None;
      - текст_промпта: то, что отправили модели (system + user);
      - полная_генерация: все ответы судьи (при samples>1 — склеены).
    """
    cand = (row.get("lean4_prediction") or "").strip()
    header = (row.get("lean4_src_header") or "").strip()
    header_block = (
        f"\n# Lean 4 header (imports / opens, shared by both)\n"
        f"```lean\n{header}\n```\n" if header else ""
    )
    user_content = USER_TEMPLATE.format(
        nl=(row.get("nl_statement") or "").strip(),
        header_block=header_block,
        ref=(row.get("lean4_formalization") or "").strip(),
        cand=cand
    )
    prompt_text = f"[SYSTEM]\n{SYSTEM_PROMPT}\n[USER]\n{user_content}"

    if not cand:
        return False, prompt_text, "<empty prediction — API call skipped>"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    verdicts, outputs = [], []
    for k in range(samples):
        resp = client.chat.completions.create(
            model=model, messages=messages,
            temperature=temperature, max_tokens=8192)
        out = resp.choices[0].message.content or ""
        verdicts.append(parse_verdict(out))
        outputs.append(out if samples == 1 else f"--- sample {k + 1} ---\n{out}")
    return vote(verdicts, voting), prompt_text, "\n\n".join(outputs)


def print_metrics(df):
    """Сравнить вердикты судьи с эталонным столбцом `correct`."""
    d = df.dropna(subset=["gold_correct", "judge_pred"])
    gold = d["gold_correct"].astype(bool)
    pred = d["judge_pred"].astype(bool)
    tp = (gold & pred).sum()
    fp = (~gold & pred).sum()
    tn = (~gold & ~pred).sum()
    fn = (gold & ~pred).sum()
    n = tp + fp + tn + fn
    acc = (tp + tn) / n if n else float("nan")
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else float("nan")
    print("\n=== Качество судьи vs эталон `correct` ===")
    print(f"оценено примеров: {n}")
    print(f"TP={tp} FP={fp} TN={tn} FN={fn}")
    print(f"accuracy={acc:.4f}  precision={prec:.4f}  "
          f"recall={rec:.4f}  F1={f1:.4f}")


# --- индикатор прогресса --------------------------------------------------- #
def progress_iter(futures_as_completed, total):
    """Обёртка с прогресс-баром tqdm, либо простой счётчик, если tqdm нет."""
    try:
        from tqdm import tqdm
        yield from tqdm(futures_as_completed, total=total, desc="judging")
    except ImportError:
        done = 0
        for item in futures_as_completed:
            done += 1
            if done % 25 == 0 or done == total:
                print(f"  {done}/{total}", file=sys.stderr)
            yield item


# --- main ------------------------------------------------------------------ #
def main():
    parser = argparse.ArgumentParser(description="LLM-судья для ProofNetVerif.")
    parser.add_argument("--model", required=True, help="id модели в OpenRouter")
    parser.add_argument("--samples", type=int, default=1,
                        help="сколько раз опрашивать судью на один пример")
    parser.add_argument("--voting", choices=["majority", "unanimous"],
                        default="majority", help="как усреднять вердикты")
    parser.add_argument("--output", default="predictions.csv",
                        help="путь для CSV с предсказаниями")
    parser.add_argument("--workers", type=int, default=8,
                        help="число параллельных потоков (запросов к API)")
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit("Задайте переменную окружения OPENROUTER_API_KEY.")
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    # температура: 0 для одного прогона, иначе включаем разнообразие для голосования
    temperature = 0.0 if args.samples == 1 else 0.6

    print(f"Загружаю {DATASET} [test]…")
    rows = list(load_dataset(DATASET, split="test"))
    print(f"Примеров: {len(rows)}. Прогоняю '{args.model}' в {args.workers} потоков…")

    # judge_one потокобезопасна (только читает аргументы), поэтому запускаем пул.
    def work(i, row):
        try:
            pred, prompt, output = judge_one(
                client, args.model, row, args.samples, temperature, args.voting)
        except Exception as exc:               # noqa: BLE001
            print(f"  [{i}] ошибка API: {exc}", file=sys.stderr)
            pred, prompt, output = None, "", f"<error: {exc}>"
        return i, pred, prompt, output

    results = [None] * len(rows)               # (pred, prompt, output) по индексу — порядок сохраняется
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(work, i, row) for i, row in enumerate(rows)]
        for fut in progress_iter(as_completed(futures), len(futures)):
            i, pred, prompt, output = fut.result()
            results[i] = (pred, prompt, output)

    df = pd.DataFrame([{
        "nl_statement": row.get("nl_statement"),
        "lean4_formalization": row.get("lean4_formalization"),
        "lean4_prediction": row.get("lean4_prediction"),
        "gold_correct": to_bool(row.get("correct")),
        "judge_pred": results[i][0],
        "prompt": results[i][1],
        "raw_output": results[i][2],
    } for i, row in enumerate(rows)])

    df.to_csv(args.output, index=False)
    print(f"\nПредсказания сохранены в {args.output}")
    print_metrics(df)


if __name__ == "__main__":
    main()
