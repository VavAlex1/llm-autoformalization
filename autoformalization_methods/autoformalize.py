#!/usr/bin/env python3
"""
Автоформализация на датасете FormalMath-formalization через OpenRouter.

Скрипт прогоняет выбранную модель (через OpenRouter) как АВТОФОРМАЛИЗАТОР: для
каждого примера он переводит формулировку на естественном языке
(`nl_statement`) в Lean 4-стейтмент. Хэдер (`lean4_src_header`) уже лежит в
датасете отдельной колонкой и подаётся модели как контекст «что уже в области
видимости»; от модели требуется только сам стейтмент БЕЗ хэдера.

На выходе — CSV со всеми колонками исходного датасета плюс отдельная колонка
`lean4_prediction` (новая формализация от модели, без хэдеров). Формат совпадает
с тем, что ожидает LLM-судья (`nl_statement`, `lean4_formalization`,
`lean4_prediction`), так что таблицу можно сразу подавать на оценивание.

Запуск:
    pip install openai datasets pandas
    export OPENROUTER_API_KEY=...
    python autoformalize.py --model "google/gemini-2.0-flash-001"
    python autoformalize.py --model "..." --attempts 4          # pass@k: 4 строки на пример
    python autoformalize.py --model "..." --workers 16 --limit 20

Датасет: AlexVav01/FormalMath-formalization (split "train", 425 примеров).
Колонки: id, nl_statement, lean4_src_header, lean4_formalization.
"""

import argparse
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from datasets import load_dataset
from openai import OpenAI

DATASET = "AlexVav01/FormalMath-formalization"

# Имя теоремы фиксированное для всех примеров (как в Kimina-Autoformalizer).
THEOREM_NAME = "my_favorite_theorem"

# --- Промпт автоформализатора (на английском — обычно даёт качество выше) ---- #
SYSTEM_PROMPT = """\
You are an expert in Lean 4 and Mathlib, specializing in AUTOFORMALIZATION:
translating natural-language mathematical statements into precise, compilable
Lean 4 theorem statements.

You will be given: (1) a mathematical problem/statement in natural language,
(2) the Lean 4 header (imports and `open` directives) that is ALREADY in scope,
and (3) the theorem name to use.

Your task: produce a SINGLE Lean 4 *statement* (the theorem signature and its
conclusion) that faithfully and EXACTLY captures the meaning of the
natural-language statement. Do NOT prove it — end the statement with
`:= by sorry`.

Guidelines:
- Output the STATEMENT ONLY. The given header is already in scope, so do NOT
  repeat any `import` or `open` lines — start directly with `theorem <name> …`.
- Use the provided theorem name.
- Assume exactly the given header is available (e.g. if `open Real` is in scope,
  you may write `sqrt` instead of `Real.sqrt`). Do not rely on anything the
  header does not provide.
- Capture the COMPLETE meaning: all hypotheses, quantifiers, and the exact
  conclusion. Do not add, drop, strengthen, or weaken any condition.
- Mind Lean 4 defaults and common pitfalls: ℕ starts at 0; Nat subtraction
  truncates at 0; `3 / 2 = 1` for naturals; integer vs. real division; pick the
  right types (ℕ, ℤ, ℚ, ℝ, ℂ) for the quantities involved.
- Be careful with parentheses in quantified / propositional formulas so the
  logic matches the statement exactly.
- For "find / compute / determine X" problems where the expected answer is
  given, formalize it as a statement asserting that the quantity equals that
  answer.
- Prefer standard Mathlib definitions and notation.

Output ONLY the Lean 4 statement inside a single ```lean fenced code block, and
nothing else (no imports, no opens, no proof)."""

USER_TEMPLATE = """\
# Natural-language statement
{nl}

# Lean 4 header already in scope (do NOT repeat it)
```lean
{header}
```

# Theorem name to use
{name}

Write the Lean 4 statement (no imports/opens, no proof). Start with
`theorem {name}` and end with `:= by sorry`. Output only the ```lean code block."""


# --- Вспомогательные функции ----------------------------------------------- #
def extract_lean(text):
    """Вытащить Lean-код из ответа модели."""
    m = re.search(r"```lean4?\s*(.*?)```", text, re.S | re.I)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\s*(.*?)```", text, re.S)
    if m:
        return m.group(1).strip()
    return text.strip()


def strip_header_lines(code):
    """Убрать ведущие import/open-строки из ответа модели.

    Хэдер уже есть в датасете и не должен дублироваться в предсказании; если
    модель всё-таки его повторила — отрезаем, чтобы хранить стейтмент без хэдера.
    """
    lines = code.splitlines()
    i = 0
    while i < len(lines) and (
            not lines[i].strip() or re.match(r"^\s*(import|open)\b", lines[i])):
        i += 1
    return "\n".join(lines[i:]).strip()


def formalize_one(client, model, row, attempts, temperature, max_tokens):
    """Сгенерировать `attempts` формализаций одного примера.

    Возвращает (preds, prompt_text, raws):
      - preds: список Lean-кандидатов (только стейтмент, без хэдера);
      - prompt_text: то, что отправили модели (system + user);
      - raws: список сырых ответов модели.
    """
    nl = (row.get("nl_statement") or "").strip()
    header = (row.get("lean4_src_header") or "").strip()
    name = THEOREM_NAME

    user_content = USER_TEMPLATE.format(nl=nl, header=header, name=name)
    prompt_text = f"[SYSTEM]\n{SYSTEM_PROMPT}\n[USER]\n{user_content}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    preds, raws = [], []
    for _ in range(attempts):
        resp = client.chat.completions.create(
            model=model, messages=messages,
            temperature=temperature, max_tokens=max_tokens,
            extra_body={"reasoning": {"effort": "high"}}
        )
        out = resp.choices[0].message.content or ""
        preds.append(strip_header_lines(extract_lean(out)))
        raws.append(out)
    return preds, prompt_text, raws


def print_summary(df):
    """Короткая сводка по сгенерированным предсказаниям."""
    n = len(df)
    nonempty = df["lean4_prediction"].fillna("").str.strip().ne("").sum()
    has_sorry = df["lean4_prediction"].fillna("").str.contains("sorry").sum()
    errors = df["raw_output"].fillna("").str.startswith("<error").sum()
    print("\n=== Сводка по автоформализации ===")
    print(f"строк всего:           {n}")
    print(f"непустых предсказаний:  {nonempty}")
    print(f"содержат `sorry`:       {has_sorry}")
    print(f"ошибок API:             {errors}")


# --- индикатор прогресса --------------------------------------------------- #
def progress_iter(futures_as_completed, total):
    """Обёртка с прогресс-баром tqdm, либо простой счётчик, если tqdm нет."""
    try:
        from tqdm import tqdm
        yield from tqdm(futures_as_completed, total=total, desc="formalizing")
    except ImportError:
        done = 0
        for item in futures_as_completed:
            done += 1
            if done % 25 == 0 or done == total:
                print(f"  {done}/{total}", file=sys.stderr)
            yield item


# --- main ------------------------------------------------------------------ #
def main():
    parser = argparse.ArgumentParser(
        description="Автоформализатор (OpenRouter) для FormalMath-formalization.")
    parser.add_argument("--model", required=True, help="id модели в OpenRouter")
    parser.add_argument("--split", default="train", help="split датасета")
    parser.add_argument("--attempts", type=int, default=1,
                        help="сколько кандидатов генерировать на пример "
                             "(>1 => pass@k: одна строка на кандидат)")
    parser.add_argument("--temperature", type=float, default=None,
                        help="температура; по умолчанию 0.0 при attempts=1, "
                             "иначе 0.7")
    parser.add_argument("--max-tokens", type=int, default=8192,
                        help="max_tokens на запрос")
    parser.add_argument("--output", default="formalizations.csv",
                        help="путь для CSV с предсказаниями")
    parser.add_argument("--workers", type=int, default=8,
                        help="число параллельных потоков (запросов к API)")
    parser.add_argument("--limit", type=int, default=None,
                        help="ограничить число примеров (для быстрых прогонов)")
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit("Задайте переменную окружения OPENROUTER_API_KEY.")
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    # температура: 0 для одиночной генерации, иначе включаем разнообразие
    temperature = args.temperature
    if temperature is None:
        temperature = 0.0 if args.attempts == 1 else 0.7

    print(f"Загружаю {DATASET} [{args.split}]…")
    rows = list(load_dataset(DATASET, split=args.split))
    if args.limit is not None:
        rows = rows[:args.limit]
    original_cols = list(rows[0].keys()) if rows else []
    print(f"Примеров: {len(rows)}. Прогоняю '{args.model}' "
          f"в {args.workers} потоков (attempts={args.attempts}, "
          f"temperature={temperature})…")

    # formalize_one потокобезопасна (только читает аргументы) — запускаем пул.
    def work(i, row):
        try:
            preds, prompt, raws = formalize_one(
                client, args.model, row, args.attempts, temperature,
                args.max_tokens)
        except Exception as exc:               # noqa: BLE001
            print(f"  [{i}] ошибка API: {exc}", file=sys.stderr)
            preds = [""] * args.attempts
            prompt = ""
            raws = [f"<error: {exc}>"] * args.attempts
        return i, preds, prompt, raws

    # (preds, prompt, raws) по индексу — порядок примеров сохраняется
    results = [None] * len(rows)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(work, i, row) for i, row in enumerate(rows)]
        for fut in progress_iter(as_completed(futures), len(futures)):
            i, preds, prompt, raws = fut.result()
            results[i] = (preds, prompt, raws)

    records = []
    for i, row in enumerate(rows):
        preds, prompt, raws = results[i]
        for k in range(len(preds)):
            rec = {col: row.get(col) for col in original_cols}  # все колонки датасета
            rec["lean4_prediction"] = preds[k]                  # новая формализация (без хэдера)
            rec["attempt"] = k + 1
            rec["prompt"] = prompt
            rec["raw_output"] = raws[k]
            records.append(rec)

    df = pd.DataFrame(records)
    # порядок колонок: сначала исходные, затем добавленные
    df = df[original_cols + ["lean4_prediction", "attempt", "prompt", "raw_output"]]
    df.to_csv(args.output, index=False)
    print(f"\nФормализации сохранены в {args.output}")
    print_summary(df)


if __name__ == "__main__":
    main()
