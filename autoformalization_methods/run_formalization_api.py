#!/usr/bin/env python3
"""
Автоформализация на датасете FormalMath-formalization через OpenRouter.

Скрипт прогоняет выбранную модель (через OpenRouter) как АВТОФОРМАЛИЗАТОР: для
каждого примера он переводит формулировку на естественном языке
(`nl_statement`) в Lean 4-стейтмент. Хэдер (`lean4_src_header`) уже лежит в
датасете отдельной колонкой и подаётся модели как контекст «что уже в области
видимости»; от модели требуется только сам стейтмент БЕЗ хэдера.

Датасет: AlexVav01/FormalMath-formalization (split "train", 425 примеров).
Колонки: id, nl_statement, lean4_src_header, lean4_formalization.
"""

import argparse
import json
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

# --- Few-shot блок --------------------------------------------------------- #
# 8 примеров (присланы в готовом виде). Хранятся как ASCII-JSON, чтобы unicode
# (ℝ, ∃, 𝓝, …) не ломался внутри файла; при --few-shot блок добавляется перед
# задачей. Включается флагом --few-shot, по умолчанию — стандартный промпт.
_FEW_SHOT_JSON = '"Example 1\\n# Natural-language statement\\nSuppose that $m$ and $n$ are positive integers with $m<n$ such that the interval $[m, n)$ contains more multiples of 2021 than multiples of 2000 . Compute the maximum possible value of $n-m$.\\n Prove that the answer is: 191999\\n\\n# Lean 4 header already in scope (do NOT repeat it)\\n```lean\\nimport Mathlib\\n```\\n\\n# Theorem name to use\\nolymid-ref-base_5848\\n\\n# Formalization\\n```lean\\ntheorem olymid_ref_base_5848 :\\n    IsGreatest {k | \\u2203 m n : \\u2115, 0 < m \\u2227 0 < n \\u2227 m < n \\u2227 (n - m) = k \\u2227\\n      {x | m \\u2264 x \\u2227 x < n \\u2227 2021 \\u2223 x}.ncard > {x | m \\u2264 x \\u2227 x < n \\u2227 2000 \\u2223 x}.ncard} 191999 := by\\n```\\nExample 2\\n# Natural-language statement\\nConsider the integral \\\\par \\\\begin{equation} I(x) = \\\\int_{0.7}^{0.8} (1.3 t^{5} - 2.7 \\\\sin{\\\\left(t \\\\right)}) e^{- x (2.6 t + 0.8)} \\\\, dt \\\\end{equation} \\\\par Develop an analytical formula for $I(x)$ that is accurate as $x \\\\to \\\\infty$.\\n Prove that the answer is: \\\\boxed{I(x) \\\\approx - \\\\frac{0.58 e^{- 2.62 x}}{x}}\\n\\n# Lean 4 header already in scope (do NOT repeat it)\\n```lean\\nimport Mathlib\\n\\nopen Real Filter Function\\nopen scoped Topology\\n```\\n\\n# Theorem name to use\\nhardmath_524\\n\\n# Formalization\\n```lean\\ntheorem hardmath_524 (I : \\u211d \\u2192 \\u211d)\\n  (hI : I = \\u03bb x => \\u222b t in (0.7)..0.8, (1.3 * t^5 - 2.7 * sin t) * exp (-x * (2.6 * t + 0.8))) :\\n  Tendsto (\\u03bb x => I x / (-0.58 * exp (-2.62 * x) / x)) atTop (\\ud835\\udcdd 1) := by\\n```\\nExample 3\\n# Natural-language statement\\nGiven positive real numbers $x, y$, and $z$ that satisfy the following system of equations:\\n\\n$$\\n\\\\begin{aligned}\\nx^{2}+y^{2}+x y & =1, \\\\\\\\\\ny^{2}+z^{2}+y z & =4, \\\\\\\\\\nz^{2}+x^{2}+z x & =5,\\n\\\\end{aligned}\\n$$\\n\\nfind $x+y+z$.\\n Prove that the answer is: \\\\sqrt{5+2 \\\\sqrt{3}}\\n\\n# Lean 4 header already in scope (do NOT repeat it)\\n```lean\\nimport Mathlib\\n\\nopen Real\\n```\\n\\n# Theorem name to use\\nolymid-ref-base_4102\\n\\n# Formalization\\n```lean\\ntheorem olymid_ref_base_4102 (x y z : \\u211d) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)\\n    (h\\u2081 : x^2 + y^2 + x * y = 1) (h\\u2082 : y^2 + z^2 + y * z = 4) (h\\u2083 : z^2 + x^2 + z * x = 5) :\\n    x + y + z = sqrt (5 + 2 * sqrt 3) := by\\n```\\nExample 4\\n# Natural-language statement\\nFour positive integers $x, y, z$, and $t$ satisfy the relations  $$ x y-z t=x+y=z+t . $$  Is it possible that both $x y$ and $z t$ are perfect squares?  (Russia)  Answer: No.\\n\\n# Lean 4 header already in scope (do NOT repeat it)\\n```lean\\nimport Mathlib\\n```\\n\\n# Theorem name to use\\nolymid-ref-base_8752\\n\\n# Formalization\\n```lean\\ntheorem olymid_ref_base_8752 : \\u00ac\\u2203 (x y z t : \\u2124), x > 0 \\u2227 y > 0 \\u2227 z > 0 \\u2227 t > 0 \\u2227 x * y - z * t = x + y \\u2227 x + y = z + t \\u2227 \\u2203 m n, m^2 = x * y \\u2227 n^2 = z * t := by\\n```\\nExample 5\\n# Natural-language statement\\nIn $\\\\triangle ABC$, prove that:\\n(1) $b\\\\cos C + c\\\\cos B = a$;\\n(2) $\\\\frac{a^{2}-b^{2}}{c^{2}}=\\\\frac{\\\\sin(A - B)}{\\\\sin C}$.\\n\\n# Lean 4 header already in scope (do NOT repeat it)\\n```lean\\nimport Mathlib\\n\\nopen Real Set\\nopen scoped BigOperators\\n```\\n\\n# Theorem name to use\\ntheorem_proving_zh_blue_952\\n\\n# Formalization\\n```lean\\ntheorem theorem_proving_zh_blue_952 (A B C : Real) (hA : A \\u2208 Ioo 0 \\u03c0)\\n  (hB : B \\u2208 Ioo 0 \\u03c0) (hC : C \\u2208 Ioo 0 \\u03c0) (hABC : A + B + C = \\u03c0)\\n  (a b c : \\u211d) (ha : a > 0) (hb : b > 0) (hc : c > 0)\\n  (h\\u2080 : a / sin A = b / sin B) (h\\u2081 : a / sin A = c / sin C)\\n  (h\\u2082 : b / sin B = c / sin C) :\\n  b * cos C + c * cos B = a \\u2227 (a ^ 2 - b ^ 2) / c ^ 2 = sin (A - B) / sin C := by\\n```\\nExample 6\\n# Natural-language statement\\nLet $P(x)$ be a polynomial with integer coefficients that satisfies $P(17)=10$ and $P(24)=17.$ Given that $P(n)=n+3$ has two distinct integer solutions $n_1$ and $n_2,$ find the product $n_1\\\\cdot n_2.$\\n Prove that the answer is: 418\\n\\n# Lean 4 header already in scope (do NOT repeat it)\\n```lean\\nimport Mathlib\\n\\nopen Polynomial\\n```\\n\\n# Theorem name to use\\naime_all_2005_II_13\\n\\n# Formalization\\n```lean\\ntheorem aime_all_2005_II_13 {P : \\u2124[X]} (hP : P.eval 17 = 10 \\u2227 P.eval 24 = 17)\\n    (n1 n2 : \\u2124) (hn1 : P.eval n1 = n1 + 3) (hn2 : P.eval n2 = n2 + 3)\\n    (hn12 : n1 \\u2260 n2) : n1 * n2 = 418 := by\\n```\\nExample 7\\n# Natural-language statement\\nThe polynomial $f(z)=az^{2018}+bz^{2017}+cz^{2016}$ has real coefficients not exceeding $2019,$ and $f\\\\left(\\\\tfrac{1+\\\\sqrt3i}{2}\\\\right)=2015+2019\\\\sqrt3i$ . Find the remainder when $f(1)$ is divided by $1000$ .\\n Prove that the answer is: 53\\n\\n# Lean 4 header already in scope (do NOT repeat it)\\n```lean\\nimport Mathlib.Tactic\\n```\\n\\n# Theorem name to use\\naime_all_2019_II_8\\n\\n# Formalization\\n```lean\\ntheorem aime_all_2019_II_8 (a b c : \\u211d) (_ : |a| \\u2264 2019) (hb : |b| \\u2264 2019) (hc : |c| \\u2264 2019)\\n    (h : ((1 + Real.sqrt 3 * Complex.I)/2)^2018*a + ((1 + Real.sqrt 3 * Complex.I)/2)^2017*b +\\n      ((1 + Real.sqrt 3 * Complex.I)/2)^2016*c = 2015 + 2019*Real.sqrt 3 *Complex.I) :\\n    (a*1^2018 + b*1^2017 + c*1^2016) % 1000 = 53 := by\\n```\\nExample 8\\n# Natural-language statement\\nAs shown in the figure![](./images/volume14/figures/fig-c8p3.png), in quadrilateral \\\\(ABCD\\\\), the ratio of the areas of \\\\(\\\\triangle ABD\\\\), \\\\(\\\\triangle BCD\\\\), and \\\\(\\\\triangle ABC\\\\) is \\\\(3:4:1\\\\). Points \\\\(M\\\\) and \\\\(N\\\\) are on \\\\(AC\\\\) and \\\\(CD\\\\) respectively, satisfying \\\\(\\\\frac{AM}{AC}=\\\\frac{CN}{CD}\\\\), and the three points \\\\(B\\\\), \\\\(M\\\\), and \\\\(N\\\\) are collinear.\\n\\nProve that \\\\(M\\\\) and \\\\(N\\\\) are the midpoints of \\\\(AC\\\\) and \\\\(CD\\\\) respectively.\\n\\n# Lean 4 header already in scope (do NOT repeat it)\\n```lean\\nimport Mathlib\\n\\nopen Real\\nopen scoped BigOperators\\n```\\n\\n# Theorem name to use\\ntheorem_proving_zh_blue_821\\n\\n# Formalization\\n```lean\\ntheorem theorem_proving_zh_blue_821 {A B C D M N : EuclideanSpace \\u211d (Fin 2)}\\n  (h\\u2080 : (MeasureTheory.volume (convexHull \\u211d {A, B, D})).toReal =\\n    3 * (MeasureTheory.volume (convexHull \\u211d {A, B, C})).toReal)\\n  (h\\u2081 : (MeasureTheory.volume (convexHull \\u211d {B, C, D})).toReal =\\n    4 * (MeasureTheory.volume (convexHull \\u211d {A, B, C})).toReal)\\n  (h\\u2082 : (MeasureTheory.volume (convexHull \\u211d {A, B, C})).toReal \\u2260 0)\\n  (h\\u2083 : M \\u2208 segment \\u211d A C)\\n  (h\\u2084 : N \\u2208 segment \\u211d C D)\\n  (h\\u2085 : dist A M / dist A C = dist C N / dist C D)\\n  (h\\u2086 : Collinear \\u211d {B, M, N}) :\\n  M = midpoint \\u211d A C \\u2227 N = midpoint \\u211d C D := by\\n```"'  # __FEWSHOT_DATA__
FEW_SHOT_BLOCK = json.loads(_FEW_SHOT_JSON)

FEW_SHOT_INTRO = (
    "Here are worked examples of this exact task "
    "(natural-language statement + header already in scope -> Lean 4 "
    "formalization). Study them, then solve the final problem the same way.\n\n")

FEW_SHOT_TRANSITION = "\nNow formalize the following problem.\n\n"


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


def build_user_content(nl, header, name, few_shot):
    """Собрать user-сообщение: стандартное или с few-shot блоком перед задачей."""
    problem = USER_TEMPLATE.format(nl=nl, header=header, name=name)
    if not few_shot or not FEW_SHOT_BLOCK:
        return problem
    return FEW_SHOT_INTRO + FEW_SHOT_BLOCK.strip() + FEW_SHOT_TRANSITION + problem


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
    """
    Убрать ведущие import/open-строки из ответа модели.
    Хэдер уже есть в датасете и не должен дублироваться в предсказании
    """
    lines = code.splitlines()
    i = 0
    while i < len(lines) and (
            not lines[i].strip() or re.match(r"^\s*(import|open)\b", lines[i])):
        i += 1
    return "\n".join(lines[i:]).strip()


def formalize_one(client, model, row, attempts, temperature, max_tokens, few_shot):
    """
    Сгенерировать `attempts` формализаций одного примера.
    Возвращает (preds, prompt_text, raws):
      - preds: список Lean-кандидатов (только стейтмент, без хэдера);
      - prompt_text: то, что отправили модели (system + user);
      - raws: список сырых ответов модели.
    """
    nl = (row.get("nl_statement") or "").strip()
    header = (row.get("lean4_src_header") or "").strip()
    name = THEOREM_NAME

    user_content = build_user_content(nl, header, name, few_shot)
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
    n = len(df)
    nonempty = df["lean4_prediction"].fillna("").str.strip().ne("").sum()
    has_sorry = df["lean4_prediction"].fillna("").str.contains("sorry").sum()
    errors = df["raw_output"].fillna("").str.startswith("<error").sum()
    print("\n=== Сводка по автоформализации ===")
    print(f"строк всего:           {n}")
    print(f"непустых предсказаний:  {nonempty}")
    print(f"содержат `sorry`:       {has_sorry}")
    print(f"ошибок API:             {errors}")


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
    parser.add_argument("--few-shot", action="store_true",
                        help="использовать промпт с few-shot примерами "
                             "(по умолчанию — стандартный промпт)")
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
          f"temperature={temperature}, "
          f"prompt={'few-shot' if args.few_shot else 'standard'})…")

    # formalize_one потокобезопасна (только читает аргументы) — запускаем пул.
    def work(i, row):
        try:
            preds, prompt, raws = formalize_one(
                client, args.model, row, args.attempts, temperature,
                args.max_tokens, args.few_shot)
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
            rec = {col: row.get(col) for col in original_cols}
            rec["lean4_prediction"] = preds[k]
            rec["attempt"] = k + 1
            rec["prompt"] = prompt
            rec["raw_output"] = raws[k]
            records.append(rec)

    df = pd.DataFrame(records)
    df = df[original_cols + ["lean4_prediction", "attempt", "prompt", "raw_output"]]
    df.to_csv(args.output, index=False)
    print(f"\nФормализации сохранены в {args.output}")
    print_summary(df)


if __name__ == "__main__":
    main()