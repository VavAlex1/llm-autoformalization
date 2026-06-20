"""
Прогон модели автоформализации через vLLM на датасете.

Два режима (--mode):
  * finetuned (по умолчанию) — под дообученную модель: промпт содержит запрос
    хэдера ("Your code should start with: ```lean4 {header}```"), ответ НЕ
    префиллится, модель сама генерирует хэдер + теорему. Из ответа достаётся
    lean-блок, хэдер срезается -> в lean4_prediction идёт голая теорема.
  * kimina — исходное поведение AI-MO/Kimina-Autoformalizer-7B: хэдер и
    "theorem my_favorite_theorem " дописываются в начало ответа ассистента,
    модель продолжает с них.

В обоих случаях в поле lean4_prediction сохраняется теорема без хэдера
(header лежит отдельно в lean4_src_header) — формат, который ждут
eval_beq_plus.py / eval_typecheck.py.
"""

from __future__ import annotations

import argparse
import re

from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


SYSTEM_PROMPT = "You are an expert in mathematics and Lean 4."
THEOREM_NAME = "my_favorite_theorem"
THEOREM_PREFIX = f"theorem {THEOREM_NAME} "

USER_PROMPT_HEAD = (
    "Please autoformalize the following problem in Lean 4 with a header. "
    f"Use the following theorem names: {THEOREM_NAME}.\n\n"
)

_CODE_BLOCK_RE = re.compile(r"```(?:[Ll]ean4?)?\s*\n(.*?)```", re.DOTALL)
_KEYWORD_RE = re.compile(r"\b(theorem|lemma|def|abbrev|example|instance)\b")


# --------------------------------------------------------------------------- #
# Извлечение формализации из ответа (для finetuned-режима)
# --------------------------------------------------------------------------- #
def extract_lean(text: str) -> str:
    """Последний lean-блок (последний — на случай будущего reasoning перед кодом)."""
    blocks = _CODE_BLOCK_RE.findall(text)
    return (blocks[-1] if blocks else text).strip()


def strip_header(code: str, header: str) -> str:
    """Убираем хэдер -> остаётся голое утверждение (header хранится отдельно)."""
    code = code.strip()
    h = header.strip()
    if h and code.startswith(h):
        return code[len(h):].strip()
    m = _KEYWORD_RE.search(code)
    return code[m.start():].strip() if m else code


# --------------------------------------------------------------------------- #
# Построение промптов
# --------------------------------------------------------------------------- #
def build_prompt_finetuned(problem: str, header: str, tokenizer) -> str:
    """Как при обучении: запрос хэдера в user-турне, ответ НЕ префиллится."""
    user = (
        USER_PROMPT_HEAD
        + f"{problem}\n\n"
        + f"Your code should start with:\n```lean4\n{header}\n```\n"
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def build_prompt_kimina(problem: str, header: str, tokenizer) -> str:
    """Исходное поведение Kimina: хэдер + THEOREM_PREFIX дописаны в начало ответа."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_HEAD + problem.strip()},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return text + header.strip() + "\n\n" + THEOREM_PREFIX


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(description="vLLM-прогон автоформализатора.")
    p.add_argument("--model", required=True, help="Путь к дообученной модели или HF id.")
    p.add_argument("--dataset", default="AlexVav01/FormalMath-formalization")
    p.add_argument("--split", default="train")
    p.add_argument("--output", default="formalizations.csv")
    p.add_argument("--mode", choices=["finetuned", "kimina"], default="finetuned")
    p.add_argument("--nl-column", default="nl_statement")
    p.add_argument("--header-column", default="lean4_src_header")
    p.add_argument("--prediction-column", default="lean4_prediction")
    # сэмплинг (для одиночного предсказания удобнее greedy: --temperature 0)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-tokens", type=int, default=2048)
    args = p.parse_args()

    dataset = load_dataset(args.dataset, split=args.split)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = LLM(args.model)

    build = build_prompt_finetuned if args.mode == "finetuned" else build_prompt_kimina
    prompts = [
        build(s[args.nl_column], s[args.header_column], tokenizer) for s in dataset
    ]

    sampling = SamplingParams(
        temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens
    )
    results = model.generate(prompts, sampling_params=sampling)

    # из ответа -> голая теорема (без хэдера)
    predictions = []
    for sample, result in zip(dataset, results):
        text = result.outputs[0].text
        if args.mode == "finetuned":
            pred = strip_header(extract_lean(text), sample[args.header_column])
        else:
            pred = THEOREM_PREFIX + text
        predictions.append(pred)

    dataset = dataset.add_column(args.prediction_column, predictions)
    dataset.to_csv(args.output)
    print(f"Сохранено {len(dataset)} записей с полем "
          f"'{args.prediction_column}' в {args.output}")


if __name__ == "__main__":
    main()
