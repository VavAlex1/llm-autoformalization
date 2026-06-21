"""
Сборка единого тренировочного датасета для автоформализации из нескольких
источников на HuggingFace. Все датасеты приводятся к общему формату из 5 колонок:

    id                   — уникальный id сэмпла (вида "<dataset>/<orig_id>")
    nl_statement         — формулировка задачи на естественном языке
    lean4_src_header     — хэдеры формализации (import / open / set_option ...)
    lean4_formalization  — формализация БЕЗ хэдеров (theorem ... := by ...)
    dataset_name         — исходный датасет

Источники:
    SphereLab/FormalMATH-All   формализация в одной колонке -> сплитим хэдер
    internlm/Lean-Workbook     formal_statement без хэдера   -> хэдер = "import Mathlib"
    PAug/ProofNetSharp         хэдер и тело уже разделены
    AI-MO/minif2f_test         формализация в одной колонке -> сплитим хэдер
    AI-MO/CombiBench           формализация в одной колонке -> сплитим хэдер

Результат сохраняется в CSV. Далее датасе обрабатывается при помощи различных фильтраций.
"""

import re

import pandas as pd
from datasets import load_dataset


# ---------------------------------------------------------------------------
# Разделение полной формализации на хэдер и тело теоремы
# ---------------------------------------------------------------------------
# первая строка-декларация: theorem / lemma / example / def / abbrev / instance
_DECL_RE = re.compile(
    r"^\s*(theorem|lemma|example|abbrev|instance|noncomputable\s+def|def)\b"
)
_BLOCK_COMMENT_RE = re.compile(r"/-.*?-/", re.DOTALL)


def split_header(formalization: str):
    """Отделить хэдер от тела теоремы в Lean 4-формализации.

    Возвращает (header, body):
      header — всё до первой строки-декларации (import / open / set_option ...);
      body   — формулировка теоремы, начиная со строки `theorem ...`.
    Если декларация не найдена — ("", вся_строка).
    """
    lines = formalization.splitlines()
    for i, line in enumerate(lines):
        if _DECL_RE.match(line):
            header = "\n".join(lines[:i]).strip()
            body = "\n".join(lines[i:]).strip()
            return header, body
    return "", formalization.strip()


def clean_header(header: str) -> str:
    """Убрать из хэдера блочные комментарии (/-- ... -/) и пустые строки."""
    header = _BLOCK_COMMENT_RE.sub("", header)
    lines = [ln for ln in header.splitlines() if ln.strip()]
    return "\n".join(lines).strip()


def strip_doc_comment(text: str) -> str:
    """Снять обёртку Lean doc-комментария /-- ... -/ с естественной формулировки."""
    text = text.strip()
    text = text.removeprefix("/--").removeprefix("/-")
    text = text.removesuffix("-/")
    return text.strip()


def normalize_ending(formalization: str) -> str:
    """Привести концовку формализации к единому виду `:= by sorry`.

    Доказательство теоремы (всё после ПОСЛЕДНЕГО `:=`) заменяется на `by sorry`.
    Берётся именно последний `:=`, поэтому предшествующие декларации
    (например `abbrev ..._solution : ℕ := sorry` в CombiBench) не затрагиваются.
    """
    formalization = formalization.strip()
    head = formalization.rsplit(":=", 1)[0].rstrip()
    return head + " := by sorry"


# ---------------------------------------------------------------------------
# Адаптеры под конкретные датасеты: row -> (orig_id, nl, header, formalization)
# ---------------------------------------------------------------------------
def adapt_formalmath(row):
    header, body = split_header(row["autoformalization"])
    return row["theorem_names"], row["refined_statement"], header, body


def adapt_leanworkbook(row):
    header, body = split_header(row["formal_statement"])
    if not header:                      # в Lean-Workbook хэдер опущен
        header = "import Mathlib"
    return row["id"], row["natural_language_statement"], header, body


def adapt_proofnet(row):
    # хэдер и тело уже разделены в исходном датасете
    return row["id"], row["nl_statement"], row["lean4_src_header"], row["lean4_formalization"]


def adapt_minif2f(row):
    header, body = split_header(row["formal_statement"])
    nl = strip_doc_comment(row["informal_prefix"])
    return row["name"], nl, header, body


def adapt_combibench(row):
    header, body = split_header(row["formal_statement"])
    return row["theorem_name"], row["natural_language"], header, body


# (имя датасета, список сплитов, адаптер)
DATASETS = [
    ("SphereLab/FormalMATH-All", ["train"],          adapt_formalmath),
    ("internlm/Lean-Workbook",   ["train"],          adapt_leanworkbook),
    ("PAug/ProofNetSharp",       ["valid", "test"],  adapt_proofnet),
    ("AI-MO/minif2f_test",       ["train"],          adapt_minif2f),
    ("AI-MO/CombiBench",         ["test"],           adapt_combibench),
]

OUTPUT_PATH = "training_dataset.csv"


def main():
    rows = []
    for name, splits, adapt in DATASETS:
        for split in splits:
            print(f"Загружаю {name} [{split}] ...")
            dataset = load_dataset(name, split=split)
            for sample in dataset:
                orig_id, nl, header, formalization = adapt(sample)
                rows.append({
                    "id": f"{name}/{orig_id}",
                    "nl_statement": (nl or "").strip(),
                    "lean4_src_header": clean_header(header or ""),
                    "lean4_formalization": normalize_ending(formalization or ""),
                    "dataset_name": name,
                })
            print(f"  -> всего собрано: {len(rows)}")

    df = pd.DataFrame(
        rows,
        columns=["id", "nl_statement", "lean4_src_header",
                 "lean4_formalization", "dataset_name"],
    )
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nГотово. {len(df)} сэмплов сохранено в {OUTPUT_PATH}")
    print("Разбивка по источникам:")
    print(df["dataset_name"].value_counts().to_string())


if __name__ == "__main__":
    main()
