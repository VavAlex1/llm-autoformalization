"""
Оценка качества автоматического метода `typecheck` как предиктора корректности
формализации.

Идея: метод считает формализацию `lean4_prediction` «корректной», если она
типизируется в Lean 4 (компилируется с заголовком `lean4_src_header` без ошибок,
тело доказательства = `sorry`). Полученные бинарные предсказания сравниваются с
человеческой разметкой `correct` из ProofNetVerif — и так измеряется,
насколько хорошо typecheck приближает человеческую оценку.

Запуск:
    python eval_typecheck.py                      # ProofNetVerif (test), 4 процесса
    python eval_typecheck.py --num-processes 8
    python eval_typecheck.py --dataset path/to/data.csv --output result.csv

Зависимости: lean-interact, datasets, scikit-learn, pandas, tqdm.
"""

from __future__ import annotations

import argparse
import functools
import os

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from lean_utils import (
    DEFAULT_LEAN_VERSION,
    DEFAULT_TIMEOUT,
    is_well_typed,
    make_lean_config,
    map_metric,
)

DEFAULT_DATASET = "AlexVav01/autoformalization-bench"


# --------------------------------------------------------------------------- #
# Метрика
# --------------------------------------------------------------------------- #
def typecheck_metric(
    record: dict,
    server,
    prediction_column: str,
    header_column: str,
    timeout: int,
) -> bool:
    """Метрика для `map_metric`: типизируется ли предсказанная формализация."""
    return is_well_typed(
        record[prediction_column],
        record[header_column],
        server,
        timeout=timeout,
    )


# --------------------------------------------------------------------------- #
# Загрузка датасета (HuggingFace id или локальный csv/jsonl)
# --------------------------------------------------------------------------- #
def load_records(dataset: str, split: str, required_columns: list[str]) -> list[dict]:
    if os.path.exists(dataset):
        if dataset.endswith(".csv"):
            df = pd.read_csv(dataset)
        elif dataset.endswith(".jsonl"):
            df = pd.read_json(dataset, lines=True)
        elif dataset.endswith(".json"):
            df = pd.read_json(dataset)
        else:
            raise ValueError(
                f"Неизвестный формат локального файла: {dataset} (ожидался .csv/.jsonl/.json)"
            )
        records = df.to_dict("records")
    else:
        # трактуем как идентификатор датасета на HuggingFace Hub
        from datasets import load_dataset

        ds = load_dataset(dataset, split=split)
        records = [dict(row) for row in ds]

    if not records:
        raise ValueError("Датасет пуст.")

    missing = [c for c in required_columns if c not in records[0]]
    if missing:
        raise KeyError(
            f"В датасете нет нужных колонок: {missing}. "
            f"Доступные: {sorted(records[0].keys())}"
        )
    return records


# --------------------------------------------------------------------------- #
# Отчёт
# --------------------------------------------------------------------------- #
def report(y_true: list[int], y_pred: list[int]) -> None:
    n = len(y_true)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    print("\n" + "=" * 48)
    print("typecheck как предиктор человеческой метки `correct`")
    print("=" * 48)
    print(f"Примеров:               {n}")
    print(f"Доля проходящих typecheck: {sum(y_pred) / n:.2%}")
    print(f"Доля корректных (人):      {sum(y_true) / n:.2%}")
    print("-" * 48)
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.2%}")
    print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.2%}")
    print(f"Recall:    {recall_score(y_true, y_pred, zero_division=0):.2%}")
    print(f"F1:        {f1_score(y_true, y_pred, zero_division=0):.2%}")
    print("-" * 48)
    print("Матрица ошибок (строки — истина, столбцы — typecheck):")
    print(f"                pred=0   pred=1")
    print(f"  true=0 (incor) {tn:>6}   {fp:>6}")
    print(f"  true=1 (corr)  {fn:>6}   {tp:>6}")
    print("=" * 48)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Typecheck-оценка формализаций в Lean 4.")
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET,
        help="HuggingFace id или путь к локальному .csv/.jsonl (по умолчанию ProofNetVerif).",
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=4,
        help="Число параллельных Lean-серверов (ограничено объёмом RAM).",
    )
    # ниже — необязательные параметры с разумными значениями по умолчанию
    parser.add_argument("--split", type=str, default="test", help="Сплит HF-датасета.")
    parser.add_argument("--prediction-column", type=str, default="lean4_prediction")
    parser.add_argument("--header-column", type=str, default="lean4_src_header")
    parser.add_argument("--label-column", type=str, default="correct")
    parser.add_argument("--lean-version", type=str, default=DEFAULT_LEAN_VERSION)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Куда сохранить результаты по примерам (csv). По умолчанию не сохраняется.",
    )
    args = parser.parse_args()

    # 1. данные
    records = load_records(
        args.dataset,
        args.split,
        required_columns=[args.prediction_column, args.header_column, args.label_column],
    )
    print(f"Загружено {len(records)} примеров из {args.dataset!r}")

    # 2. конфиг Lean (тяжёлая сборка проекта происходит здесь, один раз)
    print(f"Готовим Lean {args.lean_version} + Mathlib (первый запуск долгий)...")
    config = make_lean_config(lean_version=args.lean_version, verbose=True)

    # 3. прогон typecheck по всем примерам
    metric = functools.partial(
        typecheck_metric,
        prediction_column=args.prediction_column,
        header_column=args.header_column,
        timeout=args.timeout,
    )
    print(f"Запускаем typecheck в {args.num_processes} процессов...")
    preds = map_metric(
        records,
        metric,
        config,
        num_processes=args.num_processes,
        desc="typecheck",
    )

    # 4. метрики против человеческой разметки
    y_pred = [int(bool(p)) for p in preds]
    y_true = [int(bool(r[args.label_column])) for r in records]
    report(y_true, y_pred)

    # 5. (опционально) сохранить разбор по примерам
    if args.output:
        df = pd.DataFrame(records)
        df["typecheck"] = y_pred
        df["correct_label"] = y_true
        df.to_csv(args.output, index=False)
        print(f"Результаты по примерам сохранены в {args.output}")


if __name__ == "__main__":
    main()
