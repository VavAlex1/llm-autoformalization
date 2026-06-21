# training_methods

Сборка обучающего корпуса и дообучение модели-формализатора из **раздела 4.4** диплома ([`../docs/thesis.pdf`](../docs/thesis.pdf)).

Подход — один проход обучения с учителем (SFT) на тщательно очищенном корпусе, без многоступенчатых экспертных итераций. Основной акцент на чистоте данных.

## Скрипты

- `build_training_dataset.py` — собирает единый корпус из нескольких источников и приводит их к общей схеме (`nl_statement`, `lean4_src_header`, `lean4_formalization`, ...). Источники: FormalMATH-All, Lean-Workbook, ProofNet, miniF2F, CombiBench. Концовка каждой формализации нормализуется к `:= by sorry`.
- `train_sft.py` — дообучение с учителем (TRL `SFTTrainer`, лосс только на ответе). Раз в `--eval-steps` шагов считает eval loss и расширенный BEq на валидации.

Очистка корпуса по семантической корректности (BEq + судьи) описана в разделе 4.4; итоговый корпус опубликован как [`AlexVav01/autoformalization-clean`](https://huggingface.co/datasets/AlexVav01/autoformalization-clean).

## Запуск

```bash
pip install -r requirements.txt

# 1. собрать единый корпус
python build_training_dataset.py

# 2. дообучить (базовая модель — Qwen2.5-Coder-7B-Instruct)
python train_sft.py --dataset AlexVav01/autoformalization-clean --output-dir ./ckpt
```

Ключевые гиперпараметры (значения по умолчанию): 2 эпохи, lr `3e-5`, эффективный batch 8 (`--per-device-batch-size 2 × --grad-accum 4`), warmup `0.03`, AdamW, косинусное затухание. Логирование в W&B — флагом `--wandb`. Сетап для обучения состоял лишь из одной карты H100.

## Результат

Обученные веса: [`AlexVav01/lean4-autoformalizer-sft`](https://huggingface.co/AlexVav01/lean4-autoformalizer-sft). Динамика обучения и финальные метрики — рисунки 10–11 и таблица 8 диплома.

## Данные
Результаты прогона обученной модели на тестовом датасете хранятся в `autoformalization_sft.csv`
