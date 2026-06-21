# autoformalization_methods

Запуск формализаторов и оценка существующих решений из **раздела 4.3** диплома ([`../docs/thesis.pdf`](../docs/thesis.pdf)).

Сравниваются два способа решения задачи: крупная модель общего назначения без дообучения и компактная специализированная модель.

## Скрипты

- `run_formalization_api.py` — формализатор через **OpenRouter** (модели общего назначения, напр. DeepSeek-V4-Pro). Флаг `--few-shot` подмешивает 8 примеров в промпт. Нужен `OPENROUTER_API_KEY`.
- `run_formalization_local.py` — локальный инференс через **vLLM**. Режим `--mode finetuned` — для обученной модели в рамках данной работы (сама генерирует хэдер и теорему), `--mode kimina` - поведение `Kimina-Autoformalizer-7B`.

Оба пишут предсказание в колонку `lean4_prediction` (теорема без хэдера; хэдер хранится отдельно), готовую для скриптов из [`../evaluation_methods`](../evaluation_methods).

## Запуск

```bash
pip install openai datasets pandas vllm transformers

# модель общего назначения через API (few-shot)
export OPENROUTER_API_KEY=...
python run_formalization_api.py --model deepseek/deepseek-chat --few-shot --output preds_deepseek.csv

# локальная модель через vLLM
python run_formalization_local.py --model AlexVav01/lean4-autoformalizer-sft --mode finetuned --output preds_ours.csv
python run_formalization_local.py --model AI-MO/Kimina-Autoformalizer-7B   --mode kimina    --output preds_kimina.csv
```

Оценка полученных CSV — расширенным BEq и typecheck из [`../evaluation_methods`](../evaluation_methods).

## Данные

`autoformalization_baselines.csv` — предсказания и метрики бейзлайнов (DeepSeek zero-/few-shot, Kimina), соответствует таблицам 5–7 диплома.
