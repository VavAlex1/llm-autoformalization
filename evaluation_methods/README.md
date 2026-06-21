# evaluation_methods

Методы автоматического оценивания формализации из **главы 3** диплома ([`../docs/thesis.pdf`](../docs/thesis.pdf)).

Цель - сравнить автоматические метрики с экспертной разметкой и выбрать ту, что пригодна как метрика качества и как награда для обучения с подкреплением.

## Скрипты

- `eval_typecheck.py` — проверка типов: доля формализаций, компилирующихся в Lean 4.
- `eval_beql.py` — базовый BEq (расширенное равенство по определению).
- `eval_beq_extended.py` — **расширенный BEq**, предложенный в работе (двусторонняя проверка + расширенный набор преобразований + нормализация).
- `eval_llm_judge.py` — судья на основе LLM (через OpenRouter), с голосованием `--voting majority|unanimous`.
- `lean_utils.py` — общий слой работы с Lean (через `lean-interact`): конфиг проекта с Mathlib, нормализация теоремы, прогон метрики по датасету.

## Данные

По умолчанию используется размеченный бенчмарк [`AlexVav01/autoformalization-bench`](https://huggingface.co/datasets/AlexVav01/autoformalization-bench), где колонка `correct` — экспертная метка (0/1). Относительно неё считаются precision / recall / F1 и accuracy.

Эти же скрипты применяются и для замера формализатора (глава 4) — тогда `--dataset` указывает на CSV с предсказаниями модели.

## Запуск

```bash
pip install -r requirements.txt

# символические методы (нужен Lean 4 + Mathlib)
python eval_typecheck.py     --dataset AlexVav01/autoformalization-bench --split test
python eval_beql.py          --dataset AlexVav01/autoformalization-bench --split test
python eval_beq_extended.py  --dataset AlexVav01/autoformalization-bench --split test

# судья на основе БЯМ (нужен OPENROUTER_API_KEY)
python eval_llm_judge.py --model deepseek/deepseek-chat --voting majority
```

## Результаты

`eval_methods_final.csv` — предсказания всех методов на бенчмарке (соответствует таблицам 1–4 диплома).
