# Формализация математических теорем с помощью больших языковых моделей

Полный текст работы: [`docs/thesis.pdf`](docs/thesis.pdf).

## Два вклада

- **Расширенный BEq** — детерминированная и проверяемая метрика семантической эквивалентности формализаций (двусторонняя проверка, расширенный набор семантически сохраняющих преобразований, предварительная нормализация). Поднимает полноту базового BEq с 32,7 % до 58,6 % при почти нулевой доле ложноположительных срабатываний (раздел 3 диплома).
- **Модель-формализатор** — дообученная на очищенном по семантической корректности корпусе (~30 тыс. пар) модель на 7 млрд параметров, превосходящая по ключевой метрике BEq одно из сильнейших существующих решений (раздел 4 диплома).

## Результаты

Набор FormalMATH-Lite (*N* = 425), по одной генерации на задачу:

| Подход | Размер | typecheck, % | BEq, % |
| --- | --- | --- | --- |
| DeepSeek-V4-Pro, zero-shot | ≈1,6 трлн | 82,8 | 30,1 |
| DeepSeek-V4-Pro, few-shot (8) | ≈1,6 трлн | 88,0 | 33,4 |
| Kimina-Autoformalizer-7B | 7 млрд | 92,2 | 39,3 |
| **lean4-autoformalizer-sft** | 7 млрд | 89,9 | **42,8** |

## Артефакты на HuggingFace

|  |  |
| --- | --- |
| Веса обученной модели | [`AlexVav01/lean4-autoformalizer-sft`](https://huggingface.co/AlexVav01/lean4-autoformalizer-sft) |
| Очищенный обучающий корпус (~30 тыс. пар) | [`AlexVav01/autoformalization-clean`](https://huggingface.co/datasets/AlexVav01/autoformalization-clean) |
| Размеченный бенчмарк для оценки методов оценивания | [`AlexVav01/autoformalization-bench`](https://huggingface.co/datasets/AlexVav01/autoformalization-bench) |
| Тестовый набор формализатора (FormalMATH-Lite) | [`AlexVav01/FormalMath-formalization`](https://huggingface.co/datasets/AlexVav01/FormalMath-formalization) |

## Структура и порядок чтения

Папки соответствуют главам диплома; читать удобно в порядке оценивание → формализация → обучение.

- [`evaluation_methods/`](evaluation_methods) — методы оценивания формализации (глава 3): typecheck, базовый и расширенный BEq, судья на основе БЯМ.
- [`autoformalization_methods/`](autoformalization_methods) — запуск формализаторов и бейзлайны (раздел 4.3).
- [`training_methods/`](training_methods) — сборка обучающего корпуса и дообучение модели (раздел 4.4).
- [`docs/thesis.pdf`](docs/thesis.pdf) — полный текст работы.

## Установка

Зависимости Python указаны в `requirements.txt` внутри каждой папки (`pip install -r requirements.txt`).

Для `typecheck` и `BEq` нужен рабочий **Lean 4 + Mathlib**. Он подключается через [`lean-interact`](https://pypi.org/project/lean-interact/): временный проект с Mathlib собирается автоматически, целевая версия — `v4.19.0`.

Для скриптов, обращающихся к моделям через API (формализатор через OpenRouter и судья), задайте переменную окружения:

```bash
export OPENROUTER_API_KEY=...
```