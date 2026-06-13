"""
Вспомогательные функции для взаимодействия с Lean 4 через LeanInteract.

Модуль намеренно не привязан к конкретной метрике: здесь только «кирпичи»,
которые переиспользуются и для typecheck, и для BEq / BEq+ / прочего:

* `make_lean_config`      — собрать конфиг временного проекта с Mathlib;
* `run_command`           — безопасно выполнить команду в REPL (ловит таймауты/краши);
* `build_theorem`         — нормализовать формализацию (переименовать теорему, подставить тело);
* `is_well_typed`         — проверить, что формализация (утверждение) корректно типизируется;
* `map_metric`            — прогнать произвольную метрику по датасету в несколько процессов.

Сама метрика — это любая функция `(record: dict, server) -> bool`. Скрипты-обёртки
(например, `eval_typecheck.py`) определяют свою метрику и передают её в `map_metric`.
"""

from __future__ import annotations

import json
import multiprocessing as mp
from typing import Any, Callable, Sequence

from tqdm import tqdm

from lean_interact import AutoLeanServer, Command, LeanREPLConfig
from lean_interact.interface import CommandResponse, LeanError

# `TempRequireProject` и `clean_last_theorem_string` в разных версиях LeanInteract
# лежат в чуть разных местах — поддерживаем оба варианта импорта.
try:  # >= 0.9
    from lean_interact.project import TempRequireProject
except ImportError:  # старые версии
    from lean_interact import TempRequireProject  # type: ignore

from lean_interact.utils import clean_last_theorem_string


# Lean/Mathlib версия влияет на результат typecheck (доступность лемм, синтаксис).
# Значение по умолчанию совпадает с тем, что использовалось в исходных скриптах.
DEFAULT_LEAN_VERSION = "v4.19.0"
DEFAULT_TIMEOUT = 60  # секунд на одну команду REPL

# Тип метрики: получает строку датасета (dict) и сервер, возвращает результат.
Metric = Callable[[dict, Any], Any]


# --------------------------------------------------------------------------- #
# Конфигурация и низкоуровневое взаимодействие с REPL
# --------------------------------------------------------------------------- #
def make_lean_config(
    lean_version: str = DEFAULT_LEAN_VERSION,
    require: str = "mathlib",
    verbose: bool = False,
    **kwargs: Any,
) -> LeanREPLConfig:
    """
    Создать конфиг временного Lean-проекта с зависимостью (по умолчанию Mathlib).

    При первом вызове LeanInteract скачает и соберёт нужную версию Lean REPL и
    проект — это может занять заметное время. Конфиг стоит создавать один раз в
    главном процессе, а воркерам передавать уже готовый (см. `map_metric`).
    """
    return LeanREPLConfig(
        project=TempRequireProject(lean_version=lean_version, require=require),
        verbose=verbose,
        **kwargs,
    )


def run_command(
    server: Any,
    code: str,
    timeout: int = DEFAULT_TIMEOUT,
) -> CommandResponse | None:
    """
    Выполнить Lean-команду и вернуть `CommandResponse`, либо `None` при любой
    проблеме (ошибка элаборации REPL, таймаут, разрыв соединения, краш сервера).

    `None` означает «результат недоступен» и трактуется вызывающим кодом как
    «непригодно / не типизируется».
    """
    try:
        response = server.run(Command(cmd=code), timeout=timeout)
    except (TimeoutError, ConnectionAbortedError, json.JSONDecodeError, EOFError):
        return None
    except Exception:
        # AutoLeanServer сам восстанавливается после многих сбоев, но на всякий
        # случай не даём упасть всему прогону из-за одного проблемного примера.
        return None

    if isinstance(response, LeanError):
        return None
    return response


# --------------------------------------------------------------------------- #
# Работа с формализациями
# --------------------------------------------------------------------------- #
def build_theorem(
    formalization: str,
    thm_name: str = "theorem_to_check",
    add_sorry: bool = True,
) -> str:
    """
    Привести формализацию к каноничному виду: переименовать последнюю теорему в
    `thm_name` и заменить тело доказательства (по умолчанию на `sorry`).

    Бросает `ValueError`, если строку не удалось разобрать как теорему — это
    обёртка над `lean_interact.utils.clean_last_theorem_string`.
    """
    return clean_last_theorem_string(formalization, thm_name, add_sorry=add_sorry)


def is_well_typed(
    formalization: str,
    src_header: str,
    server: Any,
    timeout: int = DEFAULT_TIMEOUT,
    allow_sorry: bool = True,
) -> bool:
    """
    Typecheck одной формализации (утверждения).

    Логика:
      1. нормализуем теорему и подставляем `sorry` вместо доказательства;
      2. дописываем заголовок (`import Mathlib`, `open ...`) и прогоняем через REPL;
      3. считаем формализацию корректной, если нет ошибок элаборации
         (предупреждение про `sorry` допускается при `allow_sorry=True`).

    Любая непарсящаяся / некомпилируемая формализация даёт `False`.
    """
    if not formalization or not formalization.strip():
        return False

    try:
        theorem = build_theorem(formalization, thm_name="theorem_to_check", add_sorry=True)
    except ValueError:
        # даже не разобрали как теорему -> точно не типизируется
        return False

    code = (src_header or "").strip() + "\n\n" + theorem
    response = run_command(server, code, timeout=timeout)
    if response is None:
        return False

    return response.lean_code_is_valid(allow_sorry=allow_sorry)


# --------------------------------------------------------------------------- #
# Параллельный прогон метрики по датасету
# --------------------------------------------------------------------------- #
_WORKER_SERVER: Any = None
_WORKER_METRIC: Metric | None = None


def _init_worker(config: LeanREPLConfig, metric: Metric) -> None:
    """Инициализация воркера: поднимаем отдельный Lean-сервер на процесс."""
    global _WORKER_SERVER, _WORKER_METRIC
    _WORKER_SERVER = AutoLeanServer(config)
    _WORKER_METRIC = metric


def _run_worker(record: dict) -> Any:
    assert _WORKER_METRIC is not None and _WORKER_SERVER is not None
    return _WORKER_METRIC(record, _WORKER_SERVER)


def map_metric(
    records: Sequence[dict],
    metric: Metric,
    config: LeanREPLConfig,
    num_processes: int = 4,
    desc: str = "metric",
) -> list[Any]:
    """
    Прогнать `metric` по всем строкам `records` в `num_processes` процессов.

    Каждый процесс держит собственный `AutoLeanServer` (один REPL = один экземпляр
    Mathlib в памяти, поэтому число процессов ограничено объёмом RAM).
    Порядок результатов соответствует порядку `records`.

    `metric` должна быть функцией верхнего уровня (picklable), например
    `functools.partial(my_metric, prediction_column="lean4_prediction")`.
    """
    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=num_processes,
        initializer=_init_worker,
        initargs=(config, metric),
    ) as pool:
        return list(
            tqdm(
                pool.imap(_run_worker, records),
                total=len(records),
                desc=desc,
            )
        )
