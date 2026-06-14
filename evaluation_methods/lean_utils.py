"""
* `make_lean_config`      — собрать конфиг временного проекта с Mathlib;
* `run_command`           — безопасно выполнить команду в REPL (ловит таймауты/краши);
* `build_theorem`         — нормализовать формализацию (переименовать теорему, подставить тело);
* `is_well_typed`         — проверить, что формализация (утверждение) корректно типизируется;
* `map_metric`            — прогнать произвольную метрику по датасету в несколько процессов.
"""

from __future__ import annotations

import json
import multiprocessing as mp
from typing import Any, Callable, Sequence

from tqdm import tqdm

from lean_interact import AutoLeanServer, Command, LeanREPLConfig
from lean_interact.interface import CommandResponse, LeanError, Pos, message_intersects_code

# `TempRequireProject` и `clean_last_theorem_string` в разных версиях LeanInteract
# лежат в чуть разных местах — поддерживаем оба варианта импорта.
try:  # >= 0.9
    from lean_interact.project import TempRequireProject
except ImportError:  # старые версии
    from lean_interact import TempRequireProject  # type: ignore

from lean_interact.utils import clean_last_theorem_string, indent_code, split_conclusion


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
# BEqL: эквивалентность двух формализаций через двусторонний `exact?`
# --------------------------------------------------------------------------- #
def extract_exact_proof(lean_output: CommandResponse, proof_start_line: int | None = None) -> str | None:
    """
    Достать из ответа REPL терм, предложенный тактикой `exact?` (строка после
    "Try this:"). Возвращает None, если в области доказательства есть ошибка.
    """
    start = Pos(line=proof_start_line, column=0) if proof_start_line else None
    for message in lean_output.messages:
        if message_intersects_code(message, start, None):
            if message.severity == "error":
                return None
            if message.severity == "info" and message.data.startswith("Try this:"):
                return message.data.split("Try this:")[1].strip()
    return None


def check_proof_sub(
    server: Any,
    formal_code: str,
    formal_2_start_line: int,
    proof: str,
    timeout: int = DEFAULT_TIMEOUT,
    indent_level: int = 2,
) -> str | None:
    """
    Дописать тактику `proof` ко второй теореме (которая уже заканчивается на `:= by`)
    и проверить результат.

    Возвращает строку доказательства (для `exact?` — найденный терм), либо None,
    если доказательство не прошло / код некорректен / случился таймаут.
    Это переиспользуемый кирпич — на нём же можно строить BEq+.
    """
    prepended = "\nintros\nsymm_saturate\n"
    try:
        lean_output = server.run(
            Command(cmd=formal_code + indent_code(prepended + proof, indent_level)),
            timeout=timeout,
        )
    except (TimeoutError, ConnectionAbortedError, json.JSONDecodeError, EOFError):
        return None
    except Exception:
        return None

    if isinstance(lean_output, LeanError):
        return None

    start = Pos(line=formal_2_start_line, column=0)
    if proof == "sorry":
        # предварительная проверка: вторая теорема вообще корректно типизируется
        return proof if lean_output.lean_code_is_valid(start_pos=start) else None

    if lean_output.lean_code_is_valid(start_pos=start, allow_sorry=False):
        if proof == "exact?":
            return extract_exact_proof(lean_output, proof_start_line=formal_2_start_line)
        return proof
    return None


def beq(
    formalization_1: str,
    formalization_2: str,
    src_header: str,
    server: Any,
    timeout_per_proof: int = DEFAULT_TIMEOUT,
    verbose: bool = False,
) -> bool:
    """
    две формализации считаются эквивалентными, если каждая выводится
    из другой тактикой `exact?` (с предварительным `intros; symm_saturate`).

    Проверка двусторонняя, поэтому порядок аргументов на результат не влияет.
    Возвращает False, если хотя бы одна теорема не парсится или не типизируется.
    """
    base_thm_name = "base_theorem"
    reform_thm_name = "reformulated_theorem"
    res = [False, False]

    for i, (base_thm, reform_thm) in enumerate(
        [(formalization_1, formalization_2), (formalization_2, formalization_1)]
    ):
        if verbose:
            print(f"--- проверяем направление {'1 -> 2' if i == 0 else '2 -> 1'}")
        try:
            formal_1_code = (
                (src_header or "").strip()
                + "\n\n"
                + clean_last_theorem_string(base_thm, base_thm_name, add_sorry=True)
                + "\n\n"
            )
            formal_2_start_line = formal_1_code.count("\n") + 1
            formal_2_code = clean_last_theorem_string(reform_thm, reform_thm_name, add_sorry=False) + " := by"
        except ValueError:
            if verbose:
                print("не удалось разобрать одну из теорем — пара пропущена")
            break

        formal_code = formal_1_code + formal_2_code

        # 1) формализация должна быть well-typed, иначе эквивалентность бессмысленна
        if check_proof_sub(server, formal_code, formal_2_start_line, "sorry", timeout_per_proof) is None:
            if verbose:
                print("некорректная типизация — пара пропущена")
            break

        # 2) пытаемся доказать reform_thm ровно через base_thm
        proof_exact = check_proof_sub(server, formal_code, formal_2_start_line, "exact?", timeout_per_proof)
        if proof_exact and base_thm_name in proof_exact:
            res[i] = True
            if verbose:
                print("направление доказано")
        else:
            break

    return res[0] and res[1]


# beq plus
def beq_plus(
    formalization_1: str,
    formalization_2: str,
    src_header: str,
    server: Any,
    timeout_per_proof: int = DEFAULT_TIMEOUT,
    verbose: bool = False,
) -> bool:
    """
    1) exact?
    2) apply base_theorem + солверы;
    3) have <вывод базовой> := apply_rules [base_theorem]; солверы;
    4) convert base_theorem using k,  k = 0..4.
    """
    base_thm_name = "base_theorem"
    reform_thm_name = "reformulated_theorem"

    def prove_all(tactics: list[str]) -> str:
        prove_independent = " ; ".join(f"(all_goals try {t})" for t in tactics)
        prove_combined = "all_goals (" + " ; ".join(f"(try {t})" for t in tactics) + ")"
        return "all_goals intros\nfirst | (" + prove_independent + ") | (" + prove_combined + ")"

    solver_tactics_apply = ["tauto", "simp_all_arith!", "noncomm_ring", "exact?"]
    solver_tactics_have = ["tauto", "simp_all_arith!", "exact? using this"]
    proof_all_apply = prove_all(solver_tactics_apply)
    proof_all_have = prove_all(solver_tactics_have)

    res = [False, False]
    for i, (base_thm, reform_thm) in enumerate(
        [(formalization_1, formalization_2), (formalization_2, formalization_1)]
    ):
        if verbose:
            print(f"--- проверяем направление {'1 -> 2' if i == 0 else '2 -> 1'}")
        try:
            formal_1_code = (
                (src_header or "").strip()
                + "\n\n"
                + clean_last_theorem_string(base_thm, base_thm_name, add_sorry=True)
                + "\n\n"
            )
            formal_2_start_line = formal_1_code.count("\n") + 1
            formal_2_code = clean_last_theorem_string(reform_thm, reform_thm_name, add_sorry=False) + " := by"
        except ValueError:
            if verbose:
                print("не удалось разобрать одну из теорем — пара пропущена")
            break

        formal_code = formal_1_code + formal_2_code

        # формализация должна быть well-typed
        if check_proof_sub(server, formal_code, formal_2_start_line, "sorry", timeout_per_proof) is None:
            if verbose:
                print("некорректная типизация — пара пропущена")
            break

        # 1) exact?
        proof_exact = check_proof_sub(server, formal_code, formal_2_start_line, "exact?", timeout_per_proof)
        if proof_exact and base_thm_name in proof_exact:
            res[i] = True
            if verbose:
                print("доказано через exact?")
            continue

        # тривиально доказуемо `assumption` -> направление не сертифицируем
        if check_proof_sub(server, formal_code, formal_2_start_line, "assumption", timeout_per_proof):
            if verbose:
                print("цель доказуема `assumption` — направление пропущено")
            continue

        # прямое применение базовой теоремы
        proof_apply = check_proof_sub(
            server,
            formal_code,
            formal_2_start_line,
            f"apply {base_thm_name}\n" + proof_all_apply,
            timeout_per_proof,
        )
        if proof_apply:
            res[i] = True
            if verbose:
                print("доказано через apply")
            continue

        # вывод базовой теоремы как гипотеза (`have`),
        provable_without_have = False
        try:
            res_without_have = server.run(
                Command(cmd=formal_2_code + proof_all_have), timeout=timeout_per_proof
            )
            if isinstance(res_without_have, CommandResponse):
                provable_without_have = res_without_have.lean_code_is_valid(allow_sorry=False)
        except (TimeoutError, ConnectionAbortedError, json.JSONDecodeError, EOFError):
            pass
        except Exception:
            pass

        if not provable_without_have:
            idx_conclusion = split_conclusion(formal_1_code)
            if idx_conclusion:
                idx_end_conclusion = formal_1_code.rfind(":=")
                conclusion = formal_1_code[idx_conclusion:idx_end_conclusion].strip()
                have_stmt_proof = (
                    f"have {conclusion} := by\n"
                    + indent_code(f"apply_rules [{base_thm_name}]\n" + proof_all_apply, 2)
                    + "\n"
                )
                proof_have = check_proof_sub(
                    server,
                    formal_code,
                    formal_2_start_line,
                    have_stmt_proof + proof_all_have,
                    timeout_per_proof,
                )
                if proof_have:
                    res[i] = True
                    if verbose:
                        print("доказано через have")
                    continue

        # convert с возрастающим допуском по уровням подтермов
        for max_step in range(0, 5):
            proof_convert = check_proof_sub(
                server,
                formal_code,
                formal_2_start_line,
                f"convert (config := .unfoldSameFun) {base_thm_name} using {max_step}\n" + proof_all_apply,
                timeout_per_proof,
            )
            if proof_convert:
                res[i] = True
                if verbose:
                    print(f"доказано через convert (using {max_step})")
                break

        if not res[i]:
            break

    return res[0] and res[1]



# Параллельный прогон метрики по датасету
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
