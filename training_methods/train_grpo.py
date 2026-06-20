from __future__ import annotations

"""
GRPO для автоформализатора (по аналогии с train_sft.py).

Что делает скрипт:
  * грузит тот же HF-датасет (train/eval), что и SFT;
  * формирует ровно тот же prompt (Kimina-style c явным запросом хэдера);
  * стартует с SFT-чекпойнта (по умолчанию AlexVav01/lean4-autoformalizer-sft);
  * обучает GRPO (TRL GRPOTrainer) с Lean-наградой:
        format  -> модель выдала корректный ```lean4 блок;
        typecheck -> формализация (утверждение) well-typed;
        beq_plus  -> формализация эквивалентна эталону (главный сигнал).
    Награды стадийные: эквивалентная формализация автоматически well-typed,
    поэтому получает сумму всех трёх (с весами) — у политики есть градиент,
    даже пока она ещё не дотягивает до полной эквивалентности.
  * (опционально) раз в N шагов считает те же метрики на held-out eval-сплите
    жадной генерацией — чтобы ловить переобучение / reward hacking.

КЛЮЧЕВОЕ ОТЛИЧИЕ ОТ SFT:
  reward считается КАЖДЫЙ шаг, поэтому Lean-серверы (каждый = свой Mathlib в
  памяти) поднимаются ОДИН раз в постоянный пул процессов и живут всё обучение.
  В SFT же `map_metric` поднимал и гасил пул на каждой (редкой) валидации.

Зависимости (тачка с GPU и собранным Lean/Mathlib):
    pip install "trl>=0.16" transformers datasets accelerate peft
    # vLLM (сильно ускоряет генерацию в GRPO, опционально):
    pip install vllm
    + рядом lean_utils.py (тот же, что в SFT/eval_beq_plus.py)

Память: по умолчанию используется LoRA — полный FT 7B через GRPO не влезает в
одну 80GB-карту (одни только веса+градиенты+AdamW ~122GB). LoRA укладывается
в ~40–55GB. Полный FT доступен флагом --full-finetune (нужно >1 GPU / offload).

Запуск (одна H100 80GB, LoRA + vLLM colocate — рекомендуется):
    python train_grpo.py --dataset AlexVav01/autoformalization-clean \
        --output-dir ./ckpt_grpo --use-vllm

Запуск (LoRA, без vLLM — медленнее, но проще по памяти):
    python train_grpo.py --dataset AlexVav01/autoformalization-clean --output-dir ./ckpt_grpo

Колонки датасета по умолчанию совпадают с SFT:
    nl_statement, lean4_src_header, lean4_formalization
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "evaluation_methods"))

import argparse
import functools
import json
import multiprocessing as mp
import re

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

# --------------------------------------------------------------------------- #
# Промпт и работа с Lean-кодом (1:1 с train_sft.py — чтобы распределение
# совпадало с тем, на чём училась SFT-модель)
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = "You are an expert in mathematics and Lean 4."
THEOREM_NAME = "my_favorite_theorem"

_THM_RE = re.compile(r"\b(theorem|lemma)\s+[^\s({:]+")
_KEYWORD_RE = re.compile(r"\b(theorem|lemma|def|abbrev|example|instance)\b")
_CODE_BLOCK_RE = re.compile(r"```(?:[Ll]ean4?)?\s*\n(.*?)```", re.DOTALL)


def build_user_prompt(nl_statement: str, header: str) -> str:
    return (
        "Please autoformalize the following problem in Lean 4 with a header. "
        f"Use the following theorem names: {THEOREM_NAME}.\n\n"
        f"{nl_statement}\n\n"
        f"Your code should start with:\n```lean4\n{header}\n```\n"
    )


def rename_theorem(code: str, name: str = THEOREM_NAME) -> str:
    return _THM_RE.sub(r"\1 " + name, code, count=1)


def extract_lean(text: str) -> str:
    """Достаём последний lean-блок (последний — чтобы пережить reasoning)."""
    blocks = _CODE_BLOCK_RE.findall(text)
    return (blocks[-1] if blocks else text).strip()


def strip_header(code: str, header: str) -> str:
    """Убираем хэдер из сгенерированного кода -> остаётся голое утверждение."""
    code = code.strip()
    h = header.strip()
    if h and code.startswith(h):
        return code[len(h):].strip()
    m = _KEYWORD_RE.search(code)
    return code[m.start():].strip() if m else code


def completion_text(completion) -> str:
    """GRPO с conversational prompt отдаёт completion как список сообщений."""
    if isinstance(completion, list):
        return completion[-1]["content"] if completion else ""
    return completion or ""


# --------------------------------------------------------------------------- #
# Постоянный пул Lean-серверов (живёт всё обучение)
# Функции воркера — на уровне модуля, иначе multiprocessing (spawn) их не запиклит.
# --------------------------------------------------------------------------- #
_REWARD_SERVER = None  # AutoLeanServer внутри каждого воркера


def _reward_init_worker(config) -> None:
    global _REWARD_SERVER
    from lean_interact import AutoLeanServer
    _REWARD_SERVER = AutoLeanServer(config)


def _reward_eval_one(payload):
    """Считаем (typecheck, beq_plus) для одной записи. beq_plus — дорогой,
    поэтому не запускаем его, если формализация даже не типизируется."""
    record, timeout = payload
    from lean_utils import beq_plus, is_well_typed

    pred, header, gold = record["pred"], record["header"], record["gold"]
    if not pred:
        return (False, False)

    try:
        tc = bool(is_well_typed(pred, header, _REWARD_SERVER, timeout=timeout))
    except Exception:
        tc = False
    if not tc:
        return (False, False)

    try:
        bq = bool(
            beq_plus(gold, pred, header, _REWARD_SERVER, timeout_per_proof=timeout)
        )
    except Exception:
        bq = False
    return (True, bq)


class LeanRewardPool:
    """Постоянный пул процессов с поднятыми Lean-серверами."""

    def __init__(self, config, num_processes: int, timeout: int):
        ctx = mp.get_context("spawn")
        self.pool = ctx.Pool(
            processes=num_processes,
            initializer=_reward_init_worker,
            initargs=(config,),
        )
        self.timeout = timeout

    def evaluate(self, records: list[dict]) -> list[tuple[bool, bool]]:
        if not records:
            return []
        payloads = [(r, self.timeout) for r in records]
        # imap сохраняет порядок -> результаты соответствуют records
        return list(self.pool.imap(_reward_eval_one, payloads))

    def close(self) -> None:
        self.pool.close()
        self.pool.join()


# --------------------------------------------------------------------------- #
# Reward-функции
# --------------------------------------------------------------------------- #
def format_reward(prompts=None, completions=None, **kwargs) -> list[float]:
    """Дешёвая награда: есть ли корректный ```lean4 ...``` блок."""
    out = []
    for c in completions:
        out.append(1.0 if _CODE_BLOCK_RE.search(completion_text(c)) else 0.0)
    return out


class LeanRewards:
    """typecheck + beq_plus поверх постоянного пула.

    GRPO внутри одного шага зовёт каждую reward-функцию с одинаковыми
    (prompts, completions). Lean считаем один раз и кэшируем по содержимому
    completions, чтобы typecheck_reward и beq_reward не дублировали работу.
    """

    def __init__(self, pool: LeanRewardPool):
        self.pool = pool
        self._key = None
        self._cache: list[tuple[bool, bool]] = []

    def _evaluate(self, completions, headers, golds) -> list[tuple[bool, bool]]:
        texts = [completion_text(c) for c in completions]
        # ключ по содержимому шага (а не только по completions), чтобы исключить
        # любую коллизию между шагами при идентичных генерациях
        key = hash((tuple(texts), tuple(headers), tuple(golds)))
        if key == self._key:
            return self._cache
        records = []
        for t, h, g in zip(texts, headers, golds):
            pred = strip_header(extract_lean(t), h)
            records.append({"pred": pred, "header": h, "gold": g})
        self._cache = self.pool.evaluate(records)
        self._key = key
        return self._cache

    def typecheck_reward(self, prompts=None, completions=None, **kwargs) -> list[float]:
        res = self._evaluate(completions, kwargs["header"], kwargs["gold"])
        return [1.0 if tc else 0.0 for tc, _ in res]

    def beq_reward(self, prompts=None, completions=None, **kwargs) -> list[float]:
        res = self._evaluate(completions, kwargs["header"], kwargs["gold"])
        return [1.0 if bq else 0.0 for _, bq in res]


# --------------------------------------------------------------------------- #
# Подготовка датасета под GRPO: только prompt + поля для reward (gold/header)
# --------------------------------------------------------------------------- #
def make_grpo_example(row, nl_col, header_col, formal_col) -> dict:
    user = build_user_prompt(row[nl_col], row[header_col])
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        # эти колонки GRPO прокинет в reward-функции как kwargs (по одному на генерацию)
        "gold": rename_theorem(row[formal_col].strip()),
        "header": row[header_col],
    }


def prepare_split(ds, nl_col, header_col, formal_col):
    fn = functools.partial(
        make_grpo_example, nl_col=nl_col, header_col=header_col, formal_col=formal_col
    )
    return ds.map(fn, remove_columns=ds.column_names)


# --------------------------------------------------------------------------- #
# Held-out eval: жадная генерация + typecheck/BEq+ на eval-сплите
# (аналог LeanEvalCallback из SFT; переиспользует тот же пул)
# --------------------------------------------------------------------------- #
class HeldOutLeanEvalCallback(TrainerCallback):
    def __init__(self, tokenizer, eval_rows, pool: LeanRewardPool, args):
        self.tok = tokenizer
        self.pool = pool
        self.eval_steps = args.eval_steps
        self.gen_batch_size = args.gen_batch_size
        self.gen_max_new_tokens = args.max_completion_length
        self.max_prompt_len = args.max_prompt_length
        self.output_dir = args.output_dir
        self.trainer = None

        self.prompts, self.golds, self.headers = [], [], []
        for r in eval_rows:
            user = build_user_prompt(r[args.nl_column], r[args.header_column])
            text = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            self.prompts.append(text)
            self.golds.append(rename_theorem(r[args.formalization_column].strip()))
            self.headers.append(r[args.header_column])

    @torch.no_grad()
    def _generate(self, model) -> list[str]:
        was_training = model.training
        use_cache_prev = model.config.use_cache
        pad_side_prev = self.tok.padding_side
        model.eval()
        model.config.use_cache = True
        self.tok.padding_side = "left"

        device = next(model.parameters()).device
        preds = []
        for i in range(0, len(self.prompts), self.gen_batch_size):
            batch = self.prompts[i : i + self.gen_batch_size]
            enc = self.tok(
                batch, return_tensors="pt", padding=True,
                truncation=True, max_length=self.max_prompt_len,
            ).to(device)
            out = model.generate(
                **enc,
                max_new_tokens=self.gen_max_new_tokens,
                do_sample=False,
                pad_token_id=self.tok.pad_token_id,
            )
            gen = out[:, enc["input_ids"].shape[1] :]
            preds.extend(self.tok.batch_decode(gen, skip_special_tokens=True))

        self.tok.padding_side = pad_side_prev
        model.config.use_cache = use_cache_prev
        if was_training:
            model.train()
        return preds

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if self.eval_steps <= 0 or state.global_step == 0:
            return
        if state.global_step % self.eval_steps != 0:
            return
        if not state.is_world_process_zero:
            return

        raw_preds = self._generate(model)
        records = []
        for raw, gold, header in zip(raw_preds, self.golds, self.headers):
            pred = strip_header(extract_lean(raw), header)
            records.append({"pred": pred, "header": header, "gold": gold})

        res = self.pool.evaluate(records)
        n = max(len(res), 1)
        logs = {
            "eval_typecheck": sum(tc for tc, _ in res) / n,
            "eval_beq_plus": sum(bq for _, bq in res) / n,
        }
        print(
            f"[held-out] step {state.global_step}: "
            f"typecheck={logs['eval_typecheck']:.2%} beq+={logs['eval_beq_plus']:.2%}"
        )
        if self.trainer is not None:
            self.trainer.log(logs)
        os.makedirs(self.output_dir, exist_ok=True)
        with open(os.path.join(self.output_dir, "lean_eval.jsonl"), "a") as f:
            f.write(json.dumps({"step": state.global_step, **logs}) + "\n")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(description="GRPO автоформализатора с Lean-наградой (typecheck/BEq+).")
    # данные / модель
    p.add_argument("--dataset", default="AlexVav01/autoformalization-clean",
                   help="HF repo id со сплитами train/eval.")
    p.add_argument("--model", default="AlexVav01/lean4-autoformalizer-sft",
                   help="Стартовый чекпойнт (по умолчанию ваша SFT-модель).")
    p.add_argument("--output-dir", default="./ckpt_grpo")
    p.add_argument("--train-split", default="train")
    p.add_argument("--eval-split", default="eval")
    p.add_argument("--nl-column", default="nl_statement")
    p.add_argument("--header-column", default="lean4_src_header")
    p.add_argument("--formalization-column", default="lean4_formalization")
    # GRPO / обучение
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=1e-6, help="RL обычно требует LR сильно меньше SFT.")
    p.add_argument("--per-device-batch-size", type=int, default=8,
                   help="Число completions на устройство за микрошаг (кратно --num-generations).")
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--num-generations", type=int, default=8, help="Размер группы G в GRPO.")
    p.add_argument("--max-prompt-length", type=int, default=2048)
    p.add_argument("--max-completion-length", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=0.0,
                   help="KL к reference. 0 = без reference-модели (экономит память). "
                        "Классический GRPO: ~0.04.")
    p.add_argument("--num-iterations", type=int, default=1, help="mu: оптимизационных проходов на батч.")
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--logging-steps", type=int, default=5)
    p.add_argument("--save-steps", type=int, default=200)
    p.add_argument("--save-total-limit", type=int, default=2)
    # веса наград
    p.add_argument("--format-weight", type=float, default=0.1)
    p.add_argument("--typecheck-weight", type=float, default=0.4)
    p.add_argument("--beq-weight", type=float, default=1.0)
    # LoRA (включена по умолчанию — полный FT 7B не влезает в одну 80GB-карту)
    p.add_argument("--full-finetune", action="store_true",
                   help="Полный fine-tune вместо LoRA (нужно >1 GPU или offload).")
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lora-target-modules", type=str,
                   default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
                   help="Через запятую, либо 'all-linear'.")
    # vLLM (рекомендуется для GRPO)
    p.add_argument("--use-vllm", action="store_true", help="Генерация через vLLM (colocate). Нужен trl>=0.16 и vllm.")
    p.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.3)
    # held-out eval
    p.add_argument("--eval-steps", type=int, default=200, help="0 = выключить held-out eval.")
    p.add_argument("--gen-batch-size", type=int, default=8, help="Batch для held-out генерации.")
    # Lean-метрики через lean_utils
    p.add_argument("--beq-num-processes", type=int, default=24,
                   help="Число Lean-серверов в постоянном пуле (каждый = свой Mathlib в RAM).")
    p.add_argument("--beq-timeout", type=int, default=None, help="timeout_per_proof (сек).")
    p.add_argument("--lean-version", type=str, default=None)
    # wandb
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", type=str, default="autoformalization-grpo")
    p.add_argument("--run-name", type=str, default=None)

    args = p.parse_args()

    if args.wandb:
        os.environ["WANDB_PROJECT"] = args.wandb_project

    # sanity check на кратность (GRPO требует, чтобы глобальный батч делился на G)
    if args.per_device_batch_size % args.num_generations != 0:
        print(
            f"[warn] per-device-batch-size ({args.per_device_batch_size}) не кратен "
            f"num-generations ({args.num_generations}); TRL может ругнуться."
        )

    # 1. данные
    ds = load_dataset(args.dataset)
    train_raw = ds[args.train_split]
    eval_raw = ds[args.eval_split]
    print(f"train: {len(train_raw)} | eval: {len(eval_raw)}")

    train_ds = prepare_split(
        train_raw, args.nl_column, args.header_column, args.formalization_column
    ).shuffle(seed=42)

    # 2. модель и токенайзер (стартуем с SFT-чекпойнта)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    model.config.use_cache = False

    # 2b. LoRA-конфиг (по умолчанию). GRPOTrainer сам обернёт модель через
    #     get_peft_model и включит input-require-grads под gradient checkpointing.
    peft_config = None
    if not args.full_finetune:
        from peft import LoraConfig

        target = (
            args.lora_target_modules
            if args.lora_target_modules == "all-linear"
            else [m.strip() for m in args.lora_target_modules.split(",")]
        )
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target,
            bias="none",
            task_type="CAUSAL_LM",
        )
        print(f"LoRA: r={args.lora_r} alpha={args.lora_alpha} target={target}")
    else:
        print("Полный fine-tune (LoRA выключена).")

    # 3. Постоянный пул Lean-серверов (поднимается ОДИН раз)
    from lean_utils import DEFAULT_LEAN_VERSION, DEFAULT_TIMEOUT, make_lean_config

    lean_version = args.lean_version or DEFAULT_LEAN_VERSION
    timeout = args.beq_timeout or DEFAULT_TIMEOUT
    print(f"Готовим Lean {lean_version} + Mathlib (первый запуск долгий)...")
    lean_config = make_lean_config(lean_version=lean_version, verbose=True)
    reward_pool = LeanRewardPool(lean_config, args.beq_num_processes, timeout)
    print(f"Lean-пул поднят: {args.beq_num_processes} серверов.")

    lean_rewards = LeanRewards(reward_pool)
    reward_funcs = [format_reward, lean_rewards.typecheck_reward, lean_rewards.beq_reward]
    reward_weights = [args.format_weight, args.typecheck_weight, args.beq_weight]

    # 4. конфиг GRPO
    config_kwargs = dict(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_generations=args.num_generations,
        vllm_max_model_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        top_p=args.top_p,
        beta=args.beta,
        num_iterations=args.num_iterations,
        reward_weights=reward_weights,
        scale_rewards=True,
        logging_steps=args.logging_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        log_completions=True,
        report_to=("wandb" if args.wandb else "none"),
        run_name=args.run_name,
    )
    if args.use_vllm:
        config_kwargs.update(
            use_vllm=True,
            vllm_mode="colocate",
            vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        )
    config = GRPOConfig(**config_kwargs)

    # 5. held-out eval callback (опционально), переиспользует тот же пул
    callbacks = []
    held_out = None
    if args.eval_steps > 0:
        held_out = HeldOutLeanEvalCallback(tokenizer, list(eval_raw), reward_pool, args)
        callbacks.append(held_out)

    # 6. тренер
    trainer = GRPOTrainer(
        model=model,
        args=config,
        train_dataset=train_ds,
        reward_funcs=reward_funcs,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=callbacks,
    )
    if held_out is not None:
        held_out.trainer = trainer

    # 7. обучение
    try:
        trainer.train()
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
    finally:
        reward_pool.close()


if __name__ == "__main__":
    main()
