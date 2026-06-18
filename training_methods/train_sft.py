from __future__ import annotations

"""
Минимальный SFT для автоформализатора (Qwen2.5-Coder-7B-Instruct).

Что делает скрипт:
  * грузит HF-датасет со сплитами train (~30k) и eval (~400);
  * формирует prompt в стиле Kimina/StepFun-Formalizer, но с явным запросом
    хэдеров (модель сама генерирует хэдер + формализацию);
  * обучает обычным SFT (TRL SFTTrainer, лосс только на completion);
  * раз в N шагов считает eval loss (даёт TRL) И метрику BEq+ на eval-сплите
    через ту же машинерию, что и eval_beq_plus.py (lean_utils).

Зависимости (на тачке с GPU и собранным Lean/Mathlib):
    pip install "trl>=0.12" transformers datasets accelerate
    + рядом должен лежать lean_utils.py (тот же, что использует eval_beq_plus.py)

Запуск (один GPU):
    python train_sft.py --dataset <hf_repo_id> --output-dir ./ckpt

Запуск (несколько GPU, DDP):
    accelerate launch train_sft.py --dataset <hf_repo_id> --output-dir ./ckpt

Колонки датасета по умолчанию совпадают с example_samples.csv:
    nl_statement, lean4_src_header, lean4_formalization
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "evaluation_methods"))

import argparse
import functools
import json
import os
import re

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import SFTConfig, SFTTrainer

# --------------------------------------------------------------------------- #
# Промпт и работа с Lean-кодом
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = "You are an expert in mathematics and Lean 4."
THEOREM_NAME = "my_favorite_theorem"

_THM_RE = re.compile(r"\b(theorem|lemma)\s+[^\s({:]+")
_KEYWORD_RE = re.compile(r"\b(theorem|lemma|def|abbrev|example|instance)\b")
_CODE_BLOCK_RE = re.compile(r"```(?:[Ll]ean4?)?\s*\n(.*?)```", re.DOTALL)


def build_user_prompt(nl_statement: str, header: str) -> str:
    """Kimina-style + просьба сгенерировать код с конкретным хэдером."""
    return (
        "Please autoformalize the following problem in Lean 4 with a header. "
        f"Use the following theorem names: {THEOREM_NAME}.\n\n"
        f"{nl_statement}\n\n"
        f"Your code should start with a header:\n```lean4\n{header}\n```\n"
    )


def rename_theorem(code: str, name: str = THEOREM_NAME) -> str:
    """Переименовываем первое theorem/lemma в фиксированное имя (как в Kimina)."""
    return _THM_RE.sub(r"\1 " + name, code, count=1)


def build_completion(header: str, formalization: str) -> str:
    """Целевой ответ: один lean-блок c хэдером и формализацией."""
    body = rename_theorem(formalization.strip())
    return f"```lean4\n{header.strip()}\n\n{body}\n```"


def extract_lean(text: str) -> str:
    """Достаём последний lean-блок (последний — чтобы пережить будущий reasoning)."""
    blocks = _CODE_BLOCK_RE.findall(text)
    return (blocks[-1] if blocks else text).strip()


def strip_header(code: str, header: str) -> str:
    """Убираем хэдер из сгенерированного кода -> остаётся голое утверждение.

    BEq+ принимает header отдельным аргументом, поэтому хэдер тут не нужен.
    """
    code = code.strip()
    h = header.strip()
    if h and code.startswith(h):
        return code[len(h):].strip()
    m = _KEYWORD_RE.search(code)
    return code[m.start():].strip() if m else code


# --------------------------------------------------------------------------- #
# Подготовка датасета под TRL (формат prompt/completion -> лосс только на ответе)
# --------------------------------------------------------------------------- #
def make_sft_example(row: dict, nl_col: str, header_col: str, formal_col: str) -> dict:
    user = build_user_prompt(row[nl_col], row[header_col])
    completion = build_completion(row[header_col], row[formal_col])
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        "completion": [{"role": "assistant", "content": completion}],
    }


def prepare_split(ds, nl_col, header_col, formal_col):
    fn = functools.partial(
        make_sft_example, nl_col=nl_col, header_col=header_col, formal_col=formal_col
    )
    return ds.map(fn, remove_columns=ds.column_names)


# --------------------------------------------------------------------------- #
# Callback: генерация на eval + BEq+
# --------------------------------------------------------------------------- #
class BEqPlusCallback(TrainerCallback):
    """Раз в eval_steps генерирует формализации и считает долю BEq+-эквивалентных."""

    def __init__(self, tokenizer, eval_rows, lean_config, beq_metric, args):
        self.tok = tokenizer
        self.lean_config = lean_config
        self.beq_metric = beq_metric          # functools.partial(beq_plus-обёртка)
        self.map_metric = args._map_metric     # из lean_utils
        self.num_processes = args.beq_num_processes
        self.gen_batch_size = args.gen_batch_size
        self.gen_max_new_tokens = args.gen_max_new_tokens
        self.max_prompt_len = args.gen_max_prompt_len
        self.output_dir = args.output_dir
        self.trainer = None  # проставим после создания SFTTrainer (для trainer.log)

        # заранее готовим текстовые промпты и эталоны
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
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_prompt_len,
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

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        # генерим и считаем BEq+ только на главном процессе
        if not state.is_world_process_zero:
            return

        raw_preds = self._generate(model)

        records = []
        for raw, gold, header in zip(raw_preds, self.golds, self.headers):
            pred = strip_header(extract_lean(raw), header)
            records.append({"gold": gold, "pred": pred, "header": header})

        try:
            flags = self.map_metric(
                records,
                self.beq_metric,
                self.lean_config,
                num_processes=self.num_processes,
                desc="beq_plus",
            )
            rate = sum(bool(f) for f in flags) / len(flags)
        except Exception as e:  # Lean-проблемы не должны валить обучение
            print(f"[BEq+] ошибка во время оценки: {e}")
            return

        print(f"[BEq+] step {state.global_step}: {rate:.2%} эквивалентных")
        if self.trainer is not None:
            self.trainer.log({"eval_beq_plus": rate})

        # дублируем в jsonl рядом с чекпойнтами
        os.makedirs(self.output_dir, exist_ok=True)
        with open(os.path.join(self.output_dir, "beq_plus.jsonl"), "a") as f:
            f.write(json.dumps({"step": state.global_step, "beq_plus": rate}) + "\n")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(description="SFT автоформализатора с BEq+ валидацией.")
    # данные / модель
    p.add_argument("--dataset", required=True, help="HF repo id со сплитами train/eval.")
    p.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    p.add_argument("--output-dir", default="./ckpt")
    p.add_argument("--train-split", default="train")
    p.add_argument("--eval-split", default="eval")
    p.add_argument("--nl-column", default="nl_statement")
    p.add_argument("--header-column", default="lean4_src_header")
    p.add_argument("--formalization-column", default="lean4_formalization")
    # обучение
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--per-device-batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--logging-steps", type=int, default=20)
    p.add_argument("--eval-steps", type=int, default=400, help="Каждые N шагов: loss + BEq+.")
    p.add_argument("--save-steps", type=int, default=400)
    # генерация на eval
    p.add_argument("--gen-batch-size", type=int, default=16)
    p.add_argument("--gen-max-new-tokens", type=int, default=512)
    p.add_argument("--gen-max-prompt-len", type=int, default=1024)
    # BEq+
    p.add_argument("--no-beq", action="store_true", help="Только loss, без BEq+.")
    p.add_argument("--beq-num-processes", type=int, default=4)
    p.add_argument("--beq-timeout", type=int, default=None, help="timeout_per_proof (сек).")
    p.add_argument("--lean-version", type=str, default=None)
    args = p.parse_args()

    # 1. данные
    ds = load_dataset(args.dataset)
    train_raw = ds[args.train_split]
    eval_raw = ds[args.eval_split]
    print(f"train: {len(train_raw)} | eval: {len(eval_raw)}")

    train_ds = prepare_split(
        train_raw, args.nl_column, args.header_column, args.formalization_column
    )
    eval_ds = prepare_split(
        eval_raw, args.nl_column, args.header_column, args.formalization_column
    )

    # 2. модель и токенайзер
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    model.config.use_cache = False  # совместимо с gradient checkpointing

    # 3. конфиг SFT (лосс только на completion — это поведение по умолчанию TRL
    #    для prompt/completion формата)
    config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True,
        max_length=args.max_length,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        report_to="none",
    )

    # 4. BEq+ callback (через lean_utils, как в eval_beq_plus.py)
    callbacks = []
    beq_cb = None
    if not args.no_beq:
        from lean_utils import (
            DEFAULT_LEAN_VERSION,
            DEFAULT_TIMEOUT,
            beq_plus,
            make_lean_config,
            map_metric,
        )

        lean_version = args.lean_version or DEFAULT_LEAN_VERSION
        timeout = args.beq_timeout or DEFAULT_TIMEOUT
        print(f"Готовим Lean {lean_version} + Mathlib (первый запуск долгий)...")
        lean_config = make_lean_config(lean_version=lean_version, verbose=True)

        def beq_metric(record, server, _timeout=timeout):
            return beq_plus(
                record["gold"], record["pred"], record["header"], server,
                timeout_per_proof=_timeout,
            )

        args._map_metric = map_metric
        beq_cb = BEqPlusCallback(tokenizer, list(eval_raw), lean_config, beq_metric, args)
        callbacks.append(beq_cb)

    # 5. тренер
    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        callbacks=callbacks,
    )
    if beq_cb is not None:
        beq_cb.trainer = trainer  # чтобы BEq+ логировался вместе с остальными метриками

    # 6. обучение
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
