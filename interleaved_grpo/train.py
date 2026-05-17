"""Train an interleaved CoT policy with TRL GRPO."""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

from .datasets_loader import build_interleaved_messages, dataset_from_args
from .rewards import REWARD_FUNCS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--dataset", choices=["gsm8k", "math", "toolmind", "hybrid"], default="gsm8k")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--dataset-subset", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output-dir", default="./interleaved_grpo_output")
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-prompt-length", type=int, default=512)
    parser.add_argument("--max-completion-length", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--report-to", default="tensorboard")
    parser.add_argument("--wandb", action="store_true", help="Also report training metrics to Weights & Biases.")
    parser.add_argument(
        "--generation-log-file",
        default=None,
        help="JSONL path for generated completions and reward components. Defaults to OUTPUT_DIR/generations.jsonl.",
    )
    parser.add_argument(
        "--disable-generation-logging",
        action="store_true",
        help="Disable JSONL logging of generated completions.",
    )
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-vllm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--sequential-hybrid-sampler", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--reward-weights",
        type=float,
        nargs=4,
        default=[1.0, 0.5, 0.3, 1.0],
        metavar=("OUTCOME", "STEP", "TTFT", "EFFICIENCY"),
    )
    return parser


def _column_item(values: Any, index: int) -> Any:
    if values is None:
        return None
    if isinstance(values, (str, bytes, dict)):
        return values
    try:
        return values[index]
    except (IndexError, KeyError, TypeError):
        return None


def _as_list(values: Any) -> list[Any]:
    if isinstance(values, list):
        return values
    if isinstance(values, tuple):
        return list(values)
    return [values]


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


class GenerationRewardLogger:
    """Collect per-reward outputs and write one JSONL record per generated completion."""

    def __init__(
        self,
        path: str | Path,
        reward_func_names: list[str],
        reward_weights: list[float] | None = None,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.reward_func_names = reward_func_names
        weights = reward_weights or [1.0] * len(reward_func_names)
        self.reward_weights = dict(zip(reward_func_names, weights))
        self._pending_batches: dict[str, dict[str, Any]] = {}
        self._next_batch_id = 0

    def wrap(self, reward_func: Any) -> Any:
        """Wrap a TRL reward function without changing its return value."""

        @functools.wraps(reward_func)
        def wrapped(prompts: list[str], completions: list[str], **kwargs: Any) -> list[float]:
            rewards = reward_func(prompts, completions, **kwargs)
            self.record(reward_func.__name__, prompts, completions, rewards, kwargs)
            return rewards

        return wrapped

    def record(
        self,
        reward_name: str,
        prompts: list[str],
        completions: list[str],
        rewards: list[float],
        kwargs: dict[str, Any],
    ) -> None:
        prompt_list = _as_list(prompts)
        completion_list = _as_list(completions)
        key = self._batch_key(prompt_list, completion_list)
        batch = self._pending_batches.get(key)
        if batch is None:
            batch = {
                "batch_id": self._next_batch_id,
                "created_at": time.time(),
                "prompts": prompt_list,
                "completions": completion_list,
                "columns": {
                    name: kwargs.get(name)
                    for name in ("answer", "expected_answer", "task_type", "target_has_tool_call")
                    if name in kwargs
                },
                "rewards": {},
            }
            self._pending_batches[key] = batch
            self._next_batch_id += 1

        batch["rewards"][reward_name] = list(rewards)
        if all(name in batch["rewards"] for name in self.reward_func_names):
            self._flush_batch(batch)
            del self._pending_batches[key]

    def _batch_key(self, prompts: list[Any], completions: list[Any]) -> str:
        payload = json.dumps({"prompts": prompts, "completions": completions}, sort_keys=True, default=str)
        return hashlib.blake2b(payload.encode("utf-8"), digest_size=16).hexdigest()

    def _flush_batch(self, batch: dict[str, Any]) -> None:
        prompts = batch["prompts"]
        completions = batch["completions"]
        columns = batch["columns"]
        rewards_by_name = batch["rewards"]
        records = []
        for index, completion in enumerate(completions):
            reward_components = {
                name: float(values[index])
                for name, values in rewards_by_name.items()
                if index < len(values)
            }
            weighted_reward = sum(
                reward * self.reward_weights.get(name, 1.0)
                for name, reward in reward_components.items()
            )
            records.append(
                {
                    "batch_id": batch["batch_id"],
                    "generation_index": index,
                    "created_at": batch["created_at"],
                    "process_rank": os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")),
                    "prompt": _jsonable(_column_item(prompts, index)),
                    "completion": _jsonable(completion),
                    "answer": _jsonable(_column_item(columns.get("answer"), index)),
                    "expected_answer": _jsonable(_column_item(columns.get("expected_answer"), index)),
                    "task_type": _jsonable(_column_item(columns.get("task_type"), index)),
                    "target_has_tool_call": _jsonable(_column_item(columns.get("target_has_tool_call"), index)),
                    "rewards": reward_components,
                    "weighted_reward": weighted_reward,
                }
            )
        self._append_jsonl(records)

    def _append_jsonl(self, records: list[dict[str, Any]]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            try:
                import fcntl

                fcntl.flock(handle, fcntl.LOCK_EX)
            except (ImportError, OSError):
                pass

            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()

            try:
                import fcntl

                fcntl.flock(handle, fcntl.LOCK_UN)
            except (ImportError, OSError):
                pass


def generation_log_path(args: argparse.Namespace) -> str:
    return args.generation_log_file or str(Path(args.output_dir) / "generations.jsonl")


def build_reward_funcs(args: argparse.Namespace) -> list[Any]:
    """Return reward functions, optionally wrapped with generation JSONL logging."""
    if args.disable_generation_logging:
        return list(REWARD_FUNCS)
    logger = GenerationRewardLogger(
        generation_log_path(args),
        [reward_func.__name__ for reward_func in REWARD_FUNCS],
        args.reward_weights,
    )
    return [logger.wrap(reward_func) for reward_func in REWARD_FUNCS]


def resolve_report_to(args: argparse.Namespace) -> str | list[str]:
    """Resolve metric sinks from `--report-to` and the convenience `--wandb` flag."""
    if args.report_to in {None, "", "none"}:
        report_targets: list[str] = []
    elif isinstance(args.report_to, str):
        report_targets = [target.strip() for target in args.report_to.split(",") if target.strip()]
    else:
        report_targets = list(args.report_to)

    if args.wandb and "wandb" not in report_targets:
        report_targets.append("wandb")
    return report_targets if report_targets else "none"


def _model_init_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    kwargs: dict[str, Any] = {"torch_dtype": torch.bfloat16 if args.bf16 else torch.float16}
    return kwargs


def with_sequential_train_sampler(trainer_cls: Any) -> Any:
    """Return a trainer subclass that preserves dataset order for alternating hybrid rows."""

    class SequentialSamplerTrainer(trainer_cls):  # type: ignore[misc, valid-type]
        def _get_train_sampler(self) -> Any:
            if self.train_dataset is None:
                return None
            from torch.utils.data import SequentialSampler

            return SequentialSampler(self.train_dataset)

    return SequentialSamplerTrainer


def apply_interleaved_chat_template(dataset: Any, tokenizer: Any) -> Any:
    """Convert normalized questions into chat-template prompts for GRPO."""

    def process_data(examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        task_types = examples.get("task_type", ["math"] * len(examples["question"]))
        prompts = [
            tokenizer.apply_chat_template(
                build_interleaved_messages(question, task_type=task_type),
                tokenize=False,
                add_generation_prompt=True,
            )
            for question, task_type in zip(examples["question"], task_types)
        ]
        processed = {"prompt": prompts, "answer": examples["answer"], "task_type": task_types}
        if "tool_definitions" in examples:
            processed["tool_definitions"] = examples["tool_definitions"]
        if "mock_outputs" in examples:
            processed["mock_outputs"] = examples["mock_outputs"]
        if "target_has_tool_call" in examples:
            processed["target_has_tool_call"] = examples["target_has_tool_call"]
        return processed

    return dataset.map(process_data, batched=True, remove_columns=dataset.column_names)


def main() -> None:
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    args = build_parser().parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = apply_interleaved_chat_template(dataset_from_args(args), tokenizer)

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        beta=args.beta,
        reward_weights=args.reward_weights,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        report_to=resolve_report_to(args),
        seed=args.seed,
        use_vllm=args.use_vllm,
        model_init_kwargs=_model_init_kwargs(args),
    )

    trainer_cls = GRPOTrainer
    if args.dataset == "hybrid" and args.sequential_hybrid_sampler:
        trainer_cls = with_sequential_train_sampler(GRPOTrainer)

    trainer = trainer_cls(
        model=args.model_id,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=dataset,
        reward_funcs=build_reward_funcs(args),
    )
    trainer.train()


if __name__ == "__main__":
    main()
