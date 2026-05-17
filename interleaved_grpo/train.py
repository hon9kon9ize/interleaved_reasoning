"""Train an interleaved CoT policy with TRL GRPO."""

from __future__ import annotations

import argparse
import functools
import hashlib
import inspect
import json
import os
from pathlib import Path
import time
from typing import Any

from .datasets_loader import build_interleaved_messages, dataset_from_args
from .rewards import REWARD_FUNCS, language_consistency_reward_fn


BASE_REWARD_WEIGHTS = [1.0, 0.5, 0.3, 1.0]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--dataset", choices=["gsm8k", "math", "toolmind", "hybrid", "reasoning-lang"], default="gsm8k")
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
    parser.add_argument(
        "--filter-overlong-prompts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop formatted training rows whose prompt token length exceeds --max-prompt-length.",
    )
    parser.add_argument(
        "--chat-template-enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Optional passthrough for tokenizer.apply_chat_template(enable_thinking=...). Defaults to omitting it.",
    )
    parser.add_argument(
        "--normalize-prefilled-think",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For templates that prefill an opening <think>, prepend it back to completions before reward scoring.",
    )
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
    parser.add_argument("--use-lora", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated module names to adapt with LoRA.",
    )
    parser.add_argument("--lora-bias", choices=["none", "all", "lora_only"], default="none")
    parser.add_argument(
        "--lora-modules-to-save",
        default=None,
        help="Optional comma-separated module names to save alongside LoRA adapters.",
    )
    parser.add_argument("--sequential-hybrid-sampler", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--language-consistency-reward",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable the reasoning-language consistency reward. Defaults on when reasoning_lang metadata is present.",
    )
    parser.add_argument(
        "--reasoning-lang",
        "--reasoning_lang",
        "--reansoning_lang",
        dest="reasoning_lang",
        default=None,
        help="Fallback reasoning language when the dataset does not provide a reasoning_lang column.",
    )
    parser.add_argument("--language-reward-weight", type=float, default=0.2)
    parser.add_argument(
        "--reward-weights",
        type=float,
        nargs="+",
        default=None,
        metavar="WEIGHT",
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


def prompt_prefills_open_think(prompt: Any) -> bool:
    """Return whether a chat template left an opening `<think>` in the prompt."""
    text = "" if prompt is None else str(prompt)
    return text.count("<think>") > text.count("</think>")


def normalize_prefilled_think_completion(prompt: Any, completion: str) -> str:
    """
    Reconstruct a Qwen-style prefilled opening `<think>` for reward parsing.

    Some Qwen3 thinking templates append `<think>` to the generation prompt. TRL
    only passes generated tokens to reward functions, so the completion can start
    with thought text or `</think>` and otherwise look like malformed XML.
    """
    if not prompt_prefills_open_think(prompt):
        return completion
    if completion.lstrip().startswith("<think>"):
        return completion
    return "<think>" + completion


def normalize_prefilled_think_completions(
    prompts: list[Any],
    completions: list[str],
    enabled: bool = True,
) -> list[str]:
    """Normalize a batch of completions for templates that prefilled `<think>`."""
    if not enabled:
        return list(completions)
    if len(prompts) == len(completions):
        return [normalize_prefilled_think_completion(prompt, completion) for prompt, completion in zip(prompts, completions)]
    if len(prompts) == 1:
        return [normalize_prefilled_think_completion(prompts[0], completion) for completion in completions]
    return list(completions)


def with_prefilled_think_normalization(reward_func: Any, enabled: bool = True) -> Any:
    """Wrap a reward function so Qwen3 prefilled think prompts parse correctly."""

    @functools.wraps(reward_func)
    def wrapped(prompts: list[str], completions: list[str], **kwargs: Any) -> list[float]:
        prompt_list = _as_list(prompts)
        completion_list = _as_list(completions)
        normalized = normalize_prefilled_think_completions(prompt_list, completion_list, enabled=enabled)
        return reward_func(prompts, normalized, **kwargs)

    return wrapped


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
                    for name in (
                        "answer",
                        "expected_answer",
                        "task_type",
                        "target_has_tool_call",
                        "reasoning_lang",
                        "reasoning_language",
                        "language",
                    )
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
            prompt = _column_item(prompts, index)
            record = {
                "batch_id": batch["batch_id"],
                "generation_index": index,
                "created_at": batch["created_at"],
                "process_rank": os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")),
                "prompt": _jsonable(prompt),
                "completion": _jsonable(completion),
                "answer": _jsonable(_column_item(columns.get("answer"), index)),
                "expected_answer": _jsonable(_column_item(columns.get("expected_answer"), index)),
                "task_type": _jsonable(_column_item(columns.get("task_type"), index)),
                "target_has_tool_call": _jsonable(_column_item(columns.get("target_has_tool_call"), index)),
                "reasoning_lang": _jsonable(_column_item(columns.get("reasoning_lang"), index)),
                "reasoning_language": _jsonable(_column_item(columns.get("reasoning_language"), index)),
                "language": _jsonable(_column_item(columns.get("language"), index)),
                "rewards": reward_components,
                "weighted_reward": weighted_reward,
            }
            normalized_completion = normalize_prefilled_think_completion(prompt, completion)
            if normalized_completion != completion:
                record["normalized_completion"] = _jsonable(normalized_completion)
            records.append(record)
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


def dataset_has_reasoning_lang(dataset: Any) -> bool:
    names = getattr(dataset, "column_names", None)
    if names is None and isinstance(dataset, dict):
        names = dataset.keys()
    return names is not None and any(name in names for name in ("reasoning_lang", "reasoning_language"))


def use_language_consistency_reward(args: argparse.Namespace, dataset: Any | None = None) -> bool:
    if args.language_consistency_reward is not None:
        return bool(args.language_consistency_reward)
    return bool(args.reasoning_lang) or dataset_has_reasoning_lang(dataset)


def select_reward_funcs(args: argparse.Namespace, dataset: Any | None = None) -> list[Any]:
    reward_funcs = list(REWARD_FUNCS)
    if use_language_consistency_reward(args, dataset):
        reward_funcs.append(language_consistency_reward_fn)
    return reward_funcs


def resolve_reward_weights(args: argparse.Namespace, reward_funcs: list[Any]) -> list[float]:
    if args.reward_weights is not None:
        if len(args.reward_weights) != len(reward_funcs):
            raise ValueError(f"Expected {len(reward_funcs)} reward weights, got {len(args.reward_weights)}.")
        return list(args.reward_weights)

    weights = list(BASE_REWARD_WEIGHTS)
    if any(reward_func.__name__ == "language_consistency_reward_fn" for reward_func in reward_funcs):
        weights.append(args.language_reward_weight)
    return weights


def build_reward_funcs(
    args: argparse.Namespace,
    reward_funcs: list[Any] | None = None,
    reward_weights: list[float] | None = None,
) -> list[Any]:
    """Return reward functions, optionally wrapped with generation JSONL logging."""
    reward_funcs = reward_funcs or select_reward_funcs(args)
    reward_weights = reward_weights or resolve_reward_weights(args, reward_funcs)
    reward_funcs = [
        with_prefilled_think_normalization(reward_func, enabled=args.normalize_prefilled_think)
        for reward_func in reward_funcs
    ]
    if args.disable_generation_logging:
        return list(reward_funcs)
    logger = GenerationRewardLogger(
        generation_log_path(args),
        [reward_func.__name__ for reward_func in reward_funcs],
        reward_weights,
    )
    return [logger.wrap(reward_func) for reward_func in reward_funcs]


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


def _split_csv_arg(value: str | None) -> list[str] | None:
    if value is None:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or None


def build_peft_config(args: argparse.Namespace) -> Any | None:
    """Build a PEFT LoRA config when requested."""
    if not args.use_lora:
        return None

    try:
        from peft import LoraConfig
    except ImportError as exc:
        raise ImportError("LoRA support requires `peft`. Install it with `python -m pip install peft`.") from exc

    target_modules = _split_csv_arg(args.lora_target_modules)
    if not target_modules:
        raise ValueError("--lora-target-modules must include at least one module name when --use-lora is enabled.")

    modules_to_save = _split_csv_arg(args.lora_modules_to_save)
    return LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias=args.lora_bias,
        task_type="CAUSAL_LM",
        target_modules=target_modules,
        modules_to_save=modules_to_save,
    )


def trainer_accepts_peft_config(trainer_cls: Any) -> bool:
    """Return whether the installed TRL trainer can receive `peft_config` directly."""
    try:
        return "peft_config" in inspect.signature(trainer_cls.__init__).parameters
    except (TypeError, ValueError):
        return False


def build_trainer_model_kwargs(
    args: argparse.Namespace,
    trainer_cls: Any,
    peft_config: Any | None,
) -> dict[str, Any]:
    """Build model-related kwargs for GRPOTrainer with portable LoRA support."""
    if peft_config is None:
        return {"model": args.model_id}

    if trainer_accepts_peft_config(trainer_cls):
        return {"model": args.model_id, "peft_config": peft_config}

    try:
        from peft import get_peft_model
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise ImportError("Fallback LoRA model wrapping requires `peft` and `transformers`.") from exc

    model = AutoModelForCausalLM.from_pretrained(args.model_id, **_model_init_kwargs(args))
    return {"model": get_peft_model(model, peft_config)}


def with_sequential_train_sampler(trainer_cls: Any) -> Any:
    """Return a trainer subclass that preserves dataset order for alternating hybrid rows."""

    class SequentialSamplerTrainer(trainer_cls):  # type: ignore[misc, valid-type]
        def _get_train_sampler(self) -> Any:
            if self.train_dataset is None:
                return None
            from torch.utils.data import SequentialSampler

            return SequentialSampler(self.train_dataset)

    return SequentialSamplerTrainer


def apply_interleaved_chat_template(
    dataset: Any,
    tokenizer: Any,
    default_reasoning_lang: str | None = None,
    chat_template_enable_thinking: bool | None = None,
) -> Any:
    """Convert normalized questions into chat-template prompts for GRPO."""

    def process_data(examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        task_types = examples.get("task_type", ["math"] * len(examples["question"]))
        reasoning_langs = examples.get("reasoning_lang", examples.get("reasoning_language"))
        if reasoning_langs is None and default_reasoning_lang:
            reasoning_langs = [default_reasoning_lang] * len(examples["question"])
        if reasoning_langs is None:
            reasoning_langs = [None] * len(examples["question"])
        elif default_reasoning_lang:
            reasoning_langs = [
                default_reasoning_lang if value is None or str(value).strip() == "" else value
                for value in reasoning_langs
            ]
        prompts = [
            apply_chat_template_text(
                tokenizer,
                build_interleaved_messages(
                    question,
                    task_type=task_type,
                    reasoning_lang=reasoning_lang,
                ),
                enable_thinking=chat_template_enable_thinking,
            )
            for question, task_type, reasoning_lang in zip(examples["question"], task_types, reasoning_langs)
        ]
        processed = {"prompt": prompts, "answer": examples["answer"], "task_type": task_types}
        if "reasoning_lang" in examples or "reasoning_language" in examples or default_reasoning_lang:
            processed["reasoning_lang"] = reasoning_langs
        if "tool_definitions" in examples:
            processed["tool_definitions"] = examples["tool_definitions"]
        if "mock_outputs" in examples:
            processed["mock_outputs"] = examples["mock_outputs"]
        if "target_has_tool_call" in examples:
            processed["target_has_tool_call"] = examples["target_has_tool_call"]
        return processed

    return dataset.map(process_data, batched=True, remove_columns=dataset.column_names)


def apply_chat_template_text(
    tokenizer: Any,
    messages: list[dict[str, str]],
    enable_thinking: bool | None = None,
) -> str:
    """Apply the tokenizer chat template with optional Qwen thinking-mode passthrough."""
    kwargs: dict[str, Any] = {}
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **kwargs,
    )


def _prompt_token_lengths(tokenizer: Any, prompts: list[str]) -> list[int]:
    tokenized = tokenizer(prompts, add_special_tokens=False, padding=False, truncation=False)
    return [len(input_ids) for input_ids in tokenized["input_ids"]]


def filter_overlong_prompts(
    dataset: Any,
    tokenizer: Any,
    max_prompt_length: int,
    enabled: bool = True,
) -> Any:
    """Drop rows whose already-formatted prompt is too long for GRPO."""
    if not enabled:
        return dataset

    original_size = len(dataset)

    def within_prompt_limit(examples: dict[str, list[str]]) -> list[bool]:
        return [length <= max_prompt_length for length in _prompt_token_lengths(tokenizer, examples["prompt"])]

    filtered = dataset.filter(
        within_prompt_limit,
        batched=True,
        desc=f"Filtering prompts longer than {max_prompt_length} tokens",
    )
    filtered_size = len(filtered)
    dropped = original_size - filtered_size
    print(
        f"Prompt length filter kept {filtered_size}/{original_size} rows "
        f"and dropped {dropped} rows over {max_prompt_length} tokens."
    )

    if filtered_size == 0:
        raise ValueError(
            "Prompt length filtering removed every training row. "
            "Increase --max-prompt-length or pass --no-filter-overlong-prompts."
        )
    return filtered


def main() -> None:
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    args = build_parser().parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = apply_interleaved_chat_template(
        dataset_from_args(args),
        tokenizer,
        default_reasoning_lang=args.reasoning_lang,
        chat_template_enable_thinking=args.chat_template_enable_thinking,
    )
    dataset = filter_overlong_prompts(
        dataset,
        tokenizer,
        max_prompt_length=args.max_prompt_length,
        enabled=args.filter_overlong_prompts,
    )
    reward_funcs = select_reward_funcs(args, dataset)
    reward_weights = resolve_reward_weights(args, reward_funcs)
    peft_config = build_peft_config(args)

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
        reward_weights=reward_weights,
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
    model_kwargs = build_trainer_model_kwargs(args, trainer_cls, peft_config)

    trainer = trainer_cls(
        **model_kwargs,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=dataset,
        reward_funcs=build_reward_funcs(args, reward_funcs, reward_weights),
    )
    trainer.train()


if __name__ == "__main__":
    main()
