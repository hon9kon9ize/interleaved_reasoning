"""Train an interleaved CoT policy with TRL GRPO."""

from __future__ import annotations

import argparse
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
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--max-prompt-length", type=int, default=512)
    parser.add_argument("--max-completion-length", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--report-to", default="tensorboard")
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
        report_to=args.report_to,
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
        reward_funcs=REWARD_FUNCS,
    )
    trainer.train()


if __name__ == "__main__":
    main()
