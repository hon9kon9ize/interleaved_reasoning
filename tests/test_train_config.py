import json

import pytest

from interleaved_grpo.train import (
    GenerationRewardLogger,
    build_peft_config,
    build_parser,
    build_trainer_model_kwargs,
    dataset_has_reasoning_lang,
    filter_overlong_prompts,
    generation_log_path,
    normalize_prefilled_think_completion,
    prompt_prefills_open_think,
    resolve_reward_weights,
    resolve_report_to,
    select_reward_funcs,
    trainer_accepts_peft_config,
    with_prefilled_think_normalization,
    with_sequential_train_sampler,
)


def test_train_defaults_match_agentic_reward_suite():
    args = build_parser().parse_args([])

    assert args.beta == 0.01
    assert args.num_generations == 4
    assert args.filter_overlong_prompts
    assert args.chat_template_enable_thinking is None
    assert args.normalize_prefilled_think
    assert not args.use_lora
    assert args.lora_rank == 16
    assert resolve_reward_weights(args, select_reward_funcs(args)) == [1.0, 0.5, 0.3, 1.0]
    assert args.sequential_hybrid_sampler
    assert generation_log_path(args) == "interleaved_grpo_output/generations.jsonl"


def test_lora_flags_parse_without_enabling_peft_import():
    args = build_parser().parse_args(
        [
            "--use-lora",
            "--lora-rank",
            "8",
            "--lora-alpha",
            "16",
            "--lora-dropout",
            "0.1",
            "--lora-target-modules",
            "q_proj,v_proj",
            "--lora-modules-to-save",
            "lm_head",
        ]
    )

    assert args.use_lora
    assert args.lora_rank == 8
    assert args.lora_alpha == 16
    assert args.lora_dropout == 0.1
    assert args.lora_target_modules == "q_proj,v_proj"
    assert args.lora_modules_to_save == "lm_head"


def test_build_peft_config_disabled_does_not_require_peft():
    args = build_parser().parse_args([])

    assert build_peft_config(args) is None


def test_build_trainer_model_kwargs_passes_peft_config_when_supported():
    class FakeTrainerWithPeft:
        def __init__(self, model, peft_config=None):
            pass

    args = build_parser().parse_args(["--model-id", "test/model"])
    peft_config = object()

    assert trainer_accepts_peft_config(FakeTrainerWithPeft)
    assert build_trainer_model_kwargs(args, FakeTrainerWithPeft, peft_config) == {
        "model": "test/model",
        "peft_config": peft_config,
    }


def test_prefilled_think_completion_normalization_repairs_qwen3_completion():
    prompt = "<|im_start|>assistant\n<think>\n"
    completion = "Plan briefly.</think><answer>4</answer>"

    assert prompt_prefills_open_think(prompt)
    assert normalize_prefilled_think_completion(prompt, completion) == "<think>Plan briefly.</think><answer>4</answer>"
    assert normalize_prefilled_think_completion(prompt, "<think>Already tagged.</think>") == "<think>Already tagged.</think>"
    assert normalize_prefilled_think_completion("balanced <think>x</think>", completion) == completion


def test_reward_wrapper_scores_normalized_prefilled_think_completion():
    def needs_open_think(prompts, completions, **kwargs):
        return [1.0 if completion.startswith("<think>") else 0.0 for completion in completions]

    wrapped = with_prefilled_think_normalization(needs_open_think)

    assert wrapped(["assistant\n<think>\n"], ["Plan.</think>"]) == [1.0]


def test_filter_overlong_prompts_drops_rows_by_tokenized_prompt_length():
    class FakeDataset:
        def __init__(self, rows):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def filter(self, function, batched=False, desc=None):
            assert batched
            assert desc
            examples = {"prompt": [row["prompt"] for row in self.rows]}
            keep = function(examples)
            return FakeDataset([row for row, should_keep in zip(self.rows, keep) if should_keep])

    class WhitespaceTokenizer:
        def __call__(self, prompts, **kwargs):
            return {"input_ids": [prompt.split() for prompt in prompts]}

    dataset = FakeDataset(
        [
            {"prompt": "one two"},
            {"prompt": "one two three four"},
            {"prompt": "one"},
        ]
    )

    filtered = filter_overlong_prompts(dataset, WhitespaceTokenizer(), max_prompt_length=2)

    assert [row["prompt"] for row in filtered.rows] == ["one two", "one"]


def test_filter_overlong_prompts_can_be_disabled():
    dataset = object()

    assert filter_overlong_prompts(dataset, tokenizer=None, max_prompt_length=1, enabled=False) is dataset


def test_filter_overlong_prompts_rejects_empty_result():
    class FakeDataset:
        rows = [{"prompt": "too long"}]

        def __len__(self):
            return len(self.rows)

        def filter(self, function, batched=False, desc=None):
            return EmptyDataset()

    class EmptyDataset:
        def __len__(self):
            return 0

    with pytest.raises(ValueError, match="removed every training row"):
        filter_overlong_prompts(FakeDataset(), tokenizer=lambda prompts, **_: {"input_ids": [[1, 2, 3]]}, max_prompt_length=1)


def test_reasoning_lang_selects_language_reward():
    args = build_parser().parse_args(["--dataset", "reasoning-lang"])
    reward_funcs = select_reward_funcs(args, {"reasoning_lang": ["yue"]})

    assert reward_funcs[-1].__name__ == "language_consistency_reward_fn"
    assert resolve_reward_weights(args, reward_funcs) == [1.0, 0.5, 0.3, 1.0, 0.2]


def test_reasoning_lang_column_selects_language_reward_for_any_dataset():
    args = build_parser().parse_args([])
    dataset = {"prompt": ["q"], "answer": ["4"], "reasoning_lang": ["yue"]}
    reward_funcs = select_reward_funcs(args, dataset)

    assert dataset_has_reasoning_lang(dataset)
    assert reward_funcs[-1].__name__ == "language_consistency_reward_fn"


def test_reasoning_lang_cli_fallback_selects_language_reward():
    args = build_parser().parse_args(["--reansoning_lang", "yue"])
    reward_funcs = select_reward_funcs(args, {"prompt": ["q"], "answer": ["4"]})

    assert args.reasoning_lang == "yue"
    assert reward_funcs[-1].__name__ == "language_consistency_reward_fn"


def test_wandb_flag_adds_wandb_report_target():
    args = build_parser().parse_args(["--wandb"])

    assert resolve_report_to(args) == ["tensorboard", "wandb"]


def test_generation_logger_writes_one_record_per_completion(tmp_path):
    def reward_a(prompts, completions, **kwargs):
        return [1.0, 0.0]

    def reward_b(prompts, completions, **kwargs):
        return [0.5, 0.25]

    log_file = tmp_path / "generations.jsonl"
    logger = GenerationRewardLogger(log_file, ["reward_a", "reward_b"], [1.0, 2.0])

    logger.wrap(reward_a)(
        ["prompt 1", "prompt 2"],
        ["completion 1", "completion 2"],
        answer=["4", "5"],
        task_type=["math", "tool"],
    )
    assert not log_file.exists()

    logger.wrap(reward_b)(
        ["prompt 1", "prompt 2"],
        ["completion 1", "completion 2"],
        answer=["4", "5"],
        task_type=["math", "tool"],
    )

    records = [json.loads(line) for line in log_file.read_text().splitlines()]
    assert len(records) == 2
    assert records[0]["completion"] == "completion 1"
    assert records[0]["answer"] == "4"
    assert records[0]["task_type"] == "math"
    assert records[0]["rewards"] == {"reward_a": 1.0, "reward_b": 0.5}
    assert records[0]["weighted_reward"] == 2.0


def test_sequential_sampler_trainer_preserves_dataset_order():
    class FakeTrainer:
        def __init__(self, train_dataset):
            self.train_dataset = train_dataset

    trainer_cls = with_sequential_train_sampler(FakeTrainer)
    sampler = trainer_cls(["math", "tool"])._get_train_sampler()

    assert list(sampler) == [0, 1]
