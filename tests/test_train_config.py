import json

from interleaved_grpo.train import (
    GenerationRewardLogger,
    build_parser,
    generation_log_path,
    resolve_report_to,
    with_sequential_train_sampler,
)


def test_train_defaults_match_agentic_reward_suite():
    args = build_parser().parse_args([])

    assert args.beta == 0.01
    assert args.num_generations == 4
    assert args.reward_weights == [1.0, 0.5, 0.3, 1.0]
    assert args.sequential_hybrid_sampler
    assert generation_log_path(args) == "interleaved_grpo_output/generations.jsonl"


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
