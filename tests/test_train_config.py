from interleaved_grpo.train import build_parser, with_sequential_train_sampler


def test_train_defaults_match_agentic_reward_suite():
    args = build_parser().parse_args([])

    assert args.beta == 0.01
    assert args.reward_weights == [1.0, 0.5, 0.3, 1.0]
    assert args.sequential_hybrid_sampler


def test_sequential_sampler_trainer_preserves_dataset_order():
    class FakeTrainer:
        def __init__(self, train_dataset):
            self.train_dataset = train_dataset

    trainer_cls = with_sequential_train_sampler(FakeTrainer)
    sampler = trainer_cls(["math", "tool"])._get_train_sampler()

    assert list(sampler) == [0, 1]
