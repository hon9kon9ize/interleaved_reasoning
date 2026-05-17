import pytest

from interleaved_grpo.rewards import (
    correctness_reward_fn,
    efficiency_penalty_fn,
    interleaved_format_reward_fn,
    language_consistency_reward_fn,
)


def test_correctness_reward_scores_exact_normalized_answers():
    prompts = ["q1", "q2", "q3", "q4"]
    completions = [
        "<answer>42</answer>",
        "<answer>0</answer>",
        "<answer>7</answer>",
        "I cannot determine it.",
    ]
    answers = ["42", "0", "8", "9"]

    assert correctness_reward_fn(prompts, completions, answers) == [1.0, 1.0, 0.0, 0.0]


@pytest.mark.parametrize(
    ("completion", "expected"),
    [
        ("<answer>9</answer>", 0.0),
        ("<think>x</think><answer>9</answer>", 0.3),
        ("A <think>x</think> B <think>y</think><answer>9</answer>", 0.8),
        ("A <think>x</think> B <think>y</think> C <think>z</think><answer>9</answer>", 1.0),
        ("<think></think><think></think><think></think><answer>9</answer>", 0.7),
        ("A <think>x broken", 0.0),
    ],
)
def test_interleaved_format_reward_scores_structure(completion, expected):
    assert interleaved_format_reward_fn(["question"], [completion]) == [pytest.approx(expected)]


@pytest.mark.parametrize(
    ("completion", "expected"),
    [
        ("<think>short private work</think> public answer has several words", 0.0),
        ("<think>private only</think>", -0.5),
        (
            "<think>one two three four five six seven eight nine ten eleven twelve thirteen "
            "fourteen fifteen sixteen seventeen eighteen nineteen twenty</think> final",
            -0.5,
        ),
    ],
)
def test_efficiency_penalty_scores_private_public_ratio(completion, expected):
    assert efficiency_penalty_fn(["question"], [completion]) == [expected]


def test_language_consistency_reward_scores_reasoning_language():
    completions = [
        "<think>首先，我哋要計吓總數，跟住再除返每日可以完成嘅數量。</think><answer>7</answer>",
        "<think>First, I calculate the total and divide by the daily capacity.</think><answer>7</answer>",
        "<answer>7</answer>",
    ]

    assert language_consistency_reward_fn(
        ["q1", "q2", "q3"],
        completions,
        reasoning_language=["yue", "yue", "yue"],
    ) == [1.0, 0.0, 0.0]


def test_language_consistency_reward_supports_english_reasoning():
    completion = "<think>First, calculate the total number of towels, then divide by capacity.</think><answer>7</answer>"

    assert language_consistency_reward_fn(["q"], [completion], reasoning_language=["en"]) == [1.0]
