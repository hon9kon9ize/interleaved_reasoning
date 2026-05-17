from interleaved_grpo.parser import extract_math_answer, extract_tagged_answer, parse_interleaved_stream
from interleaved_grpo.rewards import (
    correctness_reward_fn,
    efficiency_penalty_fn,
    interleaved_format_reward_fn,
)


def test_parser_accepts_interleaved_stream():
    text = "Start <think>a</think> middle <think>b</think> final answer is 42"
    stats = parse_interleaved_stream(text)
    assert stats["is_valid_xml"]
    assert stats["num_transitions"] == 2
    assert stats["is_interleaved"]
    assert stats["public_text"] == "Start middle final answer is 42"


def test_parser_rejects_unclosed_and_nested_tags():
    assert not parse_interleaved_stream("A <think>missing close")["is_valid_xml"]
    assert not parse_interleaved_stream("A </think>missing open")["is_valid_xml"]
    assert not parse_interleaved_stream("<think>a <think>b</think></think>")["is_valid_xml"]


def test_extract_math_answer_prefers_public_boxed_answer():
    text = "<think>wrong \\boxed{1}</think> The final answer is \\boxed{7}."
    assert extract_math_answer(text) == "7"


def test_extract_tagged_answer_reads_only_answer_blocks():
    text = "The final answer is 7. <answer>9</answer>"
    assert extract_tagged_answer(text) == "9"
    assert extract_tagged_answer("The final answer is 7.") == ""


def test_extract_math_answer_numeric_fallback():
    assert extract_math_answer("We compute 19, so the final answer is 19.") == "19"
    assert extract_math_answer("#### 1,234") == "1234"


def test_rewards_cover_correctness_format_and_efficiency():
    completions = ["A <think>x</think> B <think>y</think> C <think>z</think><answer>9</answer>"]
    prompts = ["Question: 4+5\nAnswer:"]
    assert correctness_reward_fn(prompts, completions, ["9"]) == [1.0]
    assert interleaved_format_reward_fn(prompts, completions)[0] == 1.0
    assert efficiency_penalty_fn(prompts, completions)[0] == 0.0


def test_format_reward_does_not_reward_missing_think_tags():
    assert interleaved_format_reward_fn(["Question"], ["<answer>9</answer>"]) == [0.0]


def test_correctness_reward_accepts_zero_answers():
    completions = ["<answer>0</answer>"]
    assert correctness_reward_fn(["Question"], completions, ["0"]) == [1.0]


def test_format_reward_preserves_interleaving_premium():
    tag_spam = "<think></think><think></think><think></think><answer>9</answer>"
    interleaved = "A <think>x</think> B <think>y</think> C <think>z</think><answer>9</answer>"
    rewards = interleaved_format_reward_fn(["Question", "Question"], [tag_spam, interleaved])
    assert rewards == [0.7, 1.0]


def test_efficiency_penalty_uses_word_counts_and_caps_penalty():
    completion = (
        "<think>one two three four five six seven eight nine ten eleven twelve thirteen "
        "fourteen fifteen sixteen seventeen eighteen nineteen twenty</think> final"
    )
    assert efficiency_penalty_fn(["Question"], [completion]) == [-0.5]
