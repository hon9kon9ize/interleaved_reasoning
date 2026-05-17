from interleaved_grpo.parser import parse_interleaved_stream


REWARD_PARSER_KEYS = {
    "think_blocks",
    "public_text",
    "is_valid_xml",
    "num_transitions",
    "is_interleaved",
    "total_think_char_len",
    "total_public_char_len",
}


def test_parse_interleaved_stream_exposes_reward_contract_keys():
    stats = parse_interleaved_stream("A <think>x</think> B <think>y</think><answer>9</answer>")

    assert REWARD_PARSER_KEYS.issubset(stats)
    assert isinstance(stats["think_blocks"], list)
    assert isinstance(stats["public_text"], str)
    assert isinstance(stats["is_valid_xml"], bool)
    assert isinstance(stats["num_transitions"], int)
    assert isinstance(stats["is_interleaved"], bool)
    assert isinstance(stats["total_think_char_len"], int)
    assert isinstance(stats["total_public_char_len"], int)


def test_parse_interleaved_stream_marks_stacked_tags_as_not_interleaved():
    stats = parse_interleaved_stream("<think></think><think></think><think></think><answer>9</answer>")

    assert stats["is_valid_xml"]
    assert stats["num_transitions"] == 3
    assert not stats["is_interleaved"]
    assert stats["public_text"] == "9"


def test_parse_interleaved_stream_marks_alternating_sequence_as_interleaved():
    stats = parse_interleaved_stream("A <think>x</think> B <think>y</think> C <think>z</think> final")

    assert stats["is_valid_xml"]
    assert stats["num_transitions"] == 3
    assert stats["is_interleaved"]
    assert stats["think_blocks"] == ["x", "y", "z"]
    assert stats["public_text"] == "A B C final"


def test_parse_interleaved_stream_returns_contract_for_malformed_output():
    stats = parse_interleaved_stream("A <think>x broken")

    assert REWARD_PARSER_KEYS.issubset(stats)
    assert not stats["is_valid_xml"]
    assert not stats["is_interleaved"]


def test_parse_interleaved_stream_extracts_ordered_action_segments():
    stats = parse_interleaved_stream(
        "<think>Plan quickly.</think><answer>4</answer><think>The result 4 matches.</think>"
    )

    assert stats["is_valid_xml"]
    assert stats["segment_types"] == ["think", "answer", "think"]
    assert stats["answer_blocks"] == ["4"]
    assert stats["num_actions"] == 1
    assert stats["has_plan_action_reflection"]
    assert stats["num_reflections_referencing_actions"] == 1
