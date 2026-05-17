from interleaved_grpo.datasets_loader import (
    _conversation_to_prompt,
    _format_interleaved_tool_completion,
    _extract_replay_mock_outputs,
    _extract_toolmind_target,
    _tool_call_signature,
)
from interleaved_grpo.parser import parse_interleaved_stream


def test_extract_replay_mock_outputs_from_conversation_tool_turns():
    conversations = [
        {"role": "user", "content": "Find my order."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "get_order_details", "arguments": {"order_id": "#W1"}}}],
        },
        {"role": "tool", "content": '{"order_id":"#W1","status":"pending"}'},
        {
            "role": "assistant",
            "content": "<think>Cancel now.</think>",
            "tool_calls": [{"function": {"name": "cancel_pending_order", "arguments": {"order_id": "#W1"}}}],
        },
    ]

    outputs = _extract_replay_mock_outputs(conversations)

    assert outputs == {
        _tool_call_signature("get_order_details", {"order_id": "#W1"}): '{"order_id":"#W1","status":"pending"}'
    }


def test_extract_toolmind_target_preserves_final_tool_call_as_literal_target():
    conversations = [
        {"role": "user", "content": "Cancel order."},
        {
            "role": "assistant",
            "content": "<think>Cancel now.</think>",
            "tool_calls": [{"function": {"name": "cancel_pending_order", "arguments": {"order_id": "#W1"}}}],
        },
    ]

    target, has_tool_call = _extract_toolmind_target(conversations)
    stats = parse_interleaved_stream(target)

    assert target.startswith("&lt;think&gt;Cancel now.&lt;/think&gt;")
    assert has_tool_call
    assert stats["is_valid_xml"]
    assert stats["num_tool_calls"] == 1


def test_conversation_to_prompt_excludes_target_assistant_message():
    conversations = [
        {"role": "system", "content": "Policy"},
        {"role": "user", "content": "Cancel order."},
        {
            "role": "assistant",
            "content": "<think>Cancel now.</think>",
            "tool_calls": [{"function": {"name": "cancel_pending_order", "arguments": {"order_id": "#W1"}}}],
        },
    ]

    prompt = _conversation_to_prompt(conversations)

    assert "system: Policy" in prompt
    assert "user: Cancel order." in prompt
    assert "cancel_pending_order" not in prompt


def test_format_interleaved_tool_completion_escapes_tag_literals_in_content():
    completion = _format_interleaved_tool_completion(
        "Mention literal <tool_call> in prose.",
        [{"name": "do_work", "arguments": {}}],
    )
    stats = parse_interleaved_stream(completion)

    assert "&lt;tool_call&gt;" in completion
    assert stats["is_valid_xml"]
    assert stats["num_tool_calls"] == 1
