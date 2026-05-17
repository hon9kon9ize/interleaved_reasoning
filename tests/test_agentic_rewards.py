from interleaved_grpo.environment import InterleavedEnvironmentSimulator, answers_equivalent
from interleaved_grpo.datasets_loader import _tool_call_signature
from interleaved_grpo.rewards import (
    REWARD_FUNCS,
    conditional_step_reward_fn,
    efficiency_penalty_fn,
    outcome_correctness_reward_fn,
    structural_interleaving_reward_fn,
    ttft_reward_fn,
)


def test_reward_funcs_use_four_pillar_agentic_suite():
    assert [fn.__name__ for fn in REWARD_FUNCS] == [
        "outcome_correctness_reward_fn",
        "conditional_step_reward_fn",
        "ttft_reward_fn",
        "efficiency_penalty_fn",
    ]


def test_outcome_correctness_uses_answer_tags_and_math_equivalence():
    completions = [
        "<think>Compute half.</think><answer>1/2</answer><think>0.5 is equivalent.</think>",
        "<think>Wrong.</think><answer>2</answer>",
    ]

    assert outcome_correctness_reward_fn(["q1", "q2"], completions, answer=["0.5", "3"]) == [1.0, 0.0]


def test_outcome_correctness_uses_tool_simulator_for_tool_tasks():
    completion = (
        '<think>Use the calculator.</think>'
        '<tool_call>{"name": "calculator", "arguments": {"expression": "2 + 2"}}</tool_call>'
        '<think>The output 4 solves it.</think>'
    )

    assert outcome_correctness_reward_fn(["q"], [completion], answer=["4"], task_type=["tool"]) == [1.0]


def test_outcome_correctness_matches_literal_tool_call_targets():
    tool_definitions = [
        {
            "type": "function",
            "function": {
                "name": "cancel_pending_order",
                "parameters": {
                    "type": "object",
                    "properties": {"order_id": {"type": "string"}},
                    "required": ["order_id"],
                },
            },
        }
    ]
    target = '<tool_call>{"name":"cancel_pending_order","arguments":{"order_id":"#W1"}}</tool_call>'
    good_completion = (
        "<think>Cancel the pending order.</think>"
        '<tool_call>{"arguments":{"order_id":"#W1"},"name":"cancel_pending_order"}</tool_call>'
    )
    wrong_completion = (
        "<think>Cancel the wrong order.</think>"
        '<tool_call>{"name":"cancel_pending_order","arguments":{"order_id":"#W2"}}</tool_call>'
    )

    assert outcome_correctness_reward_fn(
        ["q1", "q2"],
        [good_completion, wrong_completion],
        answer=[target, target],
        task_type=["tool", "tool"],
        target_has_tool_call=[True, True],
        tool_definitions=[tool_definitions, tool_definitions],
    ) == [1.0, 0.0]


def test_outcome_correctness_accepts_explicit_toolmind_text_targets():
    completion = "<think>Summarize the observed status.</think>The order is pending."

    assert outcome_correctness_reward_fn(
        ["q"],
        [completion],
        answer=["The order is pending."],
        task_type=["tool"],
        target_has_tool_call=[False],
    ) == [1.0]


def test_outcome_correctness_compares_public_text_for_escaped_toolmind_traces():
    completion = "<think>Private final reasoning.</think>The order is pending."
    target = "&lt;think&gt;Reference private reasoning.&lt;/think&gt;The order is pending."

    assert outcome_correctness_reward_fn(
        ["q"],
        [completion],
        answer=[target],
        task_type=["tool"],
        target_has_tool_call=[False],
    ) == [1.0]


def test_outcome_correctness_uses_knowledge_base_for_search_tools():
    completion = (
        "<think>Search for the date.</think>"
        '<tool_call>{"name": "web_search", "arguments": {"query": "when did the berlin wall fall?"}}</tool_call>'
        "<think>The output 1989 answers the question.</think>"
    )

    assert outcome_correctness_reward_fn(
        ["q"],
        [completion],
        answer=["1989"],
        task_type=["tool"],
        kb={"berlin wall fall": "1989"},
    ) == [1.0]


def test_outcome_correctness_rejects_tool_tasks_without_tool_calls():
    completion = "<think>Guess directly.</think><answer>4</answer><think>The answer is 4.</think>"

    assert outcome_correctness_reward_fn(["q"], [completion], answer=["4"], task_type=["tool"]) == [0.0]


def test_structural_interleaving_rewards_plan_action_reflection():
    completion = "<think>Plan.</think><answer>4</answer><think>The answer 4 matches the calculation.</think>"

    assert structural_interleaving_reward_fn(["q"], [completion]) == [1.0]


def test_ttft_reward_penalizes_long_first_think():
    long_thought = " ".join(["token"] * 300)
    completion = f"<think>{long_thought}</think><answer>4</answer>"
    fast_plan = (
        "<think>Plan use arithmetic then answer after checking the sum quickly now.</think>"
        "<answer>4</answer>"
    )
    empty_plan = "<think></think><answer>4</answer>"

    assert ttft_reward_fn(["q"], [completion])[0] < 0.0
    assert ttft_reward_fn(["q"], [fast_plan]) == [0.3]
    assert ttft_reward_fn(["q"], [empty_plan]) == [0.0]


def test_conditional_step_reward_requires_correct_final_answer():
    good_structure = "<think>Plan.</think><answer>4</answer><think>The answer 4 matches.</think>"

    assert conditional_step_reward_fn(["q"], [good_structure], answer=["4"]) == [1.0]
    assert conditional_step_reward_fn(["q"], [good_structure], answer=["5"]) == [0.0]


def test_conditional_step_reward_uses_tool_simulator_for_tool_tasks():
    completion = (
        '<think>Use calculator now.</think>'
        '<tool_call>{"name": "calculator", "arguments": {"expression": "2 + 2"}}</tool_call>'
        '<think>The output 4 confirms the result.</think>'
    )

    assert conditional_step_reward_fn(["q"], [completion], answer=["4"], task_type=["tool"]) == [1.0]


def test_conditional_step_reward_rejects_failed_intermediate_tool_call():
    completion = (
        "<think>Use tools.</think>"
        '<tool_call>{"name": "unknown_tool", "arguments": {}}</tool_call>'
        '<tool_call>{"name": "final_answer", "arguments": {"value": "4"}}</tool_call>'
        "<think>The answer 4 is done.</think>"
    )

    assert conditional_step_reward_fn(["q"], [completion], answer=["4"], task_type=["tool"]) == [0.0]


def test_environment_simulator_executes_calculator_and_verifies_completion():
    simulator = InterleavedEnvironmentSimulator()
    result = simulator.execute_tool_call('{"name": "calculator", "arguments": {"expression": "2 + 2"}}')

    assert result.ok
    assert answers_equivalent(result.output, "4")
    assert simulator.verify_completion("<think>Plan.</think><answer>1/2</answer>", "0.5")


def test_environment_simulator_replay_mock_outputs_override_builtin_tools():
    from interleaved_grpo.datasets_loader import _tool_call_signature

    signature = _tool_call_signature("calculate", {"expression": "2 + 2"})
    simulator = InterleavedEnvironmentSimulator(mock_outputs={signature: "__target__"})

    result = simulator.execute_tool_call({"name": "calculate", "arguments": {"expression": "2 + 2"}})

    assert result.ok
    assert result.output == "__target__"


def test_environment_simulator_replay_mock_outputs_allow_repeated_gold_calls():
    from interleaved_grpo.datasets_loader import _tool_call_signature

    call = {"name": "custom_die", "arguments": {"sides": 26}}
    signature = _tool_call_signature("custom_die", {"sides": 26})
    simulator = InterleavedEnvironmentSimulator(mock_outputs={signature: "__target__"})

    assert [simulator.execute_tool_call(call).ok for _ in range(5)] == [True, True, True, True, True]


def test_environment_simulator_requires_all_tool_calls_to_succeed():
    simulator = InterleavedEnvironmentSimulator()
    completion = (
        "<think>Use tools.</think>"
        '<tool_call>{"name": "unknown_tool", "arguments": {}}</tool_call>'
        '<tool_call>{"name": "final_answer", "arguments": {"value": "4"}}</tool_call>'
        "<think>The answer 4 is done.</think>"
    )

    assert not simulator.verify_completion(completion, "4", task_type="tool")


def test_environment_simulator_detects_redundant_calls():
    simulator = InterleavedEnvironmentSimulator()
    call = {"name": "calculator", "arguments": {"expression": "2 + 2"}}

    assert simulator.execute_tool_call(call).ok
    assert simulator.execute_tool_call(call).ok
    third_result = simulator.execute_tool_call(call)
    assert not third_result.ok
    assert third_result.error == "redundant_call_detected"


def test_environment_simulator_search_uses_keyword_overlap():
    simulator = InterleavedEnvironmentSimulator(knowledge_base={"berlin wall fall": "1989"})

    result = simulator.execute_tool_call(
        {"name": "search_knowledge", "arguments": {"query": "when did the berlin wall fall?"}}
    )

    assert result.ok
    assert result.output == "1989"


def test_environment_simulator_validates_registered_toolmind_schema():
    tool_definitions = [
        {
            "type": "function",
            "function": {
                "name": "fetchDataWithTimestamp",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "endpoint": {"type": "string"},
                        "timestamp": {"type": "string", "pattern": r"^\d{2}-\d{2}$"},
                    },
                    "required": ["endpoint"],
                },
            },
        }
    ]
    simulator = InterleavedEnvironmentSimulator(tool_definitions=tool_definitions)

    ok_result = simulator.execute_tool_call(
        {
            "name": "fetchDataWithTimestamp",
            "arguments": {"endpoint": "https://example.com/data", "timestamp": "05-15"},
        }
    )
    missing_result = simulator.execute_tool_call({"name": "fetchDataWithTimestamp", "arguments": {}})
    pattern_result = simulator.execute_tool_call(
        {
            "name": "fetchDataWithTimestamp",
            "arguments": {"endpoint": "https://example.com/data", "timestamp": "2026-05-15"},
        }
    )

    assert ok_result.ok
    assert "mocked" in ok_result.output
    assert missing_result.error == "missing_required_argument:endpoint"
    assert pattern_result.error == "invalid_argument_pattern:timestamp"


def test_environment_simulator_accepts_toolmind_tool_names_with_spaces():
    tool_definitions = [
        {
            "type": "function",
            "function": {
                "name": "Copy Endpoint",
                "parameters": {
                    "type": "dict",
                    "properties": {"endpoint_url": {"type": "string"}},
                    "required": ["endpoint_url"],
                },
            },
        }
    ]
    simulator = InterleavedEnvironmentSimulator(tool_definitions=tool_definitions)

    result = simulator.execute_tool_call(
        {
            "name": "Copy Endpoint",
            "arguments": {"endpoint_url": "https://nguyenthanhduy178.tk/api"},
        }
    )

    assert result.ok


def test_registered_toolmind_tool_can_be_part_of_verified_trajectory():
    tool_definitions = [
        {
            "type": "function",
            "function": {
                "name": "List All Weather APIs",
                "parameters": {"type": "dict", "properties": {}, "required": []},
            },
        }
    ]
    completion = (
        "<think>List options.</think>"
        '<tool_call>{"name": "List All Weather APIs", "arguments": {}}</tool_call>'
        "<think>Pick the final answer.</think>"
        '<tool_call>{"name": "final_answer", "arguments": {"value": "WeatherAPI"}}'
        "</tool_call>"
    )
    simulator = InterleavedEnvironmentSimulator(tool_definitions=tool_definitions)

    assert simulator.verify_completion(completion, "WeatherAPI", task_type="tool")


def test_reward_passes_registered_tool_definitions_to_simulator():
    tool_definitions = [
        {
            "type": "function",
            "function": {
                "name": "List All Weather APIs",
                "parameters": {"type": "dict", "properties": {}, "required": []},
            },
        }
    ]
    completion = (
        "<think>List options.</think>"
        '<tool_call>{"name": "List All Weather APIs", "arguments": {}}</tool_call>'
        "<think>Use the listed result.</think>"
        '<tool_call>{"name": "final_answer", "arguments": {"value": "WeatherAPI"}}'
        "</tool_call>"
    )

    assert outcome_correctness_reward_fn(
        ["q"],
        [completion],
        answer=["WeatherAPI"],
        task_type=["tool"],
        tool_definitions=[tool_definitions],
    ) == [1.0]


def test_reward_accepts_json_string_tool_definitions_and_mock_outputs():
    import json

    tool_definitions = [
        {
            "type": "function",
            "function": {
                "name": "get_order_details",
                "parameters": {
                    "type": "object",
                    "properties": {"order_id": {"type": "string"}},
                    "required": ["order_id"],
                },
            },
        }
    ]
    completion = '<think>Fetch order.</think><tool_call>{"name":"get_order_details","arguments":{"order_id":"#W1"}}</tool_call>'
    signature = _tool_call_signature("get_order_details", {"order_id": "#W1"})

    assert outcome_correctness_reward_fn(
        ["q"],
        [completion],
        answer=["pending"],
        task_type=["tool"],
        tool_definitions=[json.dumps(tool_definitions)],
        mock_outputs=[json.dumps({signature: "pending"})],
    ) == [1.0]


def test_reward_uses_per_row_json_tool_definitions_in_batches():
    import json

    first_tools = [
        {
            "type": "function",
            "function": {
                "name": "first_tool",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }
    ]
    second_tools = [
        {
            "type": "function",
            "function": {
                "name": "second_tool",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }
    ]
    completions = [
        '<think>Do first.</think><tool_call>{"name":"first_tool","arguments":{}}</tool_call>',
        '<think>Do second.</think><tool_call>{"name":"second_tool","arguments":{}}</tool_call>',
    ]

    assert outcome_correctness_reward_fn(
        ["q1", "q2"],
        completions,
        answer=["ok1", "ok2"],
        task_type=["tool", "tool"],
        tool_definitions=[json.dumps(first_tools), json.dumps(second_tools)],
        mock_outputs=[json.dumps({"first_tool": "ok1"}), json.dumps({"second_tool": "ok2"})],
    ) == [1.0, 1.0]
