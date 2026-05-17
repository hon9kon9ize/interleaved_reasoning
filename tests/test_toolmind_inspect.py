from interleaved_grpo.toolmind_inspect import (
    coerce_tool_definitions,
    extract_row_tool_definitions,
    resolve_split,
    summarize_tool_definitions,
)


def test_coerce_tool_definitions_accepts_json_string_column():
    value = '[{"type": "function", "function": {"name": "get_all_coins_prices"}}]'

    definitions = coerce_tool_definitions(value)

    assert definitions == [{"type": "function", "function": {"name": "get_all_coins_prices"}}]


def test_extract_row_tool_definitions_checks_common_columns():
    row = {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "fetchDataWithTimestamp",
                    "parameters": {"type": "dict", "properties": {}, "required": ["endpoint"]},
                },
            }
        ]
    }

    assert extract_row_tool_definitions(row)[0]["function"]["name"] == "fetchDataWithTimestamp"


def test_summarize_tool_definitions_counts_unique_tools():
    rows = [
        {"tools": [{"type": "function", "function": {"name": "List All Weather APIs"}}]},
        {"tools": [{"type": "function", "function": {"name": "List All Weather APIs"}}]},
        {
            "tools": [
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
        },
    ]

    summary = summarize_tool_definitions(rows)

    assert summary["List All Weather APIs"]["count"] == 2
    assert summary["Copy Endpoint"]["required"] == ["endpoint_url"]


def test_resolve_split_falls_back_from_train():
    assert (
        resolve_split("Nanbeige/ToolMind", None, "train", split_names=["graph_syn_datasets", "open_datasets"])
        == "graph_syn_datasets"
    )
