"""Inspect ToolMind tool schemas before deciding what to mock."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from typing import Any, Iterable


TOOL_DEFINITION_COLUMNS = ("tool_definitions", "tools", "functions", "available_tools")


def coerce_tool_definitions(value: Any) -> list[dict[str, Any]]:
    """Normalize a ToolMind tool column value into a list of function definitions."""
    if value is None or value == "":
        return []
    if isinstance(value, str):
        try:
            return coerce_tool_definitions(json.loads(value))
        except json.JSONDecodeError:
            return []
    if isinstance(value, dict):
        if "function" in value or "name" in value:
            return [value]
        for key in TOOL_DEFINITION_COLUMNS:
            if key in value:
                return coerce_tool_definitions(value[key])
        return []
    if isinstance(value, list):
        definitions: list[dict[str, Any]] = []
        for item in value:
            definitions.extend(coerce_tool_definitions(item))
        return definitions
    return []


def extract_row_tool_definitions(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract tool definitions from a dataset row using common ToolMind column names."""
    for column in TOOL_DEFINITION_COLUMNS:
        if column in row:
            definitions = coerce_tool_definitions(row[column])
            if definitions:
                return definitions
    return []


def get_tool_function(definition: dict[str, Any]) -> dict[str, Any]:
    """Return the inner function schema from OpenAI-style or flat tool definitions."""
    function = definition.get("function", definition)
    return function if isinstance(function, dict) else {}


def get_required_arguments(function: dict[str, Any]) -> list[str]:
    """Extract required argument names from common schema shapes."""
    parameters = function.get("parameters") or {}
    arguments = function.get("arguments") or {}
    required = parameters.get("required") or function.get("required") or []
    if isinstance(required, list):
        return [str(item) for item in required]
    if isinstance(arguments, dict):
        return [str(key) for key, spec in arguments.items() if isinstance(spec, dict) and spec.get("required")]
    return []


def summarize_tool_definitions(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Summarize unique tools and their observed frequency from dataset rows."""
    counts: Counter[str] = Counter()
    summaries: dict[str, dict[str, Any]] = {}
    for row in rows:
        for definition in extract_row_tool_definitions(row):
            function = get_tool_function(definition)
            name = str(function.get("name", "")).strip()
            if not name:
                continue
            counts[name] += 1
            summaries.setdefault(
                name,
                {
                    "name": name,
                    "count": 0,
                    "description": function.get("description", ""),
                    "required": get_required_arguments(function),
                    "schema_keys": sorted(key for key in function.keys() if key not in {"description"}),
                },
            )

    for name, count in counts.items():
        summaries[name]["count"] = count
    return dict(sorted(summaries.items(), key=lambda item: (-item[1]["count"], item[0])))


def resolve_split(
    dataset_id: str,
    subset: str | None,
    requested_split: str,
    split_names: list[str] | None = None,
) -> str:
    """Return a valid split, falling back from `train` for ToolMind-style datasets."""
    if split_names is None:
        from datasets import get_dataset_split_names

        split_names = get_dataset_split_names(dataset_id, subset)
    if requested_split in split_names:
        return requested_split
    if requested_split == "train" and split_names:
        return split_names[0]
    raise ValueError(f'Unknown split "{requested_split}". Should be one of {split_names}.')


def load_toolmind_rows(dataset_id: str, split: str, subset: str | None, max_samples: int | None) -> list[dict[str, Any]]:
    """Download a ToolMind split and return rows for local inspection."""
    from datasets import load_dataset

    resolved_split = resolve_split(dataset_id, subset, split)
    dataset = load_dataset(dataset_id, subset, split=resolved_split)
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    return [dict(row) for row in dataset]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default="Nanbeige/ToolMind")
    parser.add_argument("--subset", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--list-splits", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.list_splits:
        from datasets import get_dataset_split_names

        print(json.dumps(get_dataset_split_names(args.dataset_id, args.subset)))
        return
    rows = load_toolmind_rows(args.dataset_id, args.split, args.subset, args.max_samples)
    print(json.dumps(summarize_tool_definitions(rows), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
