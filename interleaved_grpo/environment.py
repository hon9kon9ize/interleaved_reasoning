"""Deterministic environment helpers for agentic interleaved rewards."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .parser import extract_tagged_answer, normalize_math_answer, parse_interleaved_stream


@dataclass(frozen=True)
class ToolExecutionResult:
    """Result returned by the lightweight tool simulator."""

    ok: bool
    output: str
    error: str | None = None


def answers_equivalent(predicted: str, expected: str) -> bool:
    """Check exact normalized equality, then optional SymPy mathematical equivalence."""
    lhs = normalize_math_answer(predicted)
    rhs = normalize_math_answer(expected)
    if lhs == "" or rhs == "":
        return False
    if lhs == rhs:
        return True

    try:
        from sympy import simplify, sympify

        return bool(simplify(sympify(lhs) - sympify(rhs)) == 0)
    except Exception:
        return False


class InterleavedEnvironmentSimulator:
    """
    Small deterministic simulator for reward-time verification.

    This is intentionally conservative: it supports a tiny standard tool surface
    for tests and smoke runs while leaving real ToolMind execution to a larger
    sandbox integration.
    """

    def __init__(
        self,
        knowledge_base: dict[str, str] | None = None,
        tool_definitions: list[dict[str, Any]] | None = None,
        mock_outputs: dict[str, str] | None = None,
        fixed_time: str = "12:00",
        max_repeated_calls: int = 2,
    ) -> None:
        self.knowledge_base = knowledge_base or {}
        self.tool_schemas = _index_tool_definitions(tool_definitions or [])
        self.mock_outputs = mock_outputs or {}
        self.fixed_time = fixed_time
        self.max_repeated_calls = max_repeated_calls
        self.call_counts: dict[str, int] = {}

    def reset_call_counts(self) -> None:
        """Reset per-trajectory redundancy tracking."""
        self.call_counts.clear()

    def execute_tool_call(self, raw_tool_call: str | dict[str, Any]) -> ToolExecutionResult:
        try:
            payload = json.loads(raw_tool_call) if isinstance(raw_tool_call, str) else raw_tool_call
        except json.JSONDecodeError as exc:
            return ToolExecutionResult(False, "", f"invalid_json:{exc.msg}")

        name = str(payload.get("name", "")).strip()
        arguments = payload.get("arguments") or {}
        if not isinstance(arguments, dict):
            return ToolExecutionResult(False, "", "arguments_must_be_object")

        mock_output = self._get_mock_output(name, arguments)
        if mock_output is not None:
            return ToolExecutionResult(True, mock_output)

        signature = json.dumps({"name": name, "arguments": arguments}, sort_keys=True, default=str)
        self.call_counts[signature] = self.call_counts.get(signature, 0) + 1
        if self.call_counts[signature] > self.max_repeated_calls:
            return ToolExecutionResult(False, "", "redundant_call_detected")

        if name in {"calculator", "calculate"}:
            return self._calculate(str(arguments.get("expression", "")))
        if name in {"get_time", "get_current_time"}:
            return ToolExecutionResult(True, self.fixed_time)
        if name in {"search_knowledge", "web_search", "database_query"}:
            query = str(arguments.get("query", arguments.get("sql", ""))).strip().lower()
            return self._search_knowledge(query)
        if name in {"answer", "final_answer"}:
            value = arguments.get("value", arguments.get("answer", ""))
            return ToolExecutionResult(True, str(value))
        if name in self.tool_schemas:
            validation_error = _validate_tool_arguments(arguments, self.tool_schemas[name])
            if validation_error:
                return ToolExecutionResult(False, "", validation_error)
            return ToolExecutionResult(True, self._mock_registered_tool(name, arguments))
        return ToolExecutionResult(False, "", f"unknown_tool:{name}")

    def _calculate(self, expression: str) -> ToolExecutionResult:
        if not expression.strip():
            return ToolExecutionResult(False, "", "empty_expression")
        try:
            from sympy import simplify, sympify

            return ToolExecutionResult(True, str(simplify(sympify(expression))))
        except Exception as exc:
            return ToolExecutionResult(False, "", f"calculation_error:{exc}")

    def _search_knowledge(self, query: str) -> ToolExecutionResult:
        if not query:
            return ToolExecutionResult(False, "", "empty_query")

        normalized_kb = {key.strip().lower(): value for key, value in self.knowledge_base.items()}
        exact = normalized_kb.get(query)
        if exact is not None:
            return ToolExecutionResult(True, exact)

        query_tokens = _tokenize(query)
        best_key = ""
        best_score = 0.0
        for key in normalized_kb:
            key_tokens = _tokenize(key)
            if not key_tokens:
                continue
            score = len(query_tokens & key_tokens) / len(query_tokens | key_tokens)
            if score > best_score:
                best_key = key
                best_score = score

        if best_key and best_score >= 0.3:
            return ToolExecutionResult(True, normalized_kb[best_key])
        return ToolExecutionResult(False, "", "not_found")

    def _mock_registered_tool(self, name: str, arguments: dict[str, Any]) -> str:
        mock_output = self._get_mock_output(name, arguments)
        if mock_output is not None:
            return mock_output
        return json.dumps(
            {
                "tool": name,
                "status": "mocked",
                "arguments": arguments,
            },
            sort_keys=True,
        )

    def _get_mock_output(self, name: str, arguments: dict[str, Any]) -> str | None:
        signature = _tool_signature(name, arguments)
        if signature in self.mock_outputs:
            return self.mock_outputs[signature]
        if name in self.mock_outputs:
            return self.mock_outputs[name]
        return None

    def verify_completion(self, completion: str, expected_answer: str, task_type: str = "math") -> bool:
        """Verify a completion against a target answer or final tool response."""
        stats = parse_interleaved_stream(completion)
        if task_type == "tool" and stats["tool_calls"]:
            self.reset_call_counts()
            last_result: ToolExecutionResult | None = None
            for raw_tool_call in stats["tool_calls"]:
                last_result = self.execute_tool_call(raw_tool_call)
                if not last_result.ok:
                    return False
            return last_result is not None and answers_equivalent(last_result.output, expected_answer)
        return answers_equivalent(extract_tagged_answer(completion), expected_answer)


def _tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r"\w+", text.lower()) if len(token) > 1}


def _tool_signature(name: str, arguments: dict[str, Any]) -> str:
    return json.dumps({"name": name, "arguments": arguments}, sort_keys=True, default=str)


def _index_tool_definitions(tool_definitions: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for definition in tool_definitions:
        function = definition.get("function", definition)
        name = str(function.get("name", "")).strip()
        if name:
            indexed[name] = function
    return indexed


def _get_schema(function: dict[str, Any]) -> dict[str, Any]:
    schema = function.get("parameters") or function.get("arguments") or {}
    if schema.get("type") == "dict":
        return schema
    return {
        "type": "dict",
        "properties": schema if isinstance(schema, dict) else {},
        "required": function.get("required") or [],
    }


def _validate_tool_arguments(arguments: dict[str, Any], function: dict[str, Any]) -> str | None:
    schema = _get_schema(function)
    properties = schema.get("properties") or {}
    required = schema.get("required") or function.get("required") or []

    for key in required:
        if key not in arguments:
            return f"missing_required_argument:{key}"

    for key, value in arguments.items():
        spec = properties.get(key)
        if not isinstance(spec, dict):
            continue
        expected_type = spec.get("type")
        if expected_type and not _matches_schema_type(value, str(expected_type)):
            return f"invalid_argument_type:{key}"
        pattern = spec.get("pattern")
        if pattern and isinstance(value, str) and re.fullmatch(pattern, value) is None:
            return f"invalid_argument_pattern:{key}"
    return None


def _matches_schema_type(value: Any, expected_type: str) -> bool:
    if expected_type in {"str", "string"}:
        return isinstance(value, str)
    if expected_type in {"int", "integer"}:
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type in {"float", "number"}:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected_type in {"bool", "boolean"}:
        return isinstance(value, bool)
    if expected_type in {"dict", "object"}:
        return isinstance(value, dict)
    if expected_type == "array":
        return isinstance(value, list)
    return True
