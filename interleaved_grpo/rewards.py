"""GRPO reward functions for interleaved chain-of-thought training."""

from __future__ import annotations

import html
import json
import re
from typing import Any

from .environment import InterleavedEnvironmentSimulator, answers_equivalent
from .parser import extract_tagged_answer, parse_interleaved_stream


def _get_targets(answer: list[str] | None = None, expected_answer: list[str] | None = None, **kwargs: Any) -> list[str]:
    return answer if answer is not None else expected_answer or []


def _indexed_item(values: Any, index: int) -> Any | None:
    if isinstance(values, (str, bytes, dict)):
        return None
    try:
        if index < len(values):
            return values[index]
    except (KeyError, IndexError, TypeError):
        return None
    return None


def _get_knowledge_base(kwargs: dict[str, Any], index: int) -> dict[str, str] | None:
    knowledge_base = kwargs.get("knowledge_base", kwargs.get("kb"))
    if isinstance(knowledge_base, dict):
        return knowledge_base
    if isinstance(knowledge_base, list) and index < len(knowledge_base) and isinstance(knowledge_base[index], dict):
        return knowledge_base[index]
    item = _indexed_item(knowledge_base, index)
    if isinstance(item, dict):
        return item
    return None


def _get_tool_definitions(kwargs: dict[str, Any], index: int) -> list[dict[str, Any]] | None:
    definitions = kwargs.get("tool_definitions", kwargs.get("tools"))
    if isinstance(definitions, str):
        try:
            parsed = json.loads(definitions)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, list) else None
    if isinstance(definitions, list):
        if index < len(definitions) and isinstance(definitions[index], str):
            try:
                parsed = json.loads(definitions[index])
            except json.JSONDecodeError:
                return None
            return parsed if isinstance(parsed, list) else None
        if index < len(definitions) and isinstance(definitions[index], list):
            return definitions[index]
        if all(isinstance(item, dict) and ("function" in item or "name" in item) for item in definitions):
            return definitions
    item = _indexed_item(definitions, index)
    if isinstance(item, str):
        try:
            parsed = json.loads(item)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, list) else None
    if isinstance(item, list):
        return item
    if isinstance(item, dict) and ("function" in item or "name" in item):
        return [item]
    return None


def _get_mock_outputs(kwargs: dict[str, Any], index: int) -> dict[str, str] | None:
    mock_outputs = kwargs.get("mock_outputs", kwargs.get("tool_outputs"))
    if isinstance(mock_outputs, str):
        try:
            parsed = json.loads(mock_outputs)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    if isinstance(mock_outputs, dict):
        return mock_outputs
    if isinstance(mock_outputs, list) and index < len(mock_outputs):
        item = mock_outputs[index]
        if isinstance(item, str):
            try:
                parsed = json.loads(item)
            except json.JSONDecodeError:
                return None
            return parsed if isinstance(parsed, dict) else None
        if isinstance(item, dict):
            return item
    item = _indexed_item(mock_outputs, index)
    if isinstance(item, str):
        try:
            parsed = json.loads(item)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    if isinstance(item, dict):
        return item
    return None


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _get_target_tool_call_mode(kwargs: dict[str, Any], index: int, target: str) -> bool | None:
    flags = kwargs.get("target_has_tool_call")
    if flags is not None:
        if isinstance(flags, list):
            return _coerce_bool(flags[index]) if index < len(flags) else False
        item = _indexed_item(flags, index)
        if item is not None:
            return _coerce_bool(item)
        return _coerce_bool(flags)

    target_stats = parse_interleaved_stream(target)
    if target_stats["tool_calls"]:
        return True
    return None


def _coerce_tool_payload(raw_tool_call: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(raw_tool_call)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None

    function = payload.get("function", payload)
    if not isinstance(function, dict):
        return None

    name = str(function.get("name", "")).strip()
    arguments = function.get("arguments") or {}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None
    if not name or not isinstance(arguments, dict):
        return None
    return {"name": name, "arguments": arguments}


def _tool_signature(name: str, arguments: dict[str, Any]) -> str:
    return json.dumps({"name": name, "arguments": arguments}, sort_keys=True, default=str)


def _tool_call_signatures(text: str) -> list[str] | None:
    stats = parse_interleaved_stream(text)
    if not stats["is_valid_xml"]:
        return None

    signatures: list[str] = []
    for raw_tool_call in stats["tool_calls"]:
        payload = _coerce_tool_payload(raw_tool_call)
        if payload is None:
            return None
        signatures.append(_tool_signature(str(payload["name"]), payload["arguments"]))
    return signatures


def _normalize_public_text(text: str) -> str:
    unescaped = html.unescape(str(text))
    unescaped_stats = parse_interleaved_stream(unescaped)
    if unescaped_stats["is_valid_xml"]:
        public_text = unescaped_stats["public_text"]
    else:
        stats = parse_interleaved_stream(str(text))
        public_text = stats["public_text"] if stats["is_valid_xml"] else unescaped
    return re.sub(r"\s+", " ", public_text).strip().casefold()


def _tool_task_correct(
    completion: str,
    target: str,
    target_tool_call_mode: bool | None,
    simulator: InterleavedEnvironmentSimulator,
) -> bool:
    completion_stats = parse_interleaved_stream(completion)

    if target_tool_call_mode is True:
        target_signatures = _tool_call_signatures(target)
        completion_signatures = _tool_call_signatures(completion)
        if not target_signatures or completion_signatures != target_signatures:
            return False
        return True

    if target_tool_call_mode is False:
        return _normalize_public_text(completion) == _normalize_public_text(target)

    if not completion_stats["tool_calls"]:
        return False
    return simulator.verify_completion(completion, target, task_type="tool")


def _make_simulator(kwargs: dict[str, Any], index: int) -> InterleavedEnvironmentSimulator:
    return InterleavedEnvironmentSimulator(
        knowledge_base=_get_knowledge_base(kwargs, index),
        tool_definitions=_get_tool_definitions(kwargs, index),
        mock_outputs=_get_mock_outputs(kwargs, index),
    )


def outcome_correctness_reward_fn(
    prompts: list[str],
    completions: list[str],
    answer: list[str] | None = None,
    expected_answer: list[str] | None = None,
    **kwargs: Any,
) -> list[float]:
    """Reward final-answer correctness with exact and SymPy equivalence checks."""
    rewards: list[float] = []
    targets = _get_targets(answer=answer, expected_answer=expected_answer, **kwargs)
    task_types = kwargs.get("task_type") or ["math"] * len(completions)
    for index, (completion, ground_truth, task_type) in enumerate(zip(completions, targets, task_types)):
        if task_type == "tool":
            simulator = _make_simulator(kwargs, index)
            is_correct = _tool_task_correct(
                completion,
                ground_truth,
                _get_target_tool_call_mode(kwargs, index, ground_truth),
                simulator,
            )
            rewards.append(1.0 if is_correct else 0.0)
            continue

        is_correct = answers_equivalent(extract_tagged_answer(completion), ground_truth)
        rewards.append(1.0 if is_correct else 0.0)
    return rewards


def correctness_reward_fn(
    prompts: list[str],
    completions: list[str],
    answer: list[str],
    **kwargs: Any,
) -> list[float]:
    """Backward-compatible alias for outcome correctness."""
    return outcome_correctness_reward_fn(prompts, completions, answer=answer, **kwargs)


def structural_interleaving_reward_fn(
    prompts: list[str],
    completions: list[str],
    **kwargs: Any,
) -> list[float]:
    """Reward valid Plan-Action-Reflection structure and grounded reflections."""
    rewards: list[float] = []
    for completion in completions:
        stats = parse_interleaved_stream(completion)
        if not stats["is_valid_xml"]:
            rewards.append(0.0)
            continue

        score = 0.0
        if stats["num_transitions"] > 0:
            score += 0.2
        if stats["num_actions"] > 0:
            score += 0.2
        if stats["has_plan_action_reflection"]:
            score += 0.4
        if stats["num_reflections_referencing_actions"] > 0:
            score += 0.2
        rewards.append(min(score, 1.0))
    return rewards


def interleaved_format_reward_fn(
    prompts: list[str],
    completions: list[str],
    **kwargs: Any,
) -> list[float]:
    """Reward valid syntax and repeated public/private alternation without clipping utility."""
    rewards: list[float] = []
    for completion in completions:
        stats = parse_interleaved_stream(completion)
        if not stats["is_valid_xml"]:
            rewards.append(0.0)
            continue

        transitions = stats["num_transitions"]
        if transitions == 0:
            rewards.append(0.0)
            continue

        score = 0.2

        if transitions >= 3:
            score += 0.5
        elif transitions == 2:
            score += 0.3
        elif transitions == 1:
            score += 0.1

        if stats.get("is_interleaved", False):
            score += 0.3

        rewards.append(min(score, 1.0))
    return rewards


def ttft_reward_fn(
    prompts: list[str],
    completions: list[str],
    max_first_think_tokens: int = 256,
    min_first_think_tokens: int = 10,
    fast_plan_bonus: float = 0.3,
    **kwargs: Any,
) -> list[float]:
    """Reward fast non-empty planning and penalize delayed first actions."""
    rewards: list[float] = []
    for completion in completions:
        stats = parse_interleaved_stream(completion)
        if not stats["is_valid_xml"] or stats["num_transitions"] == 0:
            rewards.append(-0.5)
            continue
        tokens_before_close = stats["tokens_before_first_think_close"]
        if tokens_before_close < min_first_think_tokens:
            rewards.append(0.0)
            continue
        if tokens_before_close <= max_first_think_tokens:
            rewards.append(fast_plan_bonus)
            continue
        overflow = tokens_before_close - max_first_think_tokens
        rewards.append(max(-0.1 * (overflow / max_first_think_tokens), -0.5))
    return rewards


def conditional_step_reward_fn(
    prompts: list[str],
    completions: list[str],
    answer: list[str] | None = None,
    expected_answer: list[str] | None = None,
    **kwargs: Any,
) -> list[float]:
    """Reward intermediate structure only when the final answer is correct."""
    rewards: list[float] = []
    targets = _get_targets(answer=answer, expected_answer=expected_answer, **kwargs)
    task_types = kwargs.get("task_type") or ["math"] * len(completions)
    for index, (completion, ground_truth, task_type) in enumerate(zip(completions, targets, task_types)):
        stats = parse_interleaved_stream(completion)
        if task_type == "tool":
            simulator = _make_simulator(kwargs, index)
            is_correct = _tool_task_correct(
                completion,
                ground_truth,
                _get_target_tool_call_mode(kwargs, index, ground_truth),
                simulator,
            )
        else:
            is_correct = answers_equivalent(extract_tagged_answer(completion), ground_truth)

        if not is_correct:
            rewards.append(0.0)
            continue

        if not stats["is_valid_xml"]:
            rewards.append(0.0)
            continue

        score = 0.0
        if stats["num_actions"] > 0:
            score += 0.4
        if stats["has_plan_action_reflection"]:
            score += 0.4
        if stats["num_reflections_referencing_actions"] > 0:
            score += 0.2
        rewards.append(min(score, 1.0))
    return rewards


def efficiency_penalty_fn(
    prompts: list[str],
    completions: list[str],
    **kwargs: Any,
) -> list[float]:
    """Penalize bloated private reasoning relative to public answer text."""
    rewards: list[float] = []
    for completion in completions:
        stats = parse_interleaved_stream(completion)
        think_word_count = len(re.findall(r"\w+", "".join(stats["think_blocks"])))
        public_word_count = len(re.findall(r"\w+", stats["public_text"]))

        if public_word_count == 0:
            rewards.append(-0.5)
            continue

        ratio = think_word_count / max(public_word_count, 1)
        if ratio <= 2.5:
            rewards.append(0.0)
        else:
            penalty = -0.1 * (ratio - 2.5)
            rewards.append(max(penalty, -0.5))
    return rewards


def _get_language_targets(kwargs: dict[str, Any], count: int) -> list[str | None]:
    values = kwargs.get("reasoning_lang", kwargs.get("reasoning_language", kwargs.get("language")))
    if values is None:
        return [None] * count
    if isinstance(values, str):
        return [values] * count
    targets: list[str | None] = []
    for index in range(count):
        item = _indexed_item(values, index)
        targets.append(str(item) if item is not None else None)
    return targets


def _reasoning_text_for_language(completion: str) -> str:
    stats = parse_interleaved_stream(completion)
    if stats["think_blocks"]:
        return " ".join(block.strip() for block in stats["think_blocks"] if block.strip())
    if "</think>" in completion:
        return completion.split("</think>", 1)[0].replace("<think>", " ").strip()
    return ""


def _cjk_count(text: str) -> int:
    return sum(1 for char in text if "\u4e00" <= char <= "\u9fff")


def _latin_word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z]{2,}", text))


def _looks_cantonese(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False

    try:
        from cantofilter import judge as yue_judge

        judgement = yue_judge(stripped)
        if judgement in {"cantonese", "mixed"}:
            return True
        if judgement == "neutral" and _cjk_count(stripped) > 0:
            return True
    except Exception:
        pass

    cantonese_markers = (
        "嘅",
        "咗",
        "佢",
        "哋",
        "唔",
        "喺",
        "呢",
        "咁",
        "啲",
        "嚟",
        "晒",
        "冇",
        "嗰",
        "啦",
        "呀",
        "㗎",
        "咩",
    )
    cjk = _cjk_count(stripped)
    return cjk > 0 and (any(marker in stripped for marker in cantonese_markers) or cjk >= 12)


def _matches_language(text: str, language: str | None) -> bool:
    if language is None:
        return False
    normalized = language.strip().lower()
    if normalized == "":
        return False
    if normalized == "yue":
        return _looks_cantonese(text)
    if normalized in {"zh", "zh-hant", "zh-hans", "cn"}:
        return _cjk_count(text) >= 4
    if normalized == "en":
        return _latin_word_count(text) >= 4 and _cjk_count(text) == 0
    return True


def language_consistency_reward_fn(
    prompts: list[str],
    completions: list[str],
    **kwargs: Any,
) -> list[float]:
    """Reward `<think>` reasoning that matches the requested natural language."""
    targets = _get_language_targets(kwargs, len(completions))
    rewards: list[float] = []
    for completion, target_language in zip(completions, targets):
        reasoning_text = _reasoning_text_for_language(completion)
        if not reasoning_text or target_language is None:
            rewards.append(0.0)
            continue
        rewards.append(1.0 if _matches_language(reasoning_text, target_language) else 0.0)
    return rewards


REWARD_FUNCS = [
    outcome_correctness_reward_fn,
    conditional_step_reward_fn,
    ttft_reward_fn,
    efficiency_penalty_fn,
]
