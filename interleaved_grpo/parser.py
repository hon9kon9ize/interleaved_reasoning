"""Parsing and answer extraction utilities for interleaved reasoning streams."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
TAG_NAMES = ("think", "tool_call", "tool_output", "answer")
ACTION_SEGMENT_TYPES = {"tool_call", "answer"}
TAG_RE = re.compile(r"</?(think|tool_call|tool_output|answer)>")
THINK_BLOCK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
NON_REASONING_BLOCK_RE = re.compile(r"<(?:think|tool_call|tool_output)>.*?</(?:think|tool_call|tool_output)>", re.DOTALL)
ANSWER_BLOCK_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
BOXED_RE = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")
NUMBER_RE = re.compile(
    r"[-+]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)(?:/[+-]?\d+)?"
)
WORD_RE = re.compile(r"\w+")


@dataclass(frozen=True)
class _ParserState:
    is_valid_xml: bool
    error: str | None
    segments: list[dict[str, Any]]


def _parse_tag_structure(text: str) -> _ParserState:
    """Parse supported exact tags and reject nesting, mismatches, and dangling tags."""
    segments: list[dict[str, Any]] = []
    active_tag: str | None = None
    active_content_start: int | None = None
    cursor = 0

    for match in TAG_RE.finditer(text):
        token = match.group(0)
        tag_name = match.group(1)
        is_closing_tag = token.startswith("</")

        if active_tag is None:
            if match.start() > cursor:
                segments.append(
                    {
                        "type": "text",
                        "content": text[cursor : match.start()],
                        "start": cursor,
                        "end": match.start(),
                    }
                )
            if is_closing_tag:
                return _ParserState(False, f"unmatched_close_tag:{tag_name}", segments)
            active_tag = tag_name
            active_content_start = match.end()
            cursor = match.end()
            continue

        if not is_closing_tag:
            return _ParserState(False, f"nested_open_tag:{tag_name}", segments)

        if tag_name != active_tag or active_content_start is None:
            return _ParserState(False, f"mismatched_close_tag:{tag_name}", segments)

        segments.append(
            {
                "type": active_tag,
                "content": text[active_content_start : match.start()],
                "start": active_content_start,
                "end": match.start(),
            }
        )
        active_tag = None
        active_content_start = None
        cursor = match.end()

    if active_tag is not None:
        return _ParserState(False, f"unclosed_open_tag:{active_tag}", segments)

    if cursor < len(text):
        segments.append({"type": "text", "content": text[cursor:], "start": cursor, "end": len(text)})
    return _ParserState(True, None, segments)


def _nonempty_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [segment for segment in segments if segment["type"] != "text" or segment["content"].strip()]


def _count_plan_action_reflection_cycles(sequence: list[str]) -> int:
    cycles = 0
    for left, middle, right in zip(sequence, sequence[1:], sequence[2:]):
        if left == "think" and middle in ACTION_SEGMENT_TYPES and right == "think":
            cycles += 1
    return cycles


def _count_reflections_referencing_actions(segments: list[dict[str, Any]]) -> int:
    count = 0
    for index, segment in enumerate(segments[:-1]):
        if segment["type"] not in ACTION_SEGMENT_TYPES and segment["type"] != "tool_output":
            continue
        next_think = next((candidate for candidate in segments[index + 1 :] if candidate["type"] == "think"), None)
        if not next_think:
            continue
        action_words = {word.lower() for word in WORD_RE.findall(segment["content"]) if len(word) > 1}
        reflection_words = {word.lower() for word in WORD_RE.findall(next_think["content"])}
        numeric_refs = {word for word in action_words if any(char.isdigit() for char in word)}
        if numeric_refs & reflection_words:
            count += 1
            continue
        if {"result", "output", "observation", "error", "answer"} & reflection_words:
            count += 1
    return count


def parse_interleaved_stream(text: str) -> dict[str, Any]:
    """
    Parse an interleaved response into private and public segments.

    The parser intentionally recognizes only exact supported tags. It rejects
    nested tags, unmatched closing tags, mismatched tags, and unclosed opening
    tags while still returning the valid segments seen before the syntax error.
    """
    parsed = _parse_tag_structure(text)
    segments = parsed.segments
    think_blocks = [segment["content"] for segment in segments if segment["type"] == "think"]
    tool_calls = [segment["content"] for segment in segments if segment["type"] == "tool_call"]
    tool_outputs = [segment["content"] for segment in segments if segment["type"] == "tool_output"]
    answer_blocks = [segment["content"] for segment in segments if segment["type"] == "answer"]
    action_blocks = [segment["content"] for segment in segments if segment["type"] in ACTION_SEGMENT_TYPES]
    public_segments = [segment["content"] for segment in segments if segment["type"] != "think"]

    public_text = " ".join(segment.strip() for segment in public_segments if segment.strip())
    nonempty_public_segments = [segment for segment in public_segments if segment.strip()]
    nonempty_think_blocks = [block for block in think_blocks if block.strip()]
    ordered_nonempty_segments = _nonempty_segments(segments)
    segment_types = [segment["type"] for segment in ordered_nonempty_segments]

    first_think_index = next((idx for idx, segment in enumerate(segments) if segment["type"] == "think"), None)
    last_think_index = next(
        (idx for idx in range(len(segments) - 1, -1, -1) if segments[idx]["type"] == "think"),
        None,
    )
    first_action_index = next(
        (idx for idx, segment in enumerate(segments) if segment["type"] in ACTION_SEGMENT_TYPES),
        None,
    )

    has_public_before_think = (
        first_think_index is not None
        and any(segment["type"] != "think" and segment["content"].strip() for segment in segments[:first_think_index])
    )
    has_public_after_think = (
        first_think_index is not None
        and last_think_index is not None
        and any(segment["type"] != "think" and segment["content"].strip() for segment in segments[last_think_index + 1 :])
    )
    has_public_between_thinks = (
        first_think_index is not None
        and last_think_index is not None
        and any(
            segment["type"] != "think" and segment["content"].strip()
            for segment in segments[first_think_index + 1 : last_think_index]
        )
    )
    plan_action_reflection_cycles = _count_plan_action_reflection_cycles(segment_types)
    first_think_block = think_blocks[0] if think_blocks else ""
    first_think_close = text.find(THINK_CLOSE)
    text_before_first_think_close = text[:first_think_close] if first_think_close >= 0 else ""

    return {
        "segments": segments,
        "segment_types": segment_types,
        "think_blocks": think_blocks,
        "tool_calls": tool_calls,
        "tool_outputs": tool_outputs,
        "answer_blocks": answer_blocks,
        "action_blocks": action_blocks,
        "public_segments": public_segments,
        "public_text": public_text,
        "is_valid_xml": parsed.is_valid_xml,
        "error": parsed.error,
        "num_transitions": len(think_blocks),
        "num_actions": len(action_blocks),
        "num_tool_calls": len(tool_calls),
        "num_answers": len(answer_blocks),
        "num_public_segments": len(nonempty_public_segments),
        "num_nonempty_think_blocks": len(nonempty_think_blocks),
        "has_public_before_think": has_public_before_think,
        "has_public_between_thinks": has_public_between_thinks,
        "has_public_after_think": has_public_after_think,
        "is_interleaved": (
            parsed.is_valid_xml
            and len(think_blocks) >= 2
            and has_public_between_thinks
            and has_public_after_think
        ),
        "has_plan_action_reflection": parsed.is_valid_xml and plan_action_reflection_cycles > 0,
        "num_plan_action_reflection_cycles": plan_action_reflection_cycles,
        "num_reflections_referencing_actions": _count_reflections_referencing_actions(ordered_nonempty_segments),
        "first_think_word_count": len(WORD_RE.findall(first_think_block)),
        "tokens_before_first_think_close": len(WORD_RE.findall(text_before_first_think_close)),
        "first_action_index": first_action_index,
        "total_think_char_len": sum(len(block) for block in think_blocks),
        "total_public_char_len": len(public_text),
    }


def strip_think_blocks(text: str) -> str:
    """Remove complete think blocks and dangling exact think tags from text."""
    stripped = THINK_BLOCK_RE.sub(" ", text)
    return stripped.replace(THINK_OPEN, " ").replace(THINK_CLOSE, " ")


def normalize_math_answer(answer: str) -> str:
    """Normalize common math-answer formatting for deterministic string checks."""
    normalized = str(answer).strip()
    normalized = normalized.replace(",", "")
    normalized = re.sub(r"\s+", "", normalized)
    normalized = normalized.rstrip(".")
    return normalized


def extract_tagged_answer(text: str) -> str:
    """Extract only the final explicit `<answer>...</answer>` block."""
    answer_blocks = ANSWER_BLOCK_RE.findall(text)
    if answer_blocks:
        return normalize_math_answer(answer_blocks[-1])
    return ""


def extract_math_answer(text: str) -> str:
    """
    Extract the final numerical or boxed answer from public text.

    Complete `<think>...</think>` blocks are removed before extraction. If a
    malformed generation leaks dangling tags, exact tag literals are stripped
    before the fallback numeric scan.
    """
    tagged_answer = extract_tagged_answer(text)
    if tagged_answer:
        return tagged_answer

    clean_text = NON_REASONING_BLOCK_RE.sub(" ", text)
    clean_text = clean_text.replace("<answer>", " ").replace("</answer>", " ")

    boxed = BOXED_RE.findall(clean_text)
    if boxed:
        return normalize_math_answer(boxed[-1])

    final_answer_patterns = [
        r"(?:final answer|answer|therefore|thus)\s*(?:is|=|:)?\s*([$\\]?\s*[-+]?\d[\d,]*(?:\.\d+)?(?:/[+-]?\d+)?)",
        r"####\s*([-+]?\d[\d,]*(?:\.\d+)?(?:/[+-]?\d+)?)",
    ]
    for pattern in final_answer_patterns:
        matches = re.findall(pattern, clean_text, flags=re.IGNORECASE)
        if matches:
            return normalize_math_answer(matches[-1].replace("$", "").replace("\\", ""))

    numbers = [match.group(0) for match in NUMBER_RE.finditer(clean_text) if match.group(0)]
    if numbers:
        return normalize_math_answer(numbers[-1])

    return ""
