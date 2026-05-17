"""Dataset loading helpers for GRPO interleaved reasoning training."""

from __future__ import annotations

from argparse import Namespace
import json
import html
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from datasets import Dataset


DatasetName = Literal["gsm8k", "math", "toolmind", "hybrid", "reasoning-lang"]
REFERENCE_REASONING_LANG_DATA = Path("/Users/josephcheng/Projects/rl-data-geneator/data/gsm8k_yue_translated.csv")
LANGUAGE_NAMES = {
    "en": "English",
    "zh": "Chinese",
    "zh-hant": "Traditional Chinese",
    "zh-hans": "Simplified Chinese",
    "yue": "Cantonese",
}


INTERLEAVED_SYSTEM_PROMPT = (
    "You are an interleaved reasoning agent. For every step: "
    "1. Use <think> to plan. "
    "2. Perform an action, either a Math <answer>...</answer> or a <tool_call>...</tool_call>. "
    "3. Use <think> to reflect on the result. "
    "Be concise; reach the first action as quickly as possible."
)


def build_interleaved_messages(
    question: str,
    task_type: str = "math",
    reasoning_lang: str | None = None,
    reasoning_language: str | None = None,
) -> list[dict[str, str]]:
    """Build chat messages that guide the policy toward interleaved reasoning."""
    target_language = reasoning_lang or reasoning_language
    if task_type == "tool":
        user_content = (
            "Solve this tool-use task with interleaved plan-action-reflection steps. "
            "Use <think>...</think> for private planning. "
            "When taking an action, output exactly one <tool_call>...</tool_call> block containing a single valid JSON "
            'object with "name" and "arguments" keys, for example '
            '<tool_call>{"name":"tool_name","arguments":{"arg":"value"}}</tool_call>. '
            "Do not wrap tool calls in <answer>, <action>, markdown, or prose. "
            "Use <answer>...</answer> only when the task requires a final text answer rather than a tool action. "
            f"Task: {question.strip()}"
        )
    elif target_language:
        language = _language_name(target_language or "en")
        user_content = (
            f"Solve this step-by-step. Write all private reasoning inside <think>...</think> in {language}. "
            f"Put only the final math result in <answer>...</answer>: {question.strip()}"
        )
    else:
        user_content = f"Solve this step-by-step. Put the final math result in <answer>...</answer>: {question.strip()}"
    return [
        {"role": "system", "content": INTERLEAVED_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _extract_gsm8k_answer(answer: str) -> str:
    return answer.split("####")[-1].strip()


def _extract_math_answer(answer: str) -> str:
    from .parser import extract_math_answer

    extracted = extract_math_answer(answer)
    if extracted:
        return extracted
    return answer.strip()


def _language_name(language: str) -> str:
    return LANGUAGE_NAMES.get(str(language).strip().lower(), str(language).strip() or "English")


def _stringify(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _json_stringify(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _first_present(batch: dict[str, list[object]], names: tuple[str, ...], index: int) -> str:
    for name in names:
        if name in batch and batch[name][index] is not None:
            return _stringify(batch[name][index]).strip()
    return ""


def _first_present_raw(batch: dict[str, list[object]], names: tuple[str, ...], index: int) -> object:
    for name in names:
        if name in batch and batch[name][index] is not None:
            return batch[name][index]
    return []


def _tool_call_signature(name: str, arguments: dict[str, object]) -> str:
    return json.dumps({"name": name, "arguments": arguments}, sort_keys=True, default=str)


def _extract_tool_call_payloads(message: dict[str, object]) -> list[dict[str, object]]:
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list):
        return []

    payloads: list[dict[str, object]] = []
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        function = call.get("function", call)
        if not isinstance(function, dict):
            continue
        name = str(function.get("name", "")).strip()
        arguments = function.get("arguments") or {}
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}
        if name and isinstance(arguments, dict):
            payloads.append({"name": name, "arguments": arguments})
    return payloads


def _conversation_to_prompt(conversations: object) -> str:
    if not isinstance(conversations, list) or not conversations:
        return _stringify(conversations)

    lines: list[str] = []
    for message in conversations[:-1]:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "unknown"))
        content = str(message.get("content", "")).strip()
        tool_calls = _extract_tool_call_payloads(message)
        if tool_calls:
            content = f"{content}\nTool calls: {json.dumps(tool_calls, ensure_ascii=False)}".strip()
        lines.append(f"{role}: {content}")
    return "\n".join(lines).strip()


def _escape_xml_tag_literals(text: str) -> str:
    return html.escape(text, quote=False)


def _format_interleaved_tool_completion(content: str, payloads: list[dict[str, object]]) -> str:
    pieces = [_escape_xml_tag_literals(content.strip())] if content.strip() else []
    for payload in payloads:
        pieces.append(f"<tool_call>{json.dumps(payload, ensure_ascii=False, sort_keys=True)}</tool_call>")
    return "".join(pieces)


def _extract_replay_mock_outputs(conversations: object) -> dict[str, str]:
    if not isinstance(conversations, list):
        return {}

    mock_outputs: dict[str, str] = {}
    for index, message in enumerate(conversations[:-1]):
        if not isinstance(message, dict):
            continue
        payloads = _extract_tool_call_payloads(message)
        if not payloads:
            continue
        next_message = conversations[index + 1]
        if not isinstance(next_message, dict) or next_message.get("role") != "tool":
            continue
        output = str(next_message.get("content", ""))
        for payload in payloads:
            mock_outputs[_tool_call_signature(str(payload["name"]), payload["arguments"])] = output
    return mock_outputs


def _extract_toolmind_target(conversations: object) -> tuple[str, bool]:
    """Return the literal final assistant target and whether it is a tool action."""
    if not isinstance(conversations, list) or not conversations:
        return "", False

    last_message = conversations[-1]
    if not isinstance(last_message, dict):
        return "", False

    payloads = _extract_tool_call_payloads(last_message)
    if payloads:
        content = str(last_message.get("content", ""))
        return _format_interleaved_tool_completion(content, payloads), True
    return _escape_xml_tag_literals(str(last_message.get("content", "")).strip()), False


def _normalize_toolmind(dataset: "Dataset") -> "Dataset":
    question_columns = ("question", "query", "instruction", "prompt", "input", "messages", "conversations")
    answer_columns = ("answer", "final_answer", "expected_answer", "output", "response", "target")
    tool_definition_columns = ("tool_definitions", "tools", "functions", "available_tools")

    def process_toolmind(batch: dict[str, list[object]]) -> dict[str, list[object]]:
        questions: list[str] = []
        answers: list[str] = []
        tool_definitions: list[object] = []
        mock_outputs: list[str] = []
        target_has_tool_call: list[bool] = []
        batch_size = len(next(iter(batch.values()))) if batch else 0
        for index in range(batch_size):
            conversations = batch.get("conversations", [None] * batch_size)[index]
            target, has_tool_call = _extract_toolmind_target(conversations)
            replay_outputs = _extract_replay_mock_outputs(conversations)
            question = _conversation_to_prompt(conversations) or _first_present(batch, question_columns, index)
            answer = target or _first_present(batch, answer_columns, index)

            questions.append(question)
            answers.append(answer)
            tool_definitions.append(_json_stringify(_first_present_raw(batch, tool_definition_columns, index)))
            mock_outputs.append(_json_stringify(replay_outputs))
            target_has_tool_call.append(has_tool_call)
        return {
            "question": questions,
            "answer": answers,
            "task_type": ["tool"] * batch_size,
            "tool_definitions": tool_definitions,
            "mock_outputs": mock_outputs,
            "target_has_tool_call": target_has_tool_call,
        }

    dataset = dataset.map(process_toolmind, batched=True, remove_columns=dataset.column_names)
    return dataset


def _alternate_datasets(left: "Dataset", right: "Dataset") -> "Dataset":
    from datasets import Dataset

    rows: list[dict[str, object]] = []
    keys = set(left.column_names) | set(right.column_names)
    for index in range(min(len(left), len(right))):
        rows.append({key: dict(left[index]).get(key) for key in keys})
        rows.append({key: dict(right[index]).get(key) for key in keys})
    return Dataset.from_list(rows)


def _normalize_toolmind_with_target_count(dataset: "Dataset", max_samples: int | None) -> "Dataset":
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    return _normalize_toolmind(dataset)


def _resolve_reasoning_lang_data_file(subset: str | None) -> str:
    if subset:
        candidate = Path(subset).expanduser()
        if candidate.exists():
            return str(candidate)
    local_candidate = Path("data/gsm8k_yue_translated.csv")
    if local_candidate.exists():
        return str(local_candidate)
    if REFERENCE_REASONING_LANG_DATA.exists():
        return str(REFERENCE_REASONING_LANG_DATA)
    raise FileNotFoundError(
        "Could not find reasoning-lang CSV. Pass a CSV path via --dataset-subset, "
        "or place gsm8k_yue_translated.csv under ./data/."
    )


def _normalize_reasoning_lang(dataset: "Dataset", reasoning_language: str = "yue") -> "Dataset":
    question_columns = ("question_yue", "question", "problem", "prompt", "input")
    answer_columns = ("answer", "final_answer", "expected_answer", "target")

    def process_reasoning_lang(batch: dict[str, list[object]]) -> dict[str, list[str]]:
        questions: list[str] = []
        answers: list[str] = []
        languages: list[str] = []
        batch_size = len(next(iter(batch.values()))) if batch else 0
        for index in range(batch_size):
            language = _first_present(batch, ("reasoning_language", "language", "lang"), index) or reasoning_language
            question = _first_present(batch, question_columns, index)
            answer = _first_present(batch, answer_columns, index)
            questions.append(question)
            answers.append(_extract_math_answer(answer))
            languages.append(language.strip().lower())
        return {
            "question": questions,
            "answer": answers,
            "task_type": ["math"] * batch_size,
            "reasoning_lang": languages,
            "tool_definitions": ["[]" for _ in range(batch_size)],
            "mock_outputs": ["{}" for _ in range(batch_size)],
            "target_has_tool_call": [False for _ in range(batch_size)],
        }

    return dataset.map(process_reasoning_lang, batched=True, remove_columns=dataset.column_names)


def load_training_dataset(
    dataset_name: DatasetName = "gsm8k",
    split: str = "train",
    subset: str | None = None,
    max_samples: int | None = None,
    seed: int = 42,
) -> Dataset:
    """
    Load and normalize a math dataset for TRL GRPO.

    The returned dataset includes normalized `question` and `answer` columns.
    `train.py` formats `question` into a tokenizer-specific chat-template
    `prompt` after the tokenizer is loaded.
    """
    from datasets import load_dataset

    if dataset_name == "gsm8k":
        dataset = load_dataset("gsm8k", subset or "main", split=split)

        def process_gsm8k(batch: dict[str, list[str]]) -> dict[str, list[str]]:
            return {
                "question": [question.strip() for question in batch["question"]],
                "answer": [_extract_gsm8k_answer(answer) for answer in batch["answer"]],
                "task_type": ["math"] * len(batch["question"]),
                "tool_definitions": ["[]" for _ in batch["question"]],
                "mock_outputs": ["{}" for _ in batch["question"]],
                "target_has_tool_call": [False for _ in batch["question"]],
            }

        dataset = dataset.map(process_gsm8k, batched=True, remove_columns=dataset.column_names)
    elif dataset_name == "math":
        dataset = load_dataset("hendrycks/competition_math", subset or "all", split=split)

        def process_math(batch: dict[str, list[str]]) -> dict[str, list[str]]:
            return {
                "question": [problem.strip() for problem in batch["problem"]],
                "answer": [_extract_math_answer(solution) for solution in batch["solution"]],
                "task_type": ["math"] * len(batch["problem"]),
                "tool_definitions": ["[]" for _ in batch["problem"]],
                "mock_outputs": ["{}" for _ in batch["problem"]],
                "target_has_tool_call": [False for _ in batch["problem"]],
            }

        dataset = dataset.map(process_math, batched=True, remove_columns=dataset.column_names)
    elif dataset_name == "toolmind":
        if split == "train":
            split = "open_datasets"
        dataset = load_dataset("Nanbeige/ToolMind", subset, split=split)
        dataset = _normalize_toolmind_with_target_count(dataset, max_samples)
    elif dataset_name == "reasoning-lang":
        data_file = _resolve_reasoning_lang_data_file(subset)
        dataset = load_dataset("csv", data_files=data_file, split=split)
        dataset = _normalize_reasoning_lang(dataset)
    elif dataset_name == "hybrid":
        per_source_samples = max_samples // 2 if max_samples else None
        math_dataset = load_training_dataset(
            dataset_name="gsm8k",
            split=split,
            subset="main",
            max_samples=per_source_samples,
            seed=seed,
        )
        tool_dataset = load_training_dataset(
            dataset_name="toolmind",
            split=split,
            subset=subset,
            max_samples=per_source_samples,
            seed=seed,
        )
        dataset = _alternate_datasets(math_dataset, tool_dataset)
    else:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")

    if max_samples is not None and dataset_name not in {"hybrid", "toolmind"}:
        dataset = dataset.shuffle(seed=seed).select(range(min(max_samples, len(dataset))))

    return dataset


def dataset_from_args(args: Namespace) -> Dataset:
    """Build a dataset from CLI args used by `train.py`."""
    return load_training_dataset(
        dataset_name=args.dataset,
        split=args.dataset_split,
        subset=args.dataset_subset,
        max_samples=args.max_samples,
        seed=args.seed,
    )
