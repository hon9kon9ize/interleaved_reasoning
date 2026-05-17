from interleaved_grpo.datasets_loader import (
    INTERLEAVED_SYSTEM_PROMPT,
    build_interleaved_messages,
)
from interleaved_grpo.train import apply_interleaved_chat_template


class FakeDataset:
    column_names = ["question", "answer", "task_type", "reasoning_lang"]

    def __init__(self, rows):
        self.rows = rows

    def map(self, fn, batched, remove_columns):
        assert batched
        assert remove_columns == self.column_names
        batch = {
            "question": [row["question"] for row in self.rows],
            "answer": [row["answer"] for row in self.rows],
            "task_type": [row.get("task_type", "math") for row in self.rows],
            "reasoning_lang": [row.get("reasoning_lang") for row in self.rows],
        }
        return fn(batch)


class FakeTokenizer:
    def __init__(self):
        self.kwargs = None

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kwargs):
        self.kwargs = kwargs
        assert not tokenize
        assert add_generation_prompt
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == INTERLEAVED_SYSTEM_PROMPT
        assert messages[1]["role"] == "user"
        return f"<system>{messages[0]['content']}</system><user>{messages[1]['content']}</user><assistant>"


def test_build_interleaved_messages():
    messages = build_interleaved_messages("  What is 2+2?  ")
    assert messages == [
        {"role": "system", "content": INTERLEAVED_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": "Solve this step-by-step. Put the final math result in <answer>...</answer>: What is 2+2?",
        },
    ]


def test_apply_interleaved_chat_template_formats_prompt():
    dataset = FakeDataset([{"question": "What is 2+2?", "answer": "4"}])
    formatted = apply_interleaved_chat_template(dataset, FakeTokenizer())
    assert formatted["answer"] == ["4"]
    assert "reach the first action as quickly as possible" in formatted["prompt"][0]
    assert "<answer>...</answer>" in formatted["prompt"][0]
    assert formatted["prompt"][0].endswith("<assistant>")


def test_build_interleaved_messages_for_tool_tasks():
    messages = build_interleaved_messages("Look up the current time.", task_type="tool")
    assert messages[1]["content"] == (
        "Solve this tool-use task with interleaved plan-action-reflection steps: Look up the current time."
    )


def test_build_interleaved_messages_for_reasoning_language_tasks():
    messages = build_interleaved_messages("2 + 2 係幾多？", task_type="math", reasoning_lang="yue")

    assert "in Cantonese" in messages[1]["content"]
    assert "<answer>...</answer>" in messages[1]["content"]


def test_apply_interleaved_chat_template_uses_default_reasoning_lang():
    dataset = FakeDataset([{"question": "2 + 2?", "answer": "4"}])
    formatted = apply_interleaved_chat_template(dataset, FakeTokenizer(), default_reasoning_lang="yue")

    assert formatted["reasoning_lang"] == ["yue"]
    assert "in Cantonese" in formatted["prompt"][0]


def test_apply_interleaved_chat_template_can_pass_qwen_thinking_flag():
    tokenizer = FakeTokenizer()
    dataset = FakeDataset([{"question": "What is 2+2?", "answer": "4"}])

    apply_interleaved_chat_template(dataset, tokenizer, chat_template_enable_thinking=False)

    assert tokenizer.kwargs == {"enable_thinking": False}
