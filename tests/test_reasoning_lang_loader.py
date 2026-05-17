from interleaved_grpo.datasets_loader import _normalize_reasoning_lang


class FakeDataset:
    column_names = ["question_yue", "answer"]

    def __init__(self, rows):
        self.rows = rows

    def map(self, fn, batched, remove_columns):
        assert batched
        assert remove_columns == self.column_names
        batch = {
            "question_yue": [row["question_yue"] for row in self.rows],
            "answer": [row["answer"] for row in self.rows],
        }
        return fn(batch)


def test_normalize_reasoning_lang_adds_language_columns():
    dataset = FakeDataset([{"question_yue": "2 加 2 係幾多？", "answer": "4"}])

    normalized = _normalize_reasoning_lang(dataset)

    assert normalized["question"] == ["2 加 2 係幾多？"]
    assert normalized["answer"] == ["4"]
    assert normalized["task_type"] == ["math"]
    assert normalized["reasoning_lang"] == ["yue"]
