import json
import asyncio

import pytest

from agent_bench.models import ModelResponse, Task
from agent_bench.sandbox import SubprocessSandbox
from agent_bench.verifiers import (
    grade_coding,
    grade_multiple_choice,
    grade_text_recall,
    parse_model_json,
)


def _response(raw):
    return ModelResponse(task_id="T_001", model="test", raw_response=raw, latency_seconds=0.01)


def test_parse_model_json_accepts_fenced_json():
    payload, error = parse_model_json('```json\n{"answer": ["A"]}\n```')

    assert error is None
    assert payload == {"answer": ["A"]}


def test_grade_multiple_choice_exact_answer_set_for_multiselect():
    task = Task(
        id="T_001",
        category="logic",
        type="multiple_choice",
        question="Pick A and C",
        source="logic.json",
        choices=["a", "b", "c"],
        answer=["A", "C"],
    )

    result = grade_multiple_choice(task, _response('{"answer": ["C", "A"], "confidence": 0.9}'))

    assert result.passed is True
    assert result.score == 1.0
    assert result.json_valid is True


def test_grade_multiple_choice_malformed_answer_scores_zero():
    task = Task(
        id="T_001",
        category="logic",
        type="multiple_choice",
        question="Pick A",
        source="logic.json",
        choices=["a", "b"],
        answer=["A"],
    )

    result = grade_multiple_choice(task, _response('{"answer": ["A", "B"]}'))

    assert result.passed is False
    assert result.score == 0.0
    assert result.json_valid is True


def test_grade_coding_gives_partial_credit_by_passed_cases():
    task = Task(
        id="CODE_001",
        category="coding",
        type="coding",
        question="Return x except intentionally fail one case.",
        source="coding.json",
        function_name="identity",
        test_cases=[
            {"input": {"x": 1}, "output": 1},
            {"input": {"x": 2}, "output": 2},
            {"input": {"x": 3}, "output": 3},
        ],
    )
    code = "def identity(x):\n    return x if x < 3 else 0\n"

    result = asyncio.run(
        grade_coding(
            task,
            _response(json.dumps({"code": code, "confidence": 0.5})),
            SubprocessSandbox(),
            timeout_seconds=5.0,
        )
    )

    assert result.json_valid is True
    assert result.score == pytest.approx(2 / 3)
    assert result.details["passed_cases"] == 2
    assert result.details["total_cases"] == 3


def test_grade_coding_supports_class_operation_tasks():
    task = Task(
        id="CODE_002",
        category="coding",
        type="coding",
        question="Stack",
        source="coding.json",
        function_name="MinStack",
        test_cases=[
            {
                "input": {
                    "operations": ["MinStack", "push", "push", "getMin", "pop", "top"],
                    "arguments": [[], [2], [1], [], [], []],
                },
                "output": [None, None, None, 1, None, 2],
            }
        ],
    )
    code = """
class MinStack:
    def __init__(self):
        self.values = []
    def push(self, value):
        self.values.append(value)
    def pop(self):
        self.values.pop()
    def top(self):
        return self.values[-1]
    def getMin(self):
        return min(self.values)
"""

    result = asyncio.run(
        grade_coding(
            task,
            _response(json.dumps({"code": code})),
            SubprocessSandbox(),
            timeout_seconds=5.0,
        )
    )

    assert result.passed is True
    assert result.score == 1.0


def test_grade_text_recall_accepts_verbatim_answer_with_normalized_newlines():
    task = Task(
        id="CR_001",
        category="code_recall",
        type="text_recall",
        question="Return the target lines.",
        source="code_recall.json",
        expected_text="def target(value):\n    return value",
    )

    result = grade_text_recall(
        task,
        _response('{"answer": "\\r\\ndef target(value):\\r\\n    return value\\r\\n", "confidence": 0.8}'),
    )

    assert result.passed is True
    assert result.score == 1.0
    assert result.json_valid is True
    assert result.details["true_positives"] == 4
    assert result.details["false_positives"] == 0
    assert result.details["false_negatives"] == 0


def test_grade_text_recall_counts_missing_tokens_as_false_negatives():
    task = Task(
        id="CR_001",
        category="code_recall",
        type="text_recall",
        question="Return the target lines.",
        source="code_recall.json",
        expected_text="alpha beta gamma",
    )

    result = grade_text_recall(task, _response('{"answer": "alpha gamma"}'))

    assert result.passed is False
    assert result.score == pytest.approx(4 / 5)
    assert result.details["true_positives"] == 2
    assert result.details["false_positives"] == 0
    assert result.details["false_negatives"] == 1


def test_grade_text_recall_counts_hallucinated_tokens_as_false_positives():
    task = Task(
        id="CR_001",
        category="code_recall",
        type="text_recall",
        question="Return the target lines.",
        source="code_recall.json",
        expected_text="alpha beta",
    )

    result = grade_text_recall(task, _response('{"answer": "alpha beta delta"}'))

    assert result.passed is False
    assert result.score == pytest.approx(4 / 5)
    assert result.details["true_positives"] == 2
    assert result.details["false_positives"] == 1
    assert result.details["false_negatives"] == 0


def test_grade_text_recall_uses_token_multiplicity_for_f1():
    task = Task(
        id="CR_001",
        category="code_recall",
        type="text_recall",
        question="Return the target lines.",
        source="code_recall.json",
        expected_text="alpha alpha beta gamma",
    )

    result = grade_text_recall(task, _response('{"answer": "alpha beta beta delta"}'))

    assert result.passed is False
    assert result.score == pytest.approx(0.5)
    assert result.details["true_positives"] == 2
    assert result.details["false_positives"] == 2
    assert result.details["false_negatives"] == 2
