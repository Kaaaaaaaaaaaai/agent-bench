import json
import asyncio

import pytest

from agent_bench.models import ModelResponse, Task
from agent_bench.sandbox import SubprocessSandbox
from agent_bench.verifiers import (
    grade_coding,
    grade_external_benchmark,
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


def test_grade_external_benchmark_treats_unsupported_capability_as_coverage_gap():
    task = Task(
        id="PB_001",
        category="public_benchmarks",
        type="external_benchmark",
        question="Run benchmark",
        source="public_benchmarks.json",
    )
    response = _response(
        json.dumps(
            {
                "status": "error",
                "score": 0.0,
                "error": "Benchmark requires unsupported capability/capabilities: repo_patch",
                "timed_out": False,
                "details": {
                    "group": "Coding",
                    "result": {
                        "status": "skipped_unsupported_capability",
                        "unsupported_capabilities": ["repo_patch"],
                    },
                },
            }
        )
    )

    result = grade_external_benchmark(task, response)

    assert result.status == "skipped_unsupported_capability"
    assert result.error == "Benchmark requires unsupported capability/capabilities: repo_patch"
    assert result.category == "Coding"


@pytest.mark.parametrize(
    "result_payload",
    [
        {
            "status": "passed",
            "required_capabilities": ["tool_call"],
            "exposed_tools": ["final_answer"],
            "missing_tools": ["benchmark_tool"],
            "capabilities_verified": True,
        },
        {
            "status": "passed",
            "required_capabilities": ["tool_call"],
            "exposed_tools": ["benchmark_tool"],
            "missing_tools": [],
            "missing_env": ["BENCHMARK_API_KEY"],
            "capabilities_verified": True,
        },
        {
            "status": "passed",
            "required_capabilities": ["tool_call"],
            "exposed_tools": [],
            "missing_tools": [],
            "capabilities_verified": True,
        },
    ],
)
def test_grade_external_benchmark_promotes_capability_failures(result_payload):
    task = Task(
        id="PB_TOOL",
        category="public_benchmarks",
        type="external_benchmark",
        question="Run benchmark",
        source="public_benchmarks.json",
    )
    response = _response(
        json.dumps(
            {
                "status": "passed",
                "score": 1.0,
                "error": None,
                "timed_out": False,
                "details": {
                    "group": "Finance",
                    "result": result_payload,
                },
            }
        )
    )

    result = grade_external_benchmark(task, response)

    assert result.status == "failed_missing_required_tool"
    assert result.score == 0.0
    assert result.passed is False
    assert result.error == "failed missing required tool"


def test_grade_external_benchmark_promotes_all_invalid_nested_status():
    task = Task(
        id="PB_001",
        category="public_benchmarks",
        type="external_benchmark",
        question="Run benchmark",
        source="public_benchmarks.json",
    )
    response = _response(
        json.dumps(
            {
                "status": "completed",
                "score": 0.0,
                "error": None,
                "timed_out": False,
                "details": {
                    "group": "Coding",
                    "result": {
                        "status": "completed",
                        "evaluated_task_count": 3,
                        "valid_evaluated_task_count": 0,
                        "status_counts": {"failed_harness_setup": 3},
                    },
                },
            }
        )
    )

    result = grade_external_benchmark(task, response)

    assert result.status == "failed_harness_setup"
    assert result.error == "All 3 benchmark record evaluation(s) were invalid: failed harness setup"
    assert result.passed is False


def test_grade_external_benchmark_preserves_exposed_tool_harness_failure():
    task = Task(
        id="PB_009",
        category="Security",
        type="external_benchmark",
        question="Run benchmark",
        source="tasks/exploitbench/manifest.json",
    )
    response = _response(
        json.dumps(
            {
                "status": "failed_harness_setup",
                "score": 0.0,
                "error": "ExploitBench doctor failed",
                "timed_out": False,
                "details": {
                    "group": "Security",
                    "result": {
                        "status": "failed_harness_setup",
                        "score": 0.0,
                        "error": "ExploitBench doctor failed",
                        "required_capabilities": ["tool_call", "external_data_required"],
                        "required_tools": ["exploitbench"],
                        "exposed_tools": ["exploitbench"],
                        "missing_tools": [],
                        "capabilities_verified": False,
                    },
                },
            }
        )
    )

    result = grade_external_benchmark(task, response)

    assert result.status == "failed_harness_setup"
    assert result.error == "ExploitBench doctor failed"
    assert result.passed is False


def test_grade_external_benchmark_preserves_timeout_before_missing_tool_heuristic():
    task = Task(
        id="PB_009",
        category="Security",
        type="external_benchmark",
        question="Run benchmark",
        source="tasks/exploitbench/manifest.json",
    )
    response = _response(
        json.dumps(
            {
                "status": "failed_timeout",
                "score": 0.0,
                "error": "External benchmark timed out after 1200.0s",
                "timed_out": True,
                "details": {
                    "group": "Security",
                    "result": {
                        "status": "failed_timeout",
                        "score": 0.0,
                        "error": "External benchmark timed out after 1200.0s",
                        "required_capabilities": ["tool_call", "external_data_required"],
                        "required_tools": ["exploitbench"],
                        "exposed_tools": [],
                        "missing_tools": [],
                        "capabilities_verified": False,
                    },
                },
            }
        )
    )

    result = grade_external_benchmark(task, response)

    assert result.status == "failed_timeout"
    assert result.timed_out is True
    assert result.passed is False


def test_grade_external_benchmark_marks_nested_timeout_as_timed_out():
    task = Task(
        id="PB_001",
        category="public_benchmarks",
        type="external_benchmark",
        question="Run benchmark",
        source="public_benchmarks.json",
    )
    response = _response(
        json.dumps(
            {
                "status": "completed",
                "score": 0.0,
                "error": None,
                "timed_out": False,
                "details": {
                    "group": "Coding",
                    "result": {
                        "status": "timed_out",
                        "evaluated_task_count": 1,
                        "valid_evaluated_task_count": 0,
                        "status_counts": {"timed_out": 1},
                    },
                },
            }
        )
    )

    result = grade_external_benchmark(task, response)

    assert result.status == "timed_out"
    assert result.timed_out is True
    assert result.passed is False


def test_grade_external_benchmark_maps_completed_zero_score_to_failed_model_answer():
    task = Task(
        id="PB_001",
        category="public_benchmarks",
        type="external_benchmark",
        question="Run benchmark",
        source="public_benchmarks.json",
    )
    response = _response(
        json.dumps(
            {
                "status": "completed",
                "score": 0.0,
                "error": None,
                "timed_out": False,
                "details": {
                    "group": "Coding",
                    "result": {
                        "status": "completed",
                        "evaluated_task_count": 1,
                        "valid_evaluated_task_count": 1,
                    },
                },
            }
        )
    )

    result = grade_external_benchmark(task, response)

    assert result.status == "failed_model_answer"
    assert result.answer == "failed_model_answer"
    assert result.passed is False


def test_grade_external_benchmark_preserves_primary_item_model_failure_status():
    task = Task(
        id="PB_TOOL",
        category="public_benchmarks",
        type="external_benchmark",
        question="Run benchmark",
        source="public_benchmarks.json",
    )
    response = _response(
        json.dumps(
            {
                "status": "completed",
                "score": 0.0,
                "error": None,
                "timed_out": False,
                "details": {
                    "group": "Finance",
                    "result": {
                        "status": "completed",
                        "evaluated_task_count": 1,
                        "valid_evaluated_task_count": 1,
                        "status_counts": {"failed_model_tool_use": 1},
                    },
                },
            }
        )
    )

    result = grade_external_benchmark(task, response)

    assert result.status == "failed_model_tool_use"
    assert result.answer == "failed_model_tool_use"
    assert result.passed is False
