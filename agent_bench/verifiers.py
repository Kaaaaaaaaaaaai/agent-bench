import json
import re
from collections import Counter
from typing import Any

from agent_bench.models import GradeResult, ModelResponse, Task
from agent_bench.sandbox import BaseSandbox


def parse_model_json(raw_response: str) -> tuple[dict[str, Any] | None, str | None]:
    text = _strip_fences(raw_response.strip())
    if not text:
        return None, "empty response"
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return None, "response did not contain a JSON object"
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            return None, f"invalid JSON: {exc.msg}"
    if not isinstance(value, dict):
        return None, "response JSON must be an object"
    return value, None


async def grade_task(
    task: Task,
    response: ModelResponse,
    sandbox: BaseSandbox,
    timeout_seconds: float,
) -> GradeResult:
    if response.error:
        return GradeResult(
            task_id=task.id,
            category=task.category,
            kind=task.type,
            score=0.0,
            max_score=1.0,
            passed=False,
            json_valid=False,
            latency_seconds=response.latency_seconds,
            **_response_measurements(response),
            error=response.error,
        )
    if task.is_multiple_choice:
        return grade_multiple_choice(task, response)
    if task.is_coding:
        return await grade_coding(task, response, sandbox, timeout_seconds)
    if task.is_text_recall:
        return grade_text_recall(task, response)
    return GradeResult(
        task_id=task.id,
        category=task.category,
        kind=task.type,
        score=0.0,
        max_score=1.0,
        passed=False,
        json_valid=False,
        latency_seconds=response.latency_seconds,
        **_response_measurements(response),
        error=f"Unsupported task type: {task.type}",
    )


def grade_multiple_choice(task: Task, response: ModelResponse) -> GradeResult:
    payload, error = parse_model_json(response.raw_response)
    if payload is None:
        return _invalid_json_result(task, response, error)

    answer = _normalize_answer(payload.get("answer"))
    confidence = _normalize_confidence(payload.get("confidence"))
    expected = _normalize_answer(task.answer)
    if answer is None:
        return GradeResult(
            task_id=task.id,
            category=task.category,
            kind=task.type,
            score=0.0,
            max_score=1.0,
            passed=False,
            json_valid=True,
            latency_seconds=response.latency_seconds,
            **_response_measurements(response),
            confidence=confidence,
            error="answer must be a choice letter string or list of strings",
            details={"expected": expected},
        )
    passed = set(answer) == set(expected) and len(answer) == len(expected)
    return GradeResult(
        task_id=task.id,
        category=task.category,
        kind=task.type,
        score=1.0 if passed else 0.0,
        max_score=1.0,
        passed=passed,
        json_valid=True,
        latency_seconds=response.latency_seconds,
        **_response_measurements(response),
        answer=answer,
        confidence=confidence,
        details={"expected": expected},
    )


def grade_text_recall(task: Task, response: ModelResponse) -> GradeResult:
    payload, error = parse_model_json(response.raw_response)
    if payload is None:
        return _invalid_json_result(task, response, error)

    answer = payload.get("answer")
    confidence = _normalize_confidence(payload.get("confidence"))
    if not isinstance(answer, str):
        return GradeResult(
            task_id=task.id,
            category=task.category,
            kind=task.type,
            score=0.0,
            max_score=1.0,
            passed=False,
            json_valid=True,
            latency_seconds=response.latency_seconds,
            **_response_measurements(response),
            confidence=confidence,
            error="answer must be a string",
        )

    expected = _normalize_recall_text(task.expected_text or "")
    actual = _normalize_recall_text(answer)
    score, true_positives, false_positives, false_negatives = _recall_token_f1(expected, actual)
    passed = score == 1.0
    return GradeResult(
        task_id=task.id,
        category=task.category,
        kind=task.type,
        score=score,
        max_score=1.0,
        passed=passed,
        json_valid=True,
        latency_seconds=response.latency_seconds,
        **_response_measurements(response),
        answer=answer,
        confidence=confidence,
        details={
            "expected": task.expected_text or "",
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "expected_token_count": len(expected.split()),
            "actual_token_count": len(actual.split()),
        },
    )


async def grade_coding(
    task: Task,
    response: ModelResponse,
    sandbox: BaseSandbox,
    timeout_seconds: float,
) -> GradeResult:
    payload, error = parse_model_json(response.raw_response)
    if payload is None:
        return _invalid_json_result(task, response, error)
    code = payload.get("code")
    confidence = _normalize_confidence(payload.get("confidence"))
    if not isinstance(code, str) or not code.strip():
        return GradeResult(
            task_id=task.id,
            category=task.category,
            kind=task.type,
            score=0.0,
            max_score=1.0,
            passed=False,
            json_valid=True,
            latency_seconds=response.latency_seconds,
            **_response_measurements(response),
            confidence=confidence,
            error="code must be a non-empty string",
        )

    sandbox_result = await sandbox.run(task, _clean_code(code), timeout_seconds)
    total = max(sandbox_result.total_cases, len(task.test_cases), 1)
    score = sandbox_result.passed_cases / total
    return GradeResult(
        task_id=task.id,
        category=task.category,
        kind=task.type,
        score=score,
        max_score=1.0,
        passed=score == 1.0,
        json_valid=True,
        latency_seconds=response.latency_seconds,
        **_response_measurements(response),
        answer="<code>",
        confidence=confidence,
        error=sandbox_result.error,
        timed_out=sandbox_result.timed_out,
        details={
            "passed_cases": sandbox_result.passed_cases,
            "total_cases": total,
            "case_results": sandbox_result.case_results,
        },
    )


def _invalid_json_result(task: Task, response: ModelResponse, error: str | None) -> GradeResult:
    return GradeResult(
        task_id=task.id,
        category=task.category,
        kind=task.type,
        score=0.0,
        max_score=1.0,
        passed=False,
        json_valid=False,
        latency_seconds=response.latency_seconds,
        **_response_measurements(response),
        error=error,
    )


def _response_measurements(response: ModelResponse) -> dict[str, Any]:
    return {
        "time_to_first_token_seconds": response.time_to_first_token_seconds,
        "tokens_per_second": response.tokens_per_second,
        "output_token_count": response.output_token_count,
    }


def _normalize_answer(value: Any) -> list[str] | None:
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, list):
        values = value
    else:
        return None
    normalized: list[str] = []
    for item in values:
        if not isinstance(item, str):
            return None
        letter = item.strip().upper()
        if not re.fullmatch(r"[A-Z]", letter):
            return None
        normalized.append(letter)
    return normalized


def _normalize_confidence(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))
    return None


def _strip_fences(text: str) -> str:
    if text.startswith("```"):
        text = re.sub(r"^```(?:json|python)?\s*", "", text.strip(), flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text.strip())
    return text


def _clean_code(code: str) -> str:
    stripped = _strip_fences(code.strip())
    return stripped


def _normalize_recall_text(value: str) -> str:
    normalized = value.replace("\r\n", "\n").replace("\r", "\n")
    return normalized.strip("\n")


def _recall_token_f1(expected: str, actual: str) -> tuple[float, int, int, int]:
    expected_counts = Counter(expected.split())
    actual_counts = Counter(actual.split())
    true_positives = sum((expected_counts & actual_counts).values())
    false_negatives = sum((expected_counts - actual_counts).values())
    false_positives = sum((actual_counts - expected_counts).values())
    denominator = (2 * true_positives) + false_positives + false_negatives
    if denominator == 0:
        return 1.0, true_positives, false_positives, false_negatives
    return (2 * true_positives) / denominator, true_positives, false_positives, false_negatives
