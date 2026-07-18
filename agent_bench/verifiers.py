import json
import re
from collections import Counter
from typing import Any

from agent_bench.models import GradeResult, ModelResponse, Task
from agent_bench.sandbox import BaseSandbox
from agent_bench.statuses import (
    FAILED_DATASET_EXTRACTION,
    FAILED_GRADER,
    FAILED_HARNESS_SETUP,
    FAILED_INVALID_TASK_CONTEXT,
    FAILED_MISSING_ASSETS,
    FAILED_MISSING_REQUIRED_TOOL,
    FAILED_MODEL_ANSWER,
    FAILED_MODEL_FORMAT,
    FAILED_MODEL_MISSING_ARTIFACT,
    FAILED_MODEL_TOOL_USE,
    FAILED_TOKEN_BUDGET,
    FAILED_TIMEOUT,
    INVALID_EVALUATION_STATUSES,
    PASSED,
    SKIPPED_UNSUPPORTED_CAPABILITY,
    STRICT_STATUSES,
    TIMED_OUT,
    normalize_status,
)


MODEL_FAILURE_STATUSES = {
    FAILED_MODEL_ANSWER,
    FAILED_MODEL_FORMAT,
    FAILED_MODEL_MISSING_ARTIFACT,
    FAILED_MODEL_TOOL_USE,
}


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
            status=FAILED_HARNESS_SETUP,
        )
    if task.is_multiple_choice:
        return grade_multiple_choice(task, response)
    if task.is_coding:
        return await grade_coding(task, response, sandbox, timeout_seconds)
    if task.is_text_recall:
        return grade_text_recall(task, response)
    if task.is_external_benchmark:
        return grade_external_benchmark(task, response)
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
        status=FAILED_HARNESS_SETUP,
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
            details=_extraction_details(response, None, "failed") | {"expected": expected},
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
        details=_extraction_details(response, answer, "extracted") | {"expected": expected},
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
            details=_extraction_details(response, None, "failed"),
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
            **_extraction_details(response, answer, "extracted"),
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
            details=_extraction_details(response, None, "failed"),
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
        status=TIMED_OUT if sandbox_result.timed_out else "",
        details={
            **_extraction_details(response, "<code>", "extracted"),
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
        status=FAILED_MODEL_FORMAT,
        details=_extraction_details(response, None, "failed"),
    )


def _response_measurements(response: ModelResponse) -> dict[str, Any]:
    return {
        "time_to_first_token_seconds": response.time_to_first_token_seconds,
        "tokens_per_second": response.tokens_per_second,
        "output_token_count": response.output_token_count,
    }


def _extraction_details(response: ModelResponse, extracted_answer: Any, status: str) -> dict[str, Any]:
    return {
        "raw_model_response": response.raw_response,
        "extracted_answer": extracted_answer,
        "extraction_status": status,
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

def grade_external_benchmark(task: Task, response: ModelResponse) -> GradeResult:
    payload, error = parse_model_json(response.raw_response)
    if payload is None:
        return _invalid_json_result(task, response, error)
    score = payload.get("score", 0.0)
    if not isinstance(score, (int, float)):
        score = 0.0
    normalized_score = max(0.0, min(1.0, float(score)))
    benchmark_error = payload.get("error") if isinstance(payload.get("error"), str) else None
    details = dict(payload.get("details")) if isinstance(payload.get("details"), dict) else {}
    _add_benchmark_descriptor_details(task, details)
    group = details.get("group")
    result_payload = details.get("result") if isinstance(details.get("result"), dict) else {}
    result_status = result_payload.get("status") if isinstance(result_payload.get("status"), str) else ""
    status = _external_benchmark_status(
        result_status,
        payload.get("status") if isinstance(payload.get("status"), str) else "",
        result_payload,
        normalized_score,
        benchmark_error,
    )
    if status in INVALID_EVALUATION_STATUSES and not benchmark_error:
        benchmark_error = _external_status_error(status, result_payload)
    if status in INVALID_EVALUATION_STATUSES:
        normalized_score = 0.0
    return GradeResult(
        task_id=task.id,
        category=group if isinstance(group, str) and group.strip() else task.category,
        kind=task.type,
        score=normalized_score,
        max_score=1.0,
        passed=(
            normalized_score >= 1.0
            and benchmark_error is None
            and status == PASSED
        ),
        json_valid=True,
        latency_seconds=response.latency_seconds,
        **_response_measurements(response),
        answer=status,
        error=benchmark_error,
        timed_out=bool(payload.get("timed_out", False)) or status == TIMED_OUT,
        status=status,
        details=details,
    )


def _add_benchmark_descriptor_details(task: Task, details: dict[str, Any]) -> None:
    benchmark = task.benchmark if isinstance(task.benchmark, dict) else {}
    details.setdefault("benchmark_name", task.id)
    details.setdefault("benchmark", benchmark.get("name", task.id))
    details.setdefault("group", benchmark.get("group", task.category))
    details.setdefault("homepage", benchmark.get("homepage", ""))
    details.setdefault("license", benchmark.get("license", ""))
    details.setdefault("credit", benchmark.get("credit", ""))
    details.setdefault("citation", benchmark.get("citation", benchmark.get("homepage", "")))
    details["required_capabilities"] = benchmark.get("capabilities", [])
    details["required_tools"] = benchmark.get("required_tools", [])


def _external_benchmark_status(
    result_status: str,
    payload_status: str,
    result_payload: dict[str, Any],
    score: float,
    error: str | None,
) -> str:
    status = normalize_status(result_status or payload_status)
    if status in {FAILED_TIMEOUT, TIMED_OUT}:
        return status
    capability_status = _external_capability_failure_status(result_payload)
    if capability_status:
        return capability_status
    if status in STRICT_STATUSES:
        return status
    status_counts = result_payload.get("status_counts")
    if isinstance(status_counts, dict):
        evaluated_count = result_payload.get("evaluated_task_count")
        valid_count = result_payload.get("valid_evaluated_task_count")
        if isinstance(evaluated_count, int) and isinstance(valid_count, int):
            if evaluated_count > 0 and valid_count <= 0:
                for invalid_status in (
                    FAILED_HARNESS_SETUP,
                    FAILED_DATASET_EXTRACTION,
                    FAILED_GRADER,
                    FAILED_TOKEN_BUDGET,
                    FAILED_MISSING_REQUIRED_TOOL,
                    FAILED_INVALID_TASK_CONTEXT,
                    TIMED_OUT,
                    FAILED_MISSING_ASSETS,
                    SKIPPED_UNSUPPORTED_CAPABILITY,
                ):
                    if _status_count(status_counts, invalid_status):
                        return invalid_status
        primary_failure = _primary_model_failure_status(status_counts)
        if score < 1.0 and primary_failure:
            return primary_failure
    if status == "error":
        return FAILED_HARNESS_SETUP
    if error:
        return FAILED_HARNESS_SETUP
    return PASSED if score >= 1.0 else FAILED_MODEL_ANSWER


def _external_capability_failure_status(result_payload: dict[str, Any]) -> str:
    missing_tools = _string_values(result_payload.get("missing_tools"))
    missing_env = _string_values(result_payload.get("missing_env")) or _string_values(
        result_payload.get("missing_environment")
    )
    if missing_tools or missing_env:
        return FAILED_MISSING_REQUIRED_TOOL
    required = set(_string_values(result_payload.get("required_capabilities")))
    exposed_tools = _string_values(result_payload.get("exposed_tools"))
    if "tool_call" in required and not exposed_tools:
        return FAILED_MISSING_REQUIRED_TOOL
    if result_payload.get("capabilities_verified") is False:
        status = normalize_status(result_payload.get("status") if isinstance(result_payload.get("status"), str) else "")
        if status in INVALID_EVALUATION_STATUSES:
            return status
        return FAILED_HARNESS_SETUP
    return ""


def _string_values(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value else []
    if not isinstance(value, list):
        return []
    values: list[str] = []
    for item in value:
        if isinstance(item, str) and item:
            values.append(item)
    return values


def _primary_model_failure_status(status_counts: dict[str, Any]) -> str:
    counts: dict[str, int] = {}
    for key, value in status_counts.items():
        status = normalize_status(key)
        if status in MODEL_FAILURE_STATUSES and isinstance(value, int) and value > 0:
            counts[status] = counts.get(status, 0) + value
    if not counts:
        return ""
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _external_status_error(status: str, result_payload: dict[str, Any]) -> str:
    error = result_payload.get("error")
    if isinstance(error, str) and error:
        return error
    status_counts = result_payload.get("status_counts")
    if isinstance(status_counts, dict):
        total = sum(count for count in status_counts.values() if isinstance(count, int))
        if total > 0:
            return f"All {total} benchmark record evaluation(s) were invalid: {status.replace('_', ' ')}"
    return status.replace("_", " ")


def _status_count(status_counts: dict[str, Any], status: str) -> int:
    normalized = normalize_status(status)
    for key, value in status_counts.items():
        if normalize_status(key) == normalized and isinstance(value, int):
            return value
    return 0
