from collections import defaultdict
from typing import Any

from agent_bench.models import GradeResult
from agent_bench.statuses import (
    FAILED_DATASET_EXTRACTION,
    FAILED_GRADER,
    FAILED_HARNESS_SETUP,
    FAILED_MISSING_ASSETS,
    FAILED_TOKEN_BUDGET,
    INVALID_EVALUATION_STATUSES,
    PASSED,
    SKIPPED_UNSUPPORTED_CAPABILITY,
    STRICT_STATUSES,
    TIMED_OUT,
    is_skipped_like_status,
    normalize_status,
)


def aggregate_results(results: list[GradeResult], metadata: dict[str, Any]) -> dict[str, Any]:
    task_count = len(results)
    valid_results = [result for result in results if _is_valid_result(result)]
    valid_task_count = len(valid_results)
    model_valid_results = [result for result in valid_results if _capabilities_verified(result)]
    model_valid_task_count = len(model_valid_results)
    total_score = sum(result.score for result in valid_results)
    model_total_score = sum(result.score for result in model_valid_results)
    raw_total_score = sum(result.score for result in results)
    passed_count = sum(1 for result in results if result.passed)
    json_valid_count = sum(1 for result in results if result.json_valid)
    timeout_count = sum(1 for result in results if result.timed_out)
    error_count = sum(1 for result in results if result.error)
    status_counts = _status_counts(results)
    benchmark_item_status_counts = _benchmark_item_status_counts(results)
    grader_failure_count = _summary_status_count(
        results,
        status_counts,
        benchmark_item_status_counts,
        FAILED_GRADER,
        ("grader_failure_count", "judge_parse_failure_count", "judge_parse_failed_count"),
    )
    benchmark_item_skipped_count = sum(
        count for status, count in benchmark_item_status_counts.items() if is_skipped_like_status(status)
    )
    benchmark_item_setup_failure_count = benchmark_item_status_counts.get(FAILED_HARNESS_SETUP, 0)
    benchmark_item_passed_count = benchmark_item_status_counts.get(PASSED, 0)
    benchmark_item_valid_model_attempts = sum(
        count
        for status, count in benchmark_item_status_counts.items()
        if normalize_status(status) not in INVALID_EVALUATION_STATUSES
    )
    if not benchmark_item_status_counts:
        benchmark_item_valid_model_attempts = valid_task_count
    benchmark_item_passed_valid_attempts = (
        benchmark_item_status_counts.get(PASSED, 0) if benchmark_item_status_counts else passed_count
    )
    latency_values = [result.latency_seconds for result in results]
    ttft_values = [
        result.time_to_first_token_seconds
        for result in results
        if result.time_to_first_token_seconds is not None
    ]
    throughput_values = [
        result.tokens_per_second
        for result in results
        if result.tokens_per_second is not None
    ]
    task_duration_values = [
        result.task_duration_seconds
        for result in results
        if result.task_duration_seconds is not None
    ]
    output_token_values = [
        result.output_token_count for result in results if result.output_token_count is not None
    ]

    by_category: dict[str, list[GradeResult]] = defaultdict(list)
    for result in results:
        by_category[result.category].append(result)

    category_scores = {
        category: _percent(
            sum(item.score for item in _valid_items(items)),
            len(_valid_items(items)),
        )
        for category, items in sorted(by_category.items())
    }
    category_counts = {
        category: {
            "task_count": len(items),
            "passed_count": sum(1 for item in items if item.passed),
            "score": category_scores[category],
        }
        for category, items in sorted(by_category.items())
    }

    coding_results = [result for result in results if result.kind == "coding"]
    timing_by_category = _timing_by_category(results)
    timing_by_problem = _timing_by_problem(results)
    benchmark_results = _benchmark_results(results)

    return {
        "benchmark_version": "agent-bench-v0.1",
        "metadata": metadata,
        "task_count": task_count,
        "valid_task_count": valid_task_count,
        "passed_count": passed_count,
        "item_passed_count": benchmark_item_passed_count,
        "total_score": _percent(total_score, valid_task_count),
        "score_valid_tasks_only": _percent(total_score, valid_task_count),
        "model_score_valid_tasks_only": _percent(model_total_score, model_valid_task_count),
        "model_valid_task_count": model_valid_task_count,
        "raw_score": _percent(raw_total_score, task_count),
        "raw_score_all_tasks": _percent(raw_total_score, task_count),
        "pass_rate": _percent(passed_count, task_count),
        "status_counts": status_counts,
        "benchmark_item_status_counts": benchmark_item_status_counts,
        "coverage": _coverage_summary(results, status_counts, benchmark_item_status_counts),
        "harness_health": _harness_health_summary(status_counts, benchmark_item_status_counts),
        "headline": {
            "coverage": f"{valid_task_count}/{task_count} valid judged suites",
            "harness_health": _harness_health_summary(status_counts, benchmark_item_status_counts),
            "model_score": _percent(model_total_score, model_valid_task_count),
        },
        "num_suites_failed_setup": status_counts.get(FAILED_HARNESS_SETUP, 0),
        "num_items_failed_setup": benchmark_item_status_counts.get(FAILED_HARNESS_SETUP, 0),
        "num_suites_failed_extraction": status_counts.get(FAILED_DATASET_EXTRACTION, 0),
        "num_items_failed_extraction": benchmark_item_status_counts.get(FAILED_DATASET_EXTRACTION, 0),
        "num_items_failed_grader": benchmark_item_status_counts.get(FAILED_GRADER, 0),
        "num_items_failed_token_budget": benchmark_item_status_counts.get(FAILED_TOKEN_BUDGET, 0),
        "num_items_skipped_unsupported_capability": benchmark_item_status_counts.get(
            SKIPPED_UNSUPPORTED_CAPABILITY,
            0,
        ),
        "num_items_valid_model_attempts": benchmark_item_valid_model_attempts,
        "num_items_passed_valid_model_attempts": benchmark_item_passed_valid_attempts,
        "skipped_count": sum(count for status, count in status_counts.items() if is_skipped_like_status(status)),
        "missing_assets_count": status_counts.get(FAILED_MISSING_ASSETS, 0),
        "unsupported_capability_count": status_counts.get(SKIPPED_UNSUPPORTED_CAPABILITY, 0),
        "dataset_extraction_failed_count": status_counts.get(FAILED_DATASET_EXTRACTION, 0),
        "token_budget_failed_count": status_counts.get(FAILED_TOKEN_BUDGET, 0),
        "skipped_missing_assets_count": status_counts.get(FAILED_MISSING_ASSETS, 0),
        "skipped_unsupported_capability_count": status_counts.get(SKIPPED_UNSUPPORTED_CAPABILITY, 0),
        "benchmark_item_skipped_count": benchmark_item_skipped_count,
        "setup_failed_count": status_counts.get(FAILED_HARNESS_SETUP, 0),
        "harness_setup_failure_count": status_counts.get(FAILED_HARNESS_SETUP, 0),
        "benchmark_item_setup_failure_count": benchmark_item_setup_failure_count,
        "grader_failure_count": grader_failure_count,
        "judge_parse_failure_count": grader_failure_count,
        "judge_parse_failed_count": grader_failure_count,
        "json_validity_rate": _percent(json_valid_count, task_count),
        "timeout_rate": _percent(timeout_count, task_count),
        "error_rate": _percent(error_count, task_count),
        "average_latency_seconds": (sum(latency_values) / len(latency_values)) if latency_values else 0.0,
        "average_time_to_first_token_seconds": _average(ttft_values),
        "average_tokens_per_second": _average(throughput_values),
        "total_output_tokens": sum(output_token_values) if output_token_values else None,
        "total_run_duration_seconds": float(metadata.get("run_duration_seconds", 0.0)),
        "total_task_duration_seconds": sum(task_duration_values),
        "coding_pass_rate": _percent(sum(1 for result in coding_results if result.passed), len(coding_results))
        if coding_results
        else None,
        "category_scores": category_scores,
        "category_counts": category_counts,
        "benchmark_results": benchmark_results,
        "timing_by_category": timing_by_category,
        "timing_by_problem": timing_by_problem,
        "timing_note": (
            "total_task_duration_seconds is the sum of per-task elapsed time and can exceed "
            "total_run_duration_seconds when tasks run concurrently."
        ),
    }


def _percent(numerator: float, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round((numerator / denominator) * 100.0, 4)


def _unit_to_percent(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return round(float(value) * 100.0, 4)
    return None


def _average(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _valid_items(items: list[GradeResult]) -> list[GradeResult]:
    return [item for item in items if _is_valid_result(item)]


def _is_valid_result(result: GradeResult) -> bool:
    return _result_status(result) not in INVALID_EVALUATION_STATUSES


def _capabilities_verified(result: GradeResult) -> bool:
    if result.kind != "external_benchmark":
        return True
    payload = _benchmark_payload(result)
    value = payload.get("capabilities_verified")
    if isinstance(value, bool):
        return value
    unsupported = payload.get("unsupported_capabilities")
    if isinstance(unsupported, list) and unsupported:
        return False
    contract = payload.get("capability_contract")
    if isinstance(contract, dict):
        for item in contract.values():
            if isinstance(item, dict) and item.get("supported") is False:
                return False
    return _result_status(result) not in {
        FAILED_HARNESS_SETUP,
        FAILED_DATASET_EXTRACTION,
        FAILED_GRADER,
        FAILED_MISSING_ASSETS,
        SKIPPED_UNSUPPORTED_CAPABILITY,
        FAILED_TOKEN_BUDGET,
        TIMED_OUT,
    }


def _status_counts(results: list[GradeResult]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for result in results:
        counts[_result_status(result)] += 1
    return dict(sorted(counts.items()))


def _result_status(result: GradeResult) -> str:
    status = normalize_status(result.status)
    if status in STRICT_STATUSES:
        return status
    if result.status == "completed":
        return PASSED if result.passed else "failed_model_answer"
    if result.status:
        return FAILED_HARNESS_SETUP
    if result.timed_out:
        return TIMED_OUT
    if result.passed:
        return PASSED
    return "failed_model_answer"


def _benchmark_item_status_counts(results: list[GradeResult]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for result in results:
        payload = _benchmark_payload(result)
        status_counts = payload.get("status_counts")
        if not isinstance(status_counts, dict):
            continue
        for status, count in status_counts.items():
            if isinstance(status, str) and isinstance(count, int):
                counts[normalize_status(status) or status] += count
    return dict(sorted(counts.items()))


def _summary_status_count(
    results: list[GradeResult],
    status_counts: dict[str, int],
    benchmark_item_status_counts: dict[str, int],
    status: str,
    nested_keys: tuple[str, ...],
) -> int:
    nested_count = 0
    for result in results:
        payload = _benchmark_payload(result)
        for key in nested_keys:
            value = payload.get(key)
            if isinstance(value, int):
                nested_count += value
                break
    if nested_count:
        return nested_count
    return benchmark_item_status_counts.get(status, status_counts.get(status, 0))


def _benchmark_payload(result: GradeResult) -> dict[str, Any]:
    details = result.details if isinstance(result.details, dict) else {}
    payload = details.get("result")
    return payload if isinstance(payload, dict) else {}


def _payload_status_count(payload: dict[str, Any], status: str) -> int:
    status_counts = payload.get("status_counts")
    if not isinstance(status_counts, dict):
        return 0
    normalized = normalize_status(status) or status
    for key, count in status_counts.items():
        if normalize_status(key) == normalized and isinstance(count, int):
            return int(count)
    return 0


def _coverage_summary(
    results: list[GradeResult],
    status_counts: dict[str, int],
    benchmark_item_status_counts: dict[str, int],
) -> dict[str, Any]:
    task_count = len(results)
    valid_count = sum(1 for result in results if _is_valid_result(result))
    unsupported_count = status_counts.get(SKIPPED_UNSUPPORTED_CAPABILITY, 0)
    missing_assets_count = status_counts.get(FAILED_MISSING_ASSETS, 0)
    dataset_failed_count = status_counts.get(FAILED_DATASET_EXTRACTION, 0)
    return {
        "task_count": task_count,
        "valid_judged_task_count": valid_count,
        "coverage_rate": _percent(valid_count, task_count),
        "unsupported_capability_count": unsupported_count,
        "missing_assets_count": missing_assets_count,
        "dataset_extraction_failed_count": dataset_failed_count,
        "benchmark_item_valid_count": sum(
            count
            for status, count in benchmark_item_status_counts.items()
            if status not in INVALID_EVALUATION_STATUSES
        ),
        "benchmark_item_invalid_count": sum(
            count
            for status, count in benchmark_item_status_counts.items()
            if status in INVALID_EVALUATION_STATUSES
        ),
    }


def _harness_health_summary(
    status_counts: dict[str, int],
    benchmark_item_status_counts: dict[str, int],
) -> dict[str, int]:
    return {
        "setup_failure_count": status_counts.get(FAILED_HARNESS_SETUP, 0),
        "dataset_extraction_failure_count": status_counts.get(FAILED_DATASET_EXTRACTION, 0),
        "missing_assets_count": status_counts.get(FAILED_MISSING_ASSETS, 0),
        "grader_failure_count": status_counts.get(FAILED_GRADER, 0),
        "token_budget_failure_count": status_counts.get(FAILED_TOKEN_BUDGET, 0),
        "benchmark_item_setup_failure_count": benchmark_item_status_counts.get(FAILED_HARNESS_SETUP, 0),
        "benchmark_item_dataset_extraction_failure_count": benchmark_item_status_counts.get(
            FAILED_DATASET_EXTRACTION,
            0,
        ),
        "benchmark_item_grader_failure_count": benchmark_item_status_counts.get(FAILED_GRADER, 0),
    }


def _timing_by_category(results: list[GradeResult]) -> dict[str, dict[str, float | int | None]]:
    grouped: dict[str, list[GradeResult]] = defaultdict(list)
    for result in results:
        grouped[result.category].append(result)

    metrics: dict[str, dict[str, float | int | None]] = {}
    for category, items in sorted(grouped.items()):
        latency_values = [item.latency_seconds for item in items]
        task_duration_values = [
            item.task_duration_seconds
            for item in items
            if item.task_duration_seconds is not None
        ]
        ttft_values = [
            item.time_to_first_token_seconds
            for item in items
            if item.time_to_first_token_seconds is not None
        ]
        throughput_values = [
            item.tokens_per_second
            for item in items
            if item.tokens_per_second is not None
        ]
        output_tokens = [
            item.output_token_count for item in items if item.output_token_count is not None
        ]
        metrics[category] = {
            "task_count": len(items),
            "total_request_latency_seconds": sum(latency_values),
            "average_request_latency_seconds": _average(latency_values),
            "total_task_duration_seconds": sum(task_duration_values),
            "average_task_duration_seconds": _average(task_duration_values),
            "average_time_to_first_token_seconds": _average(ttft_values),
            "average_tokens_per_second": _average(throughput_values),
            "total_output_tokens": sum(output_tokens) if output_tokens else None,
        }
    return metrics


def _timing_by_problem(results: list[GradeResult]) -> dict[str, dict[str, Any]]:
    return {
        result.task_id: {
            "category": result.category,
            "kind": result.kind,
            "request_latency_seconds": result.latency_seconds,
            "task_duration_seconds": result.task_duration_seconds,
            "time_to_first_token_seconds": result.time_to_first_token_seconds,
            "tokens_per_second": result.tokens_per_second,
            "output_token_count": result.output_token_count,
        }
        for result in results
    }


def _benchmark_results(results: list[GradeResult]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        details = result.details if isinstance(result.details, dict) else {}
        benchmark = details.get("benchmark")
        if result.kind != "external_benchmark" and not isinstance(benchmark, str):
            continue
        if not isinstance(benchmark, str) or not benchmark.strip():
            benchmark = result.task_id
        group = details.get("group")
        if not isinstance(group, str) or not group.strip():
            group = result.category
        benchmark_payload = _benchmark_payload(result)
        model_eval = (
            benchmark_payload.get("model_eval")
            if isinstance(benchmark_payload.get("model_eval"), dict)
            else {}
        )
        model_evals = (
            benchmark_payload.get("model_evals")
            if isinstance(benchmark_payload.get("model_evals"), list)
            else []
        )
        grading_methods = _grading_methods(model_evals)
        rows.append(
            {
                "task_id": result.task_id,
                "group": group,
                "benchmark": benchmark,
                "score": round(result.score * 100.0, 4),
                "raw_score": _unit_to_percent(benchmark_payload.get("raw_score")),
                "valid_score": _unit_to_percent(benchmark_payload.get("valid_score")),
                "passed": result.passed,
                "status": _result_status(result),
                "error": result.error,
                "homepage": details.get("homepage", ""),
                "license": details.get("license", ""),
                "credit": details.get("credit", ""),
                "citation": details.get("citation", details.get("homepage", "")),
                "model_eval": model_eval,
                "model_evals": model_evals,
                "grading_methods": grading_methods,
                "adapter": benchmark_payload.get("adapter", details.get("adapter", "")),
                "capabilities_verified": _capabilities_verified(result),
                "capability_contract": benchmark_payload.get("capability_contract", {}),
                "supported_capabilities": benchmark_payload.get("supported_capabilities", []),
                "repository_ready": bool(benchmark_payload.get("repository_ready", False)),
                "file_count_sampled": benchmark_payload.get("file_count_sampled"),
                "extracted_task_count": benchmark_payload.get("extracted_task_count"),
                "evaluated_task_count": benchmark_payload.get("evaluated_task_count"),
                "valid_evaluated_task_count": benchmark_payload.get("valid_evaluated_task_count"),
                "evaluation_passed_count": benchmark_payload.get("evaluation_passed_count"),
                "skipped_task_count": benchmark_payload.get("skipped_task_count"),
                "setup_failed_count": _payload_status_count(benchmark_payload, "failed_harness_setup"),
                "missing_assets_count": _payload_status_count(benchmark_payload, FAILED_MISSING_ASSETS),
                "unsupported_capability_count": _payload_status_count(
                    benchmark_payload,
                    SKIPPED_UNSUPPORTED_CAPABILITY,
                ),
                "dataset_extraction_failed_count": _payload_status_count(
                    benchmark_payload,
                    FAILED_DATASET_EXTRACTION,
                ),
                "skipped_missing_assets_count": _payload_status_count(benchmark_payload, FAILED_MISSING_ASSETS),
                "skipped_unsupported_capability_count": _payload_status_count(
                    benchmark_payload,
                    SKIPPED_UNSUPPORTED_CAPABILITY,
                ),
                "grader_failure_count": benchmark_payload.get("grader_failure_count"),
                "judge_parse_failure_count": benchmark_payload.get("judge_parse_failure_count"),
                "judge_parse_repaired_count": benchmark_payload.get("judge_parse_repaired_count"),
                "status_counts": _normalized_status_counts(benchmark_payload.get("status_counts")),
                "required_capabilities": benchmark_payload.get("required_capabilities", []),
                "unsupported_capabilities": benchmark_payload.get("unsupported_capabilities", []),
                "extraction_sources": benchmark_payload.get("extraction_sources", []),
            }
        )
    return sorted(rows, key=lambda row: (str(row["group"]), str(row["benchmark"])))


def _grading_methods(model_evals: list[Any]) -> list[str]:
    methods: set[str] = set()
    for item in model_evals:
        if not isinstance(item, dict):
            continue
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        grade = item.get("grade") if isinstance(item.get("grade"), dict) else {}
        for value in (metadata.get("grading"), grade.get("method")):
            if isinstance(value, str) and value:
                methods.add(value)
    return sorted(methods)


def _normalized_status_counts(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    counts: dict[str, int] = defaultdict(int)
    for status, count in value.items():
        if isinstance(status, str) and isinstance(count, int):
            counts[normalize_status(status) or status] += count
    return dict(sorted(counts.items()))
