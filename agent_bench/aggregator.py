from collections import defaultdict
from typing import Any

from agent_bench.models import GradeResult


def aggregate_results(results: list[GradeResult], metadata: dict[str, Any]) -> dict[str, Any]:
    task_count = len(results)
    total_score = sum(result.score for result in results)
    passed_count = sum(1 for result in results if result.passed)
    json_valid_count = sum(1 for result in results if result.json_valid)
    timeout_count = sum(1 for result in results if result.timed_out)
    error_count = sum(1 for result in results if result.error)
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
        category: _percent(sum(item.score for item in items), len(items))
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

    return {
        "benchmark_version": "agent-bench-v0.1",
        "metadata": metadata,
        "task_count": task_count,
        "passed_count": passed_count,
        "total_score": _percent(total_score, task_count),
        "pass_rate": _percent(passed_count, task_count),
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
        "timing_by_category": timing_by_category,
        "timing_by_problem": timing_by_problem,
    }


def _percent(numerator: float, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round((numerator / denominator) * 100.0, 4)


def _average(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


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
