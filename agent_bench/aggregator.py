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
    benchmark_results = _benchmark_results(results)

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
        "benchmark_results": benchmark_results,
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
        benchmark_payload = details.get("result") if isinstance(details.get("result"), dict) else {}
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
                "passed": result.passed,
                "error": result.error,
                "homepage": details.get("homepage", ""),
                "license": details.get("license", ""),
                "credit": details.get("credit", ""),
                "model_eval": model_eval,
                "model_evals": model_evals,
                "grading_methods": grading_methods,
                "repository_ready": bool(benchmark_payload.get("repository_ready", False)),
                "file_count_sampled": benchmark_payload.get("file_count_sampled"),
                "extracted_task_count": benchmark_payload.get("extracted_task_count"),
                "evaluated_task_count": benchmark_payload.get("evaluated_task_count"),
                "evaluation_passed_count": benchmark_payload.get("evaluation_passed_count"),
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
