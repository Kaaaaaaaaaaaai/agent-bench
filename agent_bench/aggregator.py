from collections import defaultdict
from typing import Any

from agent_bench.models import GradeResult
from agent_bench.statuses import (
    BLOCKER_DISABLED_SCORING,
    BLOCKER_EXTERNAL_PLATFORM_UNAVAILABLE,
    BLOCKER_GIT_LFS_POINTER_STUB,
    BLOCKER_INVALID_TASK_CONTEXT,
    BLOCKER_JUDGE_PARSE_ERROR,
    BLOCKER_MISSING_ASSET,
    BLOCKER_MISSING_GRADER,
    BLOCKER_MISSING_REFERENCE_DATASET,
    BLOCKER_MISSING_REFERENCE_DOCUMENTS,
    BLOCKER_MISSING_REQUIRED_TOOL,
    BLOCKER_MISSING_TASK_INSTANCES,
    BLOCKER_OUTPUT_PARSE_ERROR,
    BLOCKER_REPO_PATCH_HARNESS_SETUP,
    BLOCKER_UNSUPPORTED_CAPABILITY,
    FAILED_DATASET_EXTRACTION,
    FAILED_GRADER,
    FAILED_HARNESS_SETUP,
    FAILED_MISSING_ASSETS,
    FAILED_INVALID_TASK_CONTEXT,
    FAILED_MODEL_ANSWER,
    FAILED_MODEL_FORMAT,
    FAILED_MODEL_TOOL_USE,
    FAILED_TOKEN_BUDGET,
    FAILED_MISSING_REQUIRED_TOOL,
    INVALID_EVALUATION_STATUSES,
    PASSED,
    FAILED_MODEL_MISSING_ARTIFACT,
    RUN_COMPLETED,
    RUN_EXECUTION_ERROR,
    RUN_INFRASTRUCTURE_ERROR,
    RUN_SKIPPED,
    SCORE_FAILED_MODEL_ANSWER,
    SCORE_NOT_APPLICABLE,
    SCORE_PARTIALLY_CORRECT,
    SCORE_PASSED,
    SCORE_UNGRADED,
    SKIPPED_UNSUPPORTED_CAPABILITY,
    STRICT_STATUSES,
    TIMED_OUT,
    is_invalid_evaluation_status,
    is_skipped_like_status,
    normalize_status,
    status_catalog_dict,
)


def aggregate_results(
    results: list[GradeResult],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    task_count = len(results)
    known_suite_count = _int_metadata(metadata, "known_suite_count", task_count)
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
            "attempted_benchmark_count": len(items),
            "successfully_scored_benchmark_count": len(_valid_items(items)),
            "failed_benchmark_count": len([item for item in items if not _is_valid_result(item)]),
            "coverage_rate": _ratio(len(_valid_items(items)), len(items)),
            "score": category_scores[category],
        }
        for category, items in sorted(by_category.items())
    }
    valid_category_scores = [
        score
        for category, score in category_scores.items()
        if any(_is_valid_result(item) for item in by_category[category])
    ]
    primary_overall_score = round(sum(valid_category_scores) / len(valid_category_scores), 4) if valid_category_scores else None

    coding_results = [result for result in results if result.kind == "coding"]
    timing_by_category = _timing_by_category(results)
    timing_by_problem = _timing_by_problem(results)
    benchmark_results = _benchmark_results(results)
    item_count, valid_item_count = _item_coverage_counts(results)
    suite_blockers = _suite_blockers(results)
    blocker_counts = _blocker_counts(results)
    excluded_suites = _merged_excluded_suites(_excluded_suites(metadata), _result_excluded_suites(results))
    excluded_suite_count = len(excluded_suites)
    summary_metadata = dict(metadata)
    summary_metadata["excluded_suites"] = excluded_suites
    summary_metadata["excluded_suite_count"] = excluded_suite_count
    profile_results = _profile_results(benchmark_results)
    parser_repair_count = _parser_repair_count(results)
    usage_summary = _usage_summary(results)

    return {
        "benchmark_version": "agent-bench-v0.1",
        "schema_version": "agent-bench.summary.v2",
        "selected_profile": metadata.get("selected_profile", "full_active"),
        "known_suite_count": known_suite_count,
        "selected_suite_count": task_count,
        "excluded_suite_count": excluded_suite_count,
        "excluded_suites": excluded_suites,
        "metadata": summary_metadata,
        "task_count": task_count,
        "valid_task_count": valid_task_count,
        "valid_judged_suite_count": valid_task_count,
        "suite_count": task_count,
        "item_count": item_count,
        "valid_judged_item_count": valid_item_count,
        "passed_count": passed_count,
        "item_passed_count": benchmark_item_passed_count,
        "valid_judged_score": _ratio(total_score, valid_task_count),
        "valid_judged_suite_score": _ratio(total_score, valid_task_count),
        "valid_judged_item_score": _ratio(
            sum(result.score for result in model_valid_results),
            valid_item_count,
        ),
        "model_verified_score": _ratio(model_total_score, model_valid_task_count),
        "conservative_selected_suite_score": _ratio(raw_total_score, task_count),
        "suite_coverage_rate": _ratio(valid_task_count, task_count),
        "item_coverage_rate": _ratio(valid_item_count, item_count),
        "conservative_all_suite_score": _ratio(raw_total_score, task_count),
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
        "coverage_summary": _official_coverage_summary(results, category_counts),
        "harness_health": _harness_health_summary(status_counts, benchmark_item_status_counts),
        "headline": {
            "valid_judged_score": _ratio(total_score, valid_task_count),
            "valid_judged_suite_score": _ratio(total_score, valid_task_count),
            "model_verified_score": _ratio(model_total_score, model_valid_task_count),
            "conservative_selected_suite_score": _ratio(raw_total_score, task_count),
            "suite_coverage_rate": _ratio(valid_task_count, task_count),
            "item_coverage_rate": _ratio(valid_item_count, item_count),
            "conservative_all_suite_score": _ratio(raw_total_score, task_count),
            "selected_suite_count": task_count,
            "excluded_suite_count": excluded_suite_count,
            "valid_judged_suite_count": valid_task_count,
            "known_suite_count": known_suite_count,
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
        "skipped_suite_count": len(suite_blockers),
        "skipped_suites": suite_blockers,
        "blocker_counts": blocker_counts,
        "skipped_count": sum(count for status, count in status_counts.items() if is_skipped_like_status(status)),
        "missing_asset_count": blocker_counts.get(BLOCKER_MISSING_ASSET, 0)
        + blocker_counts.get(BLOCKER_GIT_LFS_POINTER_STUB, 0),
        "git_lfs_pointer_stub_count": blocker_counts.get(BLOCKER_GIT_LFS_POINTER_STUB, 0),
        "missing_reference_dataset_count": blocker_counts.get(BLOCKER_MISSING_REFERENCE_DATASET, 0),
        "missing_reference_documents_count": blocker_counts.get(BLOCKER_MISSING_REFERENCE_DOCUMENTS, 0),
        "missing_required_tool_count": blocker_counts.get(BLOCKER_MISSING_REQUIRED_TOOL, 0),
        "invalid_task_context_count": blocker_counts.get(BLOCKER_INVALID_TASK_CONTEXT, 0),
        "judge_parse_error_count": blocker_counts.get(BLOCKER_JUDGE_PARSE_ERROR, 0),
        "repo_patch_harness_setup_count": blocker_counts.get(BLOCKER_REPO_PATCH_HARNESS_SETUP, 0),
        "missing_assets_count": status_counts.get(FAILED_MISSING_ASSETS, 0),
        "unsupported_capability_count": _unsupported_capability_count(results, status_counts),
        "missing_grader_count": blocker_counts.get(BLOCKER_MISSING_GRADER, 0),
        "judge_error_count": grader_failure_count,
        "parser_repair_count": parser_repair_count,
        "dataset_extraction_failed_count": status_counts.get(FAILED_DATASET_EXTRACTION, 0),
        "token_budget_failed_count": status_counts.get(FAILED_TOKEN_BUDGET, 0),
        "skipped_missing_assets_count": status_counts.get(FAILED_MISSING_ASSETS, 0),
        "skipped_unsupported_capability_count": _unsupported_capability_count(results, status_counts),
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
        **usage_summary,
        "total_output_tokens": sum(output_token_values) if output_token_values else None,
        "total_run_duration_seconds": float(metadata.get("run_duration_seconds", 0.0)),
        "total_task_duration_seconds": sum(task_duration_values),
        "coding_pass_rate": _percent(sum(1 for result in coding_results if result.passed), len(coding_results))
        if coding_results
        else None,
        "category_scores": category_scores,
        "category_counts": category_counts,
        "overall_score": primary_overall_score,
        "overall_score_fraction": round(primary_overall_score / 100.0, 6) if primary_overall_score is not None else None,
        "primary_score_method": "equal_weight_category_mean",
        "benchmark_level_mean_score": _percent(total_score, valid_task_count),
        "status_catalog": status_catalog_dict(),
        "benchmark_results": benchmark_results,
        "profile_results": profile_results,
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


def _ratio(numerator: float, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(float(numerator) / float(denominator), 6)


def _unit_to_percent(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return round(float(value) * 100.0, 4)
    return None


def _average(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _int_metadata(metadata: dict[str, Any], key: str, default: int) -> int:
    value = metadata.get(key)
    if isinstance(value, int):
        return value
    return default


def _excluded_suites(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    raw = metadata.get("excluded_suites")
    if not isinstance(raw, list):
        return []
    rows: list[dict[str, Any]] = []
    for row in raw:
        if not isinstance(row, dict):
            continue
        suite_id = row.get("suite_id")
        name = row.get("name")
        if not isinstance(suite_id, str) or not suite_id.strip():
            continue
        rows.append(
            {
                "suite_id": suite_id,
                "name": str(name or suite_id),
                "lifecycle_status": str(row.get("lifecycle_status") or "removed"),
                "exclusion_reason": str(row.get("exclusion_reason") or "removed_from_active_suite"),
                "included_in_official_score": False,
                "removal_reason": str(row.get("removal_reason") or ""),
            }
        )
    return sorted(rows, key=lambda row: row["suite_id"])


def _result_excluded_suites(results: list[GradeResult]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        status = _result_status(result)
        if _is_valid_result(result):
            continue
        rows.append(
            {
                "suite_id": result.task_id,
                "name": _benchmark_name(result),
                "lifecycle_status": "score_excluded",
                "exclusion_reason": _blocker_type(result) or status,
                "included_in_official_score": False,
                "status": status,
                "run_status": _run_status(result),
                "error": _result_error_reason(result),
            }
        )
    return rows


def _merged_excluded_suites(*groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for group in groups:
        for row in group:
            suite_id = row.get("suite_id")
            if not isinstance(suite_id, str) or not suite_id:
                continue
            existing = merged.get(suite_id, {})
            merged[suite_id] = {**existing, **row}
    return sorted(merged.values(), key=lambda row: str(row.get("suite_id", "")))


def _valid_items(items: list[GradeResult]) -> list[GradeResult]:
    return [item for item in items if _is_valid_result(item)]


def _benchmark_name(result: GradeResult) -> str:
    payload = _benchmark_payload(result)
    details = result.details if isinstance(result.details, dict) else {}
    return str(payload.get("benchmark") or details.get("benchmark") or result.task_id)


def _is_valid_result(result: GradeResult) -> bool:
    if _result_status(result) in INVALID_EVALUATION_STATUSES:
        return False
    if _explicitly_excluded_from_official(result):
        return False
    if result.kind == "external_benchmark" and not _capabilities_verified(result):
        return False
    return True


def _explicitly_excluded_from_official(result: GradeResult) -> bool:
    payload = _benchmark_payload(result)
    return payload.get("included_in_official_score") is False


def _capabilities_verified(result: GradeResult) -> bool:
    if result.kind != "external_benchmark":
        return True
    payload = _benchmark_payload(result)
    required = payload.get("required_capabilities")
    missing_tools = payload.get("missing_tools")
    missing_env = payload.get("missing_env", payload.get("missing_environment"))
    exposed_tools = payload.get("exposed_tools")
    if isinstance(missing_tools, list) and missing_tools:
        return False
    if isinstance(missing_env, list) and missing_env:
        return False
    if isinstance(required, list) and "tool_call" in {str(item) for item in required}:
        if not isinstance(exposed_tools, list) or not exposed_tools:
            return False
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


def _run_status(result: GradeResult) -> str:
    status = _result_status(result)
    if status in {
        PASSED,
        FAILED_MODEL_ANSWER,
        FAILED_MODEL_FORMAT,
        FAILED_MODEL_MISSING_ARTIFACT,
        FAILED_MODEL_TOOL_USE,
    }:
        return RUN_COMPLETED
    if status in {FAILED_MISSING_ASSETS, FAILED_MISSING_REQUIRED_TOOL, SKIPPED_UNSUPPORTED_CAPABILITY}:
        return RUN_SKIPPED
    if status in {FAILED_GRADER, FAILED_TOKEN_BUDGET, FAILED_INVALID_TASK_CONTEXT}:
        return RUN_INFRASTRUCTURE_ERROR
    if status in {FAILED_HARNESS_SETUP, FAILED_DATASET_EXTRACTION, TIMED_OUT}:
        return RUN_EXECUTION_ERROR
    return RUN_INFRASTRUCTURE_ERROR if is_invalid_evaluation_status(status) else RUN_COMPLETED


def _score_status(result: GradeResult) -> str:
    status = _result_status(result)
    if (
        status in INVALID_EVALUATION_STATUSES
        or _explicitly_excluded_from_official(result)
        or (result.kind == "external_benchmark" and not _capabilities_verified(result))
    ):
        return SCORE_NOT_APPLICABLE if _run_status(result) == RUN_SKIPPED else SCORE_UNGRADED
    if result.passed:
        return SCORE_PASSED
    if result.score > 0.0:
        return SCORE_PARTIALLY_CORRECT
    if status in {
        FAILED_MODEL_ANSWER,
        FAILED_MODEL_FORMAT,
        FAILED_MODEL_MISSING_ARTIFACT,
        FAILED_MODEL_TOOL_USE,
    }:
        return status
    return SCORE_FAILED_MODEL_ANSWER


def _blocker_type(result: GradeResult) -> str:
    status = _result_status(result)
    error = _result_error_reason(result).lower()
    payload = _benchmark_payload(result)
    explicit_blocker = _explicit_blocker_type(result, payload)
    if explicit_blocker:
        return explicit_blocker
    if _uses_smoke_score_mode(payload):
        return BLOCKER_MISSING_GRADER
    unsupported = payload.get("unsupported_capabilities")
    unsupported_values = {str(item) for item in unsupported} if isinstance(unsupported, list) else set()
    contract = payload.get("capability_contract")
    if isinstance(contract, dict):
        for capability, data in contract.items():
            if isinstance(data, dict) and data.get("supported") is False:
                unsupported_values.add(str(capability))
                if data.get("grader") is False:
                    return BLOCKER_MISSING_GRADER
    if "official patch/test grader" in error or "grader is not configured" in error:
        return BLOCKER_MISSING_GRADER
    if "scoring is disabled" in error or "grader_side_gold_labels" in unsupported_values:
        return BLOCKER_DISABLED_SCORING
    if status == FAILED_MISSING_ASSETS:
        if "git lfs pointer" in error or "pointer stub" in error:
            return BLOCKER_GIT_LFS_POINTER_STUB
        return BLOCKER_MISSING_ASSET
    if status == FAILED_MISSING_REQUIRED_TOOL:
        return BLOCKER_MISSING_REQUIRED_TOOL
    if status == FAILED_INVALID_TASK_CONTEXT:
        return BLOCKER_INVALID_TASK_CONTEXT
    if status == SKIPPED_UNSUPPORTED_CAPABILITY:
        if "kaggle" in " ".join(sorted(unsupported_values)).lower():
            return BLOCKER_EXTERNAL_PLATFORM_UNAVAILABLE
        return BLOCKER_UNSUPPORTED_CAPABILITY
    if status == FAILED_GRADER:
        return BLOCKER_JUDGE_PARSE_ERROR
    if status in {FAILED_MODEL_FORMAT, FAILED_DATASET_EXTRACTION}:
        return BLOCKER_OUTPUT_PARSE_ERROR
    return ""


def _uses_smoke_score_mode(payload: dict[str, Any]) -> bool:
    modes: set[str] = set()
    mode = payload.get("score_mode")
    if isinstance(mode, str) and mode.strip():
        modes.add(mode.strip())
    score_modes = payload.get("score_modes")
    if isinstance(score_modes, list):
        modes.update(str(item).strip() for item in score_modes if str(item).strip())
    return payload.get("official_equivalent") is False or any(mode.startswith("smoke") for mode in modes)


def _explicit_blocker_type(result: GradeResult, payload: dict[str, Any]) -> str:
    details = result.details if isinstance(result.details, dict) else {}
    containers: list[Any] = [
        payload,
        details.get("setup_details"),
        details.get("details"),
    ]
    setup = payload.get("setup_details")
    if isinstance(setup, dict):
        containers.extend([setup, setup.get("details")])
    for container in containers:
        blocker = _nested_blocker_type(container)
        if blocker:
            return blocker
    reason = _result_error_reason(result).lower()
    if "repo_patch canary" in reason or "patch executable" in reason:
        return BLOCKER_REPO_PATCH_HARNESS_SETUP
    if "git lfs pointer" in reason or "pointer stub" in reason:
        return BLOCKER_GIT_LFS_POINTER_STUB
    if "missing required tool" in reason or "missing tools" in reason:
        return BLOCKER_MISSING_REQUIRED_TOOL
    if "missing reference dataset" in reason:
        return BLOCKER_MISSING_REFERENCE_DATASET
    if "missing reference document" in reason:
        return BLOCKER_MISSING_REFERENCE_DOCUMENTS
    if "missing task instance" in reason or "concrete exploit tasks are missing" in reason:
        return BLOCKER_MISSING_TASK_INSTANCES
    return ""


def _nested_blocker_type(value: Any, *, depth: int = 0) -> str:
    if depth > 8:
        return ""
    if isinstance(value, dict):
        blocker = value.get("blocker_type")
        if isinstance(blocker, str) and blocker.strip():
            return blocker.strip()
        for nested in value.values():
            found = _nested_blocker_type(nested, depth=depth + 1)
            if found:
                return found
    elif isinstance(value, list):
        for nested in value:
            found = _nested_blocker_type(nested, depth=depth + 1)
            if found:
                return found
    return ""


def _result_error_reason(result: GradeResult) -> str:
    if isinstance(result.error, str) and result.error.strip():
        return result.error.strip()
    payload = _benchmark_payload(result)
    for key in ("error", "reason", "skip_reason", "blocker_reason"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    status = _result_status(result)
    if is_invalid_evaluation_status(status):
        return _external_status_error(status, payload)
    return ""


def _external_status_error(status: str, payload: dict[str, Any]) -> str:
    status_counts = payload.get("status_counts")
    if isinstance(status_counts, dict):
        total = sum(count for count in status_counts.values() if isinstance(count, int))
        if total > 0:
            return f"All {total} benchmark record evaluation(s) were invalid: {status.replace('_', ' ')}"
    return status.replace("_", " ")


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
    item_count, valid_item_count = _item_coverage_counts(results)
    unsupported_count = _unsupported_capability_count(results, status_counts)
    missing_assets_count = status_counts.get(FAILED_MISSING_ASSETS, 0)
    dataset_failed_count = status_counts.get(FAILED_DATASET_EXTRACTION, 0)
    return {
        "task_count": task_count,
        "valid_judged_task_count": valid_count,
        "suite_count": task_count,
        "valid_judged_suite_count": valid_count,
        "suite_coverage_rate": _ratio(valid_count, task_count),
        "item_count": item_count,
        "valid_judged_item_count": valid_item_count,
        "item_coverage_rate": _ratio(valid_item_count, item_count),
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


def _unsupported_capability_count(results: list[GradeResult], status_counts: dict[str, int]) -> int:
    explicit_count = 0
    for result in results:
        payload = _benchmark_payload(result)
        unsupported = payload.get("unsupported_capabilities")
        if isinstance(unsupported, list):
            explicit_count += len({str(item) for item in unsupported if str(item)})
    return max(status_counts.get(SKIPPED_UNSUPPORTED_CAPABILITY, 0), explicit_count)


def _official_coverage_summary(
    results: list[GradeResult],
    category_counts: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    total = len(results)
    scored = sum(1 for result in results if _is_valid_result(result))
    failed = total - scored
    return {
        "total_configured_benchmarks": total,
        "attempted_benchmarks": total,
        "successfully_scored_benchmarks": scored,
        "failed_benchmarks": failed,
        "excluded_from_score_benchmarks": failed,
        "coverage_rate": _ratio(scored, total),
        "per_category": {
            category: {
                "total_configured_benchmarks": int(data.get("task_count", 0)),
                "attempted_benchmarks": int(data.get("attempted_benchmark_count", 0)),
                "successfully_scored_benchmarks": int(data.get("successfully_scored_benchmark_count", 0)),
                "failed_benchmarks": int(data.get("failed_benchmark_count", 0)),
                "coverage_rate": data.get("coverage_rate", 0.0),
            }
            for category, data in sorted(category_counts.items())
        },
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


def _item_coverage_counts(results: list[GradeResult]) -> tuple[int, int]:
    total = 0
    valid = 0
    for result in results:
        payload = _benchmark_payload(result)
        status_counts = _normalized_status_counts(payload.get("status_counts"))
        if status_counts:
            total += sum(status_counts.values())
            valid += sum(
                count for status, count in status_counts.items() if status not in INVALID_EVALUATION_STATUSES
            )
            continue
        total += 1
        if _is_valid_result(result):
            valid += 1
    return total, valid


def _suite_blockers(results: list[GradeResult]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        if _is_valid_result(result):
            continue
        rows.append(
            {
                "task_id": result.task_id,
                "category": result.category,
                "status": _result_status(result),
                "run_status": _run_status(result),
                "blocker_type": _blocker_type(result) or "unknown",
                "error": _result_error_reason(result),
            }
        )
    return sorted(rows, key=lambda row: (str(row["category"]), str(row["task_id"])))


def _blocker_counts(results: list[GradeResult]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for result in results:
        blocker = _blocker_type(result)
        if blocker:
            counts[blocker] += 1
    return dict(sorted(counts.items()))


def _parser_repair_count(results: list[GradeResult]) -> int:
    total = 0
    for result in results:
        payload = _benchmark_payload(result)
        found_top_level = False
        for key in ("judge_parse_repaired_count", "parser_repair_count"):
            value = payload.get(key)
            if isinstance(value, int):
                total += value
                found_top_level = True
                break
        if found_top_level:
            continue
        model_evals = payload.get("model_evals")
        if isinstance(model_evals, list):
            total += sum(
                1
                for item in model_evals
                if isinstance(item, dict)
                and (
                    item.get("judge_parse_repaired") is True
                    or (
                        isinstance(item.get("grade"), dict)
                        and item["grade"].get("judge_parse_repaired") is True
                    )
                )
            )
    return total


def _usage_summary(results: list[GradeResult]) -> dict[str, Any]:
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    saw_usage = False
    for result in results:
        for usage in _usage_payloads(result):
            saw_usage = True
            prompt_tokens += _usage_value(usage, ("prompt_tokens", "input_tokens", "prompt_eval_count"))
            completion_tokens += _usage_value(usage, ("completion_tokens", "output_tokens", "eval_count"))
            total_tokens += _usage_value(usage, ("total_tokens",))
    if saw_usage and total_tokens <= 0:
        total_tokens = prompt_tokens + completion_tokens
    suite_denominator = len(results)
    item_denominator = _usage_item_denominator(results)
    finish_reason_length_count = _protocol_diagnostic_count(results, "finish_reason", "length")
    hidden_reasoning_no_final_count = _protocol_diagnostic_count(results, "hidden_reasoning_no_final", True)
    no_final_content_count = _protocol_diagnostic_count(results, "no_final_content", True)
    return {
        "total_prompt_tokens": prompt_tokens if saw_usage else None,
        "total_completion_tokens": completion_tokens if saw_usage else None,
        "total_tokens": total_tokens if saw_usage else None,
        "average_prompt_tokens_per_suite": _average_token_count(prompt_tokens, suite_denominator, saw_usage),
        "average_completion_tokens_per_suite": _average_token_count(completion_tokens, suite_denominator, saw_usage),
        "average_prompt_tokens_per_item": _average_token_count(prompt_tokens, item_denominator, saw_usage),
        "average_completion_tokens_per_item": _average_token_count(completion_tokens, item_denominator, saw_usage),
        "finish_reason_length_count": finish_reason_length_count,
        "hidden_reasoning_no_final_count": hidden_reasoning_no_final_count,
        "no_final_content_count": no_final_content_count,
        "truncated_completion_count": sum(1 for result in results if _is_truncated(result)) + finish_reason_length_count,
        "empty_response_count": sum(1 for result in results if _result_error_reason(result) == "empty response"),
        "judge_retry_count": _nested_count(results, ("judge_retry_count",)),
        "output_extraction_failure_count": sum(
            1
            for result in results
            if isinstance(result.details, dict) and result.details.get("extraction_status") == "failed"
        ),
    }


def _usage_item_denominator(results: list[GradeResult]) -> int:
    total = 0
    for result in results:
        payload = _benchmark_payload(result)
        model_evals = payload.get("model_evals")
        if isinstance(model_evals, list) and model_evals:
            total += sum(1 for item in model_evals if isinstance(item, dict))
        else:
            total += 1
    return total


def _protocol_diagnostic_count(results: list[GradeResult], key: str, expected: Any) -> int:
    total = 0
    for result in results:
        payload = _benchmark_payload(result)
        for item in payload.get("model_evals", []):
            if not isinstance(item, dict):
                continue
            diagnostics = item.get("protocol_diagnostics")
            if isinstance(diagnostics, dict) and diagnostics.get(key) == expected:
                total += 1
    return total


def _usage_payloads(result: GradeResult) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    details = result.details if isinstance(result.details, dict) else {}
    usage = details.get("usage")
    if isinstance(usage, dict):
        payloads.append(usage)
    benchmark_payload = _benchmark_payload(result)
    for item in benchmark_payload.get("model_evals", []):
        if not isinstance(item, dict):
            continue
        usage = item.get("usage")
        if isinstance(usage, dict):
            payloads.append(usage)
        grade = item.get("grade")
        if isinstance(grade, dict) and isinstance(grade.get("usage"), dict):
            payloads.append(grade["usage"])
    return payloads


def _usage_value(usage: dict[str, Any], keys: tuple[str, ...]) -> int:
    for key in keys:
        value = usage.get(key)
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
    return 0


def _average_token_count(total: int, denominator: int, saw_usage: bool) -> float | None:
    if not saw_usage or denominator <= 0:
        return None
    return total / denominator


def _nested_count(results: list[GradeResult], keys: tuple[str, ...]) -> int:
    total = 0
    for result in results:
        payload = _benchmark_payload(result)
        for key in keys:
            value = payload.get(key)
            if isinstance(value, int):
                total += value
                break
    return total


def _is_truncated(result: GradeResult) -> bool:
    details = result.details if isinstance(result.details, dict) else {}
    for container in (details, _benchmark_payload(result)):
        status = container.get("completion_status")
        if status == "truncated" or container.get("truncated") is True:
            return True
    return False


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
        error = _result_error_reason(result)
        required_capabilities = benchmark_payload.get("required_capabilities", details.get("required_capabilities", []))
        manifest = details.get("manifest") if isinstance(details.get("manifest"), dict) else {}
        setup_details = benchmark_payload.get("setup_details", details.get("setup_details", {}))
        external_setup = setup_details.get("external_harness") if isinstance(setup_details, dict) else {}
        if not isinstance(external_setup, dict):
            external_setup = {}
        rows.append(
            {
                "suite_id": result.task_id,
                "task_id": result.task_id,
                "group": group,
                "benchmark": benchmark,
                "included_in_official_score": _is_valid_result(result),
                "required_capabilities": required_capabilities,
                "profile": _profile_for_capabilities(required_capabilities),
                "score_fraction": round(result.score, 6),
                "score": round(result.score * 100.0, 4),
                "raw_score": _unit_to_percent(benchmark_payload.get("raw_score")),
                "valid_score": _unit_to_percent(benchmark_payload.get("valid_score")),
                "passed": result.passed,
                "status": _result_status(result),
                "run_status": _run_status(result),
                "score_status": _score_status(result),
                "official_equivalent": benchmark_payload.get("official_equivalent", details.get("official_equivalent")),
                "score_mode": benchmark_payload.get("score_mode", details.get("score_mode", "")),
                "score_modes": benchmark_payload.get("score_modes", details.get("score_modes", [])),
                "duration_seconds": result.task_duration_seconds,
                "blocker_type": _blocker_type(result),
                "error": error,
                "error_details": error,
                "homepage": details.get("homepage", ""),
                "official_leaderboard_url": details.get("official_leaderboard_url", ""),
                "license": details.get("license", ""),
                "credit": details.get("credit", ""),
                "citation": details.get("citation", details.get("homepage", "")),
                "docker_image": benchmark_payload.get("docker_image", details.get("docker_image", external_setup.get("image", ""))),
                "container_name": benchmark_payload.get("container_name", details.get("container_name", external_setup.get("container_name", ""))),
                "network_mode": benchmark_payload.get("network_mode", details.get("network_mode", external_setup.get("network_mode", ""))),
                "docker_socket_mount": benchmark_payload.get("docker_socket_mount", details.get("docker_socket_mount", external_setup.get("docker_socket_mount", {}))),
                "output_mount": benchmark_payload.get("output_mount", details.get("output_mount", external_setup.get("output_mount", {}))),
                "asset_cache_mount": benchmark_payload.get("asset_cache_mount", details.get("asset_cache_mount", external_setup.get("asset_cache_mount", {}))),
                "catalog_checkout_path": benchmark_payload.get(
                    "catalog_checkout_path",
                    details.get("catalog_checkout_path", external_setup.get("catalog_checkout_path", "")),
                ),
                "target_checkout_path": benchmark_payload.get(
                    "target_checkout_path",
                    details.get("target_checkout_path", external_setup.get("target_checkout_path", "")),
                ),
                "benchmark_checkout_path": benchmark_payload.get("benchmark_checkout_path", details.get("benchmark_checkout_path", external_setup.get("benchmark_checkout_path", ""))),
                "output_dir": details.get("output_dir", ""),
                "manifest": manifest,
                "asset_refs": manifest.get("assets", []) if isinstance(manifest, dict) else [],
                "official_conditions": manifest.get("official_conditions", {}) if isinstance(manifest, dict) else {},
                "source": manifest.get("source", {}) if isinstance(manifest, dict) else {},
                "model_eval": model_eval,
                "model_evals": model_evals,
                "grading_methods": grading_methods,
                "adapter": benchmark_payload.get("adapter", details.get("adapter", "")),
                "capabilities_verified": _capabilities_verified(result),
                "capability_contract": benchmark_payload.get("capability_contract", {}),
                "supported_capabilities": benchmark_payload.get("supported_capabilities", []),
                "required_tools": benchmark_payload.get("required_tools", []),
                "exposed_tools": benchmark_payload.get("exposed_tools", []),
                "missing_tools": benchmark_payload.get("missing_tools", []),
                "missing_env": benchmark_payload.get(
                    "missing_env",
                    benchmark_payload.get("missing_environment", []),
                ),
                "repository_ready": bool(benchmark_payload.get("repository_ready", False)),
                "file_count_sampled": benchmark_payload.get("file_count_sampled"),
                "extracted_task_count": benchmark_payload.get("extracted_task_count"),
                "evaluated_task_count": benchmark_payload.get("evaluated_task_count"),
                "valid_evaluated_task_count": benchmark_payload.get("valid_evaluated_task_count"),
                "evaluation_passed_count": benchmark_payload.get("evaluation_passed_count"),
                "skipped_task_count": benchmark_payload.get("skipped_task_count"),
                "setup_failed_count": _payload_status_count(benchmark_payload, "failed_harness_setup"),
                "missing_assets_count": benchmark_payload.get(
                    "missing_assets_count",
                    _payload_status_count(benchmark_payload, FAILED_MISSING_ASSETS),
                ),
                "setup_details": setup_details,
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
                "required_capabilities": required_capabilities,
                "unsupported_capabilities": benchmark_payload.get("unsupported_capabilities", []),
                "extraction_sources": benchmark_payload.get("extraction_sources", []),
            }
        )
    return sorted(rows, key=lambda row: (str(row["group"]), str(row["benchmark"])))


def _profile_results(benchmark_results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in benchmark_results:
        grouped[str(row.get("profile") or "core_text")].append(row)
    profiles: dict[str, dict[str, Any]] = {}
    for profile, rows in sorted(grouped.items()):
        valid_rows = [row for row in rows if row.get("score_status") not in {SCORE_NOT_APPLICABLE, SCORE_UNGRADED}]
        score = sum(float(row.get("score_fraction", 0.0)) for row in valid_rows)
        blockers: dict[str, int] = defaultdict(int)
        for row in rows:
            blocker = row.get("blocker_type")
            if isinstance(blocker, str) and blocker:
                blockers[blocker] += 1
        profiles[profile] = {
            "suite_count": len(rows),
            "valid_judged_suite_count": len(valid_rows),
            "suite_coverage_rate": _ratio(len(valid_rows), len(rows)),
            "valid_judged_score": _ratio(score, len(valid_rows)),
            "blocker_counts": dict(sorted(blockers.items())),
        }
    return profiles


def _profile_for_capabilities(value: Any) -> str:
    capabilities = {str(item) for item in value} if isinstance(value, list) else set()
    if "browser_or_gui" in capabilities:
        return "browser_gui"
    if "kaggle_competition_submission" in capabilities:
        return "external_submission"
    if "repo_patch" in capabilities:
        return "repo_patch"
    if "tool_call" in capabilities:
        return "tool_use"
    if capabilities & {"file_artifact", "office_document_editing"}:
        return "artifact_assets"
    return "core_text"


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
