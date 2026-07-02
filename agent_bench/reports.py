import csv
import html
import json
import math
import shutil
from pathlib import Path
from typing import Any

from agent_bench.models import GradeResult, ModelResponse
from agent_bench.statuses import (
    FAILED_DATASET_EXTRACTION,
    FAILED_GRADER,
    FAILED_HARNESS_SETUP,
    FAILED_INVALID_TASK_CONTEXT,
    FAILED_MISSING_ASSETS,
    FAILED_MISSING_REQUIRED_TOOL,
    FAILED_MODEL_ANSWER,
    FAILED_MODEL_FORMAT,
    FAILED_MODEL_TOOL_USE,
    FAILED_TOKEN_BUDGET,
    INVALID_EVALUATION_STATUSES,
    PASSED,
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
    TIMED_OUT,
    is_invalid_evaluation_status,
    normalize_status,
)


RESULT_FIELDS = [
    "task_id",
    "suite_id",
    "suite_name",
    "category",
    "kind",
    "score",
    "max_score",
    "passed",
    "json_valid",
    "latency_seconds",
    "time_to_first_token_seconds",
    "tokens_per_second",
    "output_token_count",
    "task_duration_seconds",
    "confidence",
    "answer",
    "error",
    "error_details",
    "status",
    "run_status",
    "score_status",
    "included_in_official_score",
    "blocker_type",
    "raw_score",
    "valid_score",
    "raw_model_response",
    "extracted_answer",
    "extraction_status",
    "judge_parser_status",
    "judge_parse_repaired_count",
]


def write_jsonl_line(handle: Any, payload: dict[str, Any]) -> None:
    handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    handle.flush()


def write_result_artifacts(
    output_dir: Path,
    responses: list[ModelResponse],
    results: list[GradeResult],
    summary: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_results_csv(output_dir / "results.csv", results)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.html").write_text(render_summary_html(summary, results), encoding="utf-8")


def update_latest(timestamp_dir: Path, latest_dir: Path) -> None:
    if latest_dir.exists() or latest_dir.is_symlink():
        if latest_dir.is_dir() and not latest_dir.is_symlink():
            shutil.rmtree(latest_dir)
        else:
            latest_dir.unlink()
    shutil.copytree(timestamp_dir, latest_dir)


def render_summary_html(summary: dict[str, Any], results: list[GradeResult]) -> str:
    metadata = summary.get("metadata", {})
    category_counts = summary.get("category_counts", {})
    benchmark_results = summary.get("benchmark_results", [])
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Agent Bench Summary</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #172033;
      --muted: #667085;
      --line: #d7dde8;
      --panel: #f7f9fc;
      --blue: #276ef1;
      --green: #12805c;
      --gold: #b7791f;
      --red: #b42318;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font: 14px/1.45 system-ui, -apple-system, Segoe UI, sans-serif; color: var(--ink); background: #ffffff; }}
    header {{ padding: 26px 32px 18px; border-bottom: 1px solid var(--line); }}
    h1 {{ margin: 0 0 4px; font-size: 28px; letter-spacing: 0; }}
    h2 {{ margin: 26px 0 12px; font-size: 18px; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 22px 28px 44px; }}
    .muted {{ color: var(--muted); }}
    .note {{ color: var(--muted); margin: -4px 0 12px; max-width: 920px; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin: 16px 0 24px; }}
    .card {{ border: 1px solid var(--line); border-radius: 8px; padding: 14px; background: var(--panel); }}
    .card span {{ display: block; color: var(--muted); font-size: 12px; }}
    .card strong {{ display: block; margin-top: 6px; font-size: 23px; }}
    .grid {{ display: grid; grid-template-columns: 390px minmax(0, 1fr); gap: 24px; align-items: start; }}
    table {{ width: 100%; border-collapse: collapse; border: 1px solid var(--line); }}
    th, td {{ padding: 8px 10px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }}
    th {{ background: var(--panel); white-space: nowrap; }}
    td {{ overflow-wrap: anywhere; }}
    .group-row td {{ background: #fff4f5; color: #c9334c; font-weight: 700; }}
    .score-cell {{ font-weight: 700; text-align: right; }}
    .status-pass {{ color: var(--green); font-weight: 700; }}
    .status-fail {{ color: var(--red); font-weight: 700; }}
    .radar {{ border: 1px solid var(--line); border-radius: 8px; background: var(--panel); padding: 12px; }}
    .metadata {{ display: grid; grid-template-columns: 190px 1fr; gap: 6px 12px; }}
    @media (max-width: 820px) {{ main {{ padding: 18px 14px 34px; }} .grid {{ grid-template-columns: 1fr; }} .metadata {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <header>
    <h1>Agent Bench Summary</h1>
    <div class="muted">{html.escape(str(metadata.get("model", "unknown model")))} · {html.escape(str(metadata.get("created_at_utc", "")))}</div>
  </header>
  <main>
    {_metric_cards(summary)}
    <section class="grid">
      <div>
        <h2>Average Score Radar</h2>
        <div class="radar">{_radar_svg(_average_scores(summary))}</div>
      </div>
      <div>
        <h2>Category Metrics</h2>
        {_category_table(category_counts)}
      </div>
    </section>
    <section>
      <h2>Benchmark Scores</h2>
      <p class="note">Scores are per benchmark row from the local graded benchmark sample executed by each descriptor. Repository readiness and endpoint checks are recorded in each task result.</p>
      {_benchmark_table(benchmark_results, str(metadata.get("model", "Model")))}
    </section>
    <section>
      <h2>Profile Coverage</h2>
      {_profile_table(summary.get("profile_results"))}
    </section>
    <section>
      <h2>Skipped Suites</h2>
      {_skipped_suite_table(summary.get("skipped_suites"))}
    </section>
    <section>
      <h2>Status Distribution</h2>
      {_status_distribution_table(summary)}
    </section>
    <section>
      <h2>Timing</h2>
      <p class="note">{html.escape(str(summary.get("timing_note", "")))}</p>
      {_timing_table(summary)}
    </section>
    <section>
      <h2>Run Metadata</h2>
      {_metadata_table(metadata)}
    </section>
    <section>
      <h2>Task Results</h2>
      {_results_table(results)}
    </section>
    <section>
      <h2>Benchmark Citations</h2>
      {_citation_table(benchmark_results)}
    </section>
  </main>
</body>
</html>
"""


def _write_results_csv(path: Path, results: list[GradeResult]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for result in results:
            row = _result_csv_row(result)
            writer.writerow({field: row.get(field) for field in RESULT_FIELDS})


def _result_csv_row(result: GradeResult) -> dict[str, Any]:
    row = result.to_dict()
    details = result.details if isinstance(result.details, dict) else {}
    payload = details.get("result") if isinstance(details.get("result"), dict) else {}
    row["status"] = _result_status(result)
    row["suite_id"] = details.get("suite_id", result.task_id)
    row["suite_name"] = _result_benchmark_name(result)
    row["run_status"] = _run_status(result)
    row["score_status"] = _score_status(result)
    row["included_in_official_score"] = _result_status(result) not in INVALID_EVALUATION_STATUSES
    row["blocker_type"] = _blocker_type(result)
    row["error_details"] = _result_error_reason(result)
    row["raw_score"] = _unit_to_percent(payload.get("raw_score"))
    row["valid_score"] = _unit_to_percent(payload.get("valid_score"))
    row["raw_model_response"] = details.get("raw_model_response")
    row["extracted_answer"] = _json_cell(details.get("extracted_answer"))
    row["extraction_status"] = details.get("extraction_status")
    row["judge_parser_status"] = payload.get("judge_parser_status")
    row["judge_parse_repaired_count"] = payload.get("judge_parse_repaired_count")
    return row


def _metric_cards(summary: dict[str, Any]) -> str:
    cards = [
        ("Valid Judged Score", _format_rate(summary.get("valid_judged_score"))),
        ("Suite Coverage", _format_rate(summary.get("suite_coverage_rate"))),
        ("Item Coverage", _format_rate(summary.get("item_coverage_rate"))),
        ("Conservative Score", _format_rate(summary.get("conservative_all_suite_score"))),
        ("Model Score", _format_percent(summary.get("model_score_valid_tasks_only"))),
        ("Raw Score", _format_percent(summary.get("raw_score_all_tasks", summary.get("raw_score")))),
        ("Valid Tasks", _format_integer(summary.get("valid_task_count"))),
        ("Model Valid", _format_integer(summary.get("model_valid_task_count"))),
        ("Skipped Suites", _format_integer(summary.get("skipped_suite_count", summary.get("skipped_count")))),
        ("Setup Failed", _format_integer(summary.get("setup_failed_count"))),
        ("Missing Assets", _format_integer(summary.get("missing_asset_count", summary.get("missing_assets_count")))),
        ("Missing Graders", _format_integer(summary.get("missing_grader_count"))),
        ("Judge Errors", _format_integer(summary.get("judge_error_count", summary.get("grader_failure_count")))),
        ("Parser Repairs", _format_integer(summary.get("parser_repair_count"))),
        ("Pass Rate", _format_percent(summary.get("pass_rate"))),
        ("Coding Pass", _format_percent(summary.get("coding_pass_rate"))),
        ("JSON Validity", _format_percent(summary.get("json_validity_rate"))),
        ("Timeout Rate", _format_percent(summary.get("timeout_rate"))),
        ("Avg Latency", f"{float(summary.get('average_latency_seconds', 0.0)):.2f}s"),
        ("Run Time", _format_seconds(summary.get("total_run_duration_seconds"))),
        ("Task Time Sum", _format_seconds(summary.get("total_task_duration_seconds"))),
    ]
    return '<section class="cards">' + "\n".join(
        f'<section class="card"><span>{html.escape(label)}</span><strong>{html.escape(value)}</strong></section>'
        for label, value in cards
    ) + "</section>"


def _status_distribution_table(summary: dict[str, Any]) -> str:
    rows: list[str] = []
    for label, key in (
        ("Task rows", "status_counts"),
        ("Benchmark items", "benchmark_item_status_counts"),
    ):
        counts = summary.get(key)
        if not isinstance(counts, dict) or not counts:
            continue
        for status, count in sorted(counts.items()):
            rows.append(
                "<tr>"
                f"<td>{html.escape(label)}</td>"
                f"<td>{html.escape(str(status))}</td>"
                f"<td>{int(count)}</td>"
                "</tr>"
            )
    if not rows:
        rows.append('<tr><td colspan="3">No statuses were recorded.</td></tr>')
    return (
        "<table><thead><tr><th>Scope</th><th>Status</th><th>Count</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _profile_table(value: Any) -> str:
    if not isinstance(value, dict) or not value:
        return "<table><tbody><tr><td>No benchmark profiles were recorded.</td></tr></tbody></table>"
    rows = []
    for profile, data in sorted(value.items()):
        if not isinstance(data, dict):
            continue
        blockers = data.get("blocker_counts")
        blocker_text = ", ".join(f"{key}: {count}" for key, count in sorted(blockers.items())) if isinstance(blockers, dict) else ""
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(profile))}</td>"
            f"<td>{int(data.get('valid_judged_suite_count', 0))}/{int(data.get('suite_count', 0))}</td>"
            f"<td>{_format_rate(data.get('suite_coverage_rate'))}</td>"
            f"<td>{_format_rate(data.get('valid_judged_score'))}</td>"
            f"<td>{html.escape(blocker_text)}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Profile</th><th>Runnable Suites</th><th>Coverage</th>"
        f"<th>Valid Score</th><th>Blockers</th></tr></thead><tbody>{''.join(rows)}</tbody></table>"
    )


def _skipped_suite_table(value: Any) -> str:
    if not isinstance(value, list) or not value:
        return "<table><tbody><tr><td>No skipped suites were recorded.</td></tr></tbody></table>"
    rows = []
    for row in value:
        if not isinstance(row, dict):
            continue
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('task_id', '')))}</td>"
            f"<td>{html.escape(str(row.get('category', '')))}</td>"
            f"<td>{html.escape(str(row.get('run_status', '')))}</td>"
            f"<td>{html.escape(str(row.get('status', '')))}</td>"
            f"<td>{html.escape(str(row.get('blocker_type', '')))}</td>"
            f"<td>{html.escape(_display_error(str(row.get('error', ''))))}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Suite</th><th>Category</th><th>Run</th><th>Status</th>"
        f"<th>Blocker</th><th>Reason</th></tr></thead><tbody>{''.join(rows)}</tbody></table>"
    )


def _timing_table(summary: dict[str, Any]) -> str:
    timing = summary.get("timing_by_category")
    if not isinstance(timing, dict) or not timing:
        return "<table><tbody><tr><td>No timing breakdown was recorded.</td></tr></tbody></table>"
    rows = []
    for category, data in sorted(timing.items()):
        if not isinstance(data, dict):
            continue
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(category))}</td>"
            f"<td>{int(data.get('task_count', 0))}</td>"
            f"<td>{_format_seconds(data.get('total_task_duration_seconds'))}</td>"
            f"<td>{_format_seconds(data.get('average_task_duration_seconds'))}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Category</th><th>Tasks</th><th>Total Task Time</th>"
        f"<th>Avg Task Time</th></tr></thead><tbody>{''.join(rows)}</tbody></table>"
    )


def _average_scores(summary: dict[str, Any]) -> dict[str, float]:
    scores = summary.get("category_scores")
    if isinstance(scores, dict) and scores:
        return {str(label): float(value) for label, value in scores.items()}
    benchmark_results = summary.get("benchmark_results")
    if not isinstance(benchmark_results, list):
        return {}
    grouped: dict[str, list[float]] = {}
    for row in benchmark_results:
        if not isinstance(row, dict):
            continue
        group = str(row.get("group", "Benchmarks"))
        try:
            score = float(row.get("score", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        grouped.setdefault(group, []).append(score)
    return {
        group: sum(values) / len(values)
        for group, values in grouped.items()
        if values
    }


def _radar_svg(scores: dict[str, float]) -> str:
    if not scores:
        return '<svg viewBox="0 0 320 260" role="img" aria-label="No category scores"></svg>'
    labels = list(scores.keys())
    values = [max(0.0, min(100.0, float(scores[label]))) for label in labels]
    cx, cy, radius = 160.0, 130.0, 86.0
    axes = len(labels)

    grid_polygons = []
    for fraction in (0.25, 0.5, 0.75, 1.0):
        points = [_point(cx, cy, radius * fraction, index, axes) for index in range(axes)]
        grid_polygons.append(
            f'<polygon points="{_points(points)}" fill="none" stroke="#d7dde8" stroke-width="1" />'
        )

    axis_lines = []
    label_nodes = []
    for index, label in enumerate(labels):
        end = _point(cx, cy, radius, index, axes)
        label_point = _point(cx, cy, radius + 28.0, index, axes)
        axis_lines.append(
            f'<line x1="{cx:.1f}" y1="{cy:.1f}" x2="{end[0]:.1f}" y2="{end[1]:.1f}" stroke="#b9c2d0" />'
        )
        label_nodes.append(
            f'<text x="{label_point[0]:.1f}" y="{label_point[1]:.1f}" text-anchor="middle" dominant-baseline="middle" font-size="10" fill="#172033">{html.escape(label)}</text>'
        )

    value_points = [_point(cx, cy, radius * (value / 100.0), index, axes) for index, value in enumerate(values)]
    return (
        '<svg viewBox="0 0 320 260" role="img" aria-label="Radar chart of category scores">'
        + "".join(grid_polygons)
        + "".join(axis_lines)
        + f'<polygon points="{_points(value_points)}" fill="#276ef1" fill-opacity="0.22" stroke="#276ef1" stroke-width="2" />'
        + "".join(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="#12805c" />' for x, y in value_points
        )
        + "".join(label_nodes)
        + "</svg>"
    )


def _point(cx: float, cy: float, radius: float, index: int, total: int) -> tuple[float, float]:
    angle = -math.pi / 2 + (2 * math.pi * index / total)
    return cx + radius * math.cos(angle), cy + radius * math.sin(angle)


def _points(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in points)


def _category_table(category_counts: dict[str, Any]) -> str:
    rows = []
    for category, data in category_counts.items():
        rows.append(
            "<tr>"
            f"<td>{html.escape(category)}</td>"
            f"<td>{int(data.get('task_count', 0))}</td>"
            f"<td>{int(data.get('passed_count', 0))}</td>"
            f"<td>{_format_percent(data.get('score'))}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Category</th><th>Tasks</th><th>Passed</th><th>Score</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _benchmark_table(benchmark_results: Any, model: str) -> str:
    if not isinstance(benchmark_results, list) or not benchmark_results:
        return "<table><tbody><tr><td>No benchmark rows were recorded.</td></tr></tbody></table>"
    rows: list[str] = []
    current_group: str | None = None
    for row in benchmark_results:
        if not isinstance(row, dict):
            continue
        group = str(row.get("group", "Benchmarks"))
        if group != current_group:
            rows.append(
                f'<tr class="group-row"><td colspan="8">{html.escape(group)}</td></tr>'
            )
            current_group = group
        methods = _grading_methods(row)
        evaluated = row.get("evaluated_task_count")
        passed_count = row.get("evaluation_passed_count")
        items = f"{passed_count}/{evaluated}" if evaluated is not None and passed_count is not None else "n/a"
        blocker = str(row.get("blocker_type") or "")
        reason = _display_error(str(row.get("error_details") or row.get("error") or ""))
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('benchmark', '')))}</td>"
            f"<td>{html.escape(str(row.get('profile', '')))}</td>"
            f'<td class="score-cell">{_format_percent(row.get("score"))}</td>'
            f"<td>{html.escape(items)}</td>"
            f"<td>{html.escape(methods)}</td>"
            f"<td>{html.escape(str(row.get('run_status', '')))}</td>"
            f"<td>{html.escape(str(row.get('score_status', '')))}</td>"
            f"<td>{html.escape(blocker or reason)}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Benchmark</th>"
        f"<th>Profile</th><th>{html.escape(model)}</th><th>Items</th><th>Method</th>"
        "<th>Run</th><th>Score Status</th><th>Blocker/Reason</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _grading_methods(row: dict[str, Any]) -> str:
    direct = row.get("grading_methods")
    if isinstance(direct, list):
        methods = [str(item) for item in direct if item]
    else:
        methods = []
    model_eval = row.get("model_eval") if isinstance(row.get("model_eval"), dict) else {}
    methods.extend(str(item) for item in model_eval.get("grading_methods", []) if item)
    model_evals = row.get("model_evals") if isinstance(row.get("model_evals"), list) else []
    for item in model_evals:
        if not isinstance(item, dict):
            continue
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        grade = item.get("grade") if isinstance(item.get("grade"), dict) else {}
        for value in (metadata.get("grading"), grade.get("method")):
            if isinstance(value, str) and value:
                methods.append(value)
    return ", ".join(sorted(set(methods)))


def _citation_table(benchmark_results: Any) -> str:
    if not isinstance(benchmark_results, list) or not benchmark_results:
        return "<table><tbody><tr><td>No benchmark citations were recorded.</td></tr></tbody></table>"
    rows = []
    for row in benchmark_results:
        if not isinstance(row, dict):
            continue
        source = str(row.get("citation") or row.get("homepage") or "")
        source_cell = html.escape(source)
        if source.startswith(("http://", "https://")):
            href = html.escape(source, quote=True)
            source_cell = f'<a href="{href}">{html.escape(source)}</a>'
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('benchmark', '')))}</td>"
            f"<td>{source_cell}</td>"
            f"<td>{html.escape(str(row.get('credit', '')))}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Benchmark</th><th>Source</th><th>Credit</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _metadata_table(metadata: dict[str, Any]) -> str:
    hidden_keys = {
        "api_key_env",
        "external_launcher_image",
        "sandbox",
        "sandbox_image",
        "tasks",
        "temperature",
        "timeout",
    }
    rows = []
    for key, value in sorted(metadata.items()):
        if key in hidden_keys:
            continue
        rows.append(f"<dt>{html.escape(str(key))}</dt><dd>{html.escape(str(value))}</dd>")
    return f'<dl class="metadata">{"".join(rows)}</dl>'


def _results_table(results: list[GradeResult]) -> str:
    rows = []
    for result in results:
        rows.append(
            "<tr>"
            f"<td>{html.escape(_result_benchmark_name(result))}</td>"
            f"<td>{html.escape(result.category)}</td>"
            f"<td>{result.score:.3f}</td>"
            f"<td>{'yes' if result.json_valid else 'no'}</td>"
            f"<td>{_format_seconds(result.latency_seconds)}</td>"
            f"<td>{_format_seconds(result.task_duration_seconds)}</td>"
            f"<td>{html.escape(_display_error(result.error))}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Benchmark</th><th>Category</th>"
        "<th>Score</th><th>JSON</th><th>Request Time</th>"
        "<th>Total Task Time</th><th>Error</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _result_benchmark_name(result: GradeResult) -> str:
    details = result.details if isinstance(result.details, dict) else {}
    benchmark = details.get("benchmark")
    if isinstance(benchmark, str) and benchmark.strip():
        return benchmark
    return result.task_id


def _result_status(result: GradeResult) -> str:
    status = normalize_status(result.status)
    if status:
        return status
    if result.timed_out:
        return TIMED_OUT
    if result.passed:
        return PASSED
    return FAILED_MODEL_ANSWER


def _run_status(result: GradeResult) -> str:
    status = _result_status(result)
    if status in {PASSED, FAILED_MODEL_ANSWER, FAILED_MODEL_FORMAT, FAILED_MODEL_TOOL_USE}:
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
    if status in INVALID_EVALUATION_STATUSES:
        return SCORE_NOT_APPLICABLE if _run_status(result) == RUN_SKIPPED else SCORE_UNGRADED
    if result.passed:
        return SCORE_PASSED
    if result.score > 0.0:
        return SCORE_PARTIALLY_CORRECT
    return SCORE_FAILED_MODEL_ANSWER


def _blocker_type(result: GradeResult) -> str:
    status = _result_status(result)
    error = _result_error_reason(result).lower()
    details = result.details if isinstance(result.details, dict) else {}
    payload = details.get("result") if isinstance(details.get("result"), dict) else {}
    explicit = _explicit_blocker_type(details, payload, error)
    if explicit:
        return explicit
    unsupported = payload.get("unsupported_capabilities")
    unsupported_values = {str(item) for item in unsupported} if isinstance(unsupported, list) else set()
    contract = payload.get("capability_contract")
    if isinstance(contract, dict):
        for capability, data in contract.items():
            if isinstance(data, dict) and data.get("supported") is False:
                unsupported_values.add(str(capability))
                if data.get("grader") is False:
                    return "missing_grader"
    if "official patch/test grader" in error or "grader is not configured" in error:
        return "missing_grader"
    if "scoring is disabled" in error or "grader_side_gold_labels" in unsupported_values:
        return "disabled_scoring"
    if status == FAILED_MISSING_ASSETS:
        if "git lfs pointer" in error or "pointer stub" in error:
            return "git_lfs_pointer_stub"
        return "missing_asset"
    if status == FAILED_MISSING_REQUIRED_TOOL:
        return "missing_required_tool"
    if status == FAILED_INVALID_TASK_CONTEXT:
        return "invalid_task_context"
    if status == SKIPPED_UNSUPPORTED_CAPABILITY:
        if "kaggle" in " ".join(sorted(unsupported_values)).lower():
            return "external_platform_unavailable"
        return "unsupported_capability"
    if status == FAILED_GRADER:
        return "judge_parse_error"
    if status in {FAILED_MODEL_FORMAT, FAILED_DATASET_EXTRACTION}:
        return "output_parse_error"
    return ""


def _explicit_blocker_type(details: dict[str, Any], payload: dict[str, Any], error: str) -> str:
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
    if "repo_patch canary" in error or "patch executable" in error:
        return "repo_patch_harness_setup"
    if "git lfs pointer" in error or "pointer stub" in error:
        return "git_lfs_pointer_stub"
    if "missing required tool" in error or "missing tools" in error:
        return "missing_required_tool"
    if "missing reference dataset" in error:
        return "missing_reference_dataset"
    if "missing reference document" in error:
        return "missing_reference_documents"
    if "missing task instance" in error or "concrete exploit tasks are missing" in error:
        return "missing_task_instances"
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
    details = result.details if isinstance(result.details, dict) else {}
    payload = details.get("result") if isinstance(details.get("result"), dict) else {}
    for key in ("error", "reason", "skip_reason", "blocker_reason"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    status = _result_status(result)
    if status in INVALID_EVALUATION_STATUSES:
        return status.replace("_", " ")
    return ""


def _unit_to_percent(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return round(float(value) * 100.0, 4)
    return None


def _json_cell(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _format_percent(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.2f}%"


def _format_rate(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100.0:.2f}%"


def _display_error(value: str | None) -> str:
    if not value:
        return ""
    return value.replace("No answer-keyed benchmark task records", "No benchmark task records")


def _format_seconds(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}s"


def _format_integer(value: Any) -> str:
    if value is None:
        return "n/a"
    return str(int(value))
