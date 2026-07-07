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
    FAILED_MODEL_MISSING_ARTIFACT,
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
    status_info,
)


RESULT_FIELDS = [
    "task_id",
    "suite_id",
    "suite_name",
    "category",
    "task_group",
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
    "official_equivalent",
    "score_mode",
    "score_modes",
    "included_in_official_score",
    "coverage_status",
    "required_capabilities",
    "supported_capabilities",
    "capabilities_verified",
    "required_tools",
    "exposed_tools",
    "missing_tools",
    "missing_env",
    "missing_assets_count",
    "setup_details",
    "docker_image",
    "container_name",
    "network_mode",
    "docker_socket_mount",
    "output_mount",
    "asset_cache_mount",
    "catalog_checkout_path",
    "target_checkout_path",
    "benchmark_checkout_path",
    "homepage",
    "license",
    "credit",
    "citation",
    "blocker_type",
    "raw_score",
    "valid_score",
    "raw_model_response",
    "extracted_answer",
    "extraction_status",
    "judge_parser_status",
    "judge_retry_count",
    "judge_parse_repaired_count",
]

BENCHMARK_CITATIONS_PATH = Path(__file__).resolve().parents[1] / "tasks" / "benchmark_citations.bib"


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
    metadata = summary.get("metadata") if isinstance(summary.get("metadata"), dict) else {}
    benchmark_results = _report_rows(summary, results)
    coverage = summary.get("coverage_summary", summary.get("coverage", {}))
    target = _target_metadata(metadata)
    judge = _judge_metadata(metadata)
    run_id = metadata.get("run_id") or metadata.get("output_dir") or "agent-bench-run"
    created_at = metadata.get("created_at_utc") or metadata.get("created_at") or ""
    model_name = target.get("model") or metadata.get("model") or "unknown model"
    provider = target.get("provider_type") or target.get("provider") or metadata.get("provider") or "provider n/a"
    score_label, score_display, score_fraction, score_hint = _headline_score(summary)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Agent Bench Final Report</title>
  <style>
    :root {{
      color-scheme: dark;
      --paper: #080d13;
      --surface: #101821;
      --surface-raised: #14202c;
      --ink: #ecf4f1;
      --muted: #9fb0ad;
      --soft: #182530;
      --line: #263642;
      --line-strong: #3b5362;
      --teal: #41d6c3;
      --cyan: #67d5ff;
      --green: #66e3a1;
      --amber: #f4bd61;
      --rose: #ff7f9f;
      --blue: #91b7ff;
      --shadow: 0 18px 45px rgba(0, 0, 0, 0.34);
    }}
    * {{ box-sizing: border-box; }}
    html {{ background: var(--paper); }}
    body {{ margin: 0; font: 14px/1.5 ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: var(--ink); background: var(--paper); }}
    a {{ color: var(--blue); text-decoration-thickness: 1px; text-underline-offset: 2px; }}
    h1, h2, h3, p {{ margin-top: 0; }}
    h1 {{ margin-bottom: 10px; font-size: clamp(30px, 5vw, 52px); line-height: 1.02; letter-spacing: 0; }}
    h2 {{ margin-bottom: 6px; font-size: 22px; line-height: 1.2; letter-spacing: 0; }}
    h3 {{ margin-bottom: 10px; font-size: 15px; letter-spacing: 0; }}
    code, pre {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace; }}
    .shell {{ width: min(1280px, calc(100% - 32px)); margin: 0 auto; }}
    .report-header {{ border-bottom: 1px solid var(--line); background: radial-gradient(circle at top left, rgba(65, 214, 195, 0.14), transparent 34%), #0a1118; }}
    .header-layout {{ display: grid; grid-template-columns: minmax(0, 1fr) minmax(260px, 360px); gap: 28px; align-items: end; padding: 34px 0 28px; }}
    .eyebrow {{ margin-bottom: 10px; color: var(--teal); font-size: 12px; font-weight: 800; letter-spacing: .08em; text-transform: uppercase; }}
    .dek {{ max-width: 760px; margin-bottom: 18px; color: var(--muted); font-size: 16px; }}
    .muted {{ color: var(--muted); }}
    .run-strip {{ display: flex; flex-wrap: wrap; gap: 8px; }}
    .run-strip span {{ display: inline-flex; min-height: 30px; align-items: center; border: 1px solid var(--line); border-radius: 999px; padding: 4px 10px; background: rgba(16, 24, 33, 0.78); color: #d9e7e3; }}
    .score-hero {{ border: 1px solid var(--line-strong); border-radius: 8px; padding: 18px; background: linear-gradient(180deg, #14202c, #101821); box-shadow: var(--shadow); }}
    .score-hero span {{ display: block; color: var(--muted); font-size: 12px; font-weight: 750; text-transform: uppercase; }}
    .score-hero strong {{ display: block; margin: 6px 0 10px; font-size: 44px; line-height: 1; letter-spacing: 0; }}
    .score-hero small {{ display: block; margin-top: 10px; color: var(--muted); }}
    .report-main {{ padding: 30px 0 52px; }}
    .section-block {{ margin-top: 30px; }}
    .section-block:first-child {{ margin-top: 0; }}
    .section-heading {{ display: flex; justify-content: space-between; gap: 16px; align-items: end; margin-bottom: 14px; }}
    .section-heading p {{ margin-bottom: 0; color: var(--muted); max-width: 820px; }}
    .summary-grid {{ display: grid; grid-template-columns: minmax(280px, 380px) minmax(0, 1fr); gap: 18px; align-items: stretch; }}
    .radar-panel, .metric-card, .coverage-tile {{ border: 1px solid var(--line); border-radius: 8px; background: var(--surface); box-shadow: var(--shadow); }}
    .radar-panel {{ padding: 16px; }}
    .radar-panel svg {{ display: block; width: 100%; height: auto; }}
    .radar-legend {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 8px; color: var(--muted); font-size: 12px; }}
    .radar-legend span {{ display: inline-flex; gap: 5px; align-items: center; }}
    .radar-legend i {{ display: inline-block; width: 9px; height: 9px; border-radius: 999px; background: var(--teal); }}
    .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(166px, 1fr)); gap: 12px; }}
    .metric-card {{ min-height: 112px; padding: 14px; }}
    .metric-card span {{ display: block; color: var(--muted); font-size: 12px; font-weight: 720; }}
    .metric-card strong {{ display: block; margin-top: 8px; font-size: 26px; line-height: 1.05; overflow-wrap: anywhere; }}
    .metric-card small {{ display: block; margin-top: 8px; color: var(--muted); }}
    .coverage-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin-bottom: 16px; }}
    .coverage-tile {{ padding: 14px; }}
    .coverage-tile span {{ display: block; color: var(--muted); font-size: 12px; font-weight: 720; }}
    .coverage-tile strong {{ display: block; margin-top: 6px; font-size: 24px; }}
    .progress-track {{ height: 9px; overflow: hidden; border-radius: 999px; background: #24323c; }}
    .progress-fill {{ height: 100%; border-radius: inherit; background: linear-gradient(90deg, var(--teal), var(--cyan)); }}
    .scorebar {{ display: grid; grid-template-columns: minmax(90px, 1fr) 64px; gap: 10px; align-items: center; min-width: 170px; }}
    .scorebar .progress-track {{ height: 8px; }}
    .scorebar span {{ text-align: right; font-variant-numeric: tabular-nums; font-weight: 750; }}
    .table-wrap {{ width: 100%; overflow-x: auto; border: 1px solid var(--line); border-radius: 8px; background: var(--surface); box-shadow: var(--shadow); }}
    table {{ width: 100%; border-collapse: collapse; min-width: 760px; }}
    th, td {{ padding: 10px 12px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }}
    th {{ background: #172331; color: #c9d8d5; font-size: 12px; letter-spacing: .03em; text-transform: uppercase; white-space: nowrap; }}
    td {{ overflow-wrap: anywhere; }}
    tbody tr:last-child td {{ border-bottom: 0; }}
    .group-row td {{ background: #0d151e; color: #dce9e6; font-weight: 780; }}
    .status-pill {{ display: inline-flex; align-items: center; min-height: 24px; border-radius: 999px; padding: 2px 9px; border: 1px solid var(--line); background: #101a24; color: #d5e3df; font-size: 12px; font-weight: 760; white-space: nowrap; }}
    .status-pill.success {{ border-color: rgba(102, 227, 161, 0.42); background: rgba(102, 227, 161, 0.12); color: var(--green); }}
    .status-pill.model {{ border-color: rgba(145, 183, 255, 0.42); background: rgba(145, 183, 255, 0.12); color: var(--blue); }}
    .status-pill.skipped {{ border-color: rgba(244, 189, 97, 0.42); background: rgba(244, 189, 97, 0.12); color: var(--amber); }}
    .status-pill.error {{ border-color: rgba(255, 127, 159, 0.42); background: rgba(255, 127, 159, 0.12); color: var(--rose); }}
    .metadata {{ display: grid; grid-template-columns: minmax(160px, 230px) minmax(0, 1fr); gap: 8px 16px; border: 1px solid var(--line); border-radius: 8px; padding: 14px; background: var(--surface); box-shadow: var(--shadow); }}
    .metadata dt {{ color: var(--muted); font-weight: 720; }}
    .metadata dd {{ margin: 0; overflow-wrap: anywhere; }}
    .metadata-grid {{ display: grid; grid-template-columns: minmax(280px, 430px) minmax(0, 1fr); gap: 18px; align-items: start; }}
    .bibtex {{ max-height: 180px; overflow-y: auto; overflow-x: auto; scrollbar-gutter: stable; margin: 0; border: 1px solid var(--line); border-radius: 8px; padding: 14px; background: #05080d; color: #dff8ef; white-space: pre-wrap; word-break: break-word; box-shadow: var(--shadow); }}
    .empty-state {{ border: 1px dashed var(--line-strong); border-radius: 8px; padding: 16px; background: var(--surface); color: var(--muted); }}
    @media (max-width: 1180px) {{
      .shell {{ width: min(100% - 28px, 1120px); }}
      .metric-grid {{ grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); }}
      .score-breakdown {{ min-width: 980px; }}
    }}
    @media (max-width: 900px) {{
      .shell {{ width: min(100% - 24px, 1280px); }}
      .header-layout, .summary-grid, .metadata-grid {{ grid-template-columns: 1fr; }}
      .score-hero strong {{ font-size: 38px; }}
      .section-heading {{ display: block; }}
    }}
    @media (max-width: 760px) {{
      .score-breakdown {{ min-width: 0; }}
      .score-breakdown thead {{ display: none; }}
      .score-breakdown tbody, .score-breakdown tr, .score-breakdown td {{ display: block; width: 100%; }}
      .score-breakdown tr {{ border-bottom: 1px solid var(--line); padding: 8px 0; }}
      .score-breakdown tr.group-row {{ padding: 0; }}
      .score-breakdown td {{ display: grid; grid-template-columns: minmax(112px, 42%) minmax(0, 1fr); gap: 10px; border-bottom: 0; padding: 7px 12px; }}
      .score-breakdown td::before {{ content: attr(data-label); color: var(--muted); font-size: 12px; font-weight: 760; text-transform: uppercase; }}
      .score-breakdown .group-row td {{ display: block; padding: 10px 12px; }}
      .score-breakdown .group-row td::before {{ content: none; }}
      .scorebar {{ min-width: 0; }}
    }}
    @media (max-width: 560px) {{
      h1 {{ font-size: 32px; }}
      .run-strip span {{ width: 100%; border-radius: 8px; }}
      .metric-grid, .coverage-grid {{ grid-template-columns: 1fr; }}
      .metadata {{ grid-template-columns: 1fr; }}
      th, td {{ padding: 9px 10px; }}
    }}
    @media (max-width: 420px) {{
      .shell {{ width: min(100% - 16px, 1280px); }}
      .header-layout {{ padding: 26px 0 22px; }}
      .score-hero strong {{ font-size: 34px; }}
      .metric-card, .coverage-tile, .radar-panel {{ padding: 12px; }}
      .score-breakdown td {{ grid-template-columns: 1fr; gap: 3px; }}
    }}
    @media print {{
      .report-header {{ background: #fff; }}
      .score-hero, .metric-card, .coverage-tile, .table-wrap, .metadata, .radar-panel, .bibtex {{ box-shadow: none; }}
      .table-wrap, .bibtex {{ overflow: visible; }}
    }}
  </style>
</head>
<body>
  <header class="report-header">
    <div class="shell header-layout">
      <div>
        <p class="eyebrow">Agent Bench Final Report</p>
        <h1>{html.escape(str(model_name))}</h1>
        <p class="dek">Benchmark run {html.escape(str(run_id))}{_join_if(created_at, " recorded ")} across {html.escape(_coverage_label(coverage))}.</p>
        <div class="run-strip">
          <span>Provider: {html.escape(str(provider))}</span>
          <span>Judge: {html.escape(str(judge.get("model") or judge.get("provider") or "none"))}</span>
          <span>Duration: {html.escape(_format_seconds_safe(summary.get("total_run_duration_seconds")))}</span>
        </div>
      </div>
      <div class="score-hero" aria-label="{html.escape(score_label)}">
        <span>{html.escape(score_label)}</span>
        <strong>{html.escape(score_display)}</strong>
        {_progress_bar(score_fraction)}
        <small>{html.escape(score_hint)}</small>
      </div>
    </div>
  </header>
  <main class="shell report-main">
    <section class="section-block" id="result-summary">
      <div class="section-heading">
        <div>
          <h2>Result Summary</h2>
          <p>Score and coverage are separated so a low score is not confused with an incomplete run.</p>
        </div>
      </div>
      <div class="summary-grid">
        <div class="radar-panel">
          {_radar_svg(_summary_radar_scores(summary))}
          <div class="radar-legend"><span><i></i>Category score</span><span>Scale 0-100%</span></div>
        </div>
        {_report_metric_cards(summary, benchmark_results)}
      </div>
    </section>
    <section class="section-block" id="benchmark-coverage">
      <div class="section-heading">
        <div>
          <h2>Benchmark Coverage</h2>
          <p>Configured suites, successfully scored suites, exclusions, and category-level coverage.</p>
        </div>
      </div>
      {_coverage_section(summary, coverage)}
    </section>
    <section class="section-block" id="benchmark-score-breakdown">
      <div class="section-heading">
        <div>
          <h2>Benchmark Score Breakdown</h2>
          <p>Official-score inclusion is shown separately from model performance and raw adapter output.</p>
        </div>
      </div>
      {_score_breakdown_table(benchmark_results)}
    </section>
    <section class="section-block" id="run-metadata">
      <div class="section-heading">
        <div>
          <h2>Run Metadata</h2>
          <p>Execution settings, target model, judge model, manifests, source refs, and asset coverage.</p>
        </div>
      </div>
      {_report_metadata_section(summary, metadata)}
    </section>
    <section class="section-block" id="non-model-errors">
      <div class="section-heading">
        <div>
          <h2>Non-Model Run Errors</h2>
          <p>Setup, infrastructure, asset, timeout, judge, and unsupported-capability issues encountered during the run.</p>
        </div>
      </div>
      {_non_model_error_section(benchmark_results)}
    </section>
    <section class="section-block" id="benchmark-citations">
      <div class="section-heading">
        <div>
          <h2>Benchmark Citations</h2>
          <p>BibTeX-style references for benchmark suites, kept in a compact scrollable code section.</p>
        </div>
      </div>
      {_citation_section(benchmark_results)}
    </section>
  </main>
</body>
</html>
"""


def _report_rows(summary: dict[str, Any], results: list[GradeResult]) -> list[dict[str, Any]]:
    benchmark_results = summary.get("benchmark_results")
    if isinstance(benchmark_results, list) and benchmark_results:
        return [row for row in benchmark_results if isinstance(row, dict)]
    return _fallback_benchmark_rows(results)


def _fallback_benchmark_rows(results: list[GradeResult]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        details = result.details if isinstance(result.details, dict) else {}
        payload = _benchmark_payload(result)
        rows.append(
            {
                "suite_id": result.task_id,
                "task_id": result.task_id,
                "group": details.get("group", result.category),
                "benchmark": details.get("benchmark", _result_benchmark_name(result)),
                "profile": "",
                "score": round(result.score * 100.0, 4),
                "raw_score": _unit_to_percent(payload.get("raw_score")),
                "valid_score": _unit_to_percent(payload.get("valid_score")),
                "included_in_official_score": _included_in_official_score(result),
                "status": _result_status(result),
                "run_status": _run_status(result),
                "score_status": _score_status(result),
                "duration_seconds": result.task_duration_seconds,
                "blocker_type": _blocker_type(result),
                "error": _result_error_reason(result),
                "error_details": _result_error_reason(result),
                "homepage": details.get("homepage", ""),
                "official_leaderboard_url": details.get("official_leaderboard_url", ""),
                "license": details.get("license", ""),
                "credit": details.get("credit", ""),
                "citation": details.get("citation", details.get("homepage", "")),
                "manifest": details.get("manifest", {}),
                "source": details.get("source", {}),
                "required_capabilities": payload.get("required_capabilities", details.get("required_capabilities", [])),
                "supported_capabilities": payload.get("supported_capabilities", []),
                "capabilities_verified": _capabilities_verified(result),
                "required_tools": payload.get("required_tools", []),
                "exposed_tools": payload.get("exposed_tools", []),
                "missing_tools": payload.get("missing_tools", []),
                "missing_env": payload.get("missing_env", payload.get("missing_environment", [])),
                "evaluated_task_count": payload.get("evaluated_task_count"),
                "valid_evaluated_task_count": payload.get("valid_evaluated_task_count"),
                "evaluation_passed_count": payload.get("evaluation_passed_count"),
                "grading_methods": [],
                "docker_image": payload.get("docker_image", details.get("docker_image", "")),
                "container_name": payload.get("container_name", details.get("container_name", "")),
                "network_mode": payload.get("network_mode", details.get("network_mode", "")),
                "setup_details": payload.get("setup_details", details.get("setup_details", {})),
            }
        )
    return sorted(rows, key=lambda row: (str(row.get("group", "")), str(row.get("benchmark", ""))))


def _target_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    target = metadata.get("target_model")
    if isinstance(target, dict):
        return target
    return {}


def _judge_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    judge = metadata.get("judge")
    if isinstance(judge, dict):
        return judge
    return {}


def _headline_score(summary: dict[str, Any]) -> tuple[str, str, float, str]:
    headline = summary.get("headline") if isinstance(summary.get("headline"), dict) else {}
    for key in ("valid_judged_score", "valid_judged_suite_score"):
        value = headline.get(key, summary.get(key))
        if value is not None:
            return (
                "Scored-Suite Score",
                _format_rate_auto(value),
                _fraction_from_rate(value),
                "Average score across suites that produced official-equivalent judgments.",
            )
    value = summary.get("score_valid_tasks_only", summary.get("total_score"))
    return (
        "Scored-Suite Score",
        _format_percent_display(value),
        _fraction_from_percent(value),
        "Average score across valid judged suites.",
    )


def _report_metric_cards(summary: dict[str, Any], benchmark_results: list[dict[str, Any]]) -> str:
    coverage = summary.get("coverage_summary", summary.get("coverage", {}))
    scored = _coverage_value(coverage, "successfully_scored_benchmarks", "valid_judged_suite_count", summary.get("valid_task_count", 0))
    total = _coverage_value(coverage, "total_configured_benchmarks", "suite_count", summary.get("suite_count", summary.get("task_count", 0)))
    cards = [
        (
            "Valid judged score",
            _format_rate_auto(summary.get("valid_judged_score", summary.get("valid_judged_suite_score"))),
            "Official-equivalent judged suites",
        ),
        (
            "Conservative score",
            _format_rate_auto(summary.get("conservative_all_suite_score", summary.get("conservative_selected_suite_score"))),
            "Counts exclusions as zero",
        ),
        ("Suite coverage", _format_rate_auto(summary.get("suite_coverage_rate", _coverage_rate(coverage))), f"{scored}/{total} suites scored"),
        ("Item coverage", _format_rate_auto(summary.get("item_coverage_rate")), "Valid judged benchmark items"),
        ("Non-model errors", str(len(_non_model_error_rows(benchmark_results))), "Setup, infra, judge, assets, timeout"),
        ("Run duration", _format_seconds_safe(summary.get("total_run_duration_seconds")), "Wall-clock elapsed time"),
        ("Model score", _format_percent_display(summary.get("model_score_valid_tasks_only")), "Only capability-verified attempts"),
        ("Parser repairs", _format_integer_safe(summary.get("parser_repair_count")), "Judge or output repairs"),
    ]
    return '<div class="metric-grid">' + "".join(_summary_card(label, value, hint) for label, value, hint in cards) + "</div>"


def _summary_card(label: str, value: str, hint: str) -> str:
    return (
        '<div class="metric-card">'
        f"<span>{html.escape(label)}</span>"
        f"<strong>{html.escape(value)}</strong>"
        f"<small>{html.escape(hint)}</small>"
        "</div>"
    )


def _progress_bar(fraction: float) -> str:
    width = max(0.0, min(100.0, fraction * 100.0))
    return (
        '<div class="progress-track" aria-hidden="true">'
        f'<div class="progress-fill" style="width: {width:.2f}%"></div>'
        "</div>"
    )


def _summary_radar_scores(summary: dict[str, Any]) -> dict[str, float]:
    scores = summary.get("category_scores")
    if isinstance(scores, dict) and scores:
        normalized: dict[str, float] = {}
        for label, value in scores.items():
            try:
                normalized[str(label)] = float(value)
            except (TypeError, ValueError):
                continue
        if len(normalized) <= 8:
            return normalized
        ordered = sorted(normalized.items(), key=lambda item: item[1], reverse=True)
        top = dict(ordered[:7])
        rest = [value for _, value in ordered[7:]]
        top["Other"] = sum(rest) / len(rest)
        return top
    return _average_scores(summary)


def _coverage_label(coverage: Any) -> str:
    if not isinstance(coverage, dict):
        return "coverage n/a"
    scored = coverage.get("successfully_scored_benchmarks", coverage.get("valid_judged_suite_count"))
    total = coverage.get("total_configured_benchmarks", coverage.get("suite_count"))
    if isinstance(scored, int) and isinstance(total, int):
        return f"{scored}/{total} scored"
    return "coverage n/a"


def _coverage_section(summary: dict[str, Any], coverage: Any) -> str:
    profile_results = summary.get("profile_results")
    return (
        _coverage_overview_cards(summary, coverage)
        + "<h3>Coverage By Category</h3>"
        + _coverage_table(coverage)
        + "<h3>Coverage By Profile</h3>"
        + _profile_coverage_table(profile_results)
    )


def _coverage_overview_cards(summary: dict[str, Any], coverage: Any) -> str:
    if not isinstance(coverage, dict):
        coverage = {}
    total = _coverage_value(coverage, "total_configured_benchmarks", "suite_count", summary.get("suite_count", summary.get("task_count", 0)))
    attempted = _coverage_value(coverage, "attempted_benchmarks", "attempted_suite_count", total)
    scored = _coverage_value(coverage, "successfully_scored_benchmarks", "valid_judged_suite_count", summary.get("valid_task_count", 0))
    excluded = _coverage_value(coverage, "excluded_from_score_benchmarks", "failed_benchmarks", max(0, int(total) - int(scored)))
    cards = [
        ("Configured", str(total)),
        ("Attempted", str(attempted)),
        ("Scored", str(scored)),
        ("Excluded", str(excluded)),
        ("Suite coverage", _format_rate_auto(summary.get("suite_coverage_rate", _coverage_rate(coverage)))),
        ("Item coverage", _format_rate_auto(summary.get("item_coverage_rate", coverage.get("item_coverage_rate")))),
    ]
    return (
        '<div class="coverage-grid">'
        + "".join(
            '<div class="coverage-tile">'
            f"<span>{html.escape(label)}</span><strong>{html.escape(value)}</strong>"
            "</div>"
            for label, value in cards
        )
        + "</div>"
    )


def _coverage_table(coverage: Any) -> str:
    if not isinstance(coverage, dict):
        return '<div class="empty-state">No coverage data recorded.</div>'
    per_category = coverage.get("per_category")
    if not isinstance(per_category, dict) or not per_category:
        return '<div class="empty-state">No category coverage data recorded.</div>'
    rows = []
    for category, data in sorted(per_category.items()):
        if not isinstance(data, dict):
            continue
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(category))}</td>"
            f"<td>{int(data.get('total_configured_benchmarks', 0))}</td>"
            f"<td>{int(data.get('successfully_scored_benchmarks', 0))}</td>"
            f"<td>{int(data.get('failed_benchmarks', 0))}</td>"
            f"<td>{_format_rate_auto(data.get('coverage_rate'))}</td>"
            "</tr>"
        )
    return (
        '<div class="table-wrap"><table><thead><tr><th>Category</th><th>Configured</th><th>Scored</th>'
        "<th>Failed</th><th>Coverage</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _profile_coverage_table(profile_results: Any) -> str:
    if not isinstance(profile_results, dict) or not profile_results:
        return '<div class="empty-state">No profile coverage data recorded.</div>'
    rows = []
    for profile, data in sorted(profile_results.items()):
        if not isinstance(data, dict):
            continue
        blockers = data.get("blocker_counts")
        blocker_text = ", ".join(f"{key}: {count}" for key, count in sorted(blockers.items())) if isinstance(blockers, dict) else ""
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(profile))}</td>"
            f"<td>{int(data.get('valid_judged_suite_count', 0))}/{int(data.get('suite_count', 0))}</td>"
            f"<td>{_format_rate_auto(data.get('suite_coverage_rate'))}</td>"
            f"<td>{_format_rate_auto(data.get('valid_judged_score'))}</td>"
            f"<td>{html.escape(blocker_text)}</td>"
            "</tr>"
        )
    return (
        '<div class="table-wrap"><table><thead><tr><th>Profile</th><th>Runnable Suites</th><th>Coverage</th>'
        f"<th>Valid Score</th><th>Blockers</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


def _score_breakdown_table(benchmark_results: Any) -> str:
    if not isinstance(benchmark_results, list) or not benchmark_results:
        return '<div class="empty-state">No benchmark rows were recorded.</div>'
    rows: list[str] = []
    current_group: str | None = None
    for row in benchmark_results:
        if not isinstance(row, dict):
            continue
        group = str(row.get("group", "Other"))
        if group != current_group:
            rows.append(f'<tr class="group-row"><td colspan="10">{html.escape(group)}</td></tr>')
            current_group = group
        rows.append(
            "<tr>"
            f"<td data-label=\"Benchmark\">{html.escape(str(row.get('benchmark', '')))}</td>"
            f"<td data-label=\"Profile\">{html.escape(str(row.get('profile') or ''))}</td>"
            f"<td data-label=\"Normalized 0-100\">{_score_bar(row.get('score'))}</td>"
            f"<td data-label=\"Raw Score\">{html.escape(_format_percent_display(row.get('raw_score')))}</td>"
            f"<td data-label=\"Valid Score\">{html.escape(_format_percent_display(row.get('valid_score')))}</td>"
            f"<td data-label=\"Official Score\">{'yes' if row.get('included_in_official_score') else 'no'}</td>"
            f"<td data-label=\"Status\">{_status_pill(row.get('status'))}</td>"
            f"<td data-label=\"Run\">{html.escape(str(row.get('run_status') or ''))}</td>"
            f"<td data-label=\"Score Status\">{html.escape(str(row.get('score_status') or ''))}</td>"
            f"<td data-label=\"Items\">{html.escape(_items_summary(row))}</td>"
            "</tr>"
        )
    return (
        '<div class="table-wrap"><table class="score-breakdown"><thead><tr><th>Benchmark</th><th>Profile</th>'
        "<th>Normalized 0-100</th><th>Raw Score</th><th>Valid Score</th><th>Official Score</th>"
        "<th>Status</th><th>Run</th><th>Score Status</th><th>Items</th>"
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


def _status_pill(status: Any) -> str:
    normalized = normalize_status(status)
    info = status_info(normalized)
    if normalized in {"success", "success_with_warnings", "passed"} or info.failure_class == "none":
        tone = "success"
    elif info.failure_class == "model":
        tone = "model"
    elif info.failure_class == "user_skip" or normalized.startswith("skipped"):
        tone = "skipped"
    else:
        tone = "error"
    label = normalized or "unknown"
    return f'<span class="status-pill {tone}">{html.escape(label)}</span>'


def _score_bar(value: Any) -> str:
    percent = _percent_number(value)
    display = _format_percent_display(value)
    return (
        '<div class="scorebar">'
        + _progress_bar(percent / 100.0)
        + f"<span>{html.escape(display)}</span>"
        + "</div>"
    )


def _items_summary(row: dict[str, Any]) -> str:
    evaluated = row.get("evaluated_task_count")
    valid = row.get("valid_evaluated_task_count")
    passed = row.get("evaluation_passed_count")
    if evaluated is None and valid is None and passed is None:
        return "n/a"
    parts = []
    if passed is not None and evaluated is not None:
        parts.append(f"{passed}/{evaluated} passed")
    elif evaluated is not None:
        parts.append(f"{evaluated} evaluated")
    if valid is not None:
        parts.append(f"{valid} valid")
    return "; ".join(parts)


def _report_metadata_section(summary: dict[str, Any], metadata: dict[str, Any]) -> str:
    target = _target_metadata(metadata)
    judge = _judge_metadata(metadata)
    rows = {
        "schema_version": summary.get("schema_version"),
        "benchmark_version": summary.get("benchmark_version"),
        "selected_profile": summary.get("selected_profile"),
        "run_id": metadata.get("run_id"),
        "created_at_utc": metadata.get("created_at_utc"),
        "output_dir": metadata.get("output_dir"),
        "git_commit": metadata.get("git_commit"),
        "target_provider": target.get("provider_type", target.get("provider", metadata.get("provider"))),
        "target_base_url": target.get("base_url", metadata.get("base_url")),
        "target_model": target.get("model", metadata.get("model")),
        "temperature": target.get("temperature", metadata.get("temperature")),
        "top_p": target.get("top_p", metadata.get("top_p")),
        "max_tokens": target.get("max_tokens", metadata.get("max_tokens")),
        "tool_parser": target.get("tool_parser", metadata.get("tool_parser")),
        "context_window": target.get("context_window", metadata.get("context_window")),
        "judge_provider": judge.get("provider"),
        "judge_base_url": judge.get("base_url"),
        "judge_model": judge.get("model"),
        "judge_temperature": judge.get("temperature"),
        "judge_timeout_seconds": judge.get("timeout_seconds"),
        "judge_fallback_used": judge.get("fallback_used"),
        "request_concurrency": metadata.get("request_concurrency"),
        "eval_concurrency": metadata.get("eval_concurrency"),
        "timeout_seconds": metadata.get("timeout"),
        "sandbox": metadata.get("sandbox"),
        "allow_host_docker_socket": metadata.get("allow_host_docker_socket"),
    }
    return _metadata_definition_list(_compact_metadata_rows(rows))


def _metadata_definition_list(values: dict[str, Any]) -> str:
    rows = []
    for key, value in values.items():
        rows.append(f"<dt>{html.escape(str(key))}</dt><dd>{html.escape(_display_value(value))}</dd>")
    return f'<dl class="metadata">{"".join(rows)}</dl>'


def _compact_metadata_rows(values: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value not in (None, "", [], {})}


def _non_model_error_section(benchmark_results: Any) -> str:
    rows = _non_model_error_rows(benchmark_results)
    if not rows:
        return '<div class="empty-state">No non-model run errors were recorded.</div>'
    html_rows = []
    for row in rows:
        status = normalize_status(row.get("status"))
        info = status_info(status)
        error = _display_error(str(row.get("error_details") or row.get("error") or ""))
        html_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('benchmark', '')))}</td>"
            f"<td>{html.escape(_report_failure_class(row, info.failure_class))}</td>"
            f"<td>{_status_pill(status)}</td>"
            f"<td>{html.escape(str(row.get('run_status') or ''))}</td>"
            f"<td>{html.escape(str(row.get('blocker_type') or ''))}</td>"
            f"<td>{html.escape(error or info.explanation)}</td>"
            f"<td>{html.escape(_suggested_action(row))}</td>"
            "</tr>"
        )
    return (
        '<div class="table-wrap"><table><thead><tr><th>Benchmark</th><th>Failure Class</th><th>Status</th>'
        "<th>Run</th><th>Blocker</th><th>Explanation</th><th>Suggested Action</th></tr></thead>"
        f"<tbody>{''.join(html_rows)}</tbody></table></div>"
    )


def _non_model_error_rows(benchmark_results: Any) -> list[dict[str, Any]]:
    if not isinstance(benchmark_results, list):
        return []
    rows = [row for row in benchmark_results if isinstance(row, dict) and _is_non_model_error_row(row)]
    return sorted(rows, key=lambda row: (str(row.get("group", "")), str(row.get("benchmark", ""))))


def _is_non_model_error_row(row: dict[str, Any]) -> bool:
    status = normalize_status(row.get("status"))
    error = str(row.get("error_details") or row.get("error") or "").strip()
    info = status_info(status)
    if info.failure_class in {"none", "model"}:
        return False
    if info.failure_class == "user_skip" and not error:
        return False
    if row.get("included_in_official_score") and info.counts_toward_official_score and not error:
        return False
    return bool(status or error)


def _report_failure_class(row: dict[str, Any], fallback: str) -> str:
    status = normalize_status(row.get("status"))
    blocker = str(row.get("blocker_type") or "")
    if status in {FAILED_MISSING_ASSETS, FAILED_MISSING_REQUIRED_TOOL, FAILED_INVALID_TASK_CONTEXT}:
        return "benchmark_setup"
    if any(marker in blocker for marker in ("asset", "tool", "grader", "task_context", "reference", "manifest")):
        return "benchmark_setup"
    if status in {FAILED_GRADER, FAILED_TOKEN_BUDGET, TIMED_OUT, FAILED_HARNESS_SETUP, FAILED_DATASET_EXTRACTION}:
        return "infrastructure"
    return fallback


def _suggested_action(row: dict[str, Any]) -> str:
    manifest = row.get("manifest") if isinstance(row.get("manifest"), dict) else {}
    result = row.get("result") if isinstance(row.get("result"), dict) else {}
    validation = result.get("validation") if isinstance(result.get("validation"), dict) else {}
    issues = validation.get("issues") if isinstance(validation.get("issues"), list) else []
    for issue in issues:
        if isinstance(issue, dict) and issue.get("suggestion"):
            return str(issue["suggestion"])
    blocker = str(row.get("blocker_type") or "")
    status = normalize_status(row.get("status"))
    if "asset" in blocker or status in {FAILED_MISSING_ASSETS, FAILED_DATASET_EXTRACTION}:
        return "Refresh benchmark assets and rerun validation."
    if "tool" in blocker or status == FAILED_MISSING_REQUIRED_TOOL:
        return "Install or expose the required benchmark tool and rerun."
    if status in {FAILED_GRADER, FAILED_TOKEN_BUDGET}:
        return "Inspect judge output, parser diagnostics, and token limits."
    if status == TIMED_OUT:
        return "Increase timeout or inspect benchmark/container logs."
    if manifest:
        return "Validate the benchmark manifest and official-run settings."
    return "Inspect graded results and benchmark logs."


def _citation_section(benchmark_results: Any) -> str:
    catalog = _benchmark_citation_catalog()
    if catalog:
        return _code_block(catalog)
    if not isinstance(benchmark_results, list) or not benchmark_results:
        return '<div class="empty-state">No benchmark citations were recorded.</div>'
    entries: list[str] = []
    seen: set[str] = set()
    for row in benchmark_results:
        if not isinstance(row, dict):
            continue
        entry = _bibtex_entry(row)
        key = _bibtex_entry_key(entry) or _bibtex_key(str(row.get("benchmark") or "Benchmark"))
        if key in seen:
            continue
        seen.add(key)
        entries.append(entry)
    if not entries:
        return '<div class="empty-state">No benchmark citations were recorded.</div>'
    return _code_block("\n\n".join(entries))


def _benchmark_citation_catalog() -> str:
    try:
        return BENCHMARK_CITATIONS_PATH.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _bibtex_entry(row: dict[str, Any]) -> str:
    citation = str(row.get("citation") or "").strip()
    if citation.startswith("@"):
        return citation
    title = str(row.get("benchmark") or "Benchmark")
    key = _bibtex_key(title)
    homepage = str(row.get("homepage") or "").strip()
    url = citation if citation.startswith(("http://", "https://")) else homepage
    fields = [f"  title = {{{_bibtex_value(title)}}}"]
    if url:
        fields.append(f"  url = {{{_bibtex_value(url)}}}")
    if citation and citation != url:
        fields.append(f"  note = {{{_bibtex_value(citation)}}}")
    credit = str(row.get("credit") or "").strip()
    if credit:
        fields.append(f"  author = {{{_bibtex_value(credit)}}}")
    license_name = str(row.get("license") or "").strip()
    if license_name:
        fields.append(f"  license = {{{_bibtex_value(license_name)}}}")
    return "@misc{" + key + ",\n" + ",\n".join(fields) + "\n}"


def _bibtex_entry_key(entry: str) -> str:
    first_line = entry.strip().splitlines()[0] if entry.strip() else ""
    if not first_line.startswith("@") or "{" not in first_line:
        return ""
    return first_line.split("{", 1)[1].split(",", 1)[0].strip()


def _bibtex_key(title: str) -> str:
    chars = [ch.lower() if ch.isalnum() else "_" for ch in title]
    key = "".join(chars).strip("_")
    while "__" in key:
        key = key.replace("__", "_")
    if not key or not key[0].isalpha():
        key = "benchmark_" + key
    return key


def _bibtex_value(value: str) -> str:
    return " ".join(value.split()).replace("{", "\\{").replace("}", "\\}")


def _code_block(value: str) -> str:
    return f'<pre class="bibtex"><code>{html.escape(value)}</code></pre>'


def _coverage_value(coverage: Any, primary: str, fallback: str, default: Any) -> Any:
    if not isinstance(coverage, dict):
        return default
    return coverage.get(primary, coverage.get(fallback, default))


def _coverage_rate(coverage: Any) -> Any:
    if not isinstance(coverage, dict):
        return None
    return coverage.get("coverage_rate", coverage.get("suite_coverage_rate"))


def _format_rate_auto(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    percent = numeric * 100.0 if abs(numeric) <= 1.0 else numeric
    return f"{percent:.2f}%"


def _format_percent_display(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.2f}%"
    except (TypeError, ValueError):
        return str(value)


def _format_seconds_safe(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.3f}s"
    except (TypeError, ValueError):
        return str(value)


def _format_integer_safe(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return str(value)


def _display_value(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (dict, list)):
        return _short_json(value)
    return str(value)


def _percent_number(value: Any) -> float:
    try:
        return max(0.0, min(100.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _fraction_from_rate(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, numeric if abs(numeric) <= 1.0 else numeric / 100.0))


def _fraction_from_percent(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, numeric / 100.0))


def _join_if(value: Any, prefix: str) -> str:
    if value in (None, ""):
        return ""
    return prefix + html.escape(str(value))


def _link(url: str, label: str) -> str:
    if not url:
        return ""
    if url.startswith(("http://", "https://")):
        return f'<a href="{html.escape(url, quote=True)}">{html.escape(label or url)}</a>'
    return html.escape(label or url)


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
    row["answer"] = _csv_answer(result, payload)
    row["suite_id"] = details.get("suite_id", result.task_id)
    row["suite_name"] = _result_benchmark_name(result)
    row["task_group"] = details.get("group", result.category)
    row["run_status"] = _run_status(result)
    row["score_status"] = _score_status(result)
    row["official_equivalent"] = payload.get("official_equivalent", details.get("official_equivalent"))
    row["score_mode"] = payload.get("score_mode", details.get("score_mode"))
    row["score_modes"] = _json_cell(payload.get("score_modes", details.get("score_modes", [])))
    row["included_in_official_score"] = _included_in_official_score(result)
    row["coverage_status"] = "scored" if row["included_in_official_score"] else "excluded"
    row["required_capabilities"] = _json_cell(payload.get("required_capabilities", details.get("required_capabilities", [])))
    row["supported_capabilities"] = _json_cell(payload.get("supported_capabilities", []))
    row["capabilities_verified"] = _capabilities_verified(result)
    row["required_tools"] = _json_cell(payload.get("required_tools", []))
    row["exposed_tools"] = _json_cell(payload.get("exposed_tools", []))
    row["missing_tools"] = _json_cell(payload.get("missing_tools", []))
    row["missing_env"] = _json_cell(payload.get("missing_env", payload.get("missing_environment", [])))
    row["missing_assets_count"] = payload.get("missing_assets_count", _payload_status_count(payload, FAILED_MISSING_ASSETS))
    setup_details = payload.get("setup_details", details.get("setup_details"))
    external_setup = setup_details.get("external_harness") if isinstance(setup_details, dict) else {}
    if not isinstance(external_setup, dict):
        external_setup = {}
    row["setup_details"] = _json_cell(setup_details)
    row["docker_image"] = payload.get("docker_image", details.get("docker_image", external_setup.get("image")))
    row["container_name"] = payload.get("container_name", details.get("container_name", external_setup.get("container_name")))
    row["network_mode"] = payload.get("network_mode", details.get("network_mode", external_setup.get("network_mode")))
    row["docker_socket_mount"] = _json_cell(
        payload.get("docker_socket_mount", details.get("docker_socket_mount", external_setup.get("docker_socket_mount")))
    )
    row["output_mount"] = _json_cell(
        payload.get("output_mount", details.get("output_mount", external_setup.get("output_mount")))
    )
    row["asset_cache_mount"] = _json_cell(
        payload.get("asset_cache_mount", details.get("asset_cache_mount", external_setup.get("asset_cache_mount")))
    )
    row["catalog_checkout_path"] = payload.get(
        "catalog_checkout_path",
        details.get("catalog_checkout_path", external_setup.get("catalog_checkout_path")),
    )
    row["target_checkout_path"] = payload.get(
        "target_checkout_path",
        details.get("target_checkout_path", external_setup.get("target_checkout_path")),
    )
    row["benchmark_checkout_path"] = payload.get(
        "benchmark_checkout_path",
        details.get("benchmark_checkout_path", external_setup.get("benchmark_checkout_path")),
    )
    row["homepage"] = details.get("homepage")
    row["license"] = details.get("license")
    row["credit"] = details.get("credit")
    row["citation"] = details.get("citation", details.get("homepage"))
    row["blocker_type"] = _blocker_type(result)
    row["error_details"] = _result_error_reason(result)
    row["raw_score"] = _unit_to_percent(payload.get("raw_score"))
    row["valid_score"] = _unit_to_percent(payload.get("valid_score"))
    row["raw_model_response"] = details.get("raw_model_response")
    row["extracted_answer"] = _json_cell(details.get("extracted_answer"))
    row["extraction_status"] = details.get("extraction_status")
    row["judge_parser_status"] = payload.get("judge_parser_status")
    row["judge_retry_count"] = payload.get("judge_retry_count")
    row["judge_parse_repaired_count"] = payload.get("judge_parse_repaired_count")
    return row


def _csv_answer(result: GradeResult, payload: dict[str, Any]) -> Any:
    answer = result.answer
    if result.kind != "external_benchmark":
        return answer
    if _has_answer_value(answer) and not _is_status_token(answer):
        return answer
    payload_answer = payload.get("answer")
    if _has_answer_value(payload_answer) and not _is_status_token(payload_answer):
        return payload_answer
    model_eval = payload.get("model_eval")
    if isinstance(model_eval, dict):
        model_answer = model_eval.get("answer")
        if _has_answer_value(model_answer) and not _is_status_token(model_answer):
            return model_answer
    return ""


def _has_answer_value(value: Any) -> bool:
    return value is not None and (not isinstance(value, str) or bool(value.strip()))


def _is_status_token(value: Any) -> bool:
    return isinstance(value, str) and normalize_status(value) in {
        PASSED,
        FAILED_MODEL_ANSWER,
        FAILED_MODEL_FORMAT,
        FAILED_MODEL_MISSING_ARTIFACT,
        FAILED_MODEL_TOOL_USE,
        FAILED_HARNESS_SETUP,
        FAILED_DATASET_EXTRACTION,
        FAILED_MISSING_ASSETS,
        FAILED_MISSING_REQUIRED_TOOL,
        FAILED_GRADER,
        FAILED_TOKEN_BUDGET,
        FAILED_INVALID_TASK_CONTEXT,
        SKIPPED_UNSUPPORTED_CAPABILITY,
        TIMED_OUT,
    }


def _metric_cards(summary: dict[str, Any]) -> str:
    cards = [
        ("Scored-Suite Score", _format_rate(summary.get("valid_judged_score"))),
        ("Suite Coverage", _format_rate(summary.get("suite_coverage_rate"))),
        ("Item Coverage", _format_rate(summary.get("item_coverage_rate"))),
        ("Conservative Selected-Suite", _format_rate(summary.get("conservative_all_suite_score"))),
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


def _excluded_suite_table(value: Any) -> str:
    if not isinstance(value, list) or not value:
        return "<table><tbody><tr><td>No excluded suites were recorded.</td></tr></tbody></table>"
    rows = []
    for row in value:
        if not isinstance(row, dict):
            continue
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('suite_id', '')))}</td>"
            f"<td>{html.escape(str(row.get('name', '')))}</td>"
            f"<td>{html.escape(str(row.get('lifecycle_status', '')))}</td>"
            f"<td>{html.escape(str(row.get('exclusion_reason', '')))}</td>"
            f"<td>{html.escape(str(row.get('removal_reason', '')))}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Suite</th><th>Name</th><th>Lifecycle</th><th>Exclusion</th><th>Reason</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
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
            f'<polygon points="{_points(points)}" fill="none" stroke="#314250" stroke-width="1" />'
        )

    axis_lines = []
    label_nodes = []
    for index, label in enumerate(labels):
        end = _point(cx, cy, radius, index, axes)
        label_point = _point(cx, cy, radius + 28.0, index, axes)
        axis_lines.append(
            f'<line x1="{cx:.1f}" y1="{cy:.1f}" x2="{end[0]:.1f}" y2="{end[1]:.1f}" stroke="#405667" />'
        )
        label_nodes.append(
            f'<text x="{label_point[0]:.1f}" y="{label_point[1]:.1f}" text-anchor="middle" dominant-baseline="middle" font-size="10" fill="#dbe8e5">{html.escape(label)}</text>'
        )

    value_points = [_point(cx, cy, radius * (value / 100.0), index, axes) for index, value in enumerate(values)]
    return (
        '<svg viewBox="0 0 320 260" role="img" aria-label="Radar chart of category scores">'
        + "".join(grid_polygons)
        + "".join(axis_lines)
        + f'<polygon points="{_points(value_points)}" fill="#41d6c3" fill-opacity="0.24" stroke="#67d5ff" stroke-width="2" />'
        + "".join(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="#66e3a1" />' for x, y in value_points
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


def _included_in_official_score(result: GradeResult) -> bool:
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
    return _result_status(result) not in INVALID_EVALUATION_STATUSES


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
    details = result.details if isinstance(result.details, dict) else {}
    payload = details.get("result") if isinstance(details.get("result"), dict) else {}
    explicit = _explicit_blocker_type(details, payload, error)
    if explicit:
        return explicit
    if _uses_smoke_score_mode(payload):
        return "missing_grader"
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


def _uses_smoke_score_mode(payload: dict[str, Any]) -> bool:
    modes: set[str] = set()
    mode = payload.get("score_mode")
    if isinstance(mode, str) and mode.strip():
        modes.add(mode.strip())
    score_modes = payload.get("score_modes")
    if isinstance(score_modes, list):
        modes.update(str(item).strip() for item in score_modes if str(item).strip())
    return payload.get("official_equivalent") is False or any(mode.startswith("smoke") for mode in modes)


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


def _short_json(value: Any, max_chars: int = 260) -> str:
    if value in (None, "", [], {}):
        return ""
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return text if len(text) <= max_chars else text[: max_chars - 1] + "..."


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
