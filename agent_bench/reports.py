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
    coverage = summary.get("coverage_summary") if isinstance(summary.get("coverage_summary"), dict) else {}
    if not coverage:
        coverage = summary.get("coverage") if isinstance(summary.get("coverage"), dict) else {}
    target = _target_metadata(metadata)
    judge = _judge_metadata(metadata)
    run_id = metadata.get("run_id") or metadata.get("output_dir") or "agent-bench-run"
    created_at = metadata.get("created_at_utc") or ""
    target_model = target.get("model") or metadata.get("model") or "unknown model"
    target_provider = target.get("provider_type") or metadata.get("provider") or ""
    target_base_url = target.get("base_url") or metadata.get("base_url") or ""
    judge_label = _join_if(
        [
            judge.get("provider"),
            judge.get("model"),
            "fallback used" if judge.get("fallback_used") else "",
        ],
        " / ",
    )
    radar_scores = _summary_radar_scores(summary, benchmark_results)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Agent Bench Report - {html.escape(str(target_model))}</title>
  <style>{_report_css()}</style>
</head>
<body>
  <div class="shell">
    <header class="report-header">
      <div class="header-copy">
        <p class="eyebrow">Agent Bench Report</p>
        <h1>{html.escape(str(target_model))}</h1>
        <p class="run-line">{html.escape(str(run_id))} · {html.escape(str(created_at))}</p>
        <p class="run-line">{html.escape(_join_if([target_provider, target_base_url], " · "))}</p>
      </div>
      <div class="headline-panel">
        <span>Scored-Suite Score</span>
        <strong>{html.escape(_headline_score(summary))}</strong>
        <small>{html.escape(_coverage_label(coverage))}</small>
      </div>
    </header>

    <main>
      <section id="result-summary" class="report-section">
        <div class="section-heading">
          <div>
            <p class="eyebrow">Result Summary</p>
            <h2>Result Summary</h2>
          </div>
          <p>{html.escape(judge_label or "No judge metadata recorded")}</p>
        </div>
        {_report_metric_cards(summary, coverage)}
        <div class="summary-grid">
          <div class="radar-panel">
            <h3>Capability Radar</h3>
            {_radar_svg(radar_scores)}
          </div>
          <div class="summary-panel">
            <h3>Score Context</h3>
            {_score_context(summary, coverage)}
          </div>
        </div>
      </section>

      <section id="benchmark-coverage" class="report-section">
        <div class="section-heading">
          <div>
            <p class="eyebrow">Benchmark Coverage</p>
            <h2>Benchmark Coverage</h2>
          </div>
          <p>{html.escape(_coverage_label(coverage))}</p>
        </div>
        {_coverage_section(summary, coverage)}
      </section>

      <section id="benchmark-score-breakdown" class="report-section">
        <div class="section-heading">
          <div>
            <p class="eyebrow">Benchmark Score Breakdown</p>
            <h2>Benchmark Score Breakdown</h2>
          </div>
          <p>{len(benchmark_results)} benchmark rows</p>
        </div>
        {_score_breakdown_table(benchmark_results)}
      </section>

      <section id="run-metadata" class="report-section">
        <div class="section-heading">
          <div>
            <p class="eyebrow">Run Metadata</p>
            <h2>Run Metadata</h2>
          </div>
        </div>
        {_report_metadata_section(summary, metadata)}
      </section>

      <section id="non-model-errors" class="report-section">
        <div class="section-heading">
          <div>
            <p class="eyebrow">Run Errors</p>
            <h2>Non-Model Run Errors</h2>
          </div>
        </div>
        {_non_model_error_section(benchmark_results)}
      </section>

      <section id="benchmark-citations" class="report-section citations-section">
        <div class="section-heading">
          <div>
            <p class="eyebrow">Benchmark Citations</p>
            <h2>Benchmark Citations</h2>
          </div>
        </div>
        {_citation_section(benchmark_results)}
      </section>
    </main>
  </div>
</body>
</html>
"""


def _report_css() -> str:
    return """
    :root {
      color-scheme: dark;
      --bg: #080d14;
      --bg-soft: #0d141f;
      --panel: #111a27;
      --panel-2: #162233;
      --panel-3: #1c2a3e;
      --line: #2c3b50;
      --line-strong: #3e536c;
      --text: #edf4ff;
      --muted: #9fb0c7;
      --subtle: #6f829c;
      --blue: #63a7ff;
      --green: #45d09a;
      --amber: #f4c15d;
      --red: #ff7f7f;
      --cyan: #67d7e8;
    }
    * { box-sizing: border-box; }
    html { background: var(--bg); }
    body {
      margin: 0;
      min-width: 320px;
      font: 14px/1.5 Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at 18% 0%, rgba(99, 167, 255, 0.18), transparent 34rem),
        linear-gradient(180deg, #0a1019 0%, var(--bg) 46rem);
    }
    a { color: var(--blue); }
    h1, h2, h3, p { margin: 0; }
    .shell { width: min(100%, 1440px); margin: 0 auto; padding: 24px clamp(14px, 3vw, 40px) 56px; }
    .report-header {
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(260px, 340px);
      gap: 20px;
      align-items: stretch;
      padding: clamp(22px, 4vw, 40px);
      border: 1px solid var(--line);
      border-radius: 8px;
      background: linear-gradient(135deg, rgba(17, 26, 39, 0.96), rgba(22, 34, 51, 0.88));
      box-shadow: 0 22px 70px rgba(0, 0, 0, 0.38);
    }
    .eyebrow {
      margin-bottom: 7px;
      color: var(--cyan);
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.14em;
      text-transform: uppercase;
    }
    h1 { font-size: clamp(28px, 4vw, 48px); line-height: 1.05; letter-spacing: 0; }
    h2 { font-size: clamp(22px, 2.2vw, 30px); line-height: 1.12; letter-spacing: 0; }
    h3 { font-size: 16px; line-height: 1.2; letter-spacing: 0; }
    .run-line { margin-top: 9px; color: var(--muted); overflow-wrap: anywhere; }
    .headline-panel {
      display: grid;
      align-content: center;
      gap: 8px;
      min-height: 170px;
      padding: 22px;
      border: 1px solid var(--line-strong);
      border-radius: 8px;
      background: #0b121d;
    }
    .headline-panel span,
    .headline-panel small { color: var(--muted); }
    .headline-panel strong { font-size: clamp(40px, 7vw, 68px); line-height: 0.95; letter-spacing: 0; color: var(--green); }
    main { display: grid; gap: 22px; margin-top: 22px; }
    .report-section {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: rgba(17, 26, 39, 0.92);
      overflow: hidden;
    }
    .section-heading {
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: end;
      padding: 22px 24px 16px;
      border-bottom: 1px solid var(--line);
    }
    .section-heading > p { max-width: 560px; color: var(--muted); text-align: right; }
    .cards {
      display: grid;
      grid-template-columns: repeat(6, minmax(0, 1fr));
      gap: 10px;
      padding: 20px 24px 0;
    }
    .card {
      min-width: 0;
      min-height: 110px;
      padding: 16px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel-2);
    }
    .card span { display: block; color: var(--muted); font-size: 12px; }
    .card strong { display: block; margin-top: 8px; font-size: clamp(20px, 2vw, 29px); line-height: 1.06; overflow-wrap: anywhere; }
    .summary-grid,
    .coverage-grid {
      display: grid;
      grid-template-columns: minmax(320px, 0.92fr) minmax(0, 1.38fr);
      gap: 16px;
      padding: 20px 24px 24px;
    }
    .radar-panel,
    .summary-panel,
    .coverage-panel {
      min-width: 0;
      padding: 18px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #0d1521;
    }
    .radar-panel svg { display: block; width: 100%; max-width: 440px; height: auto; margin: 10px auto 0; }
    .score-context { display: grid; gap: 12px; margin-top: 14px; }
    .context-row { display: grid; grid-template-columns: minmax(160px, 0.9fr) minmax(0, 1.2fr) auto; gap: 14px; align-items: center; }
    .context-row span { color: var(--muted); }
    .context-row strong { text-align: right; white-space: nowrap; }
    .bar {
      position: relative;
      height: 9px;
      overflow: hidden;
      border-radius: 999px;
      background: #233146;
    }
    .bar i { display: block; height: 100%; width: var(--value, 0%); border-radius: inherit; background: linear-gradient(90deg, var(--blue), var(--green)); }
    .coverage-grid { grid-template-columns: minmax(0, 1fr); }
    .coverage-cards {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
      padding: 20px 24px 0;
    }
    .coverage-card {
      min-height: 92px;
      padding: 15px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel-2);
    }
    .coverage-card span { color: var(--muted); font-size: 12px; }
    .coverage-card strong { display: block; margin-top: 7px; font-size: 24px; line-height: 1; }
    .table-wrap { width: 100%; overflow-x: auto; padding: 0 24px 24px; }
    table { width: 100%; border-collapse: collapse; min-width: 760px; }
    th, td { padding: 11px 12px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }
    th {
      position: sticky;
      top: 0;
      z-index: 1;
      color: var(--muted);
      background: var(--panel-3);
      font-size: 11px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      white-space: nowrap;
    }
    td { color: #dce8f7; overflow-wrap: anywhere; }
    tr:hover td { background: rgba(99, 167, 255, 0.06); }
    .score-breakdown { min-width: 1040px; }
    .score-value { display: grid; gap: 6px; min-width: 120px; }
    .score-value strong { font-size: 15px; color: var(--text); }
    .pill {
      display: inline-flex;
      align-items: center;
      min-height: 24px;
      max-width: 100%;
      padding: 2px 8px;
      border: 1px solid var(--line-strong);
      border-radius: 999px;
      background: #0b121d;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      white-space: normal;
    }
    .pill.pass { border-color: rgba(69, 208, 154, 0.52); color: var(--green); }
    .pill.warn { border-color: rgba(244, 193, 93, 0.55); color: var(--amber); }
    .pill.fail { border-color: rgba(255, 127, 127, 0.58); color: var(--red); }
    .official.included { color: var(--green); font-weight: 700; }
    .official.excluded { color: var(--amber); font-weight: 700; }
    .metadata {
      display: grid;
      grid-template-columns: minmax(180px, 0.28fr) minmax(0, 1fr);
      gap: 0;
      padding: 0 24px 24px;
      margin: 0;
    }
    .metadata dt,
    .metadata dd {
      min-width: 0;
      margin: 0;
      padding: 11px 12px;
      border-bottom: 1px solid var(--line);
      overflow-wrap: anywhere;
    }
    .metadata dt { color: var(--muted); background: #0d1521; }
    .metadata dd { color: var(--text); }
    .empty-state {
      margin: 0 24px 24px;
      padding: 18px;
      border: 1px solid var(--line);
      border-radius: 8px;
      color: var(--muted);
      background: #0d1521;
    }
    .citations-section { margin-bottom: 6px; }
    .bibtex {
      max-height: 180px;
      margin: 0 24px 24px;
      padding: 16px;
      overflow-y: auto;
      overflow-x: auto;
      scrollbar-gutter: stable;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #050912;
      color: #d8e7ff;
      font: 12px/1.55 ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
      white-space: pre;
    }
    .bibtex code { font: inherit; }
    @media (max-width: 1180px) {
      .cards { grid-template-columns: repeat(3, minmax(0, 1fr)); }
      .coverage-cards { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .summary-grid { grid-template-columns: 1fr; }
    }
    @media (max-width: 900px) {
      .shell { padding: 14px 10px 36px; }
      .report-header { grid-template-columns: 1fr; padding: 22px; }
      .headline-panel { min-height: 130px; }
      .section-heading { display: grid; align-items: start; padding: 20px 16px 14px; }
      .section-heading > p { text-align: left; }
      .cards, .coverage-cards { padding: 16px 16px 0; }
      .summary-grid, .coverage-grid { padding: 16px; }
      .table-wrap { padding: 0 16px 20px; }
      .metadata { grid-template-columns: 1fr; padding: 0 16px 20px; }
      .metadata dt { padding-bottom: 3px; border-bottom: 0; }
      .metadata dd { padding-top: 0; }
      .bibtex { margin: 0 16px 20px; }
    }
    @media (max-width: 760px) {
      .cards, .coverage-cards { grid-template-columns: 1fr 1fr; }
      .score-breakdown { min-width: 0; }
      .score-breakdown thead { display: none; }
      .score-breakdown,
      .score-breakdown tbody,
      .score-breakdown tr,
      .score-breakdown td { display: block; width: 100%; }
      .score-breakdown tr {
        margin-bottom: 12px;
        border: 1px solid var(--line);
        border-radius: 8px;
        background: #0d1521;
        overflow: hidden;
      }
      .score-breakdown td {
        display: grid;
        grid-template-columns: minmax(118px, 0.42fr) minmax(0, 1fr);
        gap: 12px;
        border-bottom: 1px solid var(--line);
      }
      .score-breakdown td::before {
        content: attr(data-label);
        color: var(--muted);
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
      }
    }
    @media (max-width: 560px) {
      .cards, .coverage-cards { grid-template-columns: 1fr; }
      .context-row { grid-template-columns: 1fr; gap: 6px; }
      .context-row strong { text-align: left; }
      .score-breakdown td { grid-template-columns: 1fr; gap: 6px; }
      h1 { font-size: 30px; }
    }
    """


def _report_rows(summary: dict[str, Any], results: list[GradeResult]) -> list[dict[str, Any]]:
    rows = summary.get("benchmark_results")
    if isinstance(rows, list) and rows:
        return [row for row in rows if isinstance(row, dict)]
    return _fallback_benchmark_rows(results)


def _fallback_benchmark_rows(results: list[GradeResult]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        details = result.details if isinstance(result.details, dict) else {}
        payload = details.get("result") if isinstance(details.get("result"), dict) else {}
        score = (float(result.score) / float(result.max_score) * 100.0) if result.max_score else 0.0
        rows.append(
            {
                "benchmark": _result_benchmark_name(result),
                "group": details.get("group", result.category),
                "profile": details.get("profile", result.category),
                "score": round(score, 4),
                "raw_score": _unit_to_percent(payload.get("raw_score")),
                "valid_score": _unit_to_percent(payload.get("valid_score")),
                "status": _result_status(result),
                "run_status": _run_status(result),
                "score_status": _score_status(result),
                "included_in_official_score": _included_in_official_score(result),
                "evaluated_task_count": payload.get("evaluated_task_count"),
                "evaluation_passed_count": payload.get("evaluation_passed_count"),
                "duration_seconds": result.task_duration_seconds,
                "error": result.error,
                "error_details": _result_error_reason(result),
                "blocker_type": _blocker_type(result),
                "homepage": details.get("homepage"),
                "license": details.get("license"),
                "credit": details.get("credit"),
                "citation": details.get("citation", details.get("homepage")),
            }
        )
    return rows


def _target_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    target = metadata.get("target_model")
    return target if isinstance(target, dict) else {}


def _judge_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    judge = metadata.get("judge")
    return judge if isinstance(judge, dict) else {}


def _headline_score(summary: dict[str, Any]) -> str:
    for key in ("score_valid_tasks_only", "valid_judged_score", "total_score", "benchmark_level_mean_score"):
        if summary.get(key) is not None:
            return _format_percent_display(summary.get(key))
    return "n/a"


def _report_metric_cards(summary: dict[str, Any], coverage: dict[str, Any]) -> str:
    cards = [
        ("Scored Suite", _headline_score(summary)),
        ("Conservative", _format_rate_auto(summary.get("conservative_all_suite_score"))),
        ("Suite Coverage", _format_rate_auto(coverage.get("coverage_rate", summary.get("suite_coverage_rate")))),
        ("Valid Suites", _coverage_value(coverage, "successfully_scored_benchmarks", "valid_judged_suite_count")),
        ("Items Passed", _format_integer_safe(summary.get("item_passed_count", summary.get("passed_count")))),
        ("Run Time", _format_seconds_safe(summary.get("total_run_duration_seconds", summary.get("run_duration_seconds")))),
    ]
    return '<div class="cards">' + "".join(_summary_card(label, value) for label, value in cards) + "</div>"


def _summary_card(label: str, value: str) -> str:
    return f'<div class="card"><span>{html.escape(label)}</span><strong>{html.escape(value)}</strong></div>'


def _summary_radar_scores(summary: dict[str, Any], benchmark_results: list[dict[str, Any]]) -> dict[str, float]:
    scores = summary.get("category_scores")
    if isinstance(scores, dict) and scores:
        return {str(key): _percent_number(value) for key, value in scores.items()}
    grouped: dict[str, list[float]] = {}
    for row in benchmark_results:
        group = str(row.get("group") or row.get("profile") or "Benchmarks")
        grouped.setdefault(group, []).append(_percent_number(row.get("score")))
    return {group: sum(values) / len(values) for group, values in grouped.items() if values}


def _score_context(summary: dict[str, Any], coverage: dict[str, Any]) -> str:
    rows = [
        ("Benchmark Level Mean", summary.get("benchmark_level_mean_score")),
        ("Model Valid Tasks", summary.get("model_score_valid_tasks_only")),
        ("Raw All Tasks", summary.get("raw_score_all_tasks", summary.get("raw_score"))),
        ("Item Coverage", summary.get("item_coverage_rate", coverage.get("item_coverage_rate"))),
        ("JSON Validity", summary.get("json_validity_rate")),
    ]
    rendered = []
    for label, value in rows:
        display = _format_rate_auto(value) if "Coverage" in label else _format_percent_display(value)
        rendered.append(
            '<div class="context-row">'
            f"<span>{html.escape(label)}</span>"
            f"{_progress_bar(value)}"
            f"<strong>{html.escape(display)}</strong>"
            "</div>"
        )
    return '<div class="score-context">' + "".join(rendered) + "</div>"


def _progress_bar(value: Any) -> str:
    width = max(0.0, min(100.0, _percent_number(value)))
    return f'<div class="bar" aria-hidden="true" style="--value: {width:.4f}%"><i></i></div>'


def _coverage_label(coverage: Any) -> str:
    if not isinstance(coverage, dict):
        return "coverage n/a"
    scored = coverage.get("successfully_scored_benchmarks", coverage.get("valid_judged_suite_count"))
    total = coverage.get("total_configured_benchmarks", coverage.get("suite_count"))
    if isinstance(scored, (int, float)) and isinstance(total, (int, float)) and total:
        return f"{int(scored)}/{int(total)} scored"
    rate = coverage.get("coverage_rate", coverage.get("suite_coverage_rate"))
    if rate is not None:
        return f"{_format_rate_auto(rate)} coverage"
    return "coverage n/a"


def _coverage_section(summary: dict[str, Any], coverage: dict[str, Any]) -> str:
    return (
        _coverage_overview_cards(summary, coverage)
        + '<div class="coverage-grid"><div class="coverage-panel">'
        + "<h3>Coverage By Category</h3>"
        + _coverage_table(coverage)
        + "</div></div>"
    )


def _coverage_overview_cards(summary: dict[str, Any], coverage: dict[str, Any]) -> str:
    cards = [
        ("Configured", _coverage_value(coverage, "total_configured_benchmarks", "suite_count")),
        ("Attempted", _coverage_value(coverage, "attempted_benchmarks", "task_count")),
        ("Scored", _coverage_value(coverage, "successfully_scored_benchmarks", "valid_judged_suite_count")),
        ("Failed", _coverage_value(coverage, "failed_benchmarks", "benchmark_item_invalid_count")),
    ]
    if not any(value != "n/a" for _, value in cards):
        cards = [
            ("Selected Suites", _format_integer_safe(summary.get("selected_suite_count"))),
            ("Known Suites", _format_integer_safe(summary.get("known_suite_count"))),
            ("Excluded Suites", _format_integer_safe(summary.get("excluded_suite_count"))),
            ("Coverage", _format_rate_auto(summary.get("suite_coverage_rate"))),
        ]
    return '<div class="coverage-cards">' + "".join(
        f'<div class="coverage-card"><span>{html.escape(label)}</span><strong>{html.escape(value)}</strong></div>'
        for label, value in cards
    ) + "</div>"


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
            f"<td>{_format_integer_safe(data.get('total_configured_benchmarks'))}</td>"
            f"<td>{_format_integer_safe(data.get('attempted_benchmarks'))}</td>"
            f"<td>{_format_integer_safe(data.get('successfully_scored_benchmarks'))}</td>"
            f"<td>{_format_integer_safe(data.get('failed_benchmarks'))}</td>"
            f"<td>{html.escape(_format_rate_auto(data.get('coverage_rate')))}</td>"
            "</tr>"
        )
    if not rows:
        return '<div class="empty-state">No category coverage data recorded.</div>'
    return (
        '<div class="table-wrap"><table><thead><tr><th>Category</th><th>Configured</th>'
        "<th>Attempted</th><th>Scored</th><th>Failed</th><th>Coverage</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _score_breakdown_table(benchmark_results: list[dict[str, Any]]) -> str:
    if not benchmark_results:
        return '<div class="empty-state">No benchmark score rows were recorded.</div>'
    rows = []
    for row in benchmark_results:
        benchmark = str(row.get("benchmark") or row.get("suite_id") or row.get("task_id") or "")
        profile = str(row.get("profile") or row.get("group") or "")
        score = row.get("score")
        raw_score = row.get("raw_score")
        valid_score = row.get("valid_score")
        included = bool(row.get("included_in_official_score"))
        official_class = "included" if included else "excluded"
        official_label = "Included" if included else "Excluded"
        rows.append(
            "<tr>"
            f'<td data-label="Benchmark">{html.escape(benchmark)}</td>'
            f'<td data-label="Profile">{html.escape(profile)}</td>'
            f'<td data-label="Normalized 0-100">{_score_bar(score)}</td>'
            f'<td data-label="Raw Score">{html.escape(_format_percent_display(raw_score))}</td>'
            f'<td data-label="Valid Score">{html.escape(_format_percent_display(valid_score))}</td>'
            f'<td data-label="Official Score"><span class="official {official_class}">{html.escape(official_label)}</span></td>'
            f'<td data-label="Status">{_status_pill(str(row.get("status") or ""))}</td>'
            f'<td data-label="Run">{html.escape(str(row.get("run_status") or ""))}</td>'
            f'<td data-label="Score Status">{html.escape(str(row.get("score_status") or ""))}</td>'
            f'<td data-label="Items">{html.escape(_items_summary(row))}</td>'
            "</tr>"
        )
    return (
        '<div class="table-wrap"><table class="score-breakdown"><thead><tr>'
        "<th>Benchmark</th><th>Profile</th><th>Normalized 0-100</th><th>Raw Score</th>"
        "<th>Valid Score</th><th>Official Score</th><th>Status</th><th>Run</th>"
        "<th>Score Status</th><th>Items</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _status_pill(status: str) -> str:
    normalized = normalize_status(status)
    info = status_info(normalized)
    css_class = "pass" if info.failure_class == "none" else "fail" if info.failure_class == "model" else "warn"
    if normalized in {PASSED, SCORE_PASSED}:
        css_class = "pass"
    if normalized in {SCORE_PARTIALLY_CORRECT}:
        css_class = "warn"
    label = normalized or "unknown"
    return f'<span class="pill {css_class}">{html.escape(label)}</span>'


def _score_bar(value: Any) -> str:
    percent = _percent_number(value)
    return (
        '<div class="score-value">'
        f"<strong>{html.escape(_format_percent_display(value))}</strong>"
        f"{_progress_bar(percent)}"
        "</div>"
    )


def _items_summary(row: dict[str, Any]) -> str:
    passed = row.get("evaluation_passed_count", row.get("passed_items"))
    evaluated = row.get("evaluated_task_count", row.get("total_items"))
    valid = row.get("valid_evaluated_task_count")
    if passed is not None and evaluated is not None:
        suffix = f", {int(valid)} valid" if isinstance(valid, (int, float)) else ""
        return f"{int(passed)}/{int(evaluated)}{suffix}"
    status_counts = row.get("status_counts")
    if isinstance(status_counts, dict) and status_counts:
        return ", ".join(f"{key}: {value}" for key, value in sorted(status_counts.items()))
    return "n/a"


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
        "target_provider": target.get("provider_type", metadata.get("provider")),
        "target_base_url": target.get("base_url", metadata.get("base_url")),
        "target_model": target.get("model", metadata.get("model")),
        "request_concurrency": metadata.get("request_concurrency"),
        "eval_concurrency": metadata.get("eval_concurrency"),
        "model_request_timeout": metadata.get("model_request_timeout"),
        "external_timeout": metadata.get("external_timeout"),
        "selected_profile": metadata.get("selected_profile"),
        "selected_suite_count": metadata.get("selected_suite_count", summary.get("selected_suite_count")),
        "known_suite_count": metadata.get("known_suite_count", summary.get("known_suite_count")),
        "excluded_suite_count": metadata.get("excluded_suite_count", summary.get("excluded_suite_count")),
        "temperature": target.get("temperature", metadata.get("temperature")),
        "top_p": target.get("top_p", metadata.get("top_p")),
        "max_tokens": target.get("max_tokens", metadata.get("max_tokens")),
        "tool_parser": target.get("tool_parser", metadata.get("tool_parser")),
        "context_window": target.get("context_window", metadata.get("context_window")),
        "judge_provider": judge.get("provider"),
        "judge_base_url": judge.get("base_url"),
        "judge_model": judge.get("model"),
        "judge_timeout": judge.get("timeout_seconds"),
        "judge_fallback_used": judge.get("fallback_used"),
        "request_concurrency": metadata.get("request_concurrency"),
        "eval_concurrency": metadata.get("eval_concurrency"),
        "timeout_seconds": metadata.get("timeout"),
        "sandbox": metadata.get("sandbox"),
        "allow_host_docker_socket": metadata.get("allow_host_docker_socket"),
        "git_commit": metadata.get("git_commit"),
    }
    return _metadata_definition_list(_compact_metadata_rows(rows))


def _metadata_definition_list(values: dict[str, Any]) -> str:
    rows = []
    for key, value in values.items():
        rows.append(f"<dt>{html.escape(str(key))}</dt><dd>{html.escape(_display_value(value))}</dd>")
    if not rows:
        return '<div class="empty-state">No run metadata recorded.</div>'
    return f'<dl class="metadata">{"".join(rows)}</dl>'


def _compact_metadata_rows(values: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value not in (None, "", [], {})}


def _non_model_error_section(benchmark_results: list[dict[str, Any]]) -> str:
    rows = _non_model_error_rows(benchmark_results)
    if not rows:
        return '<div class="empty-state">No non-model run errors were recorded.</div>'
    rendered = []
    for row in rows:
        status = str(row.get("status") or "")
        info = status_info(status)
        rendered.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('benchmark') or row.get('suite_id') or row.get('task_id') or ''))}</td>"
            f"<td>{_status_pill(status)}</td>"
            f"<td>{html.escape(_report_failure_class(row))}</td>"
            f"<td>{html.escape(_display_error(str(row.get('error_details') or row.get('error') or info.explanation or '')))}</td>"
            f"<td>{'no' if row.get('included_in_official_score') else 'yes'}</td>"
            f"<td>{html.escape(_suggested_action(row))}</td>"
            "</tr>"
        )
    return (
        '<div class="table-wrap"><table><thead><tr><th>Benchmark</th><th>Status</th>'
        "<th>Failure Class</th><th>Explanation</th><th>Excluded</th><th>Suggested Action</th></tr></thead>"
        f"<tbody>{''.join(rendered)}</tbody></table></div>"
    )


def _non_model_error_rows(benchmark_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in benchmark_results if _is_non_model_error_row(row)]


def _is_non_model_error_row(row: dict[str, Any]) -> bool:
    status = normalize_status(row.get("status"))
    info = status_info(status)
    if info.failure_class in {"none", "model"}:
        return False
    if not row.get("included_in_official_score"):
        return True
    if row.get("blocker_type") or row.get("error") or row.get("error_details"):
        return True
    return info.failure_class not in {"none", "model"}


def _report_failure_class(row: dict[str, Any]) -> str:
    blocker = row.get("blocker_type")
    if isinstance(blocker, str) and blocker.strip():
        return blocker.strip()
    return status_info(row.get("status")).failure_class


def _suggested_action(row: dict[str, Any]) -> str:
    blocker = _report_failure_class(row)
    if blocker in {"missing_asset", "git_lfs_pointer_stub"}:
        return "Materialize benchmark assets and rerun validation."
    if blocker in {"missing_required_tool", "unsupported_capability"}:
        return "Check adapter tool exposure and benchmark capability declarations."
    if blocker in {"missing_grader", "disabled_scoring", "judge_parse_error"}:
        return "Inspect judge/grader configuration and rerun the affected benchmark."
    if blocker in {"benchmark_setup", "repo_patch_harness_setup", "invalid_task_context"}:
        return "Fix the benchmark manifest or harness setup, then rerun."
    if blocker in {"infrastructure", "execution_error"}:
        return "Inspect container, timeout, and runner logs for the benchmark."
    return "Inspect graded_results.jsonl and benchmark logs."


def _citation_section(benchmark_results: list[dict[str, Any]]) -> str:
    catalog = _benchmark_citation_catalog()
    if not catalog:
        catalog = "\n\n".join(_bibtex_entry(row) for row in benchmark_results if isinstance(row, dict))
    catalog = catalog.strip() + "\n" if catalog.strip() else "% No benchmark citations were recorded.\n"
    return _code_block(catalog)


def _benchmark_citation_catalog() -> str:
    if not BENCHMARK_CITATIONS_PATH.exists():
        return ""
    return BENCHMARK_CITATIONS_PATH.read_text(encoding="utf-8")


def _bibtex_entry(row: dict[str, Any]) -> str:
    benchmark = str(row.get("benchmark") or row.get("suite_id") or row.get("task_id") or "benchmark")
    title = benchmark
    author = str(row.get("credit") or f"{benchmark} authors")
    url = str(row.get("citation") or row.get("homepage") or "")
    license_value = str(row.get("license") or "Unspecified")
    key = _bibtex_entry_key(benchmark)
    lines = [
        f"@misc{{{key},",
        f"  title = {{{_bibtex_value(title)}}},",
        f"  author = {{{_bibtex_value(author)}}},",
    ]
    if url:
        lines.append(f"  url = {{{_bibtex_value(url)}}},")
    lines.append(f"  license = {{{_bibtex_value(license_value)}}}")
    lines.append("}")
    return "\n".join(lines)


def _bibtex_entry_key(value: str) -> str:
    key = []
    for char in value.lower():
        if char.isalnum():
            key.append(char)
        elif key and key[-1] != "_":
            key.append("_")
    return "".join(key).strip("_") or "benchmark"


def _bibtex_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")


def _code_block(content: str) -> str:
    return f'<pre class="bibtex" tabindex="0"><code>{html.escape(content)}</code></pre>'


def _coverage_value(coverage: dict[str, Any], primary: str, fallback: str) -> str:
    value = coverage.get(primary, coverage.get(fallback))
    return _format_integer_safe(value)


def _format_rate_auto(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{_percent_number(value):.2f}%"


def _format_percent_display(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{_percent_number(value):.2f}%"


def _format_seconds_safe(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.2f}s"
    except (TypeError, ValueError):
        return "n/a"


def _format_integer_safe(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return "n/a"


def _display_value(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _percent_number(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if 0.0 <= number <= 1.0:
        return number * 100.0
    return number


def _join_if(values: list[Any], separator: str) -> str:
    return separator.join(str(value) for value in values if value not in (None, "", [], {}))


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
            f'<polygon points="{_points(points)}" fill="none" stroke="#33445c" stroke-width="1" />'
        )

    axis_lines = []
    label_nodes = []
    for index, label in enumerate(labels):
        end = _point(cx, cy, radius, index, axes)
        label_point = _point(cx, cy, radius + 28.0, index, axes)
        axis_lines.append(
            f'<line x1="{cx:.1f}" y1="{cy:.1f}" x2="{end[0]:.1f}" y2="{end[1]:.1f}" stroke="#506783" />'
        )
        label_nodes.append(
            f'<text x="{label_point[0]:.1f}" y="{label_point[1]:.1f}" text-anchor="middle" dominant-baseline="middle" font-size="10" fill="#c8d6e8">{html.escape(label)}</text>'
        )

    value_points = [_point(cx, cy, radius * (value / 100.0), index, axes) for index, value in enumerate(values)]
    return (
        '<svg viewBox="0 0 320 260" role="img" aria-label="Radar chart of category scores">'
        + "".join(grid_polygons)
        + "".join(axis_lines)
        + f'<polygon points="{_points(value_points)}" fill="#63a7ff" fill-opacity="0.24" stroke="#63a7ff" stroke-width="2" />'
        + "".join(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="#45d09a" />' for x, y in value_points
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
