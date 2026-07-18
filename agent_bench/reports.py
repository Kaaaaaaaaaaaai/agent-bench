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
    "benchmark_name",
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
    metadata = summary.get("metadata", {})
    benchmark_results = summary.get("benchmark_results", [])
    coverage = summary.get("coverage_summary", summary.get("coverage", {}))
    target = metadata.get("target_model") if isinstance(metadata.get("target_model"), dict) else {}
    run_id = metadata.get("run_id") or metadata.get("output_dir") or "agent-bench-run"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Agent Bench Report</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #090d13;
      --surface: #111821;
      --surface-2: #17212d;
      --ink: #e8edf4;
      --muted: #98a7ba;
      --line: #293646;
      --blue: #6ea8fe;
      --cyan: #55d6be;
      --green: #59d499;
      --amber: #f0bd63;
      --red: #ff7d86;
    }}
    * {{ box-sizing: border-box; }}
    html {{ background: var(--bg); }}
    body {{ margin: 0; font: 14px/1.5 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: var(--ink); background: var(--bg); }}
    header {{ border-bottom: 1px solid var(--line); background: #0d131c; }}
    .header-inner, main {{ width: min(1480px, 100%); margin: 0 auto; }}
    .header-inner {{ display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 24px; align-items: end; padding: 30px 32px 24px; }}
    h1 {{ margin: 0 0 8px; font-size: 30px; letter-spacing: 0; }}
    h2 {{ margin: 0 0 14px; font-size: 19px; letter-spacing: 0; }}
    main {{ padding: 26px 32px 52px; }}
    section {{ margin-top: 32px; }}
    a {{ color: var(--blue); }}
    .muted {{ color: var(--muted); }}
    .eyebrow {{ color: var(--cyan); font-size: 12px; font-weight: 700; text-transform: uppercase; }}
    .run-line {{ display: flex; flex-wrap: wrap; gap: 6px 18px; color: var(--muted); }}
    .headline-score {{ text-align: right; }}
    .headline-score strong {{ display: block; color: var(--cyan); font-size: 36px; line-height: 1.1; }}
    .summary-layout {{ display: grid; grid-template-columns: minmax(0, 1fr) minmax(340px, 460px); gap: 20px; align-items: stretch; }}
    .cards {{ display: grid; grid-template-columns: repeat(3, minmax(130px, 1fr)); gap: 10px; }}
    .card, .radar, .coverage-card {{ border: 1px solid var(--line); border-radius: 8px; background: var(--surface); }}
    .card {{ min-height: 100px; padding: 16px; }}
    .card span {{ display: block; color: var(--muted); font-size: 12px; }}
    .card strong {{ display: block; margin-top: 9px; font-size: 23px; font-weight: 650; }}
    .radar {{ display: grid; place-items: center; min-height: 320px; padding: 12px; }}
    .radar svg {{ width: 100%; max-width: 430px; height: auto; }}
    .coverage-overview {{ display: grid; grid-template-columns: repeat(4, minmax(140px, 1fr)); gap: 10px; margin-bottom: 12px; }}
    .coverage-card {{ padding: 13px 15px; }}
    .coverage-card span {{ display: block; color: var(--muted); font-size: 12px; }}
    .coverage-card strong {{ display: block; margin-top: 4px; font-size: 20px; }}
    table {{ width: 100%; border-collapse: collapse; border: 1px solid var(--line); background: var(--surface); }}
    th, td {{ padding: 10px 12px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: middle; }}
    th {{ background: var(--surface-2); color: #bdc9d8; font-size: 11px; text-transform: uppercase; white-space: nowrap; }}
    td {{ overflow-wrap: anywhere; }}
    tbody tr:hover {{ background: #151e29; }}
    .table-wrap {{ overflow-x: auto; border-radius: 8px; }}
    .score-bar {{ display: grid; grid-template-columns: minmax(74px, 1fr) 52px; gap: 9px; align-items: center; min-width: 138px; }}
    .score-track {{ height: 6px; overflow: hidden; border-radius: 3px; background: #273343; }}
    .score-fill {{ display: block; height: 100%; background: var(--cyan); }}
    .pill {{ display: inline-block; border: 1px solid var(--line); border-radius: 999px; padding: 2px 8px; white-space: nowrap; }}
    .pill.good {{ color: var(--green); border-color: #276149; background: #10271f; }}
    .pill.warn {{ color: var(--amber); border-color: #6b5129; background: #2a2113; }}
    .pill.bad {{ color: var(--red); border-color: #71343b; background: #2b171b; }}
    .metadata {{ display: grid; grid-template-columns: minmax(160px, 220px) minmax(0, 1fr); gap: 0; margin: 0; overflow: hidden; border: 1px solid var(--line); border-radius: 8px; background: var(--surface); }}
    .metadata dt {{ color: var(--muted); }}
    .metadata dt, .metadata dd {{ padding: 10px 14px; border-bottom: 1px solid var(--line); }}
    .metadata dd {{ margin: 0; overflow-wrap: anywhere; }}
    .error-list {{ display: grid; gap: 8px; }}
    .error-row {{ display: grid; grid-template-columns: minmax(180px, 0.8fr) minmax(170px, 0.7fr) minmax(260px, 2fr); gap: 14px; padding: 12px 14px; border: 1px solid var(--line); border-radius: 8px; background: var(--surface); }}
    .error-row strong {{ color: var(--red); }}
    pre.bibtex {{ max-height: 180px; margin: 0; overflow-y: auto; overflow-x: auto; padding: 16px; border: 1px solid var(--line); border-radius: 8px; color: #c9d5e3; background: #070a0f; font: 12px/1.55 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; white-space: pre; }}
    @media (max-width: 1180px) {{ .summary-layout {{ grid-template-columns: 1fr; }} .radar {{ min-height: 280px; }} }}
    @media (max-width: 900px) {{ .cards {{ grid-template-columns: repeat(2, minmax(130px, 1fr)); }} .coverage-overview {{ grid-template-columns: repeat(2, minmax(140px, 1fr)); }} }}
    @media (max-width: 760px) {{
      .header-inner {{ grid-template-columns: 1fr; padding: 24px 18px 20px; }} .headline-score {{ text-align: left; }} main {{ padding: 20px 14px 40px; }}
      .score-table table, .score-table tbody, .score-table tr, .score-table td {{ display: block; }} .score-table thead {{ display: none; }} .score-table table {{ border: 0; background: transparent; }}
      .score-table tr {{ margin-bottom: 10px; overflow: hidden; border: 1px solid var(--line); border-radius: 8px; background: var(--surface); }}
      .score-table td {{ display: grid; grid-template-columns: 124px minmax(0, 1fr); gap: 10px; min-height: 40px; padding: 9px 12px; }}
      .score-table td::before {{ content: attr(data-label); color: var(--muted); font-size: 11px; text-transform: uppercase; }}
      .error-row {{ grid-template-columns: 1fr; gap: 6px; }}
    }}
    @media (max-width: 560px) {{ h1 {{ font-size: 25px; }} .cards, .coverage-overview {{ grid-template-columns: 1fr; }} .metadata {{ grid-template-columns: 1fr; }} .metadata dt {{ padding-bottom: 2px; border-bottom: 0; }} .metadata dd {{ padding-top: 2px; }} }}
  </style>
</head>
<body>
  <header>
    <div class="header-inner">
      <div>
        <div class="eyebrow">Evaluation report</div>
        <h1>Agent Bench Report</h1>
        <div class="run-line"><span>{html.escape(str(target.get("model") or metadata.get("model", "unknown model")))}</span><span>Run {html.escape(str(run_id))}</span><span>{html.escape(str(metadata.get("created_at_utc", "")))}</span></div>
      </div>
      <div class="headline-score">
        <span class="muted">Scored-suite score</span>
        <strong>{_format_percent(summary.get("score_valid_tasks_only", summary.get("total_score")))}</strong>
        <span class="muted">{html.escape(_coverage_label(coverage))}</span>
      </div>
    </div>
  </header>
  <main>
    <section>
      <h2>Result Summary</h2>
      <div class="summary-layout">
        {_final_metric_cards(summary)}
        <div class="radar">{_radar_svg(_final_radar_scores(summary, benchmark_results))}</div>
      </div>
    </section>
    <section>
      <h2>Benchmark Coverage</h2>
      {_final_coverage(coverage)}
    </section>
    <section>
      <h2>Benchmark Score Breakdown</h2>
      {_final_score_table(benchmark_results)}
    </section>
    <section>
      <h2>Run Metadata</h2>
      {_final_metadata(metadata)}
    </section>
    <section>
      <h2>Non-Model Run Errors</h2>
      {_final_errors(benchmark_results)}
    </section>
    <section>
      <h2>Benchmark Citations</h2>
      <pre class="bibtex" tabindex="0" aria-label="BibTeX citations">{html.escape(_citation_catalog())}</pre>
    </section>
  </main>
</body>
</html>
"""


def _final_metric_cards(summary: dict[str, Any]) -> str:
    coverage = summary.get("coverage_summary") if isinstance(summary.get("coverage_summary"), dict) else {}
    cards = [
        ("Scored-suite score", _format_percent(summary.get("score_valid_tasks_only", summary.get("total_score")))),
        ("Conservative score", _format_rate(summary.get("conservative_all_suite_score"))),
        ("Suite coverage", _format_rate(coverage.get("coverage_rate", summary.get("suite_coverage_rate")))),
        ("Item coverage", _format_rate(summary.get("item_coverage_rate"))),
        ("Benchmarks scored", f"{coverage.get('successfully_scored_benchmarks', 0)} / {coverage.get('total_configured_benchmarks', 0)}"),
        ("Run duration", _format_seconds(summary.get("total_run_duration_seconds"))),
    ]
    return '<div class="cards">' + "".join(
        f'<div class="card"><span>{html.escape(label)}</span><strong>{html.escape(value)}</strong></div>'
        for label, value in cards
    ) + "</div>"


def _final_radar_scores(summary: dict[str, Any], benchmark_results: Any) -> dict[str, float]:
    scores = summary.get("category_scores")
    if isinstance(scores, dict) and scores:
        return scores
    grouped: dict[str, list[float]] = {}
    if isinstance(benchmark_results, list):
        for row in benchmark_results:
            if not isinstance(row, dict) or not row.get("included_in_official_score"):
                continue
            group = str(row.get("group") or "Other")
            try:
                grouped.setdefault(group, []).append(float(row.get("score") or 0.0))
            except (TypeError, ValueError):
                continue
    return {group: sum(values) / len(values) for group, values in grouped.items() if values}


def _final_coverage(coverage: Any) -> str:
    if not isinstance(coverage, dict):
        return '<div class="coverage-card">No coverage data recorded.</div>'
    overview = [
        ("Configured", coverage.get("total_configured_benchmarks", 0)),
        ("Attempted", coverage.get("attempted_benchmarks", 0)),
        ("Scored", coverage.get("successfully_scored_benchmarks", 0)),
        ("Excluded", coverage.get("excluded_from_score_benchmarks", coverage.get("failed_benchmarks", 0))),
    ]
    cards = '<div class="coverage-overview">' + "".join(
        f'<div class="coverage-card"><span>{html.escape(label)}</span><strong>{html.escape(str(value))}</strong></div>'
        for label, value in overview
    ) + "</div>"
    per_category = coverage.get("per_category")
    rows = []
    if isinstance(per_category, dict):
        for category, data in sorted(per_category.items()):
            if not isinstance(data, dict):
                continue
            rows.append(
                "<tr>"
                f"<td>{html.escape(str(category))}</td>"
                f"<td>{int(data.get('total_configured_benchmarks', 0))}</td>"
                f"<td>{int(data.get('attempted_benchmarks', 0))}</td>"
                f"<td>{int(data.get('successfully_scored_benchmarks', 0))}</td>"
                f"<td>{int(data.get('failed_benchmarks', 0))}</td>"
                f"<td>{_format_rate(data.get('coverage_rate'))}</td>"
                "</tr>"
            )
    body = "".join(rows) or '<tr><td colspan="6">No category coverage data recorded.</td></tr>'
    return cards + '<div class="table-wrap"><table><thead><tr><th>Category</th><th>Configured</th><th>Attempted</th><th>Scored</th><th>Excluded</th><th>Coverage</th></tr></thead>' + f"<tbody>{body}</tbody></table></div>"


def _final_score_table(benchmark_results: Any) -> str:
    if not isinstance(benchmark_results, list) or not benchmark_results:
        return '<div class="coverage-card">No benchmark rows were recorded.</div>'
    rows = []
    columns = ["Benchmark", "Profile", "Normalized 0-100", "Raw Score", "Valid Score", "Official Score", "Status", "Run", "Score Status", "Items"]
    for row in benchmark_results:
        if not isinstance(row, dict):
            continue
        score = _number(row.get("score"))
        values = [
            html.escape(str(row.get("benchmark") or row.get("task_id") or "")),
            html.escape(str(row.get("profile") or "")),
            _score_bar(score),
            html.escape(_format_percent(row.get("raw_score"))),
            html.escape(_format_percent(row.get("valid_score"))),
            "yes" if row.get("included_in_official_score") else "no",
            _status_pill(str(row.get("status") or "")),
            html.escape(str(row.get("run_status") or "")),
            html.escape(str(row.get("score_status") or "")),
            html.escape(_item_count(row)),
        ]
        rows.append("<tr>" + "".join(f'<td data-label="{html.escape(label, quote=True)}">{value}</td>' for label, value in zip(columns, values)) + "</tr>")
    headers = "".join(f"<th>{html.escape(column)}</th>" for column in columns)
    return f'<div class="table-wrap score-table"><table><thead><tr>{headers}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def _number(value: Any) -> float:
    try:
        return max(0.0, min(100.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _score_bar(score: float) -> str:
    return f'<div class="score-bar"><span class="score-track"><span class="score-fill" style="width:{score:.2f}%"></span></span><strong>{score:.2f}</strong></div>'


def _status_pill(status: str) -> str:
    info = status_info(status)
    tone = "good" if info.failure_class == "none" else "bad" if info.failure_class == "infrastructure" else "warn"
    return f'<span class="pill {tone}">{html.escape(status or "unknown")}</span>'


def _item_count(row: dict[str, Any]) -> str:
    for key in ("task_count", "attempted_task_count", "valid_evaluated_task_count"):
        value = row.get(key)
        if isinstance(value, (int, float)):
            return str(int(value))
    counts = row.get("status_counts")
    if isinstance(counts, dict):
        return str(sum(int(value) for value in counts.values() if isinstance(value, (int, float))))
    return "n/a"


def _final_metadata(metadata: dict[str, Any]) -> str:
    target = metadata.get("target_model") if isinstance(metadata.get("target_model"), dict) else {}
    judge = metadata.get("judge") if isinstance(metadata.get("judge"), dict) else {}
    host = metadata.get("host") if isinstance(metadata.get("host"), dict) else {}
    values = {
        "Run ID": metadata.get("run_id"),
        "Created (UTC)": metadata.get("created_at_utc"),
        "Profile": metadata.get("selected_profile"),
        "Provider": target.get("provider_type", metadata.get("provider")),
        "Model": target.get("model", metadata.get("model")),
        "Base URL": target.get("base_url", metadata.get("base_url")),
        "Temperature": target.get("temperature", metadata.get("temperature")),
        "Max tokens": target.get("max_tokens", metadata.get("max_tokens")),
        "Request concurrency": target.get("concurrency", metadata.get("request_concurrency")),
        "Evaluation concurrency": metadata.get("eval_concurrency"),
        "Judge": judge.get("model") or judge.get("provider") or "none",
        "Judge fallback used": judge.get("fallback_used"),
        "Host platform": host.get("platform"),
        "Python": host.get("python_version"),
        "Output directory": metadata.get("output_dir"),
    }
    return _metadata_definition_list(values)


def _final_errors(benchmark_results: Any) -> str:
    rows = []
    if isinstance(benchmark_results, list):
        for row in benchmark_results:
            if not isinstance(row, dict):
                continue
            status = str(row.get("status") or "")
            info = status_info(status)
            if info.failure_class in {"none", "model"}:
                continue
            explanation = _display_error(str(row.get("error_details") or row.get("error") or info.explanation))
            rows.append(
                '<div class="error-row">'
                f'<div>{html.escape(str(row.get("benchmark") or row.get("task_id") or ""))}</div>'
                f'<div><strong>{html.escape(status)}</strong><br><span class="muted">{html.escape(info.failure_class)}</span></div>'
                f'<div>{html.escape(explanation)}</div>'
                "</div>"
            )
    if not rows:
        return '<div class="coverage-card">No non-model errors were recorded.</div>'
    return '<div class="error-list">' + "".join(rows) + "</div>"


def _citation_catalog() -> str:
    try:
        return BENCHMARK_CITATIONS_PATH.read_text(encoding="utf-8").rstrip()
    except OSError:
        return "% Citation catalog unavailable: tasks/benchmark_citations.bib was not found."


def _coverage_label(coverage: Any) -> str:
    if not isinstance(coverage, dict):
        return "coverage n/a"
    scored = coverage.get("successfully_scored_benchmarks", coverage.get("valid_judged_suite_count"))
    total = coverage.get("total_configured_benchmarks", coverage.get("suite_count"))
    if isinstance(scored, int) and isinstance(total, int):
        return f"{scored}/{total} scored"
    return "coverage n/a"


def _report_metric_cards(summary: dict[str, Any]) -> str:
    coverage = summary.get("coverage_summary", {})
    metadata = summary.get("metadata", {})
    judge = metadata.get("judge") if isinstance(metadata.get("judge"), dict) else {}
    cards = [
        ("Scored-Suite", _format_percent(summary.get("score_valid_tasks_only", summary.get("total_score")))),
        ("Official Valid", f"{coverage.get('successfully_scored_benchmarks', summary.get('valid_task_count', 0))}"),
        ("Coverage", _format_rate(coverage.get("coverage_rate", summary.get("suite_coverage_rate")))),
        ("Failed", f"{coverage.get('failed_benchmarks', 0)}"),
        ("Judge", str(judge.get("model") or judge.get("provider") or "none")),
        ("Duration", _format_seconds(summary.get("total_run_duration_seconds"))),
    ]
    return '<section class="cards">' + "".join(
        f'<div class="card"><span>{html.escape(label)}</span><strong>{html.escape(value)}</strong></div>'
        for label, value in cards
    ) + "</section>"


def _coverage_table(coverage: Any) -> str:
    if not isinstance(coverage, dict):
        return "<table><tbody><tr><td>No coverage data recorded.</td></tr></tbody></table>"
    per_category = coverage.get("per_category")
    if not isinstance(per_category, dict) or not per_category:
        return "<table><tbody><tr><td>No category coverage data recorded.</td></tr></tbody></table>"
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
            f"<td>{_format_rate(data.get('coverage_rate'))}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Category</th><th>Configured</th><th>Scored</th>"
        "<th>Failed</th><th>Coverage</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _report_benchmark_table(benchmark_results: Any, metadata: dict[str, Any]) -> str:
    if not isinstance(benchmark_results, list) or not benchmark_results:
        return "<table><tbody><tr><td>No benchmark rows were recorded.</td></tr></tbody></table>"
    artifact_paths = metadata.get("artifact_paths") if isinstance(metadata.get("artifact_paths"), dict) else {}
    rows: list[str] = []
    current_group: str | None = None
    for row in benchmark_results:
        if not isinstance(row, dict):
            continue
        group = str(row.get("group", "Other"))
        if group != current_group:
            rows.append(f'<tr class="group-row"><td colspan="26">{html.escape(group)}</td></tr>')
            current_group = group
        status = str(row.get("status") or "")
        status_class = _status_class(status)
        setup_details = row.get("setup_details")
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('benchmark', '')))}</td>"
            f"<td>{html.escape(group)}</td>"
            f'<td class="{status_class}">{html.escape(status)}</td>'
            f"<td>{_format_percent(row.get('raw_score'))}</td>"
            f"<td>{_format_percent(row.get('score'))}</td>"
            f"<td>{'yes' if row.get('included_in_official_score') else 'no'}</td>"
            f"<td>{'scored' if row.get('included_in_official_score') else 'excluded'}</td>"
            f"<td>{'yes' if row.get('capabilities_verified') else 'no'}</td>"
            f"<td>{html.escape(_short_json(row.get('required_capabilities')))}</td>"
            f"<td>{html.escape(_short_json(row.get('supported_capabilities')))}</td>"
            f"<td>{html.escape(_short_json(row.get('required_tools')))}</td>"
            f"<td>{html.escape(_short_json(row.get('exposed_tools')))}</td>"
            f"<td>{html.escape(_short_json(row.get('missing_tools')))}</td>"
            f"<td>{html.escape(_short_json(row.get('missing_env')))}</td>"
            f"<td>{int(row.get('missing_assets_count') or 0)}</td>"
            f"<td>{html.escape(str(row.get('docker_image') or ''))}</td>"
            f"<td>{html.escape(str(row.get('container_name') or ''))}</td>"
            f"<td>{html.escape(str(row.get('network_mode') or ''))}</td>"
            f"<td>{html.escape(_short_json(row.get('docker_socket_mount')))}</td>"
            f"<td>{html.escape(_short_json(row.get('output_mount')))}</td>"
            f"<td>{html.escape(_short_json(row.get('asset_cache_mount')))}</td>"
            f"<td>{html.escape(_short_json(setup_details))}</td>"
            f"<td>{_format_seconds(row.get('duration_seconds'))}</td>"
            f"<td>{_artifact_link(artifact_paths.get('raw_responses'), 'raw')}</td>"
            f"<td>{_artifact_link(artifact_paths.get('graded_results'), 'graded')}</td>"
            f"<td>{html.escape(_display_error(str(row.get('error_details') or row.get('error') or '')))}</td>"
            "</tr>"
        )
    return (
        '<div class="table-wrap"><table><thead><tr><th>Benchmark</th><th>Task Group</th>'
        "<th>Status</th><th>Raw Score</th><th>Normalized 0-100</th><th>Official Score</th>"
        "<th>Coverage</th><th>Capabilities Verified</th><th>Required Capabilities</th>"
        "<th>Supported Capabilities</th><th>Required Tools</th><th>Exposed Tools</th>"
        "<th>Missing Tools</th><th>Missing Env</th><th>Missing Assets</th>"
        "<th>Image</th><th>Container</th><th>Network</th><th>Docker Socket</th>"
        "<th>Output Mount</th><th>Asset Cache</th><th>Setup Details</th>"
        "<th>Duration</th><th>Raw Responses</th><th>Graded Results</th><th>Notes</th>"
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


def _report_metadata_tables(metadata: dict[str, Any], benchmark_results: Any) -> str:
    target = metadata.get("target_model") if isinstance(metadata.get("target_model"), dict) else {}
    judge = metadata.get("judge") if isinstance(metadata.get("judge"), dict) else {}
    rows = {
        "run_id": metadata.get("run_id"),
        "git_commit": metadata.get("git_commit"),
        "target_provider": target.get("provider_type", metadata.get("provider")),
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
        "judge_timeout": judge.get("timeout_seconds"),
        "judge_fallback_used": judge.get("fallback_used"),
        "allow_host_docker_socket": metadata.get("allow_host_docker_socket"),
    }
    manifest_rows: list[str] = []
    if isinstance(benchmark_results, list):
        for row in benchmark_results:
            if not isinstance(row, dict):
                continue
            manifest = row.get("manifest") if isinstance(row.get("manifest"), dict) else {}
            source = row.get("source") if isinstance(row.get("source"), dict) else {}
            official = row.get("official_conditions") if isinstance(row.get("official_conditions"), dict) else {}
            assets = row.get("asset_refs") if isinstance(row.get("asset_refs"), list) else []
            manifest_rows.append(
                "<tr>"
                f"<td>{html.escape(str(row.get('benchmark', '')))}</td>"
                f"<td>{html.escape(str(row.get('docker_image') or manifest.get('container', {}).get('image', '')))}</td>"
                f"<td>{html.escape(_source_ref(source))}</td>"
                f"<td>{html.escape(_asset_summary(assets))}</td>"
                f"<td>{html.escape(str(official.get('official_split', '')))}</td>"
                f"<td>{html.escape(str(official.get('official_scoring_method', '')))}</td>"
                "</tr>"
            )
    manifest_body = "".join(manifest_rows) or '<tr><td colspan="6">No manifest metadata recorded.</td></tr>'
    return (
        f"{_metadata_definition_list(rows)}"
        "<h3>Benchmark Manifests</h3>"
        "<table><thead><tr><th>Benchmark</th><th>Container</th><th>Source Ref</th><th>Assets</th>"
        "<th>Official Split</th><th>Scoring</th></tr></thead>"
        f"<tbody>{manifest_body}</tbody></table>"
    )


def _report_failure_table(benchmark_results: Any) -> str:
    if not isinstance(benchmark_results, list):
        return "<table><tbody><tr><td>No failures recorded.</td></tr></tbody></table>"
    rows = []
    for row in benchmark_results:
        if not isinstance(row, dict) or row.get("included_in_official_score"):
            continue
        status = str(row.get("status") or "")
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('benchmark', '')))}</td>"
            f"<td>{html.escape(status)}</td>"
            f"<td>{html.escape(str(row.get('blocker_type') or ''))}</td>"
            f"<td>{html.escape(_display_error(str(row.get('error_details') or row.get('error') or '')))}</td>"
            "<td>yes</td>"
            f"<td>{html.escape(_suggested_action(row))}</td>"
            "</tr>"
        )
    if not rows:
        rows.append('<tr><td colspan="6">No failed or skipped benchmarks.</td></tr>')
    return (
        "<table><thead><tr><th>Benchmark</th><th>Status Code</th><th>Failure Class</th>"
        "<th>Explanation</th><th>Excluded</th><th>Suggested Action</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _report_credits_table(benchmark_results: Any) -> str:
    if not isinstance(benchmark_results, list) or not benchmark_results:
        return "<table><tbody><tr><td>No credits recorded.</td></tr></tbody></table>"
    rows = []
    for row in benchmark_results:
        if not isinstance(row, dict):
            continue
        source = row.get("source") if isinstance(row.get("source"), dict) else {}
        homepage = str(row.get("homepage") or "")
        leaderboard = str(row.get("official_leaderboard_url") or "")
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('benchmark', '')))}</td>"
            f"<td>{_link(homepage, homepage or 'n/a')}</td>"
            f"<td>{html.escape(_source_ref(source))}</td>"
            f"<td>{html.escape(str(row.get('license') or ''))}</td>"
            f"<td>{html.escape(str(row.get('credit') or ''))}</td>"
            f"<td>{_link(str(row.get('citation') or ''), str(row.get('citation') or ''))}</td>"
            f"<td>{_link(leaderboard, leaderboard)}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Benchmark</th><th>Homepage</th><th>Repository/Dataset Ref</th>"
        "<th>License</th><th>Credit</th><th>Citation</th><th>Leaderboard</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _metadata_definition_list(values: dict[str, Any]) -> str:
    rows = []
    for key, value in values.items():
        rows.append(f"<dt>{html.escape(str(key))}</dt><dd>{html.escape(str(value))}</dd>")
    return f'<dl class="metadata">{"".join(rows)}</dl>'


def _status_class(status: str) -> str:
    normalized = status.lower()
    if normalized in {"passed", "success", "success_with_warnings"}:
        return "status-success"
    if normalized.startswith("skipped"):
        return "status-skipped"
    if normalized.startswith("failed") or normalized in {"timed_out", "cancelled"}:
        return "status-failed"
    return ""


def _artifact_link(path: Any, label: str) -> str:
    if not isinstance(path, str) or not path:
        return "n/a"
    href = html.escape(Path(path).name, quote=True)
    return f'<a href="{href}">{html.escape(label)}</a>'


def _source_ref(source: dict[str, Any]) -> str:
    repository = str(source.get("repository_url") or "")
    commit = str(source.get("commit") or "")
    dataset = str(source.get("dataset_id") or "")
    revision = str(source.get("dataset_revision") or "")
    if repository:
        return f"{repository}@{commit or 'un-pinned'}"
    if dataset:
        return f"{dataset}@{revision or 'un-pinned'}"
    return ""


def _asset_summary(assets: list[Any]) -> str:
    if not assets:
        return ""
    labels = []
    for asset in assets[:3]:
        if not isinstance(asset, dict):
            continue
        path = asset.get("expected_local_path") or asset.get("source") or "asset"
        revision = asset.get("revision") or asset.get("ref") or ""
        labels.append(f"{path}@{revision}" if revision else str(path))
    if len(assets) > 3:
        labels.append(f"+{len(assets) - 3} more")
    return "; ".join(labels)


def _suggested_action(row: dict[str, Any]) -> str:
    manifest = row.get("manifest") if isinstance(row.get("manifest"), dict) else {}
    result = row.get("result") if isinstance(row.get("result"), dict) else {}
    validation = result.get("validation") if isinstance(result.get("validation"), dict) else {}
    issues = validation.get("issues") if isinstance(validation.get("issues"), list) else []
    for issue in issues:
        if isinstance(issue, dict) and issue.get("suggestion"):
            return str(issue["suggestion"])
    if manifest:
        return "Fix the benchmark manifest and rerun validation."
    return "Inspect graded_results.jsonl and benchmark logs."


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
    row["benchmark_name"] = details.get("benchmark_name", result.task_id)
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
            f"<td>{html.escape(str(row.get('benchmark_name', '')))}</td>"
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
            f'<polygon points="{_points(points)}" fill="none" stroke="#344457" stroke-width="1" />'
        )

    axis_lines = []
    label_nodes = []
    for index, label in enumerate(labels):
        end = _point(cx, cy, radius, index, axes)
        label_point = _point(cx, cy, radius + 28.0, index, axes)
        axis_lines.append(
            f'<line x1="{cx:.1f}" y1="{cy:.1f}" x2="{end[0]:.1f}" y2="{end[1]:.1f}" stroke="#405168" />'
        )
        label_nodes.append(
            f'<text x="{label_point[0]:.1f}" y="{label_point[1]:.1f}" text-anchor="middle" dominant-baseline="middle" font-size="10" fill="#c3cfdd">{html.escape(label)}</text>'
        )

    value_points = [_point(cx, cy, radius * (value / 100.0), index, axes) for index, value in enumerate(values)]
    return (
        '<svg viewBox="0 0 320 260" role="img" aria-label="Radar chart of category scores">'
        + "".join(grid_polygons)
        + "".join(axis_lines)
        + f'<polygon points="{_points(value_points)}" fill="#55d6be" fill-opacity="0.20" stroke="#55d6be" stroke-width="2" />'
        + "".join(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="#6ea8fe" />' for x, y in value_points
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
