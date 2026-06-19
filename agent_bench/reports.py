import csv
import html
import json
import math
import shutil
from pathlib import Path
from typing import Any

from agent_bench.models import GradeResult, ModelResponse


RESULT_FIELDS = [
    "task_id",
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
    timing_by_category = summary.get("timing_by_category", {})
    timing_by_problem = summary.get("timing_by_problem", {})
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
    .stack {{ display: grid; gap: 18px; }}
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
    <section class="stack">
      <div>
        <h2>Timing By Category</h2>
        {_timing_by_category_table(timing_by_category)}
      </div>
      <div>
        <h2>Timing By Problem</h2>
        {_timing_by_problem_table(timing_by_problem)}
      </div>
    </section>
    <h2>Run Metadata</h2>
    {_metadata_table(metadata)}
    <h2>Task Results</h2>
    {_results_table(results)}
  </main>
</body>
</html>
"""


def _write_results_csv(path: Path, results: list[GradeResult]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for result in results:
            row = result.to_dict()
            writer.writerow({field: row.get(field) for field in RESULT_FIELDS})


def _metric_cards(summary: dict[str, Any]) -> str:
    cards = [
        ("Total Score", _format_percent(summary.get("total_score"))),
        ("Pass Rate", _format_percent(summary.get("pass_rate"))),
        ("Coding Pass", _format_percent(summary.get("coding_pass_rate"))),
        ("JSON Validity", _format_percent(summary.get("json_validity_rate"))),
        ("Timeout Rate", _format_percent(summary.get("timeout_rate"))),
        ("Avg Latency", f"{float(summary.get('average_latency_seconds', 0.0)):.2f}s"),
        ("Avg TTFT", _format_seconds(summary.get("average_time_to_first_token_seconds"))),
        ("Avg Tokens/s", _format_rate(summary.get("average_tokens_per_second"))),
        ("Run Time", _format_seconds(summary.get("total_run_duration_seconds"))),
        ("Task Time Sum", _format_seconds(summary.get("total_task_duration_seconds"))),
        ("Output Tokens", _format_integer(summary.get("total_output_tokens"))),
    ]
    return '<section class="cards">' + "\n".join(
        f'<section class="card"><span>{html.escape(label)}</span><strong>{html.escape(value)}</strong></section>'
        for label, value in cards
    ) + "</section>"


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
                f'<tr class="group-row"><td colspan="4">{html.escape(group)}</td></tr>'
            )
            current_group = group
        methods = _grading_methods(row)
        evaluated = row.get("evaluated_task_count")
        passed_count = row.get("evaluation_passed_count")
        items = f"{passed_count}/{evaluated}" if evaluated is not None and passed_count is not None else "n/a"
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('benchmark', '')))}</td>"
            f'<td class="score-cell">{_format_percent(row.get("score"))}</td>'
            f"<td>{html.escape(items)}</td>"
            f"<td>{html.escape(methods)}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Benchmark</th>"
        f"<th>{html.escape(model)}</th><th>Items</th><th>Method</th></tr></thead>"
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


def _metadata_table(metadata: dict[str, Any]) -> str:
    rows = []
    for key, value in sorted(metadata.items()):
        rows.append(f"<dt>{html.escape(str(key))}</dt><dd>{html.escape(str(value))}</dd>")
    return f'<dl class="metadata">{"".join(rows)}</dl>'


def _timing_by_category_table(timing_by_category: dict[str, Any]) -> str:
    rows = []
    for category, data in timing_by_category.items():
        rows.append(
            "<tr>"
            f"<td>{html.escape(category)}</td>"
            f"<td>{int(data.get('task_count', 0))}</td>"
            f"<td>{_format_seconds(data.get('total_task_duration_seconds'))}</td>"
            f"<td>{_format_seconds(data.get('average_task_duration_seconds'))}</td>"
            f"<td>{_format_seconds(data.get('average_time_to_first_token_seconds'))}</td>"
            f"<td>{_format_rate(data.get('average_tokens_per_second'))}</td>"
            f"<td>{_format_integer(data.get('total_output_tokens'))}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Category</th><th>Tasks</th><th>Total Task Time</th>"
        "<th>Avg Task Time</th><th>Avg TTFT</th><th>Avg Tokens/s</th><th>Output Tokens</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _timing_by_problem_table(timing_by_problem: dict[str, Any]) -> str:
    rows = []
    for task_id, data in timing_by_problem.items():
        rows.append(
            "<tr>"
            f"<td>{html.escape(task_id)}</td>"
            f"<td>{html.escape(str(data.get('kind', '')))}</td>"
            f"<td>{_format_seconds(data.get('task_duration_seconds'))}</td>"
            f"<td>{_format_seconds(data.get('request_latency_seconds'))}</td>"
            f"<td>{_format_seconds(data.get('time_to_first_token_seconds'))}</td>"
            f"<td>{_format_rate(data.get('tokens_per_second'))}</td>"
            f"<td>{_format_integer(data.get('output_token_count'))}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Problem</th><th>Kind</th><th>Total Task Time</th>"
        "<th>Request Time</th><th>TTFT</th><th>Tokens/s</th><th>Output Tokens</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _results_table(results: list[GradeResult]) -> str:
    rows = []
    for result in results:
        rows.append(
            "<tr>"
            f"<td>{html.escape(result.task_id)}</td>"
            f"<td>{html.escape(result.category)}</td>"
            f"<td>{html.escape(result.kind)}</td>"
            f"<td>{result.score:.3f}</td>"
            f"<td>{'yes' if result.json_valid else 'no'}</td>"
            f"<td>{_format_seconds(result.latency_seconds)}</td>"
            f"<td>{_format_seconds(result.time_to_first_token_seconds)}</td>"
            f"<td>{_format_rate(result.tokens_per_second)}</td>"
            f"<td>{_format_integer(result.output_token_count)}</td>"
            f"<td>{_format_seconds(result.task_duration_seconds)}</td>"
            f"<td>{html.escape(_display_error(result.error))}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Task</th><th>Category</th><th>Kind</th>"
        "<th>Score</th><th>JSON</th><th>Request Time</th><th>TTFT</th><th>Tokens/s</th>"
        "<th>Output Tokens</th><th>Total Task Time</th><th>Error</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _format_percent(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.2f}%"


def _display_error(value: str | None) -> str:
    if not value:
        return ""
    return value.replace("No answer-keyed benchmark task records", "No benchmark task records")


def _format_seconds(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}s"


def _format_rate(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.2f}"


def _format_integer(value: Any) -> str:
    if value is None:
        return "n/a"
    return str(int(value))
