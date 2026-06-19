import json

from agent_bench.cli import main
from agent_bench.aggregator import aggregate_results
from agent_bench.models import GradeResult
from agent_bench.reports import render_summary_html


def test_render_summary_html_contains_radar_svg():
    summary = {
        "metadata": {"model": "mock", "created_at_utc": "2026-01-01T00:00:00+00:00"},
        "total_score": 50.0,
        "pass_rate": 50.0,
        "coding_pass_rate": None,
        "json_validity_rate": 100.0,
        "timeout_rate": 0.0,
        "average_latency_seconds": 0.1,
        "average_time_to_first_token_seconds": 0.03,
        "average_tokens_per_second": 42.0,
        "total_run_duration_seconds": 0.5,
        "total_task_duration_seconds": 0.4,
        "total_output_tokens": 12,
        "category_scores": {"alpha": 50.0, "beta": 100.0},
        "category_counts": {
            "alpha": {"task_count": 1, "passed_count": 0, "score": 50.0},
            "beta": {"task_count": 1, "passed_count": 1, "score": 100.0},
        },
        "benchmark_results": [
            {
                "group": "Coding",
                "benchmark": "ExampleBench",
                "score": 75.0,
                "passed": False,
                "homepage": "https://example.com",
                "file_count_sampled": 5,
                "model_eval": {"answer": "B", "expected": "A", "question": "Pick the best fix."},
            }
        ],
        "timing_by_category": {
            "alpha": {
                "task_count": 1,
                "total_task_duration_seconds": 0.4,
                "average_task_duration_seconds": 0.4,
                "average_time_to_first_token_seconds": 0.03,
                "average_tokens_per_second": 42.0,
                "total_output_tokens": 12,
            }
        },
        "timing_by_problem": {
            "A": {
                "kind": "multiple_choice",
                "task_duration_seconds": 0.4,
                "request_latency_seconds": 0.1,
                "time_to_first_token_seconds": 0.03,
                "tokens_per_second": 42.0,
                "output_token_count": 12,
            }
        },
    }
    html = render_summary_html(
        summary,
        [
            GradeResult(
                task_id="A",
                category="alpha",
                kind="multiple_choice",
                score=0.0,
                max_score=1.0,
                passed=False,
                json_valid=True,
                latency_seconds=0.1,
                time_to_first_token_seconds=0.03,
                tokens_per_second=42.0,
                output_token_count=12,
                task_duration_seconds=0.4,
            )
        ],
    )

    assert "Average Score Radar" in html
    assert "Benchmark Scores" in html
    assert "Evaluation Methodology" not in html
    assert "ExampleBench" in html
    assert "<th>mock</th>" in html
    assert "<th>Benchmark</th><th>mock</th><th>Items</th><th>Method</th>" in html
    assert "<th>Method</th><th>Status</th>" not in html
    assert "<th>Method</th><th>Answer</th>" not in html
    assert "<th>Method</th><th>Expected</th>" not in html
    assert "<th>Kind</th><th>Status</th>" not in html
    assert "Timing By Category" in html
    assert "Timing By Problem" in html
    assert "<th>Category</th><th>Tasks</th><th>Total Task Time</th>" in html
    assert "Avg TTFT" in html
    assert '<svg viewBox="0 0 320 260"' in html
    assert "alpha" in html


def test_cli_mock_smoke_writes_expected_artifacts(tmp_path):
    task_dir = tmp_path / "tasks"
    out_dir = tmp_path / "runs" / "latest"
    task_dir.mkdir()
    (task_dir / "sample.json").write_text(
        json.dumps(
            [
                {
                    "id": "S_001",
                    "type": "multiple_choice",
                    "question": "Pick A",
                    "choices": ["yes", "no"],
                    "answer": ["A"],
                }
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "run",
            "--provider",
            "mock",
            "--tasks",
            str(task_dir),
            "--out",
            str(out_dir),
            "--sandbox",
            "subprocess",
        ]
    )

    assert exit_code == 0
    assert (out_dir / "summary.json").exists()
    assert (out_dir / "summary.html").exists()
    assert (out_dir / "results.csv").exists()
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["task_count"] == 1
    assert summary["total_score"] == 100.0
    assert "total_run_duration_seconds" in summary
    assert "timing_by_category" in summary
    assert "timing_by_problem" in summary


def test_cli_mock_smoke_runs_all_bundled_benchmarks(tmp_path):
    out_dir = tmp_path / "runs" / "latest"

    exit_code = main(
        [
            "run",
            "--provider",
            "mock",
            "--tasks",
            "tasks",
            "--out",
            str(out_dir),
            "--sandbox",
            "subprocess",
        ]
    )

    assert exit_code == 0
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    html = (out_dir / "summary.html").read_text(encoding="utf-8")
    assert summary["task_count"] == 32
    assert len(summary["benchmark_results"]) == 32
    assert "public_benchmarks" not in html
    assert "SWE-Bench Verified" in html
    assert "Terminal Bench 2.1" in html
    assert "MCP Atlas" in html
    assert "VideoMME (w/ sub)" in html
    assert "USAMO 2026" in html
    assert "<th>Benchmark</th><th>mock-perfect</th><th>Items</th><th>Method</th>" in html


def test_aggregate_results_groups_timing_by_category():
    summary = aggregate_results(
        [
            GradeResult(
                task_id="A",
                category="alpha",
                kind="multiple_choice",
                score=1.0,
                max_score=1.0,
                passed=True,
                json_valid=True,
                latency_seconds=0.1,
                task_duration_seconds=0.2,
                output_token_count=5,
            ),
            GradeResult(
                task_id="B",
                category="alpha",
                kind="coding",
                score=0.0,
                max_score=1.0,
                passed=False,
                json_valid=True,
                latency_seconds=0.2,
                task_duration_seconds=0.5,
                output_token_count=7,
            ),
            GradeResult(
                task_id="C",
                category="beta",
                kind="multiple_choice",
                score=1.0,
                max_score=1.0,
                passed=True,
                json_valid=True,
                latency_seconds=0.3,
                task_duration_seconds=0.7,
                output_token_count=11,
            ),
        ],
        {"run_duration_seconds": 1.4},
    )

    assert set(summary["timing_by_category"]) == {"alpha", "beta"}
    assert summary["timing_by_category"]["alpha"]["task_count"] == 2
    assert summary["timing_by_category"]["alpha"]["total_task_duration_seconds"] == 0.7
    assert summary["timing_by_category"]["alpha"]["total_output_tokens"] == 12
    assert summary["timing_by_category"]["beta"]["task_count"] == 1
    assert summary["benchmark_results"] == []


def test_aggregate_results_emits_external_benchmark_rows():
    summary = aggregate_results(
        [
            GradeResult(
                task_id="PB_001",
                category="Coding",
                kind="external_benchmark",
                score=0.75,
                max_score=1.0,
                passed=False,
                json_valid=True,
                latency_seconds=0.1,
                details={
                    "benchmark": "ExampleBench",
                    "group": "Coding",
                    "homepage": "https://example.com",
                    "license": "MIT",
                    "credit": "Example authors",
                    "result": {
                        "repository_ready": True,
                        "file_count_sampled": 12,
                        "model_eval": {"answer": "A", "expected": "B"},
                    },
                },
            )
        ],
        {"run_duration_seconds": 1.0},
    )

    assert summary["category_counts"] == {
        "Coding": {"task_count": 1, "passed_count": 0, "score": 75.0}
    }
    assert summary["benchmark_results"][0]["benchmark"] == "ExampleBench"
    assert summary["benchmark_results"][0]["score"] == 75.0
    assert summary["benchmark_results"][0]["model_eval"] == {"answer": "A", "expected": "B"}
