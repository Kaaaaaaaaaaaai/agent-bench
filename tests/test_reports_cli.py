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
        "timing_by_category": {
            "alpha": {
                "task_count": 1,
                "total_task_duration_seconds": 0.4,
                "average_task_duration_seconds": 0.4,
            }
        },
        "timing_note": "total_task_duration_seconds is the sum of per-task elapsed time and can exceed total_run_duration_seconds when tasks run concurrently.",
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
                "citation": "https://example.com/citation",
                "license": "MIT",
                "credit": "Example authors",
                "file_count_sampled": 5,
                "model_eval": {"answer": "B", "expected": "A", "question": "Pick the best fix."},
            }
        ],
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
                details={"benchmark": "ExampleBench"},
            )
        ],
    )

    assert "Average Score Radar" in html
    assert "Benchmark Scores" in html
    assert "Benchmark Citations" in html
    assert "Evaluation Methodology" not in html
    assert "ExampleBench" in html
    assert "https://example.com/citation" in html
    assert "<th>mock</th>" in html
    assert "<th>Benchmark</th><th>Profile</th><th>mock</th><th>Items</th><th>Method</th>" in html
    assert "<th>Run</th><th>Score Status</th><th>Blocker/Reason</th>" in html
    assert "<th>Benchmark</th><th>Source</th><th>Credit</th>" in html
    assert "<th>Group</th>" not in html
    assert "<th>License</th>" not in html
    assert html.rfind("Benchmark Citations") > html.rfind("Task Results")
    assert "<th>Benchmark</th><th>Category</th><th>Score</th>" in html
    assert "<th>Method</th><th>Status</th>" not in html
    assert "<th>Method</th><th>Answer</th>" not in html
    assert "<th>Method</th><th>Expected</th>" not in html
    assert "<th>Kind</th>" not in html
    assert "<h2>Timing</h2>" in html
    assert "Timing By Problem" not in html
    assert "TTFT" not in html
    assert "Tokens/s" not in html
    assert "Output Tokens" not in html
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
    assert summary["timing_by_category"]["sample"]["task_count"] == 1
    assert summary["timing_by_problem"]["S_001"]["kind"] == "multiple_choice"
    assert "concurrently" in summary["timing_note"]


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
    assert summary["task_count"] == 15
    assert summary["selected_suite_count"] == 15
    assert summary["known_suite_count"] == 15
    assert len(summary["benchmark_results"]) == 15
    assert summary["suite_coverage_rate"] == 1.0
    assert summary["conservative_all_suite_score"] == 1.0
    assert "public_benchmarks" not in html
    assert "SWE-bench" in html
    assert "GDPval" in html
    assert "Humanity&#x27;s Last Exam" not in html
    assert "BioMystery Bench" in html
    assert "EDINET-Bench" not in html
    assert "MLE-bench" not in html
    assert "Benchmark Citations" in html
    assert "<th>Benchmark</th><th>Profile</th><th>mock-perfect</th><th>Items</th><th>Method</th>" in html


def test_aggregate_results_emits_timing_breakdowns():
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

    assert summary["timing_by_category"]["alpha"]["task_count"] == 2
    assert summary["timing_by_category"]["alpha"]["total_task_duration_seconds"] == 0.7
    assert summary["timing_by_problem"]["A"]["task_duration_seconds"] == 0.2
    assert "concurrently" in summary["timing_note"]
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
                    "citation": "https://example.com/cite",
                    "result": {
                        "repository_ready": True,
                        "file_count_sampled": 12,
                        "raw_score": 0.5,
                        "valid_score": 0.75,
                        "status_counts": {"passed": 2, "failed_harness_setup": 1},
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
    assert summary["benchmark_results"][0]["score_fraction"] == 0.75
    assert summary["benchmark_results"][0]["run_status"] == "completed"
    assert summary["benchmark_results"][0]["score_status"] == "partially_correct"
    assert summary["benchmark_results"][0]["raw_score"] == 50.0
    assert summary["benchmark_results"][0]["valid_score"] == 75.0
    assert summary["benchmark_results"][0]["setup_failed_count"] == 1
    assert summary["benchmark_results"][0]["citation"] == "https://example.com/cite"
    assert summary["benchmark_results"][0]["model_eval"] == {"answer": "A", "expected": "B"}
    assert summary["num_suites_failed_setup"] == 0
    assert summary["num_items_failed_setup"] == 1
    assert summary["num_items_valid_model_attempts"] == 2
    assert summary["num_items_passed_valid_model_attempts"] == 2
    assert summary["headline"]["coverage"] == "1/1 valid judged suites"


def test_aggregate_results_separates_skipped_from_valid_scores():
    summary = aggregate_results(
        [
            GradeResult(
                task_id="valid",
                category="alpha",
                kind="multiple_choice",
                score=1.0,
                max_score=1.0,
                passed=True,
                json_valid=True,
                latency_seconds=0.1,
            ),
            GradeResult(
                task_id="skipped",
                category="alpha",
                kind="external_benchmark",
                score=0.0,
                max_score=1.0,
                passed=False,
                json_valid=True,
                latency_seconds=0.1,
                status="failed_unsupported_capability",
            ),
        ],
        {"run_duration_seconds": 1.0},
    )

    assert summary["raw_score"] == 50.0
    assert summary["total_score"] == 100.0
    assert summary["valid_judged_score"] == 1.0
    assert summary["suite_coverage_rate"] == 0.5
    assert summary["item_coverage_rate"] == 0.5
    assert summary["conservative_all_suite_score"] == 0.5
    assert summary["valid_task_count"] == 1
    assert summary["skipped_suite_count"] == 1
    assert summary["skipped_suites"][0]["blocker_type"] == "unsupported_capability"
    assert summary["skipped_suites"][0]["error"]
    assert summary["skipped_count"] == 1


def test_aggregate_results_surfaces_nested_lfs_blocker_type():
    summary = aggregate_results(
        [
            GradeResult(
                task_id="PB_003",
                category="Research",
                kind="external_benchmark",
                score=0.0,
                max_score=1.0,
                passed=False,
                json_valid=True,
                latency_seconds=0.1,
                status="failed_missing_assets",
                error="All benchmark record evaluations were invalid: failed missing assets",
                details={
                    "benchmark": "PaperBench",
                    "group": "Research",
                    "result": {
                        "status": "failed_missing_assets",
                        "status_counts": {"failed_missing_assets": 1},
                        "model_evals": [
                            {
                                "status": "failed_missing_assets",
                                "setup_details": {
                                    "details": {
                                        "blocker_type": "git_lfs_pointer_stub",
                                        "asset_errors": [
                                            {
                                                "blocker_type": "git_lfs_pointer_stub",
                                                "reason": "git_lfs_pointer_stub",
                                            }
                                        ],
                                    }
                                },
                            }
                        ],
                    },
                },
            )
        ],
        {"run_duration_seconds": 1.0},
    )

    assert summary["skipped_suites"][0]["blocker_type"] == "git_lfs_pointer_stub"
    assert summary["benchmark_results"][0]["blocker_type"] == "git_lfs_pointer_stub"


def test_aggregate_results_surfaces_repo_patch_canary_blocker_type():
    summary = aggregate_results(
        [
            GradeResult(
                task_id="PB_001",
                category="Coding",
                kind="external_benchmark",
                score=0.0,
                max_score=1.0,
                passed=False,
                json_valid=True,
                latency_seconds=0.1,
                status="failed_harness_setup",
                error="All benchmark record evaluations were invalid: failed harness setup",
                details={
                    "benchmark": "SWE-bench",
                    "group": "Coding",
                    "result": {
                        "status": "failed_harness_setup",
                        "capability_contract": {
                            "repo_patch": {
                                "supported": False,
                                "canary": {
                                    "passed": False,
                                    "blocker_type": "repo_patch_harness_setup",
                                    "reason": "repo_patch requires the `patch` executable",
                                },
                            }
                        },
                    },
                },
            )
        ],
        {"run_duration_seconds": 1.0},
    )

    assert summary["skipped_suites"][0]["blocker_type"] == "repo_patch_harness_setup"
    assert summary["benchmark_results"][0]["blocker_type"] == "repo_patch_harness_setup"


def test_aggregate_results_uses_nested_judge_parse_and_model_valid_counts():
    summary = aggregate_results(
        [
            GradeResult(
                task_id="PB_017",
                category="Finance",
                kind="external_benchmark",
                score=0.0,
                max_score=1.0,
                passed=False,
                json_valid=True,
                latency_seconds=0.1,
                details={
                    "benchmark": "FinanceMath",
                    "group": "Finance",
                    "result": {
                        "status": "completed",
                        "capabilities_verified": False,
                        "evaluated_task_count": 1,
                        "valid_evaluated_task_count": 0,
                        "judge_parse_failure_count": 1,
                        "status_counts": {"failed_grader": 1},
                    },
                },
            )
        ],
        {"run_duration_seconds": 1.0},
    )

    assert summary["valid_task_count"] == 1
    assert summary["model_valid_task_count"] == 0
    assert summary["model_score_valid_tasks_only"] == 0.0
    assert summary["judge_parse_failure_count"] == 1
    assert summary["judge_parse_failed_count"] == 1
    assert summary["benchmark_item_status_counts"] == {"failed_grader": 1}


def test_aggregate_results_counts_nested_passed_items():
    summary = aggregate_results(
        [
            GradeResult(
                task_id="PB_008",
                category="Science",
                kind="external_benchmark",
                score=0.5,
                max_score=1.0,
                passed=False,
                json_valid=True,
                latency_seconds=0.1,
                details={
                    "benchmark": "BioMystery",
                    "group": "Science",
                    "result": {
                        "status": "completed",
                        "status_counts": {"passed": 2, "failed_model_answer": 1},
                        "evaluated_task_count": 3,
                        "evaluation_passed_count": 2,
                    },
                },
            )
        ],
        {"run_duration_seconds": 1.0},
    )

    assert summary["passed_count"] == 0
    assert summary["item_passed_count"] == 2
    assert summary["benchmark_item_status_counts"]["passed"] == 2
