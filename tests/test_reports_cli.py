import json
from pathlib import Path

import pytest

from agent_bench.cli import main
from agent_bench.aggregator import aggregate_results
from agent_bench.models import GradeResult
from agent_bench.reports import _result_csv_row, render_summary_html
from agent_bench.statuses import status_info


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

    assert "Radar Chart" in html
    assert "Benchmark Scores" in html
    assert "Credits Citations Licenses" in html
    assert "Evaluation Methodology" not in html
    assert "ExampleBench" in html
    assert "https://example.com/citation" in html
    assert "<th>Status</th><th>Raw Score</th><th>Normalized 0-100</th>" in html
    assert "<th>Image</th><th>Container</th><th>Network</th>" in html
    assert "<th>Missing Tools</th><th>Missing Env</th><th>Missing Assets</th>" in html
    assert "<th>Asset Cache</th><th>Setup Details</th>" in html
    assert "<th>Benchmark</th><th>Status Code</th><th>Failure Class</th>" in html
    assert "<th>Benchmark</th><th>Homepage</th><th>Repository/Dataset Ref</th>" in html
    assert html.rfind("Credits Citations Licenses") > html.rfind("Failures And Status")
    assert "<th>Benchmark</th><th>Category</th><th>Score</th>" not in html
    assert "<th>Method</th><th>Status</th>" not in html
    assert "<th>Method</th><th>Answer</th>" not in html
    assert "<th>Method</th><th>Expected</th>" not in html
    assert "<th>Kind</th>" not in html
    assert "<h2>Metadata</h2>" in html
    assert "Timing By Problem" not in html
    assert "TTFT" not in html
    assert "Tokens/s" not in html
    assert "Output Tokens" not in html
    assert "Scored-Suite Score" in html
    assert "Conservative selected-suite" in html
    assert "Overall Score" not in html
    assert '<svg viewBox="0 0 320 260"' in html
    assert "alpha" in html


def test_failed_missing_required_tool_is_catalogued_as_setup_exclusion():
    info = status_info("failed_missing_required_tool")

    assert info.counts_toward_official_score is False
    assert info.failure_class == "benchmark_setup"


def test_cli_mock_smoke_writes_expected_artifacts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    task_dir = tmp_path / "tasks"
    out_dir = Path("runs") / "latest"
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
    assert (tmp_path / out_dir / "summary.json").exists()
    assert (tmp_path / out_dir / "summary.html").exists()
    assert (tmp_path / out_dir / "results.csv").exists()
    summary = json.loads((tmp_path / out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["task_count"] == 1
    assert summary["total_score"] == 100.0
    assert "total_run_duration_seconds" in summary
    assert summary["timing_by_category"]["sample"]["task_count"] == 1
    assert summary["timing_by_problem"]["S_001"]["kind"] == "multiple_choice"
    assert "concurrently" in summary["timing_note"]


def test_cli_mock_smoke_logs_progress_to_stderr(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    task_dir = tmp_path / "tasks"
    out_dir = Path("runs") / "latest"
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

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "INFO Starting Agent Bench run" in captured.err
    assert "Configuration: provider=mock" in captured.err
    assert "Task S_001 started" in captured.err
    assert "Progress: 1/1 task(s) completed" in captured.err
    assert "Run complete: 1/1 task(s) passed" in captured.err


def test_cli_quiet_suppresses_progress_logs(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    task_dir = tmp_path / "tasks"
    out_dir = Path("runs") / "latest"
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
            "--quiet",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""


def test_cli_mock_smoke_runs_all_bundled_benchmarks(tmp_path, monkeypatch):
    bundled_tasks = Path("tasks").resolve()
    monkeypatch.chdir(tmp_path)
    out_dir = Path("runs") / "latest"

    exit_code = main(
        [
            "run",
            "--provider",
            "mock",
            "--tasks",
            str(bundled_tasks),
            "--out",
            str(out_dir),
            "--sandbox",
            "subprocess",
        ]
    )

    assert exit_code == 0
    summary = json.loads((tmp_path / out_dir / "summary.json").read_text(encoding="utf-8"))
    html = (tmp_path / out_dir / "summary.html").read_text(encoding="utf-8")
    assert summary["task_count"] == 20
    assert summary["selected_suite_count"] == 20
    assert summary["known_suite_count"] == 20
    assert summary["excluded_suite_count"] == 4
    assert len(summary["excluded_suites"]) == 4
    assert len(summary["benchmark_results"]) == 20
    assert summary["suite_coverage_rate"] == pytest.approx(16 / 20)
    assert summary["coverage_summary"]["successfully_scored_benchmarks"] == 16
    assert summary["conservative_all_suite_score"] == pytest.approx(16 / 20)
    assert "public_benchmarks" not in html
    assert "SWE-bench" in html
    assert "Failures And Status" in html
    assert "Humanity&#x27;s Last Exam" in html
    assert "EDINET-Bench" not in html
    assert "MLE-bench" not in html
    assert "Credits Citations Licenses" in html
    assert "<th>Status</th><th>Raw Score</th><th>Normalized 0-100</th>" in html


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
                    "required_capabilities": ["repo_patch"],
                    "setup_details": {
                        "external_harness": {
                            "image": "example-benchmark:local",
                            "container_name": "agent-bench-pb-001",
                            "network_mode": "bridge",
                            "docker_socket_mount": {"enabled": False},
                            "output_mount": {"host_path": "/tmp/out", "container_path": "/outputs"},
                            "asset_cache_mount": {"cache_key": "examplebench", "container_path": "/benchmark/assets"},
                            "benchmark_checkout_path": "/workspace/repo",
                        }
                    },
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

    assert summary["category_counts"]["Coding"]["task_count"] == 1
    assert summary["category_counts"]["Coding"]["passed_count"] == 0
    assert summary["category_counts"]["Coding"]["score"] == 75.0
    assert summary["category_counts"]["Coding"]["successfully_scored_benchmark_count"] == 1
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
    assert summary["benchmark_results"][0]["required_capabilities"] == ["repo_patch"]
    assert summary["benchmark_results"][0]["docker_image"] == "example-benchmark:local"
    assert summary["benchmark_results"][0]["container_name"] == "agent-bench-pb-001"
    assert summary["benchmark_results"][0]["network_mode"] == "bridge"
    assert summary["benchmark_results"][0]["docker_socket_mount"] == {"enabled": False}
    assert summary["benchmark_results"][0]["asset_cache_mount"] == {
        "cache_key": "examplebench",
        "container_path": "/benchmark/assets",
    }
    assert summary["benchmark_results"][0]["benchmark_checkout_path"] == "/workspace/repo"
    assert summary["num_suites_failed_setup"] == 0
    assert summary["num_items_failed_setup"] == 1
    assert summary["num_items_valid_model_attempts"] == 2
    assert summary["num_items_passed_valid_model_attempts"] == 2
    assert summary["headline"]["coverage"] == "1/1 valid judged suites"


def test_external_benchmark_csv_row_keeps_answer_separate_from_status():
    result = GradeResult(
        task_id="PB_TOOL",
        category="Finance",
        kind="external_benchmark",
        score=0.0,
        max_score=1.0,
        passed=False,
        json_valid=True,
        latency_seconds=0.1,
        answer="failed_model_tool_use",
        status="failed_model_tool_use",
        details={
            "benchmark": "ExampleToolBench",
            "group": "Finance",
            "setup_details": {
                "external_harness": {
                    "image": "agent-bench-external:python3.12",
                    "container_name": "agent-bench-pb-tool",
                    "network_mode": "bridge",
                    "docker_socket_mount": {"enabled": False},
                    "output_mount": {"host_path": "/tmp/out", "container_path": "/outputs"},
                    "asset_cache_mount": {"cache_key": "exampletoolbench"},
                    "benchmark_checkout_path": "/workspace/repo",
                }
            },
            "result": {
                "model_eval": {"answer": "0/1", "expected": "1/1"},
                "status_counts": {"failed_model_tool_use": 1},
            },
        },
    )

    row = _result_csv_row(result)

    assert row["status"] == "failed_model_tool_use"
    assert row["score_status"] == "failed_model_tool_use"
    assert row["answer"] == "0/1"
    assert row["docker_image"] == "agent-bench-external:python3.12"
    assert row["container_name"] == "agent-bench-pb-tool"
    assert row["network_mode"] == "bridge"
    assert row["docker_socket_mount"] == '{"enabled": false}'
    assert row["asset_cache_mount"] == '{"cache_key": "exampletoolbench"}'
    assert row["benchmark_checkout_path"] == "/workspace/repo"


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
    assert summary["excluded_suite_count"] == 1
    assert summary["skipped_suites"][0]["blocker_type"] == "unsupported_capability"
    assert summary["skipped_suites"][0]["error"]
    assert summary["skipped_count"] == 1


def test_aggregate_results_surfaces_nested_lfs_blocker_type():
    summary = aggregate_results(
        [
            GradeResult(
                task_id="ASSET_001",
                category="Artifact",
                kind="external_benchmark",
                score=0.0,
                max_score=1.0,
                passed=False,
                json_valid=True,
                latency_seconds=0.1,
                status="failed_missing_assets",
                error="All benchmark record evaluations were invalid: failed missing assets",
                details={
                    "benchmark": "AssetBench",
                    "group": "Artifact",
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
                task_id="PB_016",
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

    assert summary["valid_task_count"] == 0
    assert summary["model_valid_task_count"] == 0
    assert summary["model_score_valid_tasks_only"] == 0.0
    assert summary["coverage_summary"]["successfully_scored_benchmarks"] == 0
    assert summary["benchmark_results"][0]["included_in_official_score"] is False
    assert summary["benchmark_results"][0]["capabilities_verified"] is False
    assert summary["judge_parse_failure_count"] == 1
    assert summary["judge_parse_failed_count"] == 1
    assert summary["benchmark_item_status_counts"] == {"failed_grader": 1}


def test_aggregate_results_excludes_tool_call_without_exposed_tools():
    summary = aggregate_results(
        [
            GradeResult(
                task_id="PB_TOOL",
                category="Finance",
                kind="external_benchmark",
                score=1.0,
                max_score=1.0,
                passed=True,
                json_valid=True,
                latency_seconds=0.1,
                status="passed",
                details={
                    "benchmark": "ExampleToolBench",
                    "group": "Finance",
                    "result": {
                        "status": "passed",
                        "capabilities_verified": True,
                        "required_capabilities": ["tool_call"],
                        "exposed_tools": [],
                        "missing_tools": [],
                        "status_counts": {"passed": 1},
                    },
                },
            )
        ],
        {"run_duration_seconds": 1.0},
    )

    assert summary["valid_task_count"] == 0
    assert summary["benchmark_results"][0]["included_in_official_score"] is False
    assert summary["benchmark_results"][0]["capabilities_verified"] is False


def test_aggregate_results_excludes_missing_environment_requirements():
    summary = aggregate_results(
        [
            GradeResult(
                task_id="PB_015",
                category="Finance",
                kind="external_benchmark",
                score=1.0,
                max_score=1.0,
                passed=True,
                json_valid=True,
                latency_seconds=0.1,
                status="passed",
                details={
                    "benchmark": "Finance Agent v2",
                    "group": "Finance",
                    "result": {
                        "status": "passed",
                        "required_capabilities": ["tool_call"],
                        "required_tools": ["web_search"],
                        "exposed_tools": ["web_search"],
                        "missing_tools": [],
                        "missing_env": ["FINANCE_AGENT_V2_CACHE"],
                        "status_counts": {"passed": 1},
                    },
                },
            )
        ],
        {"run_duration_seconds": 1.0},
    )

    assert summary["valid_task_count"] == 0
    assert summary["coverage_summary"]["successfully_scored_benchmarks"] == 0
    assert summary["excluded_suite_count"] == 1
    assert summary["benchmark_results"][0]["included_in_official_score"] is False
    assert summary["benchmark_results"][0]["capabilities_verified"] is False
    assert summary["benchmark_results"][0]["missing_env"] == ["FINANCE_AGENT_V2_CACHE"]


def test_aggregate_results_excludes_non_official_smoke_scores():
    summary = aggregate_results(
        [
            GradeResult(
                task_id="PB_004",
                category="Coding",
                kind="external_benchmark",
                score=0.0,
                max_score=1.0,
                passed=False,
                json_valid=True,
                latency_seconds=0.1,
                status="failed_model_missing_artifact",
                error="empty file: model.patch contains no repository diff",
                details={
                    "benchmark": "SWE-Lancer",
                    "group": "Coding",
                    "result": {
                        "status": "failed_model_missing_artifact",
                        "capabilities_verified": True,
                        "included_in_official_score": False,
                        "official_equivalent": False,
                        "score_mode": "smoke_patch_presence",
                        "status_counts": {"failed_model_missing_artifact": 1},
                    },
                },
            )
        ],
        {"run_duration_seconds": 1.0, "excluded_suite_count": 0, "excluded_suites": []},
    )

    assert summary["valid_task_count"] == 0
    assert summary["excluded_suite_count"] == 1
    assert summary["coverage_summary"]["excluded_from_score_benchmarks"] == 1
    assert summary["metadata"]["excluded_suite_count"] == 1
    assert summary["metadata"]["excluded_suites"][0]["suite_id"] == "PB_004"
    assert summary["benchmark_results"][0]["included_in_official_score"] is False
    assert summary["benchmark_results"][0]["official_equivalent"] is False
    assert summary["benchmark_results"][0]["score_mode"] == "smoke_patch_presence"
    assert summary["benchmark_results"][0]["blocker_type"] == "missing_grader"
    assert summary["benchmark_results"][0]["score_status"] == "ungraded"


def test_aggregate_results_counts_payload_unsupported_capabilities():
    summary = aggregate_results(
        [
            GradeResult(
                task_id="PB_TOOL",
                category="Finance",
                kind="external_benchmark",
                score=0.0,
                max_score=1.0,
                passed=False,
                json_valid=True,
                latency_seconds=0.1,
                status="failed_missing_required_tool",
                details={
                    "benchmark": "Example Tool Bench",
                    "group": "Finance",
                    "result": {
                        "status": "failed_missing_required_tool",
                        "capabilities_verified": False,
                        "unsupported_capabilities": ["tool_call"],
                        "missing_tools": ["quote_lookup"],
                        "status_counts": {"failed_missing_required_tool": 1},
                    },
                },
            ),
            GradeResult(
                task_id="PB_015",
                category="Finance",
                kind="external_benchmark",
                score=0.0,
                max_score=1.0,
                passed=False,
                json_valid=True,
                latency_seconds=0.1,
                status="failed_missing_required_tool",
                details={
                    "benchmark": "Finance Agent v2",
                    "group": "Finance",
                    "result": {
                        "status": "failed_missing_required_tool",
                        "capabilities_verified": False,
                        "unsupported_capabilities": ["external_data_required", "tool_call"],
                        "missing_tools": ["web_search"],
                        "status_counts": {"failed_missing_required_tool": 1},
                    },
                },
            ),
        ],
        {"run_duration_seconds": 1.0},
    )

    assert summary["unsupported_capability_count"] == 3
    assert summary["coverage"]["unsupported_capability_count"] == 3
    assert summary["excluded_suite_count"] == 2


def test_aggregate_results_counts_nested_passed_items():
    summary = aggregate_results(
        [
            GradeResult(
                task_id="STATIC_001",
                category="Science",
                kind="external_benchmark",
                score=0.5,
                max_score=1.0,
                passed=False,
                json_valid=True,
                latency_seconds=0.1,
                details={
                    "benchmark": "StaticBench",
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


def test_usage_summary_uses_benchmark_item_denominator_and_hidden_reasoning_counts():
    summary = aggregate_results(
        [
            GradeResult(
                task_id="PB_TOOL",
                category="Finance",
                kind="external_benchmark",
                score=0.0,
                max_score=1.0,
                passed=False,
                json_valid=True,
                latency_seconds=0.1,
                details={
                    "benchmark": "ExampleToolBench",
                    "group": "Finance",
                    "result": {
                        "status": "completed",
                        "status_counts": {"failed_model_format": 2},
                        "model_evals": [
                            {
                                "status": "failed_model_format",
                                "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
                                "protocol_diagnostics": {
                                    "finish_reason": "length",
                                    "hidden_reasoning_no_final": True,
                                    "no_final_content": True,
                                },
                            },
                            {
                                "status": "failed_model_format",
                                "usage": {"prompt_tokens": 30, "completion_tokens": 40, "total_tokens": 70},
                                "protocol_diagnostics": {},
                            },
                        ],
                    },
                },
            )
        ],
        {"run_duration_seconds": 1.0},
    )

    assert summary["average_prompt_tokens_per_suite"] == 40.0
    assert summary["average_completion_tokens_per_suite"] == 60.0
    assert summary["average_prompt_tokens_per_item"] == 20.0
    assert summary["average_completion_tokens_per_item"] == 30.0
    assert summary["finish_reason_length_count"] == 1
    assert summary["hidden_reasoning_no_final_count"] == 1
    assert summary["no_final_content_count"] == 1
