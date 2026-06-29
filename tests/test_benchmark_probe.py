import importlib.util
import json
import sys
import zipfile
from pathlib import Path


def _load_probe_module():
    path = Path(__file__).resolve().parents[1] / "docker" / "benchmark_probe.py"
    spec = importlib.util.spec_from_file_location("benchmark_probe", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_probe_extracts_answer_keyed_json_records(tmp_path):
    probe = _load_probe_module()
    data = [
        {
            "question": "Which option fixes the regression?",
            "choices": ["Only explain it", "Patch the bug"],
            "answer": "B",
        }
    ]
    (tmp_path / "tasks.json").write_text(json.dumps(data), encoding="utf-8")

    items, errors = probe.extract_benchmark_items(tmp_path, limit=3)

    assert errors == []
    assert len(items) == 1
    assert items[0].question == "Which option fixes the regression?"
    assert items[0].expected == "B"
    assert items[0].choices == {"A": "Only explain it", "B": "Patch the bug"}


def test_probe_json_records_get_unique_sources_and_item_dirs(tmp_path, monkeypatch):
    probe = _load_probe_module()
    monkeypatch.setenv("AGENT_BENCH_OUTPUT_DIR", str(tmp_path / "outputs"))
    data = [
        {"question": "Question one?", "answer": "A"},
        {"question": "Question two?", "answer": "B"},
    ]
    (tmp_path / "tasks.json").write_text(json.dumps(data), encoding="utf-8")

    items, errors = probe.extract_benchmark_items(tmp_path, limit=3)
    item_dirs = [probe._item_output_dir(item) for item in items]

    assert errors == []
    assert [item.source for item in items] == [
        "tasks.json:record-000001",
        "tasks.json:record-000002",
    ]
    assert item_dirs[0] != item_dirs[1]
    assert "__sha256-" in item_dirs[0].name


def test_probe_extracts_csv_records_and_scores_choices(tmp_path):
    probe = _load_probe_module()
    (tmp_path / "sample.csv").write_text(
        "prompt,A,B,label\nPick the answer,wrong,right,B\n",
        encoding="utf-8",
    )

    items, errors = probe.extract_benchmark_items(tmp_path, limit=3)

    assert errors == []
    assert len(items) == 1
    assert probe.score_answer("right", items[0].expected, items[0].choices) == 1.0
    assert probe.score_answer("A", items[0].expected, items[0].choices) == 0.0


def test_probe_reports_no_items_for_unanswerable_repository(tmp_path):
    probe = _load_probe_module()
    (tmp_path / "README.md").write_text("# Benchmark\nNo answer key here.\n", encoding="utf-8")

    items, errors = probe.extract_benchmark_items(tmp_path, limit=3)

    assert items == []
    assert errors == []


def test_probe_reads_explicit_huggingface_dataset_id(monkeypatch):
    probe = _load_probe_module()
    monkeypatch.setenv("AGENT_BENCH_REPOSITORY", "https://github.com/SWE-bench/SWE-bench.git")
    monkeypatch.setenv("AGENT_BENCH_DATASET_ID", "princeton-nlp/SWE-bench")

    assert probe._huggingface_dataset_ids() == ["princeton-nlp/SWE-bench"]


def test_probe_uses_configurable_model_request_timeout(monkeypatch):
    probe = _load_probe_module()

    monkeypatch.setenv("AGENT_BENCH_MODEL_REQUEST_TIMEOUT", "1800")

    assert probe._model_request_timeout() == 1800.0


def test_probe_uses_configurable_agent_max_tokens(monkeypatch):
    probe = _load_probe_module()

    assert probe._max_answer_tokens(probe.BenchmarkItem("q", "a", "source")) == 1024

    monkeypatch.setenv("AGENT_BENCH_MAX_TOKENS", "32768")

    assert probe._max_answer_tokens(probe.BenchmarkItem("q", "a", "source")) == 1024
    assert (
        probe._max_answer_tokens(
            probe.BenchmarkItem("fix the repo", "diff --git a/a.py b/a.py", "huggingface:swe-bench/test:1")
        )
        == 4096
    )
    assert (
        probe._max_answer_tokens(
            probe.BenchmarkItem("write report", "must be correct", "source", metadata={"grading": "rubric"})
        )
        == 2048
    )


def test_probe_caps_agent_max_tokens_with_environment(monkeypatch):
    probe = _load_probe_module()
    monkeypatch.setenv("AGENT_BENCH_MAX_TOKENS", "768")

    assert probe._max_answer_tokens(probe.BenchmarkItem("fix the repo", "diff --git a/a.py b/a.py", "source")) == 768


def test_probe_builds_readiness_fallback_from_readme(tmp_path):
    probe = _load_probe_module()
    (tmp_path / "README.md").write_text(
        "# ExampleBench\n\n"
        "This benchmark evaluates agents on public tasks with a local Docker harness. "
        "Agents must produce a submission artifact that the grader can execute. " * 3,
        encoding="utf-8",
    )

    item = probe.readiness_fallback_item(tmp_path, "ExampleBench")

    assert item is not None
    assert "local-readiness task" in item.question
    assert item.metadata["grading"] == "task_compliance"
    assert item.metadata["fallback"] is True


def test_probe_disables_readiness_fallback_by_default(monkeypatch):
    probe = _load_probe_module()

    monkeypatch.delenv("AGENT_BENCH_ALLOW_READINESS_FALLBACK", raising=False)

    assert probe._allow_readiness_fallback() is False


def test_probe_all_invalid_evaluations_fail_top_level_status():
    probe = _load_probe_module()
    evaluations = [
        {"status": "failed_harness_setup", "passed": False, "score": 0.0},
        {"status": "failed_harness_setup", "passed": False, "score": 0.0},
    ]

    status, error = probe._overall_status_and_error(evaluations, "")

    assert status == "failed_harness_setup"
    assert "All 2 benchmark record evaluation(s) were invalid" in error


def test_probe_model_failures_remain_completed_benchmark_status():
    probe = _load_probe_module()
    evaluations = [
        {"status": "failed_model_answer", "passed": False, "score": 0.0},
    ]

    status, error = probe._overall_status_and_error(evaluations, "")

    assert status == "completed"
    assert error == ""


def test_probe_all_timeout_evaluations_fail_top_level_status():
    probe = _load_probe_module()
    evaluations = [
        {"status": "timed_out", "passed": False, "score": 0.0, "timed_out": True},
    ]

    status, error = probe._overall_status_and_error(evaluations, "")

    assert status == "timed_out"
    assert "timed out" in error


def test_probe_status_counts_include_dataset_extraction_failure_when_no_items(tmp_path, monkeypatch):
    probe = _load_probe_module()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENT_BENCH_PROVIDER", "openai-compatible")
    monkeypatch.setenv("AGENT_BENCH_BASE_URL", "http://model.test/v1")
    monkeypatch.setenv("AGENT_BENCH_MODEL", "model")
    monkeypatch.setenv("AGENT_BENCH_OUTPUT_DIR", str(tmp_path / "outputs"))
    monkeypatch.setattr("sys.argv", ["agent-bench-probe", "--benchmark", "ExampleBench"])
    (tmp_path / "README.md").write_text("# Empty benchmark\nNo rows here.\n", encoding="utf-8")

    exit_code = probe.main()

    assert exit_code == 2
    payload = json.loads((tmp_path / "outputs" / "agent_bench_result.json").read_text(encoding="utf-8"))
    assert payload["status"] == "failed_dataset_extraction"
    assert payload["status_counts"] == {"failed_dataset_extraction": 1}


def test_probe_recovers_wrapped_judge_json():
    probe = _load_probe_module()

    assert probe.parse_judge_json('{"score": 0.8, "passed": true, "reason": "ok"}')["score"] == 0.8
    assert probe.parse_judge_json('  {"score": 1, "passed": true, "reason": "ok"}  ')["passed"] is True
    assert probe.parse_judge_json('{"score": 0.5, "passed": false, "reason": "partial"}</think>')[
        "reason"
    ] == "partial"
    assert probe.parse_judge_json(
        '{"score": 0.1, "passed": false, "reason": "first"}</think>'
        '{"score": 0.9, "passed": true, "reason": "second"}'
    )["reason"] == "first"
    assert probe.parse_judge_json(
        'service artifact </think> {"score": 0.7, "passed": false, "reason": "usable"} trailing text'
    )["score"] == 0.7


def test_probe_extracts_provider_content_variants():
    probe = _load_probe_module()

    assert (
        probe.extract_openai_content(
            {"choices": [{"message": {"content": [{"type": "text", "text": "{\"answer\":\"A\"}"}]}}]}
        )
        == '{"answer":"A"}'
    )
    assert (
        probe.extract_openai_content(
            {"choices": [{"message": {"content": "", "reasoning_content": '{"score":1,"passed":true,"reason":"ok"}'}}]}
        )
        == '{"score":1,"passed":true,"reason":"ok"}'
    )


def test_probe_rejects_judge_text_without_json():
    probe = _load_probe_module()

    try:
        probe.parse_judge_json("not json")
    except ValueError as exc:
        assert "judge content was not a JSON object" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_probe_classifies_unsupported_capabilities(monkeypatch):
    probe = _load_probe_module()
    monkeypatch.setenv("AGENT_BENCH_REQUIRED_CAPABILITIES", "repo_patch,chat_answer,browser_or_gui")

    required = probe._required_capabilities("ExampleBench")

    assert required == {"repo_patch", "chat_answer", "browser_or_gui"}
    assert sorted(required - probe.HARNESS_SUPPORTED_CAPABILITIES) == ["browser_or_gui"]


def test_probe_does_not_treat_generic_tools_as_tool_call_support(monkeypatch):
    probe = _load_probe_module()
    monkeypatch.setenv("AGENT_BENCH_REQUIRED_CAPABILITIES", "tool_call,external_data_required")

    required = probe._required_capabilities("ToolBench")
    adapter = probe.select_adapter(required)
    contract = probe.capability_contract_for(required, adapter)

    assert "tool_call" not in probe.HARNESS_SUPPORTED_CAPABILITIES
    assert adapter.supported_capabilities() == {"chat_answer", "external_data_required"}
    assert contract["tool_call"]["supported"] is False
    assert "benchmark-native" in contract["tool_call"]["reason"]


def test_probe_base_payload_reports_only_required_supported_capabilities():
    probe = _load_probe_module()
    payload = probe._base_result_payload(
        args=type("Args", (), {"benchmark": "codeneedle", "kind": "repository"})(),
        files=["README.md"],
        markers=["README.md"],
        sample_limit=3,
        required_capabilities=["chat_answer"],
        unsupported_capabilities=[],
        adapter=probe.ChatAnswerAdapter(),
        capability_contract={"chat_answer": {"supported": True}},
    )

    assert payload["supported_capabilities"] == ["chat_answer"]


def test_probe_base_payload_uses_verified_capability_contract():
    probe = _load_probe_module()
    payload = probe._base_result_payload(
        args=type("Args", (), {"benchmark": "SWE-bench", "kind": "repository"})(),
        files=["README.md"],
        markers=["README.md"],
        sample_limit=3,
        required_capabilities=["repo_patch"],
        unsupported_capabilities=[],
        adapter=probe.RepoPatchAdapter(),
        capability_contract={"repo_patch": {"supported": False}},
    )

    assert payload["supported_capabilities"] == []


def test_probe_repo_patch_contract_missing_metadata_is_setup_failure(monkeypatch):
    probe = _load_probe_module()
    monkeypatch.delenv("AGENT_BENCH_ALLOW_TARGET_CHECKOUT", raising=False)
    item = probe.BenchmarkItem(
        "Fix the bug.",
        "diff --git a/a.py b/a.py",
        "huggingface:swe-lancer/test:1",
        metadata={"grading": "exact"},
    )

    contract = probe.RepoPatchAdapter().capability_contract({"repo_patch"}, [item])

    assert contract["repo_patch"]["supported"] is False
    assert contract["repo_patch"]["workspace"] is False
    assert contract["repo_patch"]["missing_metadata_count"] == 1
    assert "missing target repo/base_commit metadata" in contract["repo_patch"]["reason"]


def test_probe_repo_patch_missing_target_checkout_fails_setup(monkeypatch):
    probe = _load_probe_module()
    monkeypatch.delenv("AGENT_BENCH_ALLOW_TARGET_CHECKOUT", raising=False)
    monkeypatch.delenv("AGENT_BENCH_TARGET_REPO_ROOT", raising=False)
    item = probe.BenchmarkItem(
        "Fix the bug.",
        "diff --git a/a.py b/a.py",
        "huggingface:swe-bench/test:1",
        metadata={"repo": "astropy/astropy", "base_commit": "abc123", "grading": "exact"},
    )

    result = probe.run_model_on_item("SWE-bench", item, probe.RepoPatchAdapter())

    assert result["status"] == "failed_harness_setup"
    assert result["passed"] is False
    assert "target repository checkout was not materialized" in result["error"]


def test_probe_repo_patch_contract_requires_canary(monkeypatch):
    probe = _load_probe_module()
    monkeypatch.setattr(probe, "_repo_patch_checkout_available", lambda items: True)
    monkeypatch.setattr(probe, "_repo_patch_canary", lambda: {"passed": False, "reason": "repo canary failed"})
    item = probe.BenchmarkItem(
        "Fix the bug.",
        "diff --git a/a.py b/a.py",
        "huggingface:swe-bench/test:1",
        metadata={"repo": "astropy/astropy", "base_commit": "abc123", "grading": "exact"},
    )

    contract = probe.RepoPatchAdapter().capability_contract({"repo_patch"}, [item])

    assert contract["repo_patch"]["supported"] is False
    assert contract["repo_patch"]["reason"] == "repo canary failed"
    assert contract["repo_patch"]["canary"]["passed"] is False


def test_probe_repo_patch_grading_rejects_reference_diff(monkeypatch):
    probe = _load_probe_module()
    monkeypatch.delenv("AGENT_BENCH_REPO_PATCH_GRADER", raising=False)
    item = probe.BenchmarkItem(
        "Fix the bug.",
        "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n",
        "huggingface:swe-bench/test:1",
        metadata={"repo": "astropy/astropy", "base_commit": "abc123", "reference_patch": "same"},
    )

    score, grade = probe.RepoPatchAdapter().grade(
        "SWE-bench",
        item,
        probe.OutputBundle(patch="diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n"),
    )

    assert score == 0.0
    assert grade["status"] == "failed_harness_setup"
    assert grade["method"] == "official_patch_tests"
    assert "reference-diff exact matching is intentionally not used" in grade["reason"]


def test_probe_file_artifact_missing_assets_fails_preflight(tmp_path, monkeypatch):
    probe = _load_probe_module()
    monkeypatch.chdir(tmp_path)

    result = probe.run_model_on_item(
        "GDPval",
        probe.BenchmarkItem("Create the spreadsheet.", "rubric", "gdpval:1", metadata={"grading": "rubric"}),
        probe.FileArtifactAdapter(),
    )

    assert result["status"] == "failed_missing_assets"
    assert result["passed"] is False


def test_probe_file_artifact_contract_requires_provisioned_assets(tmp_path, monkeypatch):
    probe = _load_probe_module()
    monkeypatch.chdir(tmp_path)
    item = probe.BenchmarkItem(
        "Create the spreadsheet.",
        "rubric",
        "gdpval:1",
        metadata={"grading": "rubric"},
    )

    contract = probe.FileArtifactAdapter().capability_contract({"file_artifact"}, [item])

    assert contract["file_artifact"]["supported"] is False
    assert contract["file_artifact"]["workspace"] is False
    assert contract["file_artifact"]["assets_provisioned_count"] == 0


def test_probe_file_artifact_preflight_skips_without_model_call(tmp_path, monkeypatch):
    probe = _load_probe_module()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENT_BENCH_PROVIDER", "openai-compatible")
    monkeypatch.setenv("AGENT_BENCH_BASE_URL", "http://model.test/v1")
    monkeypatch.setenv("AGENT_BENCH_MODEL", "model")
    monkeypatch.setenv("AGENT_BENCH_OUTPUT_DIR", str(tmp_path / "outputs"))
    monkeypatch.setenv("AGENT_BENCH_REQUIRED_CAPABILITIES", "file_artifact")
    monkeypatch.setattr("sys.argv", ["agent-bench-probe", "--benchmark", "PaperBench"])
    (tmp_path / "tasks.json").write_text(
        json.dumps(
            [
                {
                    "question": "Create the report from the missing assets.",
                    "rubric": "Generated report must satisfy the task.",
                }
            ]
        ),
        encoding="utf-8",
    )

    def fail_model_call(*args, **kwargs):
        raise AssertionError("model evaluation should be skipped by preflight")

    monkeypatch.setattr(probe, "run_model_evaluations", fail_model_call)

    exit_code = probe.main()
    payload = json.loads((tmp_path / "outputs" / "agent_bench_result.json").read_text(encoding="utf-8"))
    item_dir = next((tmp_path / "outputs" / "items").iterdir())

    assert exit_code == 2
    assert payload["status"] == "failed_missing_assets"
    assert payload["status_counts"] == {"failed_missing_assets": 1}
    assert payload["model_evals"][0]["status"] == "failed_missing_assets"
    assert (item_dir / "item.json").is_file()
    assert (item_dir / "setup_error.json").is_file()
    assert (item_dir / "item_result.json").is_file()


def test_probe_incomplete_prompt_template_fails_dataset_preflight(tmp_path, monkeypatch):
    probe = _load_probe_module()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENT_BENCH_PROVIDER", "openai-compatible")
    monkeypatch.setenv("AGENT_BENCH_BASE_URL", "http://model.test/v1")
    monkeypatch.setenv("AGENT_BENCH_MODEL", "model")
    monkeypatch.setenv("AGENT_BENCH_OUTPUT_DIR", str(tmp_path / "outputs"))
    monkeypatch.setattr("sys.argv", ["agent-bench-probe", "--benchmark", "EDINET-Bench"])
    (tmp_path / "tasks.json").write_text(
        json.dumps(
            [
                {
                    "question": "Extract the required financial facts; the report is as follows:",
                    "answer": '{"company_name":"..."}',
                }
            ]
        ),
        encoding="utf-8",
    )

    def fail_model_call(*args, **kwargs):
        raise AssertionError("model evaluation should be skipped by preflight")

    monkeypatch.setattr(probe, "run_model_evaluations", fail_model_call)

    exit_code = probe.main()
    payload = json.loads((tmp_path / "outputs" / "agent_bench_result.json").read_text(encoding="utf-8"))

    assert exit_code == 2
    assert payload["status"] == "failed_dataset_extraction"
    assert payload["model_evals"][0]["status"] == "failed_dataset_extraction"


def test_probe_file_artifact_requires_generated_output(tmp_path, monkeypatch):
    probe = _load_probe_module()
    monkeypatch.chdir(tmp_path)
    (tmp_path / "input.txt").write_text("source data", encoding="utf-8")

    def fake_loop(benchmark, item, tools=None):
        return {"answer": "done", "content": '{"answer":"done"}', "usage": {}, "tool_trace": []}

    monkeypatch.setattr(probe, "run_agent_loop", fake_loop)
    item = probe.BenchmarkItem(
        "Create the report.",
        "rubric",
        "artifact:1",
        metadata={"grading": "rubric", "input_files": ["input.txt"]},
    )

    result = probe.run_model_on_item("PaperBench", item, probe.FileArtifactAdapter())

    assert result["status"] == "failed_model_answer"
    assert "did not produce any files" in result["error"]


def test_probe_file_artifact_rejects_corrupt_xlsx(tmp_path, monkeypatch):
    probe = _load_probe_module()
    monkeypatch.chdir(tmp_path)
    (tmp_path / "input.txt").write_text("source data", encoding="utf-8")

    def fake_loop(benchmark, item, tools=None):
        output_dir = Path.cwd() / "agent_bench_outputs" / probe._safe_slug(item.source)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "report.xlsx").write_text("not a zip file", encoding="utf-8")
        return {"answer": "done", "content": '{"answer":"done"}', "usage": {}, "tool_trace": []}

    monkeypatch.setattr(probe, "run_agent_loop", fake_loop)
    item = probe.BenchmarkItem(
        "Create the spreadsheet.",
        "rubric",
        "artifact:1",
        metadata={"grading": "rubric", "input_files": ["input.txt"]},
    )

    result = probe.run_model_on_item("GDPval", item, probe.FileArtifactAdapter())

    assert result["status"] == "failed_model_tool_use"
    assert result["grade"]["method"] == "artifact_integrity"


def test_probe_file_artifact_persists_inputs_outputs_and_item_files(tmp_path, monkeypatch):
    probe = _load_probe_module()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENT_BENCH_OUTPUT_DIR", str(tmp_path / "outputs"))
    (tmp_path / "input.txt").write_text("source data", encoding="utf-8")

    def fake_loop(benchmark, item, tools=None):
        output_dir = Path.cwd() / "agent_bench_outputs" / probe._safe_slug(item.source)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "report.txt").write_text("answer artifact", encoding="utf-8")
        return {"answer": "done", "content": '{"answer":"done"}', "usage": {"total_tokens": 3}, "tool_trace": []}

    def fake_judge(benchmark, item, answer, method):
        return 1.0, {"method": method, "score": 1.0, "status": "passed", "passed": True, "reason": "ok"}

    monkeypatch.setattr(probe, "run_agent_loop", fake_loop)
    monkeypatch.setattr(probe, "judge_answer", fake_judge)
    item = probe.BenchmarkItem(
        "Create the report.",
        "rubric",
        "artifact:1",
        metadata={"grading": "rubric", "input_files": ["input.txt"]},
    )

    result = probe.run_model_on_item("PaperBench", item, probe.FileArtifactAdapter())
    item_dir = probe._item_output_dir(item)

    assert result["status"] == "passed"
    assert (item_dir / "item.json").is_file()
    assert (item_dir / "workspace.json").is_file()
    assert (item_dir / "agent_run.json").is_file()
    assert (item_dir / "output_bundle.json").is_file()
    assert (item_dir / "item_result.json").is_file()
    assert (item_dir / "artifacts" / "report.txt").read_text(encoding="utf-8") == "answer artifact"
    workspace = json.loads((item_dir / "workspace.json").read_text(encoding="utf-8"))
    assert workspace["metadata"]["input_assets"] == ["agent_bench_task_inputs/artifact-1/input.txt"]


def test_probe_file_artifact_uses_cached_assets_outside_repo(tmp_path, monkeypatch):
    probe = _load_probe_module()
    asset_root = tmp_path / "asset-cache"
    cached = asset_root / "paperbench" / "paper-1"
    cached.mkdir(parents=True)
    (cached / "paper.pdf").write_bytes(b"%PDF-1.4\n")
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    monkeypatch.setenv("AGENT_BENCH_ASSET_ROOT", str(asset_root))
    monkeypatch.setenv("AGENT_BENCH_BENCHMARK_NAME", "PaperBench")
    item = probe.BenchmarkItem("Write a review.", "rubric", "paper:1", metadata={"grading": "rubric"})

    workspace = probe.FileArtifactAdapter().prepare_task(item)

    assert workspace.metadata["input_assets"] == ["agent_bench_task_inputs/paper-1/paper.pdf"]
    assert (workspace.root / "agent_bench_task_inputs" / "paper-1" / "paper.pdf").is_file()


def test_probe_chat_adapter_uses_sanitized_workspace(tmp_path, monkeypatch):
    probe = _load_probe_module()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENT_BENCH_OUTPUT_DIR", str(tmp_path / "outputs"))
    (tmp_path / "problems.csv").write_text("question,answer_rubric\nq,secret rubric\n", encoding="utf-8")
    item = probe.BenchmarkItem(
        "Answer from the visible task only.",
        "secret rubric",
        "problems.csv:2",
        metadata={"grading": "exact", "expected_key": "answer_rubric"},
    )

    workspace = probe.ChatAnswerAdapter().prepare_task(item)

    assert workspace.root != tmp_path
    assert (workspace.root / "TASK.md").is_file()
    assert not (workspace.root / "problems.csv").exists()
    assert "secret rubric" not in (workspace.root / "TASK.md").read_text(encoding="utf-8")


def test_probe_file_artifact_workspace_exposes_only_task_inputs(tmp_path, monkeypatch):
    probe = _load_probe_module()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENT_BENCH_OUTPUT_DIR", str(tmp_path / "outputs"))
    (tmp_path / "COA.xlsx").write_bytes(b"fake xlsx input")
    (tmp_path / "answer_key.txt").write_text("gold", encoding="utf-8")
    item = probe.BenchmarkItem(
        "Use attached file COA.xlsx to create the report.",
        "rubric",
        "gdpval:1",
        metadata={"grading": "rubric", "input_files": ["COA.xlsx"]},
    )

    workspace = probe.FileArtifactAdapter().prepare_task(item)

    assert (workspace.root / "COA.xlsx").exists()
    assert not (workspace.root / "answer_key.txt").exists()
    manifest = json.loads((workspace.root / "agent_bench_task_inputs" / "manifest.json").read_text())
    assert manifest["files"][0]["display_filename"] == "COA.xlsx"
    assert manifest["files"][0]["tool_path"] == "agent_bench_task_inputs/gdpval-1/COA.xlsx"


def test_probe_file_artifact_contract_requires_canary(tmp_path, monkeypatch):
    probe = _load_probe_module()
    monkeypatch.chdir(tmp_path)
    (tmp_path / "input.txt").write_text("source", encoding="utf-8")
    monkeypatch.setattr(
        probe,
        "_file_artifact_canary",
        lambda: {"passed": False, "reason": "file canary failed"},
    )
    item = probe.BenchmarkItem(
        "Create an artifact.",
        "rubric",
        "artifact:1",
        metadata={"grading": "rubric", "input_files": ["input.txt"]},
    )

    contract = probe.FileArtifactAdapter().capability_contract({"file_artifact"}, [item])

    assert contract["file_artifact"]["supported"] is False
    assert contract["file_artifact"]["reason"] == "file canary failed"


def test_probe_file_artifact_canary_passes_in_minimal_environment(tmp_path, monkeypatch):
    probe = _load_probe_module()
    monkeypatch.setenv("AGENT_BENCH_OUTPUT_DIR", str(tmp_path / "outputs"))

    result = probe._file_artifact_canary()

    assert result["passed"] is True
    assert result["collected_count"] == 2


def test_probe_read_spreadsheet_reads_xlsx_without_openpyxl_dependency(tmp_path, monkeypatch):
    probe = _load_probe_module()
    monkeypatch.chdir(tmp_path)
    probe._write_minimal_xlsx(tmp_path / "sample.xlsx", [["name", "value"], ["canary", "ok"]])

    preview = probe.tool_read_spreadsheet({"path": "sample.xlsx", "max_rows": 5})

    assert "canary" in preview
    assert "openpyxl is not installed" not in preview


def test_probe_extracts_canonical_answer_from_rubric_for_grader_side_use():
    probe = _load_probe_module()
    item = probe.item_from_record(
        {
            "question": "Identify the species from the hidden biological clues.",
            "answer_rubric": "The answer is Homo sapiens Score 1. Any other answer Score 0.",
        },
        "biomystery:1",
    )
    compact_item = probe.item_from_record(
        {"question": "Identify the regulator from the clues.", "answer_rubric": "CTCF Score 1"},
        "biomystery:2",
    )

    assert item is not None
    assert item.expected == "Homo sapiens"
    assert item.metadata["grading"] == "exact"
    assert probe.score_answer('{"answer":"Homo sapiens","confidence":0.95}', item.expected, {}) == 1.0
    assert compact_item is not None
    assert compact_item.expected == "CTCF"
    assert probe.score_answer("CTCF", compact_item.expected, {}) == 1.0


def test_probe_disables_biomystery_scoring_from_local_zip(tmp_path, monkeypatch):
    probe = _load_probe_module()
    monkeypatch.setenv("AGENT_BENCH_BENCHMARK_NAME", "BioMystery Bench")
    monkeypatch.setenv("AGENT_BENCH_DATASET_ID", "Anthropic/BioMysteryBench-preview")
    with zipfile.ZipFile(tmp_path / "data.zip", "w") as archive:
        archive.writestr(
            "records.jsonl",
            json.dumps(
                {
                    "question": "Identify the species from the hidden biological clues.",
                    "answer_rubric": "The answer is Homo sapiens Score 1. Any other answer Score 0.",
                }
            )
            + "\n",
        )

    items, errors = probe.extract_benchmark_items(tmp_path, limit=3)

    assert items == []
    assert errors == [
        "BioMystery Bench scoring is disabled until answer_rubric/gold labels are kept grader-side only"
    ]


def test_probe_text_tools_skip_binary_files(tmp_path, monkeypatch):
    probe = _load_probe_module()
    monkeypatch.chdir(tmp_path)
    (tmp_path / "notes.txt").write_text("needle\n", encoding="utf-8")
    (tmp_path / "archive.zip").write_bytes(b"needle\x00payload")

    assert "notes.txt:1: needle" in probe.tool_search_files({"query": "needle"})
    assert "archive.zip" not in probe.tool_search_files({"query": "needle"})
    assert probe.tool_read_file({"path": "archive.zip"}).startswith("Refusing to read binary/archive")


def test_probe_run_command_uses_argv_and_rejects_shell_syntax(tmp_path, monkeypatch):
    probe = _load_probe_module()
    monkeypatch.chdir(tmp_path)

    ok = probe.tool_run_command({"argv": [sys.executable, "-c", "print('ok')"]})
    rejected = probe.tool_run_command({"command": "python - <<'PY'\nprint('bad')\nPY"})

    assert "exit_code=0" in ok
    assert "ok" in ok
    assert "legacy command field is disabled" in rejected


def test_probe_final_answer_keeps_first_non_empty_answer():
    probe = _load_probe_module()
    messages = []
    trace = []
    answer = probe.handle_native_tool_calls(
        [
            {
                "id": "call_1",
                "function": {"name": "final_answer", "arguments": json.dumps({"answer": "A"})},
            },
            {
                "id": "call_2",
                "function": {"name": "final_answer", "arguments": json.dumps({"answer": ""})},
            },
        ],
        messages,
        trace,
    )

    assert answer == "A"
    assert [item["tool"] for item in trace] == ["final_answer"]


def test_probe_does_not_extract_benchmark_plan_as_task(tmp_path):
    probe = _load_probe_module()
    (tmp_path / "benchmark_plan.md").write_text(
        "# Benchmark Plan\n\n"
        "This document explains implementation requirements and evaluation strategy. " * 4,
        encoding="utf-8",
    )

    items, errors = probe.extract_benchmark_items(tmp_path, limit=3)

    assert items == []
    assert errors == []


def test_probe_extracts_python_task_records_with_blank_answer(tmp_path):
    probe = _load_probe_module()
    task_file = tmp_path / "tasks.py"
    task_file.write_text(
        '''
SYSTEM_PROMPT = "Use the available tools."

def get_task():
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Find Jordan's email and update her Salesforce phone number."},
        ],
        "answer": "",
    }
''',
        encoding="utf-8",
    )

    items, errors = probe.extract_benchmark_items(tmp_path, limit=3)

    assert errors == []
    assert len(items) == 1
    assert "Jordan" in items[0].question
    assert items[0].metadata["grading"] == "task_compliance"


def test_probe_extracts_benchmark_description_text(tmp_path):
    probe = _load_probe_module()
    path = tmp_path / "mlebench" / "competitions" / "sample" / "description.md"
    path.parent.mkdir(parents=True)
    path.write_text(
        "Build a model for this competition. " * 8
        + "Submit predictions in the required CSV format.",
        encoding="utf-8",
    )

    items, errors = probe.extract_benchmark_items(tmp_path, limit=3)

    assert errors == []
    assert len(items) == 1
    assert "Build a model" in items[0].question
    assert items[0].metadata["grading"] == "rubric"


def test_probe_extracts_investorbench_market_records(tmp_path, monkeypatch):
    probe = _load_probe_module()
    monkeypatch.setenv("AGENT_BENCH_BENCHMARK_NAME", "InvestorBench")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "abc.json").write_text(
        json.dumps(
            {
                "2026-01-01": {"prices": 100, "news": ["Positive product launch."]},
                "2026-01-02": {"prices": 103, "news": []},
            }
        ),
        encoding="utf-8",
    )

    items, errors = probe.extract_benchmark_items(tmp_path, limit=3)

    assert errors == []
    assert len(items) == 1
    assert "InvestorBench trading decision" in items[0].question
    assert items[0].expected == "buy"
    assert items[0].choices == {"A": "buy", "B": "sell", "C": "hold"}


def test_probe_does_not_synthesize_hle_format_task(tmp_path, monkeypatch):
    probe = _load_probe_module()
    monkeypatch.setenv("AGENT_BENCH_BENCHMARK_NAME", "Humanity's Last Exam")
    (tmp_path / "README.md").write_text("HLE benchmark repository\n" * 10, encoding="utf-8")
    runner = tmp_path / "hle_eval" / "run_model_predictions.py"
    runner.parent.mkdir()
    runner.write_text('SYSTEM_PROMPT = "Use Explanation/Answer/Confidence."\n', encoding="utf-8")

    items, errors = probe.extract_benchmark_items(tmp_path, limit=3)

    assert items == []
    assert errors == ["Humanity's Last Exam requires accessible dataset records; format-only fallback is disabled"]


def test_probe_classifies_model_request_timeout(monkeypatch):
    probe = _load_probe_module()

    def fake_loop(benchmark, item):
        raise probe.ChatCompletionTimeoutError(12.0)

    monkeypatch.setattr(probe, "run_agent_loop", fake_loop)

    result = probe.run_model_on_item("ExampleBench", probe.BenchmarkItem("q", "a", "source"))

    assert result["status"] == "timed_out"
    assert result["timed_out"] is True
    assert result["error"] == "model request timed out after 12.0s"


def test_probe_classifies_judge_request_timeout(monkeypatch):
    probe = _load_probe_module()

    def fake_post(base_url, payload, headers):
        raise probe.ChatCompletionTimeoutError(12.0)

    monkeypatch.setenv("AGENT_BENCH_BASE_URL", "http://model.test/v1")
    monkeypatch.setenv("AGENT_BENCH_MODEL", "model")
    monkeypatch.setattr(probe, "post_chat_completion", fake_post)

    score, grade = probe.judge_answer(
        "ExampleBench",
        probe.BenchmarkItem("q", "rubric", "source", metadata={"grading": "rubric"}),
        "answer",
        "rubric",
    )

    assert score == 0.0
    assert grade["status"] == "timed_out"
    assert grade["timed_out"] is True


def test_probe_rejects_placeholder_judge_reason(monkeypatch):
    probe = _load_probe_module()

    def fake_post(base_url, payload, headers):
        return json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "content": '{"score":1.0,"passed":true,"reason":"short reason"}'
                        }
                    }
                ]
            }
        )

    monkeypatch.setenv("AGENT_BENCH_BASE_URL", "http://model.test/v1")
    monkeypatch.setenv("AGENT_BENCH_MODEL", "model")
    monkeypatch.setattr(probe, "post_chat_completion", fake_post)

    score, grade = probe.judge_answer(
        "ExampleBench",
        probe.BenchmarkItem("q", "rubric", "source", metadata={"grading": "rubric"}),
        "answer",
        "rubric",
    )

    assert score == 0.0
    assert grade["status"] == "failed_grader"
    assert "placeholder reason" in grade["reason"]


def test_probe_financemath_uses_deterministic_numeric_grader(monkeypatch):
    probe = _load_probe_module()
    monkeypatch.setenv("AGENT_BENCH_BENCHMARK_NAME", "FinanceMath")
    item = probe.item_from_record(
        {
            "question": "What is revenue in dollars?",
            "ground_truth": "$1.2 million",
        },
        "finance_math.jsonl:1",
    )

    assert item is not None
    assert item.metadata["grading"] == "numeric"
    assert probe.grade_answer("FinanceMath", item, '{"answer":"1,200,000"}')[0] == 1.0
    assert probe.grade_answer("FinanceMath", item, '{"answer":"short reason"}')[0] == 0.0


def test_probe_financemath_rejects_non_numeric_solution(monkeypatch):
    probe = _load_probe_module()
    monkeypatch.setenv("AGENT_BENCH_BENCHMARK_NAME", "FinanceMath")

    item = probe.item_from_record(
        {
            "question": "Compute the value.",
            "python_solution": "def solve():\n    return result\n",
        },
        "finance_math.jsonl:2",
    )

    assert item is None


def test_probe_token_budget_preflight_stops_before_model_call(monkeypatch):
    probe = _load_probe_module()
    monkeypatch.setenv("AGENT_BENCH_PROVIDER", "openai-compatible")
    monkeypatch.setenv("AGENT_BENCH_BASE_URL", "http://model.test/v1")
    monkeypatch.setenv("AGENT_BENCH_MODEL", "model")
    monkeypatch.setenv("AGENT_BENCH_MODEL_CONTEXT_LIMIT", "2048")

    def fail_post(*args, **kwargs):
        raise AssertionError("model call should not happen")

    monkeypatch.setattr(probe, "post_chat_completion_with_variant", fail_post)

    result = probe.run_agent_loop("ExampleBench", probe.BenchmarkItem("q", "a", "source"))

    assert result["diagnostics"]["status"] == "failed_token_budget"
    assert result["answer"] == ""


def test_probe_protocol_limit_detects_repeated_identical_tool_calls():
    probe = _load_probe_module()
    trace = [
        {"tool": "read_file", "arguments": {"path": "a.txt"}, "failed": False},
        {"tool": "read_file", "arguments": {"path": "a.txt"}, "failed": False},
        {"tool": "read_file", "arguments": {"path": "a.txt"}, "failed": False},
        {"tool": "read_file", "arguments": {"path": "a.txt"}, "failed": False},
    ]

    reason = probe._protocol_limit_failure(trace, final_answer_count=0, final_content="")

    assert "repeated an identical tool call" in reason


def test_agent_loop_executes_native_tool_calls(tmp_path, monkeypatch):
    probe = _load_probe_module()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENT_BENCH_PROVIDER", "openai-compatible")
    monkeypatch.setenv("AGENT_BENCH_BASE_URL", "http://model.test/v1")
    monkeypatch.setenv("AGENT_BENCH_MODEL", "model")
    (tmp_path / "answer.txt").write_text("answer: 42\n", encoding="utf-8")
    calls = []

    def fake_post(base_url, payload, headers):
        calls.append(payload)
        if len(calls) == 1:
            return (
                json.dumps(
                    {
                        "choices": [
                            {
                                "message": {
                                    "content": "",
                                    "tool_calls": [
                                        {
                                            "id": "call_1",
                                            "type": "function",
                                            "function": {
                                                "name": "read_file",
                                                "arguments": json.dumps({"path": "answer.txt"}),
                                            },
                                        }
                                    ],
                                }
                            }
                        ],
                        "usage": {"total_tokens": 10},
                    }
                ),
                payload,
            )
        return (
            json.dumps(
                {
                    "choices": [{"message": {"content": '{"answer":"42","confidence":1.0}'}}],
                    "usage": {"total_tokens": 5},
                }
            ),
            payload,
        )

    monkeypatch.setattr(probe, "post_chat_completion_with_variant", fake_post)

    result = probe.run_agent_loop("ExampleBench", probe.BenchmarkItem("Read the answer.", "42", "test"))

    assert result["answer"] == "42"
    assert calls[0]["tools"]
    assert calls[1]["messages"][-1]["role"] == "tool"
    assert result["tool_trace"][0]["tool"] == "read_file"
    assert result["usage"]["total_tokens"] == 15


def test_agent_loop_executes_text_tool_requests_when_native_tools_are_unavailable(tmp_path, monkeypatch):
    probe = _load_probe_module()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENT_BENCH_PROVIDER", "openai-compatible")
    monkeypatch.setenv("AGENT_BENCH_BASE_URL", "http://model.test/v1")
    monkeypatch.setenv("AGENT_BENCH_MODEL", "model")
    (tmp_path / "answer.txt").write_text("answer: blue\n", encoding="utf-8")
    calls = []

    def fake_post(base_url, payload, headers):
        calls.append(payload)
        if len(calls) == 1:
            used_payload = dict(payload)
            used_payload.pop("tools", None)
            used_payload.pop("tool_choice", None)
            return (
                json.dumps(
                    {
                        "choices": [
                            {
                                "message": {
                                    "content": '{"tool":"read_file","arguments":{"path":"answer.txt"}}'
                                }
                            }
                        ]
                    }
                ),
                used_payload,
            )
        return json.dumps({"choices": [{"message": {"content": '{"answer":"blue"}'}}]}), payload

    monkeypatch.setattr(probe, "post_chat_completion_with_variant", fake_post)

    result = probe.run_agent_loop("ExampleBench", probe.BenchmarkItem("Read the answer.", "blue", "test"))

    assert result["answer"] == "blue"
    assert "tools" in calls[0]
    assert "tools" not in calls[1]
    assert calls[1]["messages"][-1]["content"].startswith("Tool result for read_file")
