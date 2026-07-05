import importlib.util
import json
import subprocess
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


def test_probe_fails_before_model_call_when_required_tool_missing(tmp_path, monkeypatch):
    probe = _load_probe_module()
    monkeypatch.setenv("AGENT_BENCH_OUTPUT_DIR", str(tmp_path / "outputs"))

    class MissingToolAdapter(probe.BenchmarkAdapter):
        def available_tools(self, item, workspace):
            return [{"type": "function", "function": {"name": "quote_lookup"}}]

        def run_agent_loop(self, benchmark, item, workspace, tools):
            raise AssertionError("agent loop should be skipped by required-tool preflight")

    item = probe.BenchmarkItem(
        "Call the benchmark-selected finance API.",
        "Use the selected tool.",
        "synthetic:1",
        metadata={"required_tools": ["earnings_calendar"]},
    )

    result = MissingToolAdapter().evaluate_item("ToolBench", item)

    assert result["status"] == "failed_missing_required_tool"
    assert result["setup_details"]["blocker_type"] == "missing_required_tool"
    assert result["setup_details"]["missing_tools"] == ["earnings_calendar"]
    assert result["setup_details"]["exposed_tools"] == ["quote_lookup"]
    assert result["included_in_official_score"] is False


def test_probe_descriptor_required_tools_participate_in_preflight(tmp_path, monkeypatch):
    probe = _load_probe_module()
    monkeypatch.setenv("AGENT_BENCH_OUTPUT_DIR", str(tmp_path / "outputs"))
    monkeypatch.setenv(
        "AGENT_BENCH_BENCHMARK_JSON",
        json.dumps({"name": "ExploitBench", "required_tools": ["exploitbench"]}),
    )

    def fail_model_call(*args, **kwargs):
        raise AssertionError("model evaluation should be skipped by descriptor required-tool preflight")

    monkeypatch.setattr(probe.ToolCallAdapter, "run_agent_loop", fail_model_call)
    item = probe.BenchmarkItem(
        "Run the upstream challenge.",
        "Use upstream oracle.",
        "benchmarks/v8.yaml",
        metadata={"live_tools_required": True},
    )

    result = probe.ToolCallAdapter().evaluate_item("ExploitBench", item)

    assert result["status"] == "failed_missing_required_tool"
    assert result["required_tools"] == ["exploitbench"]
    assert result["missing_tools"] == ["exploitbench"]
    assert result["included_in_official_score"] is False


def test_probe_exploitbench_missing_backend_is_listed_and_excluded(tmp_path, monkeypatch):
    probe = _load_probe_module()
    monkeypatch.setenv("AGENT_BENCH_OUTPUT_DIR", str(tmp_path / "outputs"))
    monkeypatch.setenv("AGENT_BENCH_REQUIRED_CAPABILITIES", "tool_call,external_data_required")

    def fail_model_call(*args, **kwargs):
        raise AssertionError("model evaluation should be skipped by missing exploit backend preflight")

    monkeypatch.setattr(probe.ToolCallAdapter, "run_agent_loop", fail_model_call)
    item = probe.BenchmarkItem(
        "Run ExploitBench challenge.",
        "Use upstream oracle.",
        "benchmarks/v8.yaml",
        metadata={"required_tools": ["exploitbench"], "live_tools_required": True},
    )

    result = probe.ToolCallAdapter().evaluate_item("ExploitBench", item)

    assert result["status"] == "failed_missing_required_tool"
    assert result["missing_tools"] == ["exploitbench"]
    assert result["included_in_official_score"] is False
    assert result["capabilities_verified"] is False


def test_probe_finance_agent_v2_missing_tools_are_exact_and_excluded(tmp_path, monkeypatch):
    probe = _load_probe_module()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENT_BENCH_OUTPUT_DIR", str(tmp_path / "outputs"))
    monkeypatch.setenv("AGENT_BENCH_BENCHMARK_NAME", "Finance Agent v2")
    monkeypatch.setenv(probe.FINANCE_AGENT_V2_FIXTURE_ROOT_ENV, str(tmp_path / "missing-fixtures"))
    for env_name in probe.FINANCE_AGENT_V2_REQUIRED_ENV:
        monkeypatch.delenv(env_name, raising=False)

    def fail_model_call(*args, **kwargs):
        raise AssertionError("model evaluation should be skipped when Finance Agent v2 tools are missing")

    monkeypatch.setattr(probe.FinanceAgentV2Adapter, "run_agent_loop", fail_model_call)
    item = probe.BenchmarkItem(
        "Research the company and answer.",
        "Use the official finance-agent tools.",
        "tasks.json:1",
        metadata={"required_tools": sorted(probe.FINANCE_AGENT_V2_REQUIRED_TOOLS), "live_tools_required": True},
    )

    result = probe.FinanceAgentV2Adapter().evaluate_item("Finance Agent v2", item)

    assert result["status"] == "failed_missing_required_tool"
    assert result["missing_tools"] == sorted(probe.FINANCE_AGENT_V2_REQUIRED_TOOLS)
    assert result["capabilities_verified"] is False
    assert result["included_in_official_score"] is False


def test_probe_finance_agent_v2_fixture_mode_exposes_required_tools(tmp_path, monkeypatch):
    probe = _load_probe_module()
    fixture_root = Path(__file__).resolve().parents[1] / "tasks" / "finance-agent-v2" / "fixtures" / "finance_agent_v2"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENT_BENCH_BENCHMARK_NAME", "Finance Agent v2")
    monkeypatch.setenv(probe.FINANCE_AGENT_V2_FIXTURE_ROOT_ENV, str(fixture_root))
    for env_name in probe.FINANCE_AGENT_V2_REQUIRED_ENV:
        monkeypatch.delenv(env_name, raising=False)

    item = probe.BenchmarkItem(
        "For NASDAQ:CRWD, describe AI tailwinds and AI risks.",
        "rubric",
        "data/public.csv:record-000001",
        metadata={"required_tools": sorted(probe.FINANCE_AGENT_V2_REQUIRED_TOOLS), "live_tools_required": True},
    )
    workspace = probe.TaskWorkspace(root=tmp_path, output_dir=tmp_path / "outputs")
    tools = probe.FinanceAgentV2Adapter().available_tools(item, workspace)
    contract = probe.FinanceAgentV2Adapter().capability_contract({"tool_call", "external_data_required"}, [item])
    status = probe._finance_agent_v2_backend_status(tmp_path)

    assert probe._tool_schema_names(tools) == sorted(probe.FINANCE_AGENT_V2_REQUIRED_TOOLS)
    assert status["ready"] is True
    assert all(status["canaries"].values())
    assert contract["tool_call"]["benchmark_required_tools_available"] is True
    assert contract["tool_call"]["missing_tools"] == []
    assert probe.validate_finance_agent_v2_item(item, tmp_path, tools) is None


def test_probe_finance_agent_v2_fixture_tools_return_crwd_evidence(tmp_path, monkeypatch):
    probe = _load_probe_module()
    fixture_root = Path(__file__).resolve().parents[1] / "tasks" / "finance-agent-v2" / "fixtures" / "finance_agent_v2"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(probe.FINANCE_AGENT_V2_FIXTURE_ROOT_ENV, str(fixture_root))

    raw = probe._execute_finance_agent_v2_tool(
        "retrieve_information",
        {"query": "CRWD Charlotte AI AI Detection and Response adversarial AI regulatory compliance"},
    )
    payload = json.loads(raw)
    result = payload["result"]

    assert result["fixture_valid"] is True
    assert result["canary_passed"] is True
    text = json.dumps(result["results"])
    assert "Charlotte AI" in text
    assert "AI Detection and Response" in text
    assert "adversarial AI" in text


def test_probe_finance_agent_v2_malformed_fixture_fails_preflight(tmp_path, monkeypatch):
    probe = _load_probe_module()
    fixture_root = tmp_path / "finance-fixtures"
    fixture_root.mkdir()
    (fixture_root / "manifest.json").write_text(
        json.dumps({"version": 1, "ticker": "CRWD", "files": {}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("AGENT_BENCH_BENCHMARK_NAME", "Finance Agent v2")
    monkeypatch.setenv(probe.FINANCE_AGENT_V2_FIXTURE_ROOT_ENV, str(fixture_root))

    item = probe.BenchmarkItem(
        "Research CRWD AI.",
        "rubric",
        "data/public.csv:record-000001",
        metadata={"required_tools": sorted(probe.FINANCE_AGENT_V2_REQUIRED_TOOLS), "live_tools_required": True},
    )
    result = probe.validate_finance_agent_v2_item(item, tmp_path, probe._finance_agent_v2_tool_schemas())

    assert result is not None
    assert result[0] == "failed_missing_required_tool"
    assert result[2]["blocker_type"] == "missing_required_tool_backend"
    assert result[2]["backend_canary"]["ready"] is False


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


def test_probe_scores_verbose_choice_final_decision():
    probe = _load_probe_module()
    choices = {"A": "buy", "B": "sell", "C": "hold"}
    answer = """
    * A (Buy): possible bullish interpretation.
    * B (Sell): possible bearish interpretation.

    * Action: Buy.
    * Choice A is buy
    """

    assert probe.score_answer(answer, "buy", choices) == 1.0
    assert probe.score_answer(answer, "sell", choices) == 0.0


def test_probe_scores_embedded_exact_code_line():
    probe = _load_probe_module()
    answer = (
        "The most salient API entry point is HTTPServer. "
        "The line that defines it is `class HTTPServer(socketserver.TCPServer):`."
    )

    assert probe.score_answer(answer, "class HTTPServer(socketserver.TCPServer):", {}) == 1.0


def test_probe_exact_items_use_direct_answer_path():
    probe = _load_probe_module()

    assert probe._is_direct_answer_item(
        probe.BenchmarkItem(
            "codeneedle retrieval task. Return the exact line.",
            "class A:",
            "fixtures/sample.py:codeneedle",
            metadata={"grading": "exact", "expected_key": "fixture_needle"},
        )
    )
    assert probe._is_direct_answer_item(
        probe.BenchmarkItem(
            "Identify the gene from the clues.",
            "CTCF",
            "data.zip!records.jsonl:1",
            metadata={"grading": "exact", "expected_key": "answer_rubric"},
        )
    )
    assert not probe._is_direct_answer_item(
        probe.BenchmarkItem("Read the answer from answer.txt.", "42", "source", metadata={"grading": "exact"})
    )


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


def test_probe_gdpval_requires_office_document_capability(monkeypatch):
    probe = _load_probe_module()
    monkeypatch.delenv("AGENT_BENCH_REQUIRED_CAPABILITIES", raising=False)

    capabilities = probe._required_capabilities("GDPval")

    assert "office_document_editing" in capabilities
    assert "office_document_editing" in probe.HARNESS_SUPPORTED_CAPABILITIES


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
    message = {"content": "", "reasoning_content": '{"score":1,"passed":true,"reason":"ok"}'}
    assert probe.extract_openai_content({"choices": [{"message": message}]}) == ""
    assert probe._message_has_hidden_reasoning(message) is True


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
    monkeypatch.setenv("AGENT_BENCH_REQUIRED_CAPABILITIES", "repo_patch,chat_answer,kaggle_competition_submission")

    required = probe._required_capabilities("ExampleBench")

    assert required == {"repo_patch", "chat_answer", "kaggle_competition_submission"}
    assert sorted(required - probe.HARNESS_SUPPORTED_CAPABILITIES) == ["kaggle_competition_submission"]


def test_probe_supports_tool_call_with_stateful_tool_adapter(monkeypatch):
    probe = _load_probe_module()
    monkeypatch.setenv("AGENT_BENCH_REQUIRED_CAPABILITIES", "tool_call,external_data_required")

    required = probe._required_capabilities("ToolBench")
    adapter = probe.select_adapter(required)
    contract = probe.capability_contract_for(required, adapter)

    assert "tool_call" in probe.HARNESS_SUPPORTED_CAPABILITIES
    assert adapter.supported_capabilities() == {"chat_answer", "external_data_required", "tool_call"}
    assert contract["tool_call"]["supported"] is True


def test_static_finance_contract_reports_degraded_missing_benchmark_tools():
    probe = _load_probe_module()
    item = probe.BenchmarkItem(
        "Call selected finance API.",
        "$8.70",
        "synthetic",
        metadata={"select_tools": ["companies_balance_sheet_statements"], "live_tools_required": False},
    )

    contract = probe.StaticFinanceReasoningAdapter().capability_contract(
        {"tool_call", "external_data_required"},
        [item],
    )
    payload = probe._base_result_payload(
        args=type("Args", (), {"benchmark": "Finance Agent v2", "kind": "repository"})(),
        files=["README.md"],
        markers=["README.md"],
        sample_limit=3,
        required_capabilities=["tool_call", "external_data_required"],
        unsupported_capabilities=[],
        adapter=probe.StaticFinanceReasoningAdapter(),
        capability_contract=contract,
    )

    assert contract["tool_call"]["supported"] is True
    assert contract["tool_call"]["tools"] is False
    assert contract["tool_call"]["static_degraded_mode"] is True
    assert contract["tool_call"]["benchmark_required_tools_available"] is False
    assert contract["tool_call"]["required_benchmark_tools"] == ["companies_balance_sheet_statements"]
    assert payload["benchmark_required_tools_available"] is False
    assert payload["static_degraded_mode"] is True
    assert payload["supported_capabilities"] == []


def test_bundled_descriptor_capabilities_are_supported():
    probe = _load_probe_module()
    task_file = Path(__file__).resolve().parents[1] / "tasks" / "public_benchmarks.json"
    tasks = json.loads(task_file.read_text(encoding="utf-8"))
    declared = {
        capability
        for task in tasks
        for capability in task["benchmark"].get("capabilities", [])
    }

    assert sorted(declared - probe.HARNESS_SUPPORTED_CAPABILITIES) == []


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


def test_probe_payload_reports_contract_unsupported_capabilities():
    probe = _load_probe_module()

    unsupported = probe._payload_unsupported_capabilities(
        {"repo_patch", "chat_answer"},
        {
            "chat_answer": {"supported": True},
            "repo_patch": {"supported": False, "reason": "official patch/test grader missing"},
        },
        "skipped_unsupported_capability",
        "All benchmark records were invalid",
        "SWE-bench",
    )

    assert unsupported == ["repo_patch"]


def test_probe_payload_reports_degraded_benchmark_tools_as_unsupported():
    probe = _load_probe_module()

    unsupported = probe._payload_unsupported_capabilities(
        {"tool_call", "external_data_required"},
        {
            "tool_call": {"supported": True, "benchmark_required_tools_available": False},
            "external_data_required": {"supported": True, "benchmark_required_tools_available": False},
        },
        "completed",
        "",
        "ExampleFinanceToolBench",
    )

    assert unsupported == ["external_data_required", "tool_call"]


def test_probe_repo_patch_contract_missing_metadata_is_setup_failure(monkeypatch):
    probe = _load_probe_module()
    monkeypatch.delenv("AGENT_BENCH_ALLOW_TARGET_CHECKOUT", raising=False)
    monkeypatch.setenv("AGENT_BENCH_REPO_PATCH_GRADER", "grader")
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


def test_probe_repo_patch_missing_metadata_preflight_is_invalid_context(monkeypatch):
    probe = _load_probe_module()
    monkeypatch.delenv("AGENT_BENCH_ALLOW_TARGET_CHECKOUT", raising=False)
    monkeypatch.setenv("AGENT_BENCH_REPO_PATCH_GRADER", "grader")
    item = probe.BenchmarkItem(
        "Fix the bug.",
        "diff --git a/a.py b/a.py",
        "huggingface:swe-lancer/test:1",
        metadata={"grading": "exact"},
    )

    contract = probe.RepoPatchAdapter().capability_contract({"repo_patch"}, [item])
    status, reason = probe._preflight_failure_from_contract({"repo_patch"}, contract)

    assert status == "failed_invalid_task_context"
    assert "missing target repo/base_commit metadata" in reason


def test_probe_repo_patch_missing_metadata_direct_run_is_invalid_context(tmp_path, monkeypatch):
    probe = _load_probe_module()
    monkeypatch.setenv("AGENT_BENCH_OUTPUT_DIR", str(tmp_path / "outputs"))
    monkeypatch.setenv("AGENT_BENCH_REPO_PATCH_GRADER", "grader")
    item = probe.BenchmarkItem(
        "Fix the bug.",
        "diff --git a/a.py b/a.py",
        "huggingface:swe-lancer/test:1",
        metadata={"grading": "exact"},
    )

    result = probe.run_model_on_item("SWE-Lancer", item, probe.RepoPatchAdapter())

    assert result["status"] == "failed_invalid_task_context"
    assert result["included_in_official_score"] is False


def test_probe_swelancer_rejects_catalog_as_target_repo(tmp_path, monkeypatch):
    probe = _load_probe_module()
    catalog = tmp_path / "frontier-evals" / "project" / "swelancer"
    catalog.mkdir(parents=True)
    monkeypatch.chdir(catalog)
    monkeypatch.setenv("AGENT_BENCH_BENCHMARK_NAME", "SWE-Lancer")
    item = probe.BenchmarkItem(
        "Fix the issue.",
        "A patch should pass tests.",
        "all_swelancer_tasks.csv:2",
        metadata={"target_repo": "/workspace/repo/project/swelancer", "base_commit": "abc123"},
    )

    error = probe.validate_swelancer_item(item, catalog)

    assert error is not None
    assert error[0] == "failed_invalid_task_context"
    assert "catalog checkout is not the target repository" in error[1]


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


def test_repo_patch_prepare_uses_isolated_clean_checkouts(tmp_path, monkeypatch):
    probe = _load_probe_module()
    source = tmp_path / "source"
    source.mkdir()
    subprocess.run(["git", "init"], cwd=source, check=True, capture_output=True, text=True)
    subprocess.run(["git", "config", "user.email", "agent-bench@example.invalid"], cwd=source, check=True)
    subprocess.run(["git", "config", "user.name", "Agent Bench"], cwd=source, check=True)
    (source / "app.py").write_text("print('base')\n", encoding="utf-8")
    subprocess.run(["git", "add", "app.py"], cwd=source, check=True)
    subprocess.run(["git", "commit", "-m", "base"], cwd=source, check=True, capture_output=True, text=True)
    base_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=source,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    monkeypatch.setenv("AGENT_BENCH_OUTPUT_DIR", str(tmp_path / "outputs"))

    first = probe.BenchmarkItem("Fix one.", "patch", "item:1", metadata={"target_repo": str(source), "base_commit": base_commit})
    second = probe.BenchmarkItem("Fix two.", "patch", "item:2", metadata={"target_repo": str(source), "base_commit": base_commit})

    first_workspace = probe.RepoPatchAdapter().prepare_task(first)
    (first_workspace.root / "leaked.txt").write_text("should not leak\n", encoding="utf-8")
    second_workspace = probe.RepoPatchAdapter().prepare_task(second)

    assert first_workspace.root != second_workspace.root
    assert not (second_workspace.root / "leaked.txt").exists()
    assert (second_workspace.root / "app.py").read_text(encoding="utf-8") == "print('base')\n"


def test_probe_repo_patch_contract_requires_canary(monkeypatch):
    probe = _load_probe_module()
    monkeypatch.setenv("AGENT_BENCH_REPO_PATCH_GRADER", "grader")
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


def test_probe_repo_patch_without_grader_uses_model_judge_fallback(monkeypatch):
    probe = _load_probe_module()
    monkeypatch.delenv("AGENT_BENCH_REPO_PATCH_GRADER", raising=False)
    monkeypatch.setattr(probe, "_repo_patch_checkout_available", lambda items: True)
    monkeypatch.setattr(probe, "_repo_patch_canary", lambda: {"passed": True})
    item = probe.BenchmarkItem(
        "Fix the bug.",
        "diff --git a/a.py b/a.py",
        "huggingface:swe-bench/test:1",
        metadata={"repo": "astropy/astropy", "base_commit": "abc123", "grading": "exact"},
    )

    contract = probe.RepoPatchAdapter().capability_contract({"repo_patch"}, [item])
    status, reason = probe._preflight_failure_from_contract({"repo_patch"}, contract)

    assert contract["repo_patch"]["supported"] is True
    assert contract["repo_patch"]["official_grader"] is False
    assert contract["repo_patch"]["official_equivalent"] is False
    assert contract["repo_patch"]["score_mode"] == "smoke_fallback"
    assert contract["repo_patch"]["fallback_grader"] == "model_judge_task_compliance"
    assert status == ""
    assert reason == ""


def test_probe_repo_patch_grading_rejects_reference_diff(monkeypatch):
    probe = _load_probe_module()
    monkeypatch.delenv("AGENT_BENCH_REPO_PATCH_GRADER", raising=False)
    calls = []

    def fake_judge(benchmark, item, answer, method):
        calls.append((benchmark, item, answer, method))
        return 0.0, {"method": method, "score": 0.0, "status": "failed_model_answer", "reason": "not sufficient"}

    monkeypatch.setattr(probe, "judge_answer", fake_judge)
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
    assert grade["status"] == "failed_model_answer"
    assert grade["method"] == "task_compliance_fallback"
    assert grade["official_grader"] is False
    assert grade["official_equivalent"] is False
    assert grade["score_mode"] == "smoke_task_compliance_fallback"
    assert grade["included_in_official_score"] is False
    assert calls[0][3] == "task_compliance"
    assert "Model patch:" in calls[0][2]


def test_probe_repo_patch_missing_diff_is_missing_artifact(monkeypatch):
    probe = _load_probe_module()
    monkeypatch.delenv("AGENT_BENCH_REPO_PATCH_GRADER", raising=False)
    item = probe.BenchmarkItem(
        "Fix the bug.",
        "diff --git a/a.py b/a.py\n",
        "huggingface:swe-bench/test:1",
        metadata={"repo": "astropy/astropy", "base_commit": "abc123"},
    )

    score, grade = probe.RepoPatchAdapter().grade("SWE-bench", item, probe.OutputBundle(patch=""))

    assert score == 0.0
    assert grade["status"] == "failed_model_missing_artifact"
    assert grade["method"] == "repo_patch_output_presence"
    assert grade["official_equivalent"] is False
    assert grade["score_mode"] == "smoke_patch_presence"
    assert grade["included_in_official_score"] is False


def test_probe_swelancer_contract_uses_builtin_official_task_tests(tmp_path, monkeypatch):
    probe = _load_probe_module()
    issue_dir = tmp_path / "issues" / "16912_4"
    issue_dir.mkdir(parents=True)
    (issue_dir / "test.py").write_text("def test_placeholder():\n    assert True\n", encoding="utf-8")
    monkeypatch.delenv("AGENT_BENCH_REPO_PATCH_GRADER", raising=False)
    monkeypatch.setenv("AGENT_BENCH_BENCHMARK_NAME", "SWE-Lancer")
    monkeypatch.setattr(probe, "_repo_patch_checkout_available", lambda items: True)
    monkeypatch.setattr(probe, "_repo_patch_canary", lambda: {"passed": True})
    item = probe.BenchmarkItem(
        "Fix the issue.",
        "A patch should pass tests.",
        "all_swelancer_tasks.csv:2",
        metadata={
            "target_repo": "https://github.com/Expensify/App.git",
            "base_commit": "abc123",
            "instance_id": "16912_4",
            "issue_dir": str(issue_dir),
        },
    )

    contract = probe.RepoPatchAdapter().capability_contract({"repo_patch"}, [item])

    assert contract["repo_patch"]["supported"] is True
    assert contract["repo_patch"]["official_grader"] is True
    assert contract["repo_patch"]["official_equivalent"] is True
    assert contract["repo_patch"]["score_mode"] == "official_swelancer_task_tests"
    assert contract["repo_patch"]["swelancer_task_tests"]["issue_ids"] == ["16912_4"]


def test_probe_swelancer_empty_patch_counts_as_official_artifact_failure(tmp_path, monkeypatch):
    probe = _load_probe_module()
    issue_dir = tmp_path / "issues" / "16912_4"
    issue_dir.mkdir(parents=True)
    (issue_dir / "test.py").write_text("def test_placeholder():\n    assert True\n", encoding="utf-8")
    monkeypatch.delenv("AGENT_BENCH_REPO_PATCH_GRADER", raising=False)
    item = probe.BenchmarkItem(
        "Fix the issue.",
        "A patch should pass tests.",
        "all_swelancer_tasks.csv:2",
        metadata={
            "target_repo": "https://github.com/Expensify/App.git",
            "base_commit": "abc123",
            "instance_id": "16912_4",
            "issue_dir": str(issue_dir),
        },
    )

    score, grade = probe.RepoPatchAdapter().grade("SWE-Lancer", item, probe.OutputBundle(patch=""))

    assert score == 0.0
    assert grade["status"] == "failed_model_missing_artifact"
    assert grade["method"] == "pre_grader_artifact_check"
    assert grade["official_grader_not_run_reason"] == "empty_patch"
    assert grade["official_grader"] is True
    assert grade["official_equivalent"] is True
    assert grade["score_mode"] == "official_swelancer_task_tests"
    assert grade["included_in_official_score"] is True


def test_probe_swelancer_official_grader_applies_patch_and_runs_task_test(tmp_path, monkeypatch):
    probe = _load_probe_module()
    source = tmp_path / "source"
    source.mkdir()
    subprocess.run(["git", "init"], cwd=source, check=True, capture_output=True, text=True)
    subprocess.run(["git", "config", "user.email", "agent-bench@example.invalid"], cwd=source, check=True)
    subprocess.run(["git", "config", "user.name", "Agent Bench"], cwd=source, check=True)
    (source / "app.py").write_text("print('base')\n", encoding="utf-8")
    subprocess.run(["git", "add", "app.py"], cwd=source, check=True)
    subprocess.run(["git", "commit", "-m", "base"], cwd=source, check=True, capture_output=True, text=True)
    base_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=source,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    issue_dir = tmp_path / "issues" / "16912_4"
    issue_dir.mkdir(parents=True)
    (issue_dir / "test.py").write_text(
        "from pathlib import Path\n\n"
        "def test_model_patch_applied():\n"
        "    assert Path('app.py').read_text(encoding='utf-8') == \"print('fixed')\\n\"\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    patch = (
        "diff --git a/app.py b/app.py\n"
        "--- a/app.py\n"
        "+++ b/app.py\n"
        "@@ -1 +1 @@\n"
        "-print('base')\n"
        "+print('fixed')\n"
    )
    patch_path = output_dir / "model.patch"
    patch_path.write_text(patch, encoding="utf-8")
    monkeypatch.delenv("AGENT_BENCH_REPO_PATCH_GRADER", raising=False)
    monkeypatch.setenv("AGENT_BENCH_SWELANCER_GRADER_TIMEOUT", "30")
    item = probe.BenchmarkItem(
        "Fix the issue.",
        "A patch should pass tests.",
        "all_swelancer_tasks.csv:2",
        metadata={
            "target_repo": str(source),
            "base_commit": base_commit,
            "instance_id": "16912_4",
            "issue_dir": str(issue_dir),
        },
    )

    score, grade = probe.RepoPatchAdapter().grade(
        "SWE-Lancer",
        item,
        probe.OutputBundle(
            patch=patch,
            metadata={
                "patch_path": str(patch_path),
                "target_checkout_path": str(source),
                "base_commit": base_commit,
            },
        ),
    )

    assert score == 1.0
    assert grade["status"] == "passed"
    assert grade["method"] == "official_swelancer_task_tests"
    assert grade["official_grader"] is True
    assert grade["official_equivalent"] is True
    assert grade["included_in_official_score"] is True
    assert grade["official_test_exit_code"] == 0
    assert (output_dir / "official_swelancer_grader_checkout" / "app.py").read_text(encoding="utf-8") == "print('fixed')\n"


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
    monkeypatch.setattr("sys.argv", ["agent-bench-probe", "--benchmark", "ExampleBench"])
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

def test_probe_file_artifact_grades_text_fallback_without_generated_output(tmp_path, monkeypatch):
    probe = _load_probe_module()
    monkeypatch.chdir(tmp_path)
    (tmp_path / "input.txt").write_text("source data", encoding="utf-8")

    def fake_loop(benchmark, item, tools=None):
        return {"answer": "done", "content": '{"answer":"done"}', "usage": {}, "tool_trace": []}

    def fake_judge(benchmark, item, answer, method):
        return 0.25, {
            "method": method,
            "score": 0.25,
            "status": "failed_model_answer",
            "passed": False,
            "reason": "text fallback was partially relevant",
        }

    monkeypatch.setattr(probe, "run_agent_loop", fake_loop)
    monkeypatch.setattr(probe, "judge_answer", fake_judge)
    item = probe.BenchmarkItem(
        "Create the report.",
        "rubric",
        "artifact:1",
        metadata={"grading": "rubric", "input_files": ["input.txt"]},
    )

    result = probe.run_model_on_item("PaperBench", item, probe.FileArtifactAdapter())

    assert result["status"] == "failed_model_answer"
    assert result["score"] == 0.25
    assert result["grade"]["output_collection"] == "text_answer_fallback"


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


def test_probe_artifact_integrity_reports_path_outside_allowed_output(tmp_path):
    probe = _load_probe_module()
    allowed = tmp_path / "outputs" / "items" / "case" / "artifacts"
    allowed.mkdir(parents=True)
    outside = tmp_path / "outside.txt"
    outside.write_text("not allowed\n", encoding="utf-8")

    errors = probe._artifact_integrity_errors([str(outside)], allowed_root=allowed)

    assert errors == [{"path": str(outside), "error": "path outside allowed output directory"}]


def test_probe_file_artifact_collects_file_replacing_output_dir(tmp_path, monkeypatch):
    probe = _load_probe_module()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENT_BENCH_OUTPUT_DIR", str(tmp_path / "outputs"))
    (tmp_path / "input.txt").write_text("source data", encoding="utf-8")

    def fake_loop(benchmark, item, tools=None):
        output_path = Path.cwd() / "agent_bench_outputs" / probe._safe_slug(item.source)
        if output_path.exists():
            output_path.rmdir()
        output_path.write_text("answer artifact", encoding="utf-8")
        return {"answer": "done", "content": '{"answer":"done"}', "usage": {}, "tool_trace": []}

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

    result = probe.run_model_on_item("FileArtifactBench", item, probe.FileArtifactAdapter())

    assert result["status"] == "passed"
    assert result["output_bundle"]["artifact_paths"]


def test_probe_write_text_file_redirects_directory_path(tmp_path, monkeypatch):
    probe = _load_probe_module()
    monkeypatch.chdir(tmp_path)
    (tmp_path / "agent_bench_outputs" / "item").mkdir(parents=True)

    result = probe.tool_write_file({"path": "agent_bench_outputs/item", "content": "answer"})

    assert "agent_bench_outputs/item/response.md" in result
    assert (tmp_path / "agent_bench_outputs" / "item" / "response.md").read_text(encoding="utf-8") == "answer"


def test_probe_artifact_previews_include_text_content(tmp_path):
    probe = _load_probe_module()
    artifact = tmp_path / "response.md"
    artifact.write_text("concrete deliverable text", encoding="utf-8")

    preview = probe._artifact_previews([str(artifact)])

    assert "response.md" in preview
    assert "concrete deliverable text" in preview


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


def test_probe_chat_adapter_exposes_record_tables_but_not_answers(tmp_path, monkeypatch):
    probe = _load_probe_module()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENT_BENCH_OUTPUT_DIR", str(tmp_path / "outputs"))
    monkeypatch.setenv("AGENT_BENCH_BENCHMARK_NAME", "FinanceMath")
    item = probe.item_from_record(
        {
            "question": "Use Exhibit 1 to calculate beta.",
            "tables": [{"title": "Exhibit 1", "rows": [["asset", "covariance"], ["real estate", "0.578"]]}],
            "ground_truth": "0.578",
            "python_solution": "def solve():\n    return 0.578",
        },
        "data/validation.json:record-000003",
    )
    assert item is not None

    workspace = probe.ChatAnswerAdapter().prepare_task(item)
    task_text = (workspace.root / "TASK.md").read_text(encoding="utf-8")

    assert "Benchmark record context" in task_text
    assert "Exhibit 1" in task_text
    assert "real estate" in task_text
    assert "ground_truth" not in task_text
    assert "python_solution" not in task_text


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


def test_probe_extracts_biomystery_answer_rubric_from_local_zip(tmp_path, monkeypatch):
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

    assert errors == []
    assert len(items) == 1
    assert items[0].expected == "Homo sapiens"
    assert items[0].metadata["expected_key"] == "answer_rubric"
    assert items[0].metadata["grading"] == "exact"


def test_probe_biomystery_workspace_hides_answer_rubric(tmp_path, monkeypatch):
    probe = _load_probe_module()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENT_BENCH_OUTPUT_DIR", str(tmp_path / "outputs"))
    item = probe.BenchmarkItem(
        "Identify the species from the hidden biological clues.",
        "Homo sapiens",
        "data.zip!records.jsonl:1",
        metadata={"grading": "exact", "expected_key": "answer_rubric"},
    )

    workspace = probe.ChatAnswerAdapter().prepare_task(item)
    task_text = (workspace.root / "TASK.md").read_text(encoding="utf-8")

    assert "Identify the species" in task_text
    assert "Homo sapiens" not in task_text
    assert "answer_rubric" not in task_text


def test_probe_biomystery_answer_rubric_scores_exact_answer(tmp_path, monkeypatch):
    probe = _load_probe_module()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENT_BENCH_PROVIDER", "openai-compatible")
    monkeypatch.setenv("AGENT_BENCH_BASE_URL", "http://model.test/v1")
    monkeypatch.setenv("AGENT_BENCH_MODEL", "model")
    monkeypatch.setenv("AGENT_BENCH_OUTPUT_DIR", str(tmp_path / "outputs"))
    monkeypatch.setenv("AGENT_BENCH_BENCHMARK_NAME", "BioMystery Bench")
    monkeypatch.setenv("AGENT_BENCH_DATASET_ID", "Anthropic/BioMysteryBench-preview")
    monkeypatch.setattr("sys.argv", ["agent-bench-probe", "--benchmark", "BioMystery Bench"])
    with zipfile.ZipFile(tmp_path / "data.zip", "w") as archive:
        archive.writestr(
            "records.jsonl",
            '{"question":"q","answer_rubric":"The answer is X Score 1"}\n',
        )

    monkeypatch.setattr(
        probe,
        "run_agent_loop",
        lambda benchmark, item, tools=None: {
            "answer": "X",
            "content": '{"answer":"X"}',
            "usage": {},
            "tool_trace": [],
        },
    )

    exit_code = probe.main()
    payload = json.loads((tmp_path / "outputs" / "agent_bench_result.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert payload["status"] == "completed"
    assert payload["unsupported_capabilities"] == []
    assert payload["model_evals"][0]["status"] == "passed"


def test_probe_biomystery_terminal_extraction_does_not_fall_through_to_csv(tmp_path, monkeypatch):
    probe = _load_probe_module()
    monkeypatch.setenv("AGENT_BENCH_BENCHMARK_NAME", "BioMystery Bench")
    with zipfile.ZipFile(tmp_path / "data.zip", "w") as archive:
        archive.writestr("records.jsonl", '{"question":"q","answer_rubric":"The answer is X Score 1"}\n')
    (tmp_path / "problems.csv").write_text("question,answer_rubric\nq,The answer is X Score 1\n", encoding="utf-8")

    items, errors = probe.extract_benchmark_items(tmp_path, limit=3)

    assert errors == []
    assert len(items) == 1
    assert items[0].source == "data.zip!records.jsonl:1"


def test_probe_extracts_codeneedle_fixture_tasks(tmp_path, monkeypatch):
    probe = _load_probe_module()
    monkeypatch.setenv("AGENT_BENCH_BENCHMARK_NAME", "codeneedle")
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    (fixtures / "sample.py").write_text("import os\n\ndef target_api(value):\n    return value\n", encoding="utf-8")

    items, errors = probe.extract_benchmark_items(tmp_path, limit=3)

    assert errors == []
    assert len(items) == 1
    assert items[0].expected == "def target_api(value):"
    assert items[0].metadata["grading"] == "exact"


def test_probe_extracts_stockbench_cached_financial_tasks(tmp_path, monkeypatch):
    probe = _load_probe_module()
    monkeypatch.setenv("AGENT_BENCH_BENCHMARK_NAME", "StockBench")
    financials = tmp_path / "storage" / "cache" / "financials"
    financials.mkdir(parents=True)
    (financials / "AAPL.annual.json").write_text(json.dumps({"revenue": [1, 2, 3]}), encoding="utf-8")
    (tmp_path / "stockbench" / "agents" / "prompts").mkdir(parents=True)
    (tmp_path / "stockbench" / "agents" / "prompts" / "decision_agent_v1.txt").write_text(
        "prompt template {{ input }}",
        encoding="utf-8",
    )

    items, errors = probe.extract_benchmark_items(tmp_path, limit=3)

    assert errors == []
    assert len(items) == 1
    assert "AAPL" in items[0].question
    assert items[0].metadata["input_files"] == ["storage/cache/financials/AAPL.annual.json"]


def test_probe_extracts_swelancer_repo_patch_metadata(tmp_path, monkeypatch):
    probe = _load_probe_module()
    monkeypatch.setenv("AGENT_BENCH_BENCHMARK_NAME", "SWE-Lancer")
    (tmp_path / "all_swelancer_tasks.csv").write_text(
        "question_id,title,description,target_repo,base_commit\n"
        "12155_1,Fix bug,Patch the failing behavior,https://github.com/example/project.git,abc123\n",
        encoding="utf-8",
    )
    issue = tmp_path / "issues" / "12155_1"
    issue.mkdir(parents=True)
    (issue / "issue_data.json").write_text(json.dumps({"issue": "Detailed issue"}), encoding="utf-8")
    (issue / "commit_id.txt").write_text("issuecommit\n", encoding="utf-8")

    items, errors = probe.extract_benchmark_items(tmp_path, limit=3)

    assert errors == []
    assert len(items) == 1
    assert items[0].metadata["target_repo"] == "https://github.com/example/project.git"
    assert items[0].metadata["base_commit"] == "abc123"
    assert items[0].metadata["issue_commit_id"] == "issuecommit"
    assert items[0].metadata["target_repo_source"] == "record_metadata"
    assert items[0].metadata["base_commit_source"] == "record_metadata"


def test_probe_infers_swelancer_repo_patch_metadata_from_official_assets(tmp_path, monkeypatch):
    probe = _load_probe_module()
    monkeypatch.setenv("AGENT_BENCH_BENCHMARK_NAME", "SWE-Lancer")
    (tmp_path / "all_swelancer_tasks.csv").write_text(
        "question_id,title,description,cwd\n"
        "16912_4,Fix zip hint,Patch the failing behavior,/app/expensify\n",
        encoding="utf-8",
    )
    issue = tmp_path / "issues" / "16912_4"
    issue.mkdir(parents=True)
    (issue / "issue_data.json").write_text(
        json.dumps({"issue_repo_steps": "Steps to reproduce the issue"}),
        encoding="utf-8",
    )
    (issue / "commit_id.txt").write_text("2b791c9f3053c1682ddcb50ab036deb3e55a7542\n", encoding="utf-8")

    items, errors = probe.extract_benchmark_items(tmp_path, limit=3)

    assert errors == []
    assert len(items) == 1
    assert items[0].metadata["target_repo"] == "https://github.com/Expensify/App.git"
    assert items[0].metadata["target_repo_source"] == "cwd:/app/expensify"
    assert items[0].metadata["base_commit"] == "2b791c9f3053c1682ddcb50ab036deb3e55a7542"
    assert items[0].metadata["base_commit_source"] == "16912_4/commit_id.txt"
    assert items[0].metadata["issue_commit_id"] == "2b791c9f3053c1682ddcb50ab036deb3e55a7542"
    assert items[0].metadata["official_workspace_cwd"] == "/app/expensify"
    assert items[0].metadata["metadata_association"] == "official_swelancer_workspace"
    assert "invalid_task_context_reason" not in items[0].metadata
    assert probe.validate_swelancer_item(items[0], tmp_path) is None


def test_probe_extracts_swelancer_lite_csv_prompt_layout(tmp_path, monkeypatch):
    probe = _load_probe_module()
    monkeypatch.setenv("AGENT_BENCH_BENCHMARK_NAME", "SWE-Lancer")
    (tmp_path / "swelancer_tasks_lite.csv").write_text(
        "question_id,prompt,cwd\n"
        "17387_1,Fix the modal regression,/app/expensify\n",
        encoding="utf-8",
    )
    issue = tmp_path / "issues" / "17387_1"
    issue.mkdir(parents=True)
    (issue / "commit_id.txt").write_text("abc123def456\n", encoding="utf-8")

    items, errors = probe.extract_benchmark_items(tmp_path, limit=3)

    assert errors == []
    assert len(items) == 1
    assert items[0].question == "Fix the modal regression"
    assert items[0].source == "swelancer_tasks_lite.csv:2"
    assert items[0].metadata["target_repo"] == "https://github.com/Expensify/App.git"
    assert items[0].metadata["base_commit"] == "abc123def456"


def test_probe_swelancer_catalog_checkout_is_invalid_target_context(tmp_path, monkeypatch):
    probe = _load_probe_module()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENT_BENCH_BENCHMARK_NAME", "SWE-Lancer")
    item = probe.BenchmarkItem(
        "Fix the issue.",
        "A repository patch should satisfy the SWE-Lancer issue tests.",
        "all_swelancer_tasks.csv:2",
        metadata={"target_repo": str(tmp_path), "base_commit": "abc123", "grading": "task_compliance"},
    )

    status, reason, details = probe._item_preflight_failure(item, {"repo_patch"}, set(), set())

    assert status == "failed_invalid_task_context"
    assert "catalog checkout is not the target repository" in reason
    assert details["target_repo"] == str(tmp_path)


def test_probe_swelancer_missing_target_metadata_is_invalid_context(tmp_path, monkeypatch):
    probe = _load_probe_module()
    monkeypatch.setenv("AGENT_BENCH_BENCHMARK_NAME", "SWE-Lancer")
    (tmp_path / "all_swelancer_tasks.csv").write_text(
        "question_id,title,description\n12155_1,Fix bug,Patch the failing behavior\n",
        encoding="utf-8",
    )
    (tmp_path / "issues" / "12155_1").mkdir(parents=True)

    items, errors = probe.extract_benchmark_items(tmp_path, limit=3)
    status, reason, _details = probe._item_preflight_failure(items[0], {"repo_patch"}, set(), set())

    assert errors == []
    assert "target_repo" not in items[0].metadata
    assert status == "failed_invalid_task_context"
    assert "target repository and base commit" in reason


def test_probe_extracts_quantcode_repo_patch_metadata(tmp_path, monkeypatch):
    probe = _load_probe_module()
    monkeypatch.setenv("AGENT_BENCH_BENCHMARK_NAME", "QuantCode-Bench")

    def fake_run(command, **kwargs):
        return type("Completed", (), {"returncode": 0, "stdout": "def456\n", "stderr": ""})()

    monkeypatch.setattr(probe.subprocess, "run", fake_run)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "benchmark_tasks_multiframe.json").write_text(
        json.dumps([{"id": "task-1", "reformulated_task": "Implement SMA crossover"}]),
        encoding="utf-8",
    )

    items, errors = probe.extract_benchmark_items(tmp_path, limit=3)

    assert errors == []
    assert len(items) == 1
    assert items[0].metadata["target_repo"] == str(tmp_path)
    assert items[0].metadata["base_commit"] == "def456"


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


def test_probe_extracts_paperbench_leaf_rubric_tasks_first(tmp_path, monkeypatch):
    probe = _load_probe_module()
    monkeypatch.setenv("AGENT_BENCH_BENCHMARK_NAME", "PaperBench")
    paper_dir = tmp_path / "data" / "papers" / "sample-paper"
    paper_dir.mkdir(parents=True)
    (paper_dir / "paper.md").write_text("Paper text", encoding="utf-8")
    (paper_dir / "rubric.json").write_text(
        json.dumps(
            {
                "id": "root",
                "requirements": "The whole paper is reproduced.",
                "sub_tasks": [
                    {
                        "id": "models",
                        "requirements": "All model components are available.",
                        "sub_tasks": [
                            {
                                "id": "vit",
                                "requirements": "ViT-Base loading code is implemented.",
                                "sub_tasks": [],
                                "weight": 1,
                            },
                            {
                                "id": "resnet",
                                "requirements": "ResNet-50 loading code is implemented.",
                                "sub_tasks": [],
                                "weight": 1,
                            },
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    items, errors = probe.extract_benchmark_items(tmp_path, limit=2)

    assert errors == []
    assert [item.question for item in items] == [
        "ViT-Base loading code is implemented.",
        "ResNet-50 loading code is implemented.",
    ]
    assert all("rubric.json:leaf-" in item.source for item in items)
    assert "The whole paper is reproduced" in items[0].metadata["visible_context"]
    assert paper_dir / "paper.md" in probe._artifact_asset_paths(items[0], tmp_path)


def test_probe_paperbench_lfs_pointer_assets_are_missing_assets(tmp_path, monkeypatch):
    probe = _load_probe_module()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENT_BENCH_BENCHMARK_NAME", "PaperBench")
    paper_dir = tmp_path / "data" / "papers" / "sample-paper"
    paper_dir.mkdir(parents=True)
    (paper_dir / "rubric.json").write_text("{}", encoding="utf-8")
    (paper_dir / "config.yaml").write_text("id: sample-paper\n", encoding="utf-8")
    (paper_dir / "paper.md").write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:abc\n"
        "size 12345\n",
        encoding="utf-8",
    )
    item = probe.BenchmarkItem(
        "Implement the method.",
        "Candidate answer should satisfy the task requirements from the benchmark prompt.",
        "data/papers/sample-paper/rubric.json:leaf-000001",
        metadata={"grading": "task_compliance"},
    )

    status, reason, details = probe._item_preflight_failure(item, {"file_artifact"}, set(), set())

    assert status == "failed_missing_assets"
    assert "Git LFS pointer" in reason
    assert details["blocker_type"] == "git_lfs_pointer_stub"
    assert details["asset_errors"][0]["reason"] == "git_lfs_pointer_stub"


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


def test_probe_classifies_model_http_error_as_endpoint_failure(monkeypatch):
    probe = _load_probe_module()

    def fake_loop(self, benchmark, item, workspace, tools):
        raise probe.ChatCompletionHTTPError(404, '{"error":{"message":"model not found"}}')

    monkeypatch.setattr(probe.ChatAnswerAdapter, "run_agent_loop", fake_loop)

    result = probe.ChatAnswerAdapter().evaluate_item(
        "ExampleBench",
        probe.BenchmarkItem("q", "a", "source"),
    )

    assert result["status"] == "failed_model_endpoint"
    assert result["included_in_official_score"] is False


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


def test_probe_unparseable_judge_text_is_grader_failure(monkeypatch):
    probe = _load_probe_module()

    def fake_post(base_url, payload, headers):
        return json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "content": "The candidate does not provide the required label, so the score should be low."
                        }
                    }
                ],
                "usage": {"total_tokens": 7},
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
    assert grade["judge_parser_status"] == "judge_parse_error"
    assert grade["judge_retry_count"] == 2


def test_probe_accepts_repaired_judge_json(monkeypatch):
    probe = _load_probe_module()

    def fake_post(base_url, payload, headers):
        return json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "content": '```json\n{"score":0.0,"passed":false,"reason":"not complete"}\n```'
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
    assert grade["status"] == "failed_model_answer"
    assert grade["judge_parse_repaired"] is True


def test_probe_finmcp_accepts_base_judge_json(monkeypatch):
    probe = _load_probe_module()

    def fake_post(base_url, payload, headers):
        return json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "content": '{"score":1.0,"passed":true,"reason":"complete"}'
                        }
                    }
                ]
            }
        )

    monkeypatch.setenv("AGENT_BENCH_BASE_URL", "http://model.test/v1")
    monkeypatch.setenv("AGENT_BENCH_MODEL", "model")
    monkeypatch.setattr(probe, "post_chat_completion", fake_post)

    score, grade = probe.judge_answer(
        "FinMCP-Bench",
        probe.BenchmarkItem("q", "rubric", "source", metadata={"grading": "task_compliance"}),
        "answer",
        "task_compliance",
    )

    assert score == 1.0
    assert grade["status"] == "passed"
    assert grade["judge_parser_status"] == "parsed"


def test_probe_accepts_explicit_numeric_prose_judge_score_after_retries(monkeypatch):
    probe = _load_probe_module()

    def fake_post(base_url, payload, headers):
        return json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "content": "The candidate gives the right label but no rationale. Score: 0.5"
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

    assert score == 0.5
    assert grade["status"] == "failed_model_answer"
    assert grade["judge_parser_status"] == "prose_score"
    assert grade["judge_retry_count"] == 2


def test_probe_accepts_fractional_prose_judge_score(monkeypatch):
    probe = _load_probe_module()

    def fake_post(base_url, payload, headers):
        return json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "content": (
                                "The candidate directly satisfies the task. Score: 5/5 (or 1.0). "
                                "Passed: true. Reason: complete."
                            )
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

    assert score == 1.0
    assert grade["status"] == "passed"
    assert grade["judge_parser_status"] == "prose_score"


def test_probe_accepts_positive_qualitative_prose_judge(monkeypatch):
    probe = _load_probe_module()

    def fake_post(base_url, payload, headers):
        return json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "content": "The candidate answer is a concise and accurate summary of the prompt."
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

    assert score == 0.5
    assert grade["status"] == "failed_model_answer"
    assert grade["judge_parser_status"] == "prose_score"


def test_probe_accepts_negative_qualitative_prose_judge(monkeypatch):
    probe = _load_probe_module()

    def fake_post(base_url, payload, headers):
        return json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "content": "The candidate answer is incomplete and fails to provide the required output."
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
    assert grade["status"] == "failed_model_answer"
    assert grade["judge_parser_status"] == "prose_score"


def test_probe_gives_partial_credit_for_valid_choice_when_judge_scores_zero(monkeypatch):
    probe = _load_probe_module()

    def fake_post(base_url, payload, headers):
        return json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"score":0.0,"passed":false,"label_correct":true,'
                                '"rationale_present":false,"rationale_quality":"inadequate",'
                                '"reason":"missing rationale"}'
                            )
                        }
                    }
                ]
            }
        )

    monkeypatch.setenv("AGENT_BENCH_BASE_URL", "http://model.test/v1")
    monkeypatch.setenv("AGENT_BENCH_MODEL", "model")
    monkeypatch.setattr(probe, "post_chat_completion", fake_post)
    item = probe.BenchmarkItem(
        "Choose stronger, weaker, or mixed.",
        "rubric",
        "stockbench",
        choices={"A": "stronger", "B": "weaker", "C": "mixed"},
        metadata={"grading": "task_compliance"},
    )

    score, grade = probe.judge_answer("StockBench", item, "C", "task_compliance")

    assert score == 0.5
    assert grade["status"] == "failed_model_answer"
    assert "valid choice label" in grade["reason"]


def test_probe_gives_partial_credit_for_valid_choice_when_judge_unparsed(monkeypatch):
    probe = _load_probe_module()

    def fake_post(base_url, payload, headers):
        return json.dumps({"choices": [{"message": {"content": "The answer lacks a rationale."}}]})

    monkeypatch.setenv("AGENT_BENCH_BASE_URL", "http://model.test/v1")
    monkeypatch.setenv("AGENT_BENCH_MODEL", "model")
    monkeypatch.setattr(probe, "post_chat_completion", fake_post)
    item = probe.BenchmarkItem(
        "Choose stronger, weaker, or mixed.",
        "rubric",
        "stockbench",
        choices={"A": "stronger", "B": "weaker", "C": "mixed"},
        metadata={"grading": "task_compliance"},
    )

    score, grade = probe.judge_answer("StockBench", item, "mixed", "task_compliance")

    assert score == 0.0
    assert grade["status"] == "failed_grader"
    assert grade["judge_parser_status"] == "judge_parse_error"


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
    assert probe.grade_answer("FinanceMath", item, "computed value is $1.2 million")[0] == 1.0
    assert probe.score_answer("beta = 0.57822", "0.578", {}) == 1.0
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


def test_probe_detects_intermediate_agent_progress_message():
    probe = _load_probe_module()

    assert probe._looks_like_intermediate_agent_message("Okay, the directory exists. Now I will write the file.")
    assert not probe._looks_like_intermediate_agent_message('{"answer":"done"}')
    assert probe._contains_tool_syntax("<|tool_call>call:write_text_file{content:<|\"|>x")


def test_probe_parses_angle_bracket_tool_call():
    probe = _load_probe_module()

    parsed = probe.parse_text_tool_request(
        '<|tool_call>call:run_command{argv:[<|"|>python3<|"|>,<|"|>-c<|"|>,<|"|>print("ok")<|"|>]}'
    )

    assert parsed == ("run_command", {"argv": ["python3", "-c", 'print("ok")']})

    suffix_parsed = probe.parse_text_tool_request('call:read_file{path:<|"|>TASK.md<|"|>}<tool_call|>')

    assert suffix_parsed == ("read_file", {"path": "TASK.md"})


def test_probe_parses_qwen_tagged_json_tool_call():
    probe = _load_probe_module()

    parsed = probe.parse_text_tool_request(
        '<tool_call>\n{"name": "read_file", "arguments": {"path": "TASK.md"}}\n</tool_call>'
    )

    assert parsed == ("read_file", {"path": "TASK.md"})


def test_probe_parses_gemma_parameter_alias_tool_call():
    probe = _load_probe_module()

    parsed = probe.parse_text_tool_request(
        '```json\n{"tool_name":"search_files","parameters":{"query":"answer_rubric","path":"."}}\n```'
    )

    assert parsed == ("search_files", {"query": "answer_rubric", "path": "."})


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


def test_agent_loop_respects_empty_tool_list_for_static_adapters(tmp_path, monkeypatch):
    probe = _load_probe_module()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENT_BENCH_PROVIDER", "openai-compatible")
    monkeypatch.setenv("AGENT_BENCH_BASE_URL", "http://model.test/v1")
    monkeypatch.setenv("AGENT_BENCH_MODEL", "model")
    calls = []

    def fake_post(base_url, payload, headers):
        calls.append(payload)
        return json.dumps({"choices": [{"message": {"content": '{"answer":"done","confidence":1.0}'}}]}), payload

    monkeypatch.setattr(probe, "post_chat_completion_with_variant", fake_post)
    item = probe.BenchmarkItem(
        "Answer without tools.",
        "rubric",
        "test",
        metadata={"grading": "task_compliance"},
    )

    result = probe.run_agent_loop("ExampleBench", item, tools=[])

    assert result["answer"] == "done"
    assert "tools" not in calls[0]
    assert "tool_choice" not in calls[0]
    assert calls[0]["response_format"] == {"type": "json_object"}
    assert "Do not use tools" in calls[0]["messages"][0]["content"]


def test_tool_manifest_records_sent_tools_separately_for_direct_answer():
    probe = _load_probe_module()
    item = probe.BenchmarkItem(
        "Return the exact numeric answer.",
        "42",
        "synthetic",
        metadata={"grading": "numeric"},
    )
    available = probe.agent_tool_schemas()
    sent = probe.ChatAnswerAdapter().tools_sent_to_model(item, available)

    manifest = probe._tool_manifest(item, available, sent)

    assert manifest["tools"] == []
    assert manifest["sent_to_model"] == []
    assert manifest["available_in_runner"]
    assert manifest["suppressed_for_direct_answer"] is True


def test_agent_loop_rejects_hidden_reasoning_without_final_content(tmp_path, monkeypatch):
    probe = _load_probe_module()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENT_BENCH_PROVIDER", "openai-compatible")
    monkeypatch.setenv("AGENT_BENCH_BASE_URL", "http://model.test/v1")
    monkeypatch.setenv("AGENT_BENCH_MODEL", "model")

    def fake_post(base_url, payload, headers):
        return (
            json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "content": None,
                                "reasoning": '{"answer":"hidden"}',
                            }
                        }
                    ]
                }
            ),
            payload,
        )

    monkeypatch.setattr(probe, "post_chat_completion_with_variant", fake_post)

    result = probe.run_agent_loop("ExampleBench", probe.BenchmarkItem("Answer.", "hidden", "test"), tools=[])

    assert result["answer"] == ""
    assert result["diagnostics"]["status"] == "failed_model_format"
    assert "hidden reasoning" in result["diagnostics"]["reason"]


def test_execute_agent_tool_supports_ls_alias(tmp_path, monkeypatch):
    probe = _load_probe_module()
    monkeypatch.chdir(tmp_path)
    (tmp_path / "answer.txt").write_text("42\n", encoding="utf-8")

    result = probe.execute_agent_tool("ls", {"path": "."})

    assert "answer.txt" in result


def test_exploitbench_does_not_extract_spec_markdown(tmp_path, monkeypatch):
    probe = _load_probe_module()
    monkeypatch.setenv("AGENT_BENCH_BENCHMARK_NAME", "ExploitBench")
    (tmp_path / "benchmarks").mkdir()
    (tmp_path / "benchmarks" / "bench-v8").mkdir()
    (tmp_path / "benchmarks" / "v8.yaml").write_text("target_image: ghcr.io/exploitbench/v8:latest\n", encoding="utf-8")
    (tmp_path / "benchmarks" / "v8-small.yaml").write_text("image: ghcr.io/exploitbench/v8-small:latest\n", encoding="utf-8")
    (tmp_path / "benchmarks-bench-v8-SPEC.md").write_text("# spec\nThis is methodology.\n", encoding="utf-8")

    items, errors = probe.extract_benchmark_items(tmp_path, limit=3)

    assert errors == []
    assert items
    assert all("SPEC.md" not in item.source for item in items)
    assert all(item.metadata.get("target_image") or item.metadata.get("environment") for item in items)


def test_exploitbench_preflight_requires_upstream_runner(tmp_path, monkeypatch):
    probe = _load_probe_module()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENT_BENCH_BENCHMARK_NAME", "ExploitBench")
    monkeypatch.delenv("AGENT_BENCH_EXPLOITBENCH_UPSTREAM_READY", raising=False)
    monkeypatch.setattr(probe.shutil, "which", lambda name: None)
    (tmp_path / "benchmarks" / "bench-v8").mkdir(parents=True)
    (tmp_path / "benchmarks" / "v8.yaml").write_text("target_image: ghcr.io/exploitbench/v8:latest\n", encoding="utf-8")
    (tmp_path / "benchmarks" / "v8-small.yaml").write_text("target_image: ghcr.io/exploitbench/v8-small:latest\n", encoding="utf-8")
    item = probe.BenchmarkItem(
        "Use upstream runner.",
        "oracle",
        "benchmarks/v8.yaml",
        metadata={
            "benchmark_config": "benchmarks/v8.yaml",
            "target_image": "ghcr.io/exploitbench/v8:latest",
            "oracle": "upstream_capability_oracle",
        },
    )

    result = probe.validate_exploitbench_item(item)

    assert result is not None
    assert result[0] == "failed_missing_required_tool"
    assert result[2]["blocker_type"] == "missing_required_tool_backend"
    assert result[2]["missing_tools"] == ["exploitbench"]


def test_finance_agent_v2_rejects_task_md_only_item(tmp_path, monkeypatch):
    probe = _load_probe_module()
    monkeypatch.setenv("AGENT_BENCH_BENCHMARK_NAME", "Finance Agent v2")
    item = probe.BenchmarkItem(
        "TASK.md-only prompt",
        "requires official evidence",
        "TASK.md",
        metadata={"task_md_only": True},
    )

    result = probe.validate_finance_agent_v2_item(item, tmp_path, [])

    assert result is not None
    assert result[0] == "failed_missing_required_tool"
    assert result[2]["blocker_type"] == "missing_required_tool_backend"
    assert result[2]["missing_tools"] == sorted(probe.FINANCE_AGENT_V2_REQUIRED_TOOLS)


def test_finance_agent_v2_static_public_item_skips_backend_credentials_only_when_descriptor_static(tmp_path, monkeypatch):
    probe = _load_probe_module()
    monkeypatch.setenv("AGENT_BENCH_BENCHMARK_NAME", "Finance Agent v2")
    for key in probe.FINANCE_AGENT_V2_REQUIRED_ENV:
        monkeypatch.delenv(key, raising=False)
    item = probe.BenchmarkItem(
        "Answer from the public prompt.",
        "rubric",
        "data/public.csv:record-000001",
        metadata={
            "live_tools_required": False,
            "required_tools": sorted(probe.FINANCE_AGENT_V2_REQUIRED_TOOLS),
        },
    )

    result = probe.validate_finance_agent_v2_item(item, tmp_path, probe.agent_tool_schemas())

    assert result is not None
    assert result[0] == "failed_missing_required_tool"

    monkeypatch.setenv(
        "AGENT_BENCH_BENCHMARK_JSON",
        json.dumps(
            {
                "name": "Finance Agent v2",
                "adapter_mode": "static_gold_answer",
                "live_tools_required": False,
            }
        ),
    )
    result = probe.validate_finance_agent_v2_item(item, tmp_path, probe.agent_tool_schemas())

    assert result is None


def test_missing_required_tool_helper_blocks_model_call():
    probe = _load_probe_module()
    item = probe.BenchmarkItem(
        "Call selected tool.",
        "Use tool.",
        "synthetic",
        metadata={"select_tools": ["companies_balance_sheet_statements"]},
    )
    tools = [{"type": "function", "function": {"name": "final_answer"}}]

    assert probe._missing_required_tools(item, tools) == ["companies_balance_sheet_statements"]


def test_finmcp_static_item_validation_rejects_live_tool_call_metadata():
    probe = _load_probe_module()
    item = probe.BenchmarkItem(
        "query\ntranscript",
        "answer",
        "record",
        metadata={
            "source_dataset": "DianJin/FinMCP-Bench",
            "live_tools_required": False,
            "required_capabilities": ["tool_call"],
            "transcript": "tool output",
        },
    )

    result = probe.validate_finmcp_static_item(item)

    assert result is not None
    assert result[0] == "failed_invalid_task_context"


def test_probe_parses_negative_qualitative_judge_prose():
    probe = _load_probe_module()

    grade = probe.parse_qualitative_prose_judge_grade(
        "The candidate answer is not a direct response to the user. "
        "It is a meta-analysis and fails to provide the requested final answer."
    )

    assert grade == {
        "score": 0.0,
        "passed": False,
        "reason": "judge prose described the candidate as not directly satisfying the task",
    }


def test_probe_parses_positive_qualitative_judge_prose():
    probe = _load_probe_module()

    grade = probe.parse_qualitative_prose_judge_grade(
        "The candidate answer is accurate and directly addresses the task."
    )

    assert grade == {
        "score": 0.5,
        "passed": False,
        "reason": "judge prose described the candidate as partially task-relevant",
    }
