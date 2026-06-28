import importlib.util
import json
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
    monkeypatch.setenv("AGENT_BENCH_REQUIRED_CAPABILITIES", "repo_patch,chat_answer")

    required = probe._required_capabilities("ExampleBench")

    assert required == {"repo_patch", "chat_answer"}
    assert sorted(required - probe.HARNESS_SUPPORTED_CAPABILITIES) == ["repo_patch"]


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
