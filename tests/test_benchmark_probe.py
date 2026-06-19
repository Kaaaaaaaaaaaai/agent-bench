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
