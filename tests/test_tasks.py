import json
from pathlib import Path

import pytest

from agent_bench.tasks import TaskLoadError, load_task_registry, load_tasks


REPO_TASKS_DIR = Path(__file__).resolve().parents[1] / "tasks"


def test_load_tasks_discovers_json_files_and_derives_category(tmp_path):
    task_dir = tmp_path / "tasks"
    task_dir.mkdir()
    (task_dir / "extra_reasoning.json").write_text(
        json.dumps(
            [
                {
                    "id": "EX_001",
                    "type": "multiple_choice",
                    "question": "Pick A",
                    "choices": ["yes", "no"],
                    "answer": ["A"],
                }
            ]
        ),
        encoding="utf-8",
    )

    tasks = load_tasks(task_dir)

    assert len(tasks) == 1
    assert tasks[0].id == "EX_001"
    assert tasks[0].category == "extra_reasoning"
    assert tasks[0].source == "extra_reasoning.json"


def test_load_tasks_filters_by_category_and_limit(tmp_path):
    task_dir = tmp_path / "tasks"
    task_dir.mkdir()
    for name in ("alpha", "beta"):
        (task_dir / f"{name}.json").write_text(
            json.dumps(
                [
                    {
                        "id": f"{name}_001",
                        "type": "multiple_choice",
                        "question": "Pick A",
                        "choices": ["yes", "no"],
                        "answer": ["A"],
                    }
                ]
            ),
            encoding="utf-8",
        )

    tasks = load_tasks(task_dir, include={"beta"}, limit=1)

    assert [task.category for task in tasks] == ["beta"]


def test_load_tasks_rejects_missing_required_fields(tmp_path):
    task_dir = tmp_path / "tasks"
    task_dir.mkdir()
    (task_dir / "bad.json").write_text(json.dumps([{"id": "BAD"}]), encoding="utf-8")

    with pytest.raises(TaskLoadError):
        load_tasks(task_dir)


def test_load_tasks_supports_text_recall(tmp_path):
    task_dir = tmp_path / "tasks"
    task_dir.mkdir()
    ref_dir = task_dir / "ref"
    ref_dir.mkdir()
    (ref_dir / "example.py").write_text("def target(value):\n    return value\n", encoding="utf-8")
    (task_dir / "code_recall.json").write_text(
        json.dumps(
            [
                {
                    "id": "CR_001",
                    "type": "text_recall",
                    "question": "Recall the requested function header.",
                    "expected_text": "def target(value):\n    return value",
                    "reference_path": "ref/example.py",
                }
            ]
        ),
        encoding="utf-8",
    )

    tasks = load_tasks(task_dir)

    assert len(tasks) == 1
    assert tasks[0].is_text_recall is True
    assert tasks[0].expected_text == "def target(value):\n    return value"
    assert tasks[0].reference_path == "ref/example.py"
    assert tasks[0].reference_text == "def target(value):\n    return value\n"




def test_bundled_public_benchmarks_are_external_tasks_with_credits():
    tasks = json.loads((REPO_TASKS_DIR / "public_benchmarks.json").read_text(encoding="utf-8"))

    assert len(tasks) == 12
    assert {task["type"] for task in tasks} == {"external_benchmark"}
    assert all(task["benchmark"].get("license") for task in tasks)
    assert all(task["benchmark"].get("credit") for task in tasks)
    assert all(task["benchmark"].get("citation") for task in tasks)
    assert all(task["benchmark"].get("group") for task in tasks)
    assert [task["id"] for task in tasks] == [
        "PB_001",
        "PB_004",
        "PB_005",
        "PB_009",
        "PB_010",
        "PB_011",
        "PB_012",
        "PB_013",
        "PB_014",
        "PB_015",
        "PB_016",
        "PB_017",
    ]
    assert [task["benchmark"]["name"] for task in tasks] == [
        "SWE-bench",
        "SWE-Lancer",
        "SWE-bench Verified",
        "ExploitBench",
        "codeneedle",
        "StockBench",
        "InvestorBench",
        "QuantCode-Bench",
        "FinMCP-Bench",
        "FinToolBench",
        "Finance Agent v2",
        "FinanceMath",
    ]

    registry = load_task_registry(REPO_TASKS_DIR)
    loaded = load_tasks(REPO_TASKS_DIR)
    loaded_ids = [task.id for task in loaded]
    assert len(registry) == 12
    assert len(loaded) == 12
    assert loaded_ids == [task["id"] for task in tasks]
    assert "public_benchmarks" not in {task.category for task in loaded}
    assert {task.category for task in loaded} == {
        "Coding",
        "Finance",
        "Long Context",
        "Security",
    }


def test_full_active_profile_selects_all_current_public_benchmarks():
    loaded = load_tasks(REPO_TASKS_DIR, profile="full_active")

    assert len(loaded) == 12
    assert [task.id for task in loaded] == [
        "PB_001",
        "PB_004",
        "PB_005",
        "PB_009",
        "PB_010",
        "PB_011",
        "PB_012",
        "PB_013",
        "PB_014",
        "PB_015",
        "PB_016",
        "PB_017",
    ]


def test_finmcp_descriptor_is_static_not_live_tool_call():
    tasks = json.loads((REPO_TASKS_DIR / "public_benchmarks.json").read_text(encoding="utf-8"))
    descriptor = next(task for task in tasks if task["id"] == "PB_014")

    assert "tool_call" not in descriptor["benchmark"]["capabilities"]
    assert descriptor["benchmark"]["adapter"] == "static_transcript_reasoning"


def test_load_tasks_supports_external_benchmark(tmp_path):
    task_dir = tmp_path / "tasks"
    task_dir.mkdir()
    (task_dir / "benchmarks.json").write_text(
        json.dumps(
            [
                {
                    "id": "PB_001",
                    "type": "external_benchmark",
                    "question": "Run benchmark",
                    "benchmark": {
                        "name": "ExampleBench",
                        "group": "Coding",
                        "homepage": "https://example.com",
                        "repository": "https://example.com/repo.git",
                        "license": "MIT",
                        "credit": "Example authors",
                        "docker": {"image": "example", "command": "echo ok"},
                    },
                }
            ]
        ),
        encoding="utf-8",
    )

    tasks = load_tasks(task_dir)

    assert len(tasks) == 1
    assert tasks[0].is_external_benchmark is True
    assert tasks[0].category == "Coding"
    assert tasks[0].benchmark["name"] == "ExampleBench"


def test_load_tasks_keeps_non_mit_license_as_metadata(tmp_path):
    task_dir = tmp_path / "tasks"
    task_dir.mkdir()
    (task_dir / "benchmarks.json").write_text(
        json.dumps(
            [
                {
                    "id": "PB_001",
                    "type": "external_benchmark",
                    "question": "Run benchmark",
                    "benchmark": {
                        "name": "ExampleBench",
                        "homepage": "https://example.com",
                        "license": "Apache-2.0",
                        "credit": "Example authors",
                        "docker": {"image": "example", "command": "echo ok"},
                    },
                }
            ]
        ),
        encoding="utf-8",
    )

    tasks = load_tasks(task_dir)

    assert tasks[0].benchmark["license"] == "Apache-2.0"
