import json
from pathlib import Path

import pytest

from agent_bench.tasks import TaskLoadError, load_tasks


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


def test_bundled_task_categories_are_balanced_to_twenty_items():
    counts: dict[str, int] = {}
    for path in sorted(REPO_TASKS_DIR.glob("*.json")):
        counts[path.stem] = len(json.loads(path.read_text(encoding="utf-8")))

    assert counts == {
        "analytical_reasoning": 20,
        "code_recall": 20,
        "coding": 20,
        "mathematical_reasoning": 20,
        "planning": 20,
        "probability": 20,
        "system_design": 20,
    }


def test_bundled_task_ids_are_sequential_per_category():
    prefixes = {
        "analytical_reasoning": "AR",
        "code_recall": "CR",
        "coding": "CO",
        "mathematical_reasoning": "MR",
        "planning": "PL",
        "probability": "PR",
        "system_design": "SD",
    }

    for category, prefix in prefixes.items():
        path = REPO_TASKS_DIR / f"{category}.json"
        tasks = json.loads(path.read_text(encoding="utf-8"))

        assert [task["id"] for task in tasks] == [
            f"{prefix}_{index:03d}" for index in range(1, 21)
        ]


def test_coding_config_arrays_are_compactly_formatted():
    lines = (REPO_TASKS_DIR / "coding.json").read_text(encoding="utf-8").splitlines()
    compact_array_keys = ("operations", "arguments", "output", "outputs")

    for line in lines:
        stripped = line.strip()
        if stripped.startswith('"input": {'):
            assert stripped.endswith(("},", "}"))
        if any(stripped.startswith(f'"{key}": [') for key in compact_array_keys):
            assert stripped.endswith(("],", "]"))
