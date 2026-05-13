import json
from pathlib import Path
from typing import Any

from agent_bench.models import Task


class TaskLoadError(ValueError):
    """Raised when a task file is malformed."""


def load_tasks(
    tasks_dir: str | Path,
    include: set[str] | None = None,
    limit: int | None = None,
) -> list[Task]:
    root = Path(tasks_dir)
    if not root.exists():
        raise TaskLoadError(f"Task directory does not exist: {root}")
    if not root.is_dir():
        raise TaskLoadError(f"Task path is not a directory: {root}")

    tasks: list[Task] = []
    for path in sorted(root.glob("*.json")):
        category = path.stem
        tasks.extend(_load_task_file(path, category))

    if include:
        include_normalized = {item.strip() for item in include if item.strip()}
        tasks = [
            task
            for task in tasks
            if task.category in include_normalized
            or task.id in include_normalized
            or task.source in include_normalized
        ]

    if limit is not None:
        if limit < 0:
            raise TaskLoadError("limit must be non-negative")
        tasks = tasks[:limit]

    if not tasks:
        raise TaskLoadError("No tasks were loaded")
    return tasks


def _load_task_file(path: Path, category: str) -> list[Task]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TaskLoadError(f"{path}: invalid JSON: {exc}") from exc

    if not isinstance(payload, list):
        raise TaskLoadError(f"{path}: top-level JSON value must be a list")

    tasks: list[Task] = []
    seen_ids: set[str] = set()
    for index, raw in enumerate(payload):
        if not isinstance(raw, dict):
            raise TaskLoadError(f"{path}: item {index} must be an object")
        task = _parse_task(raw, category, path.name, path.parent, index)
        if task.id in seen_ids:
            raise TaskLoadError(f"{path}: duplicate task id {task.id}")
        seen_ids.add(task.id)
        tasks.append(task)
    return tasks


def _parse_task(
    raw: dict[str, Any],
    category: str,
    source: str,
    base_dir: Path,
    index: int,
) -> Task:
    task_type = _required_str(raw, "type", source, index)
    task_id = _required_str(raw, "id", source, index)
    question = _required_str(raw, "question", source, index)

    if task_type == "multiple_choice":
        choices = raw.get("choices")
        answer = raw.get("answer")
        if not isinstance(choices, list) or not all(isinstance(item, str) for item in choices):
            raise TaskLoadError(f"{source}: {task_id} choices must be a list of strings")
        if not isinstance(answer, list) or not all(isinstance(item, str) for item in answer):
            raise TaskLoadError(f"{source}: {task_id} answer must be a list of strings")
        return Task(
            id=task_id,
            category=category,
            type=task_type,
            question=question,
            source=source,
            choices=choices,
            answer=answer,
        )

    if task_type == "coding":
        function_name = _required_str(raw, "function_name", source, index)
        test_cases = raw.get("test_cases")
        if not isinstance(test_cases, list) or not test_cases:
            raise TaskLoadError(f"{source}: {task_id} test_cases must be a non-empty list")
        for case_index, case in enumerate(test_cases):
            if not isinstance(case, dict) or "input" not in case or "output" not in case:
                raise TaskLoadError(
                    f"{source}: {task_id} test case {case_index} must contain input and output"
                )
            if not isinstance(case["input"], dict):
                raise TaskLoadError(f"{source}: {task_id} test case {case_index} input must be an object")
        return Task(
            id=task_id,
            category=category,
            type=task_type,
            question=question,
            source=source,
            title=raw.get("title"),
            function_name=function_name,
            test_cases=test_cases,
            comparison=raw.get("comparison"),
        )

    if task_type == "text_recall":
        expected_text = _required_str(raw, "expected_text", source, index)
        reference_path = _required_str(raw, "reference_path", source, index)
        reference_text = _load_reference_text(base_dir, reference_path, source, task_id)
        return Task(
            id=task_id,
            category=category,
            type=task_type,
            question=question,
            source=source,
            expected_text=expected_text,
            reference_path=reference_path,
            reference_text=reference_text,
        )

    raise TaskLoadError(f"{source}: {task_id} unsupported task type {task_type!r}")


def _required_str(raw: dict[str, Any], key: str, source: str, index: int) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise TaskLoadError(f"{source}: item {index} requires non-empty string field {key!r}")
    return value


def _load_reference_text(base_dir: Path, reference_path: str, source: str, task_id: str) -> str:
    root = base_dir.resolve()
    candidate = (base_dir / reference_path).resolve()
    if not candidate.is_relative_to(root):
        raise TaskLoadError(f"{source}: {task_id} reference_path must stay inside {base_dir}")
    if not candidate.is_file():
        raise TaskLoadError(f"{source}: {task_id} reference file does not exist: {reference_path}")
    return candidate.read_text(encoding="utf-8")
