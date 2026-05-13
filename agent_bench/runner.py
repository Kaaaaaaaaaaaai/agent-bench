import asyncio
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agent_bench.aggregator import aggregate_results
from agent_bench.clients import make_client
from agent_bench.models import GradeResult, ModelResponse, Task
from agent_bench.reports import update_latest, write_jsonl_line, write_result_artifacts
from agent_bench.sandbox import make_sandbox
from agent_bench.tasks import load_tasks
from agent_bench.verifiers import grade_task


@dataclass(slots=True)
class RunConfig:
    provider: str = "mock"
    base_url: str | None = None
    model: str | None = None
    api_key_env: str | None = None
    tasks_dir: Path = Path("tasks")
    out: Path = Path("runs/latest")
    request_concurrency: int = 8
    eval_concurrency: int = 4
    timeout: float = 60.0
    limit: int | None = None
    include: set[str] | None = None
    temperature: float = 0.0
    max_tokens: int = 4096
    json_mode: str = "auto"
    sandbox: str = "docker"
    sandbox_image: str = "agent-bench-python:3.12"


async def run_benchmark(config: RunConfig) -> dict[str, Any]:
    run_started = time.perf_counter()
    tasks = load_tasks(config.tasks_dir, include=config.include, limit=config.limit)
    output_dir, latest_dir = _prepare_output_paths(config.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    client = make_client(
        provider=config.provider,
        base_url=config.base_url,
        model=config.model,
        api_key_env=config.api_key_env,
        timeout=config.timeout,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        json_mode=config.json_mode,
    )
    sandbox = make_sandbox(config.sandbox, config.sandbox_image)
    request_sem = asyncio.Semaphore(max(1, config.request_concurrency))
    eval_sem = asyncio.Semaphore(max(1, config.eval_concurrency))
    raw_lock = asyncio.Lock()
    graded_lock = asyncio.Lock()
    responses: list[ModelResponse] = []
    grades: list[GradeResult] = []

    raw_path = output_dir / "raw_responses.jsonl"
    graded_path = output_dir / "graded_results.jsonl"

    try:
        with raw_path.open("w", encoding="utf-8") as raw_handle, graded_path.open(
            "w", encoding="utf-8"
        ) as graded_handle:
            coroutines = [
                _process_task(
                    task=task,
                    client=client,
                    sandbox=sandbox,
                    timeout=config.timeout,
                    request_sem=request_sem,
                    eval_sem=eval_sem,
                    raw_lock=raw_lock,
                    graded_lock=graded_lock,
                    raw_handle=raw_handle,
                    graded_handle=graded_handle,
                    responses=responses,
                )
                for task in tasks
            ]
            for completed in asyncio.as_completed(coroutines):
                grades.append(await completed)
    finally:
        await client.aclose()

    grades.sort(key=lambda result: _task_order(tasks, result.task_id))
    responses.sort(key=lambda response: _task_order(tasks, response.task_id))
    run_duration_seconds = time.perf_counter() - run_started
    metadata = _metadata(config, len(tasks), output_dir, run_duration_seconds)
    summary = aggregate_results(grades, metadata)
    write_result_artifacts(output_dir, responses, grades, summary)
    if latest_dir is not None:
        update_latest(output_dir, latest_dir)
    return summary


async def _process_task(
    task: Task,
    client: Any,
    sandbox: Any,
    timeout: float,
    request_sem: asyncio.Semaphore,
    eval_sem: asyncio.Semaphore,
    raw_lock: asyncio.Lock,
    graded_lock: asyncio.Lock,
    raw_handle: Any,
    graded_handle: Any,
    responses: list[ModelResponse],
) -> GradeResult:
    task_started = time.perf_counter()
    async with request_sem:
        response = await client.complete(task)
    responses.append(response)
    async with raw_lock:
        write_jsonl_line(raw_handle, response.to_dict())

    async with eval_sem:
        grade = await grade_task(task, response, sandbox, timeout)
    grade.task_duration_seconds = time.perf_counter() - task_started
    async with graded_lock:
        write_jsonl_line(graded_handle, grade.to_dict())
    return grade


def _prepare_output_paths(out: Path) -> tuple[Path, Path | None]:
    out = Path(out)
    if out.name == "latest":
        timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")
        return out.parent / timestamp, out
    return out, None


def _metadata(
    config: RunConfig,
    task_count: int,
    output_dir: Path,
    run_duration_seconds: float,
) -> dict[str, Any]:
    return {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "provider": config.provider,
        "base_url": config.base_url or "",
        "model": config.model or ("mock-perfect" if config.provider == "mock" else ""),
        "api_key_env": config.api_key_env or "",
        "tasks": str(config.tasks_dir),
        "task_count": task_count,
        "output_dir": str(output_dir),
        "run_duration_seconds": run_duration_seconds,
        "request_concurrency": config.request_concurrency,
        "eval_concurrency": config.eval_concurrency,
        "timeout": config.timeout,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "json_mode": config.json_mode,
        "sandbox": config.sandbox,
        "sandbox_image": config.sandbox_image if config.sandbox == "docker" else "",
    }


def _task_order(tasks: list[Task], task_id: str) -> int:
    for index, task in enumerate(tasks):
        if task.id == task_id:
            return index
    return len(tasks)
