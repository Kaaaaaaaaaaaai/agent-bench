import asyncio
import json
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agent_bench.aggregator import aggregate_results
from agent_bench.clients import make_client
from agent_bench.external import ExternalBenchmarkConfig, ExternalBenchmarkRunner
from agent_bench.models import GradeResult, ModelResponse, Task
from agent_bench.reports import update_latest, write_jsonl_line, write_result_artifacts
from agent_bench.sandbox import make_sandbox
from agent_bench.tasks import load_tasks
from agent_bench.verifiers import grade_task


MAX_EMPTY_RESPONSES_PER_TASK = 3
DEFAULT_TASK_TIMEOUT_SECONDS = 60.0
DEFAULT_EXTERNAL_TIMEOUT_SECONDS = 6 * 60 * 60.0
DEFAULT_MODEL_REQUEST_TIMEOUT_SECONDS = 10 * 60.0


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
    timeout: float = DEFAULT_TASK_TIMEOUT_SECONDS
    external_timeout: float = DEFAULT_EXTERNAL_TIMEOUT_SECONDS
    model_request_timeout: float = DEFAULT_MODEL_REQUEST_TIMEOUT_SECONDS
    limit: int | None = None
    include: set[str] | None = None
    temperature: float = 0.0
    max_tokens: int = 4096
    json_mode: str = "auto"
    sandbox: str = "docker"
    sandbox_image: str = "agent-bench-python:3.12"
    external_launcher_image: str = "agent-bench-external:python3.12"


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
    external_runner = ExternalBenchmarkRunner()
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
                    external_runner=external_runner,
                    config=config,
                    output_dir=output_dir,
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
    external_runner: ExternalBenchmarkRunner,
    config: RunConfig,
    output_dir: Path,
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
    if task.is_external_benchmark and config.provider != "mock":
        response = await _run_external_benchmark(task, external_runner, config, output_dir)
    else:
        empty_response_count = 0
        while True:
            async with request_sem:
                response = await client.complete(task)

            if not _is_empty_model_response(response):
                break

            empty_response_count += 1
            if empty_response_count >= MAX_EMPTY_RESPONSES_PER_TASK:
                break

    responses.append(response)
    async with raw_lock:
        write_jsonl_line(raw_handle, response.to_dict())

    async with eval_sem:
        grade = await grade_task(task, response, sandbox, timeout)
    grade.task_duration_seconds = time.perf_counter() - task_started
    async with graded_lock:
        write_jsonl_line(graded_handle, grade.to_dict())
    return grade



async def _run_external_benchmark(
    task: Task,
    external_runner: ExternalBenchmarkRunner,
    config: RunConfig,
    output_dir: Path,
) -> ModelResponse:
    result = await external_runner.run(
        task,
        ExternalBenchmarkConfig(
            provider=config.provider,
            base_url=config.base_url or "",
            model=config.model or "",
            api_key_env=config.api_key_env or "",
            output_dir=output_dir,
            timeout=config.external_timeout,
            limit=config.limit,
            model_request_timeout=config.model_request_timeout,
            launcher_image=config.external_launcher_image,
            source_root=Path(os.environ.get("AGENT_BENCH_SOURCE_ROOT", Path.cwd())),
        ),
    )
    details = dict(result.details)
    details["output_tail"] = result.output
    return ModelResponse(
        task_id=task.id,
        model=config.model or config.provider,
        raw_response=json.dumps(
            {
                "status": "completed" if result.error is None else "error",
                "score": result.score,
                "error": result.error,
                "timed_out": result.timed_out,
                "details": details,
            },
            ensure_ascii=False,
        ),
        latency_seconds=result.latency_seconds,
        error=None,
    )

def _is_empty_model_response(response: ModelResponse) -> bool:
    return response.error is None and not response.raw_response.strip()


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
        "external_timeout": config.external_timeout,
        "model_request_timeout": config.model_request_timeout,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "json_mode": config.json_mode,
        "sandbox": config.sandbox,
        "sandbox_image": config.sandbox_image if config.sandbox == "docker" else "",
        "external_launcher_image": config.external_launcher_image,
    }


def _task_order(tasks: list[Task], task_id: str) -> int:
    for index, task in enumerate(tasks):
        if task.id == task_id:
            return index
    return len(tasks)
