import asyncio
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agent_bench.aggregator import aggregate_results
from agent_bench.clients import make_client
from agent_bench.external import ExternalBenchmarkConfig, ExternalBenchmarkRunner
from agent_bench.manifest import JudgeMetadata, ModelMetadata, load_manifest_tasks
from agent_bench.models import GradeResult, ModelResponse, Task
from agent_bench.proxy import JsonlRecorder, OpenAIProxyConfig, OpenAIRecordingProxy, redact_url
from agent_bench.reports import update_latest, write_result_artifacts
from agent_bench.sandbox import make_sandbox
from agent_bench.tasks import TaskLoadError, load_task_registry, select_tasks
from agent_bench.verifiers import grade_task


MAX_EMPTY_RESPONSES_PER_TASK = 3
DEFAULT_TASK_TIMEOUT_SECONDS = 60.0
DEFAULT_EXTERNAL_TIMEOUT_SECONDS = 6 * 60 * 60.0
DEFAULT_MODEL_REQUEST_TIMEOUT_SECONDS = 30 * 60.0
DEFAULT_MAX_TOKENS = 16384
DEFAULT_EXTERNAL_ASSET_ROOT = Path("agent-bench-assets")


@dataclass(slots=True)
class RunConfig:
    provider: str = "mock"
    base_url: str | None = None
    model: str | None = None
    api_key_env: str | None = None
    tasks_dir: Path = Path("tasks")
    benchmark_root: Path = Path("benchmarks")
    out: Path = Path("runs/latest")
    request_concurrency: int = 8
    eval_concurrency: int = 4
    timeout: float = DEFAULT_TASK_TIMEOUT_SECONDS
    external_timeout: float = DEFAULT_EXTERNAL_TIMEOUT_SECONDS
    model_request_timeout: float = DEFAULT_MODEL_REQUEST_TIMEOUT_SECONDS
    limit: int | None = None
    include: set[str] | None = None
    temperature: float = 0.0
    top_p: float | None = None
    max_tokens: int = DEFAULT_MAX_TOKENS
    seed: int | None = None
    max_retries: int = 2
    tool_parser: str = "auto"
    context_window: int | None = None
    stop: list[str] | None = None
    json_mode: str = "auto"
    sandbox: str = "docker"
    sandbox_image: str = "agent-bench-python:3.12"
    external_launcher_image: str = "agent-bench-external:python3.12"
    asset_root: Path = DEFAULT_EXTERNAL_ASSET_ROOT
    profile: str = "full_active"
    suite_ids: set[str] | None = None
    judge_provider: str = "none"
    judge_base_url: str | None = None
    judge_model: str | None = None
    judge_api_key_env: str | None = None
    judge_temperature: float = 0.0
    judge_timeout: float = DEFAULT_MODEL_REQUEST_TIMEOUT_SECONDS
    judge_max_retries: int = 2
    judge_fallback: str = "same-as-target"
    allow_host_docker_socket: bool = False
    cli_args: list[str] | None = None


async def run_benchmark(config: RunConfig) -> dict[str, Any]:
    run_started = time.perf_counter()
    manifest_tasks = load_manifest_tasks(config.benchmark_root)
    try:
        registry = load_task_registry(config.tasks_dir)
    except TaskLoadError:
        if not manifest_tasks:
            raise
        registry = []
    registry.extend(manifest_tasks)
    tasks = select_tasks(
        registry,
        include=config.include,
        limit=config.limit,
        suite_ids=config.suite_ids,
        profile=config.profile,
    )
    output_dir, latest_dir = _prepare_output_paths(config.out)
    _reset_output_dir(output_dir)

    client = make_client(
        provider=config.provider,
        base_url=config.base_url,
        model=config.model,
        api_key_env=config.api_key_env,
        timeout=config.timeout,
        temperature=config.temperature,
        top_p=config.top_p,
        max_tokens=config.max_tokens,
        seed=config.seed,
        stop=list(config.stop or []),
        max_retries=config.max_retries,
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
    proxy: OpenAIRecordingProxy | None = None
    judge_proxy: OpenAIRecordingProxy | None = None
    external_base_url = config.base_url or ""
    judge_base_url = ""
    judge_fallback_used = False
    api_key = os.environ.get(config.api_key_env or "", "") if config.api_key_env else ""
    judge_api_key = os.environ.get(config.judge_api_key_env or "", "") if config.judge_api_key_env else ""

    try:
        with JsonlRecorder(raw_path) as raw_recorder, JsonlRecorder(graded_path) as graded_recorder:
            if _needs_model_proxy(config, tasks):
                proxy = OpenAIRecordingProxy(
                    OpenAIProxyConfig(
                        upstream_base_url=config.base_url or "http://localhost:8000/v1",
                        model=config.model or "",
                        api_key=api_key,
                        label="target",
                        timeout_seconds=config.model_request_timeout,
                        tool_parser=config.tool_parser,
                    ),
                    raw_recorder,
                )
                proxy.start()
                external_base_url = proxy.container_base_url
            if config.judge_provider == "openai-compatible" and config.judge_base_url:
                judge_proxy = OpenAIRecordingProxy(
                    OpenAIProxyConfig(
                        upstream_base_url=config.judge_base_url,
                        model=config.judge_model or "",
                        api_key=judge_api_key,
                        label="judge",
                        timeout_seconds=config.judge_timeout,
                        tool_parser="none",
                    ),
                    raw_recorder,
                )
                judge_proxy.start()
                judge_base_url = judge_proxy.container_base_url
            elif config.judge_provider == "same-as-target" or (
                config.judge_provider in {"openai-compatible", "none"}
                and config.judge_fallback == "same-as-target"
                and external_base_url
            ):
                judge_base_url = external_base_url
                judge_fallback_used = config.judge_provider != "same-as-target"

            coroutines = [
                _process_task(
                    task=task,
                    client=client,
                    sandbox=sandbox,
                    external_runner=external_runner,
                    config=config,
                    output_dir=output_dir,
                    external_base_url=external_base_url,
                    judge_base_url=judge_base_url,
                    judge_fallback_used=judge_fallback_used,
                    timeout=config.timeout,
                    request_sem=request_sem,
                    eval_sem=eval_sem,
                    raw_lock=raw_lock,
                    graded_lock=graded_lock,
                    raw_recorder=raw_recorder,
                    graded_recorder=graded_recorder,
                    responses=responses,
                )
                for task in tasks
            ]
            for completed in asyncio.as_completed(coroutines):
                grades.append(await completed)
    finally:
        if proxy is not None:
            proxy.stop()
        if judge_proxy is not None:
            judge_proxy.stop()
        await client.aclose()

    grades.sort(key=lambda result: _task_order(tasks, result.task_id))
    responses.sort(key=lambda response: _task_order(tasks, response.task_id))
    run_duration_seconds = time.perf_counter() - run_started
    metadata = _metadata(
        config,
        task_count=len(tasks),
        registry_count=len(registry),
        output_dir=output_dir,
        run_duration_seconds=run_duration_seconds,
        judge_base_url=judge_base_url or config.judge_base_url or "",
        judge_fallback_used=judge_fallback_used,
    )
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
    external_base_url: str,
    judge_base_url: str,
    judge_fallback_used: bool,
    timeout: float,
    request_sem: asyncio.Semaphore,
    eval_sem: asyncio.Semaphore,
    raw_lock: asyncio.Lock,
    graded_lock: asyncio.Lock,
    raw_recorder: JsonlRecorder,
    graded_recorder: JsonlRecorder,
    responses: list[ModelResponse],
) -> GradeResult:
    task_started = time.perf_counter()
    if task.is_external_benchmark and config.provider != "mock":
        async with request_sem:
            response = await _run_external_benchmark(
                task,
                external_runner,
                config,
                output_dir,
                external_base_url=external_base_url,
                judge_base_url=judge_base_url,
                judge_fallback_used=judge_fallback_used,
            )
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
        raw_recorder.write(_raw_response_record(task, response))

    async with eval_sem:
        grade = await grade_task(task, response, sandbox, timeout)
    if response.usage:
        grade.details.setdefault("usage", response.usage)
    grade.task_duration_seconds = time.perf_counter() - task_started
    async with graded_lock:
        graded_recorder.write(_graded_result_record(task, grade))
    return grade



async def _run_external_benchmark(
    task: Task,
    external_runner: ExternalBenchmarkRunner,
    config: RunConfig,
    output_dir: Path,
    *,
    external_base_url: str,
    judge_base_url: str,
    judge_fallback_used: bool,
) -> ModelResponse:
    result = await external_runner.run(
        task,
        ExternalBenchmarkConfig(
            provider=config.provider,
            base_url=external_base_url or config.base_url or "",
            model=config.model or "",
            api_key_env=config.api_key_env or "",
            output_dir=output_dir,
            timeout=config.external_timeout,
            limit=config.limit,
            model_request_timeout=config.model_request_timeout,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            seed=config.seed,
            stop=list(config.stop or []),
            tool_parser=config.tool_parser,
            context_window=config.context_window,
            judge_base_url=judge_base_url,
            judge_model=config.judge_model or config.model or "",
            judge_temperature=config.judge_temperature,
            judge_timeout=config.judge_timeout,
            judge_max_retries=config.judge_max_retries,
            judge_fallback_used=judge_fallback_used,
            asset_root=config.asset_root,
            launcher_image=config.external_launcher_image,
            source_root=Path(os.environ.get("AGENT_BENCH_SOURCE_ROOT", Path.cwd())),
            allow_host_docker_socket=config.allow_host_docker_socket,
            pass_api_key_to_container=False,
        ),
    )
    details = dict(result.details)
    details["output_tail"] = result.output
    result_payload = details.get("result") if isinstance(details.get("result"), dict) else {}
    benchmark_status = result_payload.get("status") if isinstance(result_payload.get("status"), str) else ""
    return ModelResponse(
        task_id=task.id,
        model=config.model or config.provider,
        raw_response=json.dumps(
            {
                "status": benchmark_status or ("completed" if result.error is None else "error"),
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


def _needs_model_proxy(config: RunConfig, tasks: list[Task]) -> bool:
    return config.provider == "openai-compatible" and any(task.is_external_benchmark for task in tasks)


def _raw_response_record(task: Task, response: ModelResponse) -> dict[str, Any]:
    return {
        "record_type": "benchmark_wrapper_response" if task.is_external_benchmark else "model_response",
        "benchmark_id": task.id if task.is_external_benchmark else "",
        "task_id": task.id,
        "request_id": f"task_{task.id}",
        "timestamp": datetime.now(UTC).isoformat(),
        "target_model": response.model,
        "model": response.model,
        "raw_response": response.raw_response,
        "latency_seconds": response.latency_seconds,
        "time_to_first_token_seconds": response.time_to_first_token_seconds,
        "tokens_per_second": response.tokens_per_second,
        "output_token_count": response.output_token_count,
        "usage": response.usage,
        "error": response.error,
    }


def _graded_result_record(task: Task, grade: GradeResult) -> dict[str, Any]:
    details = grade.details if isinstance(grade.details, dict) else {}
    result = details.get("result") if isinstance(details.get("result"), dict) else {}
    return {
        "record_type": "graded_result",
        "benchmark_id": task.id if task.is_external_benchmark else "",
        "task_id": grade.task_id,
        "raw_score": result.get("raw_score"),
        "normalized_score": grade.score * 100.0,
        "pass": grade.passed,
        "grade": grade.to_dict(),
        "grader_metadata": {
            "status": grade.status,
            "kind": grade.kind,
            "json_valid": grade.json_valid,
        },
        "judge_metadata": result.get("judge_metadata", {}),
        "status": grade.status,
        "error": grade.error,
    }


def _prepare_output_paths(out: Path) -> tuple[Path, Path | None]:
    out = Path(out)
    if out.name == "latest":
        timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")
        return out.parent / timestamp, out
    return out, None


def _reset_output_dir(output_dir: Path) -> None:
    if output_dir.exists() or output_dir.is_symlink():
        if output_dir.is_dir() and not output_dir.is_symlink():
            shutil.rmtree(output_dir)
        else:
            output_dir.unlink()
    output_dir.mkdir(parents=True, exist_ok=False)


def _metadata(
    config: RunConfig,
    task_count: int,
    registry_count: int,
    output_dir: Path,
    run_duration_seconds: float,
    judge_base_url: str,
    judge_fallback_used: bool,
) -> dict[str, Any]:
    target_model = ModelMetadata(
        provider_type="openai-compatible" if config.provider != "mock" else "mock",
        base_url=redact_url(config.base_url or ""),
        model=config.model or ("mock-perfect" if config.provider == "mock" else ""),
        temperature=config.temperature,
        top_p=config.top_p,
        max_tokens=config.max_tokens,
        seed=config.seed,
        request_timeout_seconds=config.model_request_timeout,
        max_retries=config.max_retries,
        concurrency=config.request_concurrency,
        tool_parser=config.tool_parser,
        context_window=config.context_window,
        stop=list(config.stop or []),
    )
    judge = JudgeMetadata(
        provider=config.judge_provider,
        base_url=redact_url(judge_base_url),
        model=config.judge_model or (config.model if judge_fallback_used else "") or "",
        temperature=config.judge_temperature,
        timeout_seconds=config.judge_timeout,
        max_retries=config.judge_max_retries,
        prompt_version="agent-bench-json-judge-v1" if config.judge_provider != "none" else "",
        fallback_used=judge_fallback_used,
        fallback_policy=config.judge_fallback,
    )
    return {
        "run_id": output_dir.name,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "cli_command": _redacted_cli_command(config.cli_args),
        "host": {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
        },
        "provider": config.provider,
        "base_url": redact_url(config.base_url or ""),
        "model": config.model or ("mock-perfect" if config.provider == "mock" else ""),
        "target_model": target_model.to_dict(),
        "judge": judge.to_dict(),
        "task_count": task_count,
        "known_suite_count": registry_count,
        "excluded_suites": [],
        "excluded_suite_count": 0,
        "selected_profile": config.profile,
        "selected_suite_count": task_count,
        "output_dir": str(output_dir),
        "run_duration_seconds": run_duration_seconds,
        "request_concurrency": config.request_concurrency,
        "eval_concurrency": config.eval_concurrency,
        "external_timeout": config.external_timeout,
        "model_request_timeout": config.model_request_timeout,
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "seed": config.seed,
        "max_retries": config.max_retries,
        "tool_parser": config.tool_parser,
        "context_window": config.context_window,
        "stop": list(config.stop or []),
        "json_mode": config.json_mode,
        "allow_host_docker_socket": config.allow_host_docker_socket,
        "artifact_paths": {
            "raw_responses": str(output_dir / "raw_responses.jsonl"),
            "graded_results": str(output_dir / "graded_results.jsonl"),
            "results_csv": str(output_dir / "results.csv"),
            "summary_json": str(output_dir / "summary.json"),
            "summary_html": str(output_dir / "summary.html"),
        },
    }

def _task_order(tasks: list[Task], task_id: str) -> int:
    for index, task in enumerate(tasks):
        if task.id == task_id:
            return index
    return len(tasks)


def _git_commit() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError:
        return ""
    return completed.stdout.strip() if completed.returncode == 0 else ""


def _redacted_cli_command(cli_args: list[str] | None) -> list[str]:
    args = list(cli_args if cli_args is not None else sys.argv[1:])
    redacted: list[str] = []
    redact_next = False
    secret_flags = {"--api-key", "--judge-api-key", "--password", "--token"}
    for arg in args:
        if redact_next:
            redacted.append("<redacted>")
            redact_next = False
            continue
        if arg in secret_flags:
            redacted.append(arg)
            redact_next = True
            continue
        if any(arg.startswith(f"{flag}=") for flag in secret_flags):
            key, _, _value = arg.partition("=")
            redacted.append(f"{key}=<redacted>")
            continue
        redacted.append(arg)
    return redacted
