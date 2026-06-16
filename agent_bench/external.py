import asyncio
import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent_bench.models import Task


DEFAULT_EXTERNAL_IMAGE = "agent-bench-external:python3.12"


@dataclass(slots=True)
class ExternalBenchmarkResult:
    score: float
    passed: bool
    latency_seconds: float
    output: str = ""
    error: str | None = None
    timed_out: bool = False
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExternalBenchmarkConfig:
    provider: str
    base_url: str
    model: str
    api_key_env: str
    output_dir: Path
    timeout: float
    limit: int | None = None
    docker_bin: str = "docker"
    launcher_image: str = DEFAULT_EXTERNAL_IMAGE
    source_root: Path = Path(".")


class ExternalBenchmarkRunner:
    async def run(self, task: Task, config: ExternalBenchmarkConfig) -> ExternalBenchmarkResult:
        return await asyncio.to_thread(self._run_sync, task, config)

    def _run_sync(self, task: Task, config: ExternalBenchmarkConfig) -> ExternalBenchmarkResult:
        started = time.perf_counter()
        if shutil.which(config.docker_bin) is None:
            return ExternalBenchmarkResult(
                score=0.0,
                passed=False,
                latency_seconds=0.0,
                error="Docker is required for external benchmark evaluation but was not found",
            )

        task_output_dir = config.output_dir / "external" / task.id
        task_output_dir.mkdir(parents=True, exist_ok=True)

        benchmark = task.benchmark
        docker = benchmark["docker"]
        launcher_image = docker.get("image") or config.launcher_image
        env = _docker_env(task, benchmark, docker, config)
        image_error = _ensure_launcher_image(config, launcher_image)
        if image_error:
            return ExternalBenchmarkResult(
                score=0.0,
                passed=False,
                latency_seconds=time.perf_counter() - started,
                error=image_error,
            )

        command = [
            config.docker_bin,
            "run",
            "--rm",
            "--network",
            "host",
            "-v",
            "/var/run/docker.sock:/var/run/docker.sock",
            "-v",
            f"{task_output_dir.resolve()}:/outputs",
        ]
        for volume in docker.get("volumes", []):
            command.extend(["-v", volume])
        for key, value in env.items():
            command.extend(["-e", f"{key}={value}"])
        command.append(launcher_image)

        try:
            completed = subprocess.run(
                command,
                text=True,
                capture_output=True,
                timeout=config.timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return ExternalBenchmarkResult(
                score=0.0,
                passed=False,
                latency_seconds=time.perf_counter() - started,
                output=(exc.stdout or "") + (exc.stderr or ""),
                error=f"External benchmark timed out after {config.timeout:.1f}s",
                timed_out=True,
                details={"output_dir": str(task_output_dir)},
            )

        result_file = task_output_dir / "agent_bench_result.json"
        payload = _load_result_payload(result_file)
        output = (completed.stdout or "") + (completed.stderr or "")
        score = _coerce_score(payload.get("score"), completed.returncode == 0)
        error = payload.get("error") if isinstance(payload.get("error"), str) else None
        if completed.returncode != 0 and not error:
            error = f"External benchmark exited with code {completed.returncode}"
        details = {
            "benchmark": benchmark["name"],
            "homepage": benchmark["homepage"],
            "license": benchmark["license"],
            "credit": benchmark["credit"],
            "docker_image": launcher_image,
            "output_dir": str(task_output_dir),
            "result": payload,
        }
        return ExternalBenchmarkResult(
            score=score,
            passed=completed.returncode == 0 and score >= 1.0 and error is None,
            latency_seconds=time.perf_counter() - started,
            output=output[-20000:],
            error=error,
            details=details,
        )


def _ensure_launcher_image(config: ExternalBenchmarkConfig, image: str | None = None) -> str | None:
    image = image or config.launcher_image
    inspect = subprocess.run(
        [config.docker_bin, "image", "inspect", image],
        text=True,
        capture_output=True,
        check=False,
    )
    if inspect.returncode == 0:
        return None
    dockerfile = config.source_root / "docker" / "external-benchmark.Dockerfile"
    if not dockerfile.is_file():
        return f"External benchmark Dockerfile was not found at {dockerfile}"
    build = subprocess.run(
        [config.docker_bin, "build", "-f", str(dockerfile), "-t", image, str(config.source_root)],
        text=True,
        capture_output=True,
        check=False,
    )
    if build.returncode != 0:
        return (build.stderr or build.stdout or "Unable to build external benchmark launcher image").strip()
    return None


def _docker_env(task: Task, benchmark: dict[str, Any], docker: dict[str, Any], config: ExternalBenchmarkConfig) -> dict[str, str]:
    env = {
        "AGENT_BENCH_TASK_ID": task.id,
        "AGENT_BENCH_BENCHMARK_NAME": benchmark["name"],
        "AGENT_BENCH_REPOSITORY": benchmark.get("repository", benchmark["homepage"]),
        "AGENT_BENCH_REPOSITORY_REF": benchmark.get("ref", "main"),
        "AGENT_BENCH_SUBDIR": benchmark.get("subdir", ""),
        "AGENT_BENCH_DOCKER_IMAGE": docker.get("image", config.launcher_image),
        "AGENT_BENCH_SETUP": "\n".join(docker.get("setup", [])),
        "AGENT_BENCH_COMMAND": docker["command"],
        "AGENT_BENCH_PROVIDER": config.provider,
        "AGENT_BENCH_BASE_URL": config.base_url,
        "AGENT_BENCH_MODEL": config.model,
        "AGENT_BENCH_OUTPUT_DIR": "/outputs",
    }
    if config.api_key_env:
        env["AGENT_BENCH_API_KEY_ENV"] = config.api_key_env
        env["AGENT_BENCH_API_KEY"] = os.environ.get(config.api_key_env, "")
    if config.limit is not None:
        env["AGENT_BENCH_LIMIT"] = str(config.limit)
    for item in docker.get("environment", []):
        key, _, value = item.partition("=")
        env[key] = value
    return env


def _load_result_payload(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _coerce_score(value: Any, success: bool) -> float:
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))
    if success:
        return 1.0
    return 0.0
