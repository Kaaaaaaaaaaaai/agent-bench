import asyncio
import hashlib
import json
import os
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent_bench.models import Task
from agent_bench.statuses import FAILED_HARNESS_SETUP, TIMED_OUT


DEFAULT_EXTERNAL_IMAGE = "agent-bench-external:python3.12"
CONTAINER_OUTPUT_DIR = "/outputs"
DEFAULT_ASSET_ROOT = Path("/tmp/agent-bench-assets")
CONTAINER_ASSET_ROOT = "/asset-cache"
IMAGE_FINGERPRINT_LABEL = "agent-bench.external-fingerprint"
_IMAGE_BUILD_LOCK = threading.Lock()


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
    model_request_timeout: float = 1800.0
    max_tokens: int = 16384
    asset_root: Path = field(default_factory=lambda: Path(os.environ.get("AGENT_BENCH_EXTERNAL_ASSET_ROOT", DEFAULT_ASSET_ROOT)))
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
        config.asset_root.mkdir(parents=True, exist_ok=True)

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

        container_name = f"agent-bench-{task.id.lower()}-{uuid.uuid4().hex[:12]}"
        command = [
            config.docker_bin,
            "run",
            "--name",
            container_name,
            "--network",
            "host",
            "-v",
            "/var/run/docker.sock:/var/run/docker.sock",
            "-v",
            f"{config.asset_root}:{CONTAINER_ASSET_ROOT}",
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
            _remove_container(config, container_name)
            result_payload = _synthetic_result_payload(
                task,
                TIMED_OUT,
                f"External benchmark timed out after {config.timeout:.1f}s",
                score=0.0,
            )
            _write_result_payload(task_output_dir / "agent_bench_result.json", result_payload)
            return ExternalBenchmarkResult(
                score=0.0,
                passed=False,
                latency_seconds=time.perf_counter() - started,
                output=_timeout_output(exc),
                error=f"External benchmark timed out after {config.timeout:.1f}s",
                timed_out=True,
                details={
                    "benchmark": task.benchmark["name"],
                    "group": task.benchmark.get("group", task.category),
                    "homepage": task.benchmark["homepage"],
                    "license": task.benchmark["license"],
                    "credit": task.benchmark["credit"],
                    "citation": task.benchmark.get("citation", task.benchmark["homepage"]),
                    "output_dir": str(task_output_dir),
                    "result": result_payload,
                },
            )

        copy_error = _copy_container_outputs(config, container_name, task_output_dir)
        _remove_container(config, container_name)
        result_file = task_output_dir / "agent_bench_result.json"
        payload = _load_result_payload(result_file)
        output = (completed.stdout or "") + (completed.stderr or "")
        if copy_error:
            output = f"{output}{copy_error}\n"
        score = _coerce_score(payload.get("score"), completed.returncode == 0 and bool(payload))
        error = payload.get("error") if isinstance(payload.get("error"), str) else None
        if not payload and not error:
            error = "External benchmark did not produce agent_bench_result.json"
            payload = _synthetic_result_payload(task, FAILED_HARNESS_SETUP, error, score=0.0)
            _write_result_payload(result_file, payload)
        if copy_error and not error:
            error = copy_error
            if not payload:
                payload = _synthetic_result_payload(task, FAILED_HARNESS_SETUP, error, score=0.0)
                _write_result_payload(result_file, payload)
        if completed.returncode != 0 and not error:
            error = f"External benchmark exited with code {completed.returncode}"
            if not payload:
                payload = _synthetic_result_payload(task, FAILED_HARNESS_SETUP, error, score=0.0)
                _write_result_payload(result_file, payload)
        details = {
            "benchmark": benchmark["name"],
            "group": benchmark.get("group", task.category),
            "homepage": benchmark["homepage"],
            "license": benchmark["license"],
            "credit": benchmark["credit"],
            "citation": benchmark.get("citation", benchmark["homepage"]),
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
    if _image_is_current(config, image):
        return None
    with _IMAGE_BUILD_LOCK:
        if _image_is_current(config, image):
            return None
        return _build_launcher_image(config, image)


def _image_is_current(config: ExternalBenchmarkConfig, image: str) -> bool:
    inspect = subprocess.run(
        [config.docker_bin, "image", "inspect", image],
        text=True,
        capture_output=True,
        check=False,
    )
    if inspect.returncode != 0:
        return False
    if image != config.launcher_image:
        return True
    try:
        image_config = json.loads(inspect.stdout)[0]
        labels = image_config.get("Config", {}).get("Labels", {})
    except (IndexError, json.JSONDecodeError, TypeError):
        return False
    if not isinstance(labels, dict):
        return False
    return labels.get(IMAGE_FINGERPRINT_LABEL) == _launcher_image_fingerprint(config)


def _build_launcher_image(config: ExternalBenchmarkConfig, image: str) -> str | None:
    dockerfile = config.source_root / "docker" / "external-benchmark.Dockerfile"
    if not dockerfile.is_file():
        return f"External benchmark Dockerfile was not found at {dockerfile}"
    build = subprocess.run(
        [
            config.docker_bin,
            "build",
            "-f",
            str(dockerfile),
            "-t",
            image,
            "--label",
            f"{IMAGE_FINGERPRINT_LABEL}={_launcher_image_fingerprint(config)}",
            str(config.source_root),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if build.returncode != 0:
        return (build.stderr or build.stdout or "Unable to build external benchmark launcher image").strip()
    return None


def _launcher_image_fingerprint(config: ExternalBenchmarkConfig) -> str:
    digest = hashlib.sha256()
    for relative in (
        "docker/external-benchmark.Dockerfile",
        "docker/external_launcher.sh",
        "docker/benchmark_probe.py",
    ):
        path = config.source_root / relative
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        if path.is_file():
            digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _copy_container_outputs(config: ExternalBenchmarkConfig, container_name: str, output_dir: Path) -> str | None:
    copy = subprocess.run(
        [config.docker_bin, "cp", f"{container_name}:{CONTAINER_OUTPUT_DIR}/.", str(output_dir)],
        text=True,
        capture_output=True,
        check=False,
    )
    if copy.returncode == 0:
        return None
    return (copy.stderr or copy.stdout or "Unable to copy external benchmark outputs").strip()


def _timeout_output(exc: subprocess.TimeoutExpired) -> str:
    return f"{_to_text(exc.stdout)}{_to_text(exc.stderr)}"


def _to_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _remove_container(config: ExternalBenchmarkConfig, container_name: str) -> None:
    subprocess.run(
        [config.docker_bin, "rm", "-f", container_name],
        text=True,
        capture_output=True,
        check=False,
    )


def _docker_env(task: Task, benchmark: dict[str, Any], docker: dict[str, Any], config: ExternalBenchmarkConfig) -> dict[str, str]:
    env = {
        "AGENT_BENCH_TASK_ID": task.id,
        "AGENT_BENCH_BENCHMARK_NAME": benchmark["name"],
        "AGENT_BENCH_BENCHMARK_GROUP": benchmark.get("group", task.category),
        "AGENT_BENCH_REPOSITORY": benchmark.get("repository", benchmark["homepage"]),
        "AGENT_BENCH_REPOSITORY_REF": benchmark.get("ref", "main"),
        "AGENT_BENCH_SUBDIR": benchmark.get("subdir", ""),
        "AGENT_BENCH_DATASET_ID": benchmark.get("dataset_id", ""),
        "AGENT_BENCH_BENCHMARK_HOMEPAGE": benchmark["homepage"],
        "AGENT_BENCH_BENCHMARK_LICENSE": benchmark["license"],
        "AGENT_BENCH_BENCHMARK_CREDIT": benchmark["credit"],
        "AGENT_BENCH_BENCHMARK_CITATION": benchmark.get("citation", benchmark["homepage"]),
        "AGENT_BENCH_REQUIRED_CAPABILITIES": ",".join(_benchmark_capabilities(benchmark)),
        "AGENT_BENCH_DOCKER_IMAGE": docker.get("image", config.launcher_image),
        "AGENT_BENCH_SETUP": "\n".join(docker.get("setup", [])),
        "AGENT_BENCH_COMMAND": docker["command"],
        "AGENT_BENCH_PROVIDER": config.provider,
        "AGENT_BENCH_BASE_URL": config.base_url,
        "AGENT_BENCH_MODEL": config.model,
        "AGENT_BENCH_MODEL_REQUEST_TIMEOUT": str(config.model_request_timeout),
        "AGENT_BENCH_MAX_TOKENS": str(config.max_tokens),
        "AGENT_BENCH_OUTPUT_DIR": CONTAINER_OUTPUT_DIR,
        "AGENT_BENCH_ASSET_ROOT": CONTAINER_ASSET_ROOT,
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


def _benchmark_capabilities(benchmark: dict[str, Any]) -> list[str]:
    capabilities = benchmark.get("capabilities", [])
    if not isinstance(capabilities, list):
        return []
    return [item.strip() for item in capabilities if isinstance(item, str) and item.strip()]


def _load_result_payload(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_result_payload(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _synthetic_result_payload(task: Task, status: str, error: str, score: float = 0.0) -> dict[str, Any]:
    benchmark = task.benchmark
    required_capabilities = _benchmark_capabilities(benchmark)
    capability_contract = _synthetic_capability_contract(required_capabilities)
    return {
        "benchmark": benchmark.get("name", task.id),
        "group": benchmark.get("group", task.category),
        "status": status,
        "score": score,
        "raw_score": score,
        "valid_score": 0.0,
        "error": error,
        "repository": benchmark.get("repository", benchmark.get("homepage", "")),
        "repository_ref": benchmark.get("ref", ""),
        "required_capabilities": required_capabilities,
        "supported_capabilities": [
            capability
            for capability, support in capability_contract.items()
            if support.get("supported") is True
        ],
        "capability_contract": capability_contract,
        "unsupported_capabilities": [],
        "capabilities_verified": False,
        "extracted_task_count": 0,
        "evaluated_task_count": 0,
        "valid_evaluated_task_count": 0,
        "evaluation_passed_count": 0,
        "skipped_task_count": 0,
        "judge_parse_failure_count": 0,
        "judge_parse_repaired_count": 0,
        "model_evals": [],
        "model_eval": {},
        "status_counts": {status: 1},
    }


def _synthetic_capability_contract(required_capabilities: list[str]) -> dict[str, dict[str, Any]]:
    supported = {"chat_answer", "external_data_required"}
    reasons = {
        "browser_or_gui": "No real desktop/browser environment adapter is implemented",
        "tool_call": "No benchmark-native stateful tool adapter is implemented",
        "repo_patch": "External benchmark did not complete before repo_patch workspace/grader verification",
        "file_artifact": "External benchmark did not complete before file_artifact asset/output verification",
    }
    contract: dict[str, dict[str, Any]] = {}
    for capability in required_capabilities:
        is_supported = capability in supported
        contract[capability] = {
            "capability": capability,
            "workspace": is_supported,
            "tools": is_supported,
            "output_collection": is_supported,
            "grader": is_supported,
            "native": is_supported,
            "supported": is_supported,
            "reason": "" if is_supported else reasons.get(capability, "External benchmark did not complete"),
        }
    return contract


def _coerce_score(value: Any, success: bool) -> float:
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))
    if success:
        return 1.0
    return 0.0
