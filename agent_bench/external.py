import asyncio
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent_bench.manifest import BenchmarkManifest, ValidationResult, manifest_from_task
from agent_bench.models import Task
from agent_bench.statuses import (
    FAILED_CONTAINER_BUILD,
    FAILED_CONTAINER_RUNTIME,
    FAILED_MANIFEST_VALIDATION,
    FAILED_TIMEOUT,
    FAILED_HARNESS_SETUP,
    TIMED_OUT,
)


DEFAULT_EXTERNAL_IMAGE = "agent-bench-external:python3.12"
CONTAINER_OUTPUT_DIR = "/outputs"
DEFAULT_ASSET_ROOT = Path("agent-bench-assets")
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
    temperature: float = 0.0
    top_p: float | None = None
    seed: int | None = None
    stop: list[str] = field(default_factory=list)
    tool_parser: str = "auto"
    context_window: int | None = None
    judge_base_url: str = ""
    judge_model: str = ""
    judge_temperature: float = 0.0
    judge_timeout: float = 1800.0
    judge_max_retries: int = 2
    judge_fallback_used: bool = False
    asset_root: Path = field(default_factory=lambda: Path(os.environ.get("AGENT_BENCH_EXTERNAL_ASSET_ROOT", DEFAULT_ASSET_ROOT)))
    docker_bin: str = "docker"
    launcher_image: str = DEFAULT_EXTERNAL_IMAGE
    source_root: Path = Path(".")
    allow_host_docker_socket: bool = False
    pass_api_key_to_container: bool = False


class ExternalBenchmarkRunner:
    async def run(self, task: Task, config: ExternalBenchmarkConfig) -> ExternalBenchmarkResult:
        return await asyncio.to_thread(self._run_sync, task, config)

    def _run_sync(self, task: Task, config: ExternalBenchmarkConfig) -> ExternalBenchmarkResult:
        started = time.perf_counter()
        manifest = manifest_from_task(task)
        validation = manifest.validate(allow_host_docker_socket=config.allow_host_docker_socket)
        task_output_dir = config.output_dir / "external" / task.id
        task_output_dir.mkdir(parents=True, exist_ok=True)
        config.asset_root.mkdir(parents=True, exist_ok=True)
        if not validation.ok:
            result_payload = _manifest_validation_payload(task, manifest, validation)
            _write_result_payload(task_output_dir / "agent_bench_result.json", result_payload)
            return ExternalBenchmarkResult(
                score=0.0,
                passed=False,
                latency_seconds=time.perf_counter() - started,
                error=validation.error_message,
                details=_benchmark_details(task, manifest, task_output_dir, result_payload),
            )

        if shutil.which(config.docker_bin) is None:
            result_payload = _synthetic_result_payload(
                task,
                FAILED_CONTAINER_RUNTIME,
                "Docker is required for external benchmark evaluation but was not found",
                score=0.0,
            )
            _write_result_payload(task_output_dir / "agent_bench_result.json", result_payload)
            return ExternalBenchmarkResult(
                score=0.0,
                passed=False,
                latency_seconds=time.perf_counter() - started,
                error="Docker is required for external benchmark evaluation but was not found",
                details=_benchmark_details(task, manifest, task_output_dir, result_payload),
            )

        benchmark = task.benchmark
        docker = benchmark.get("docker", {}) if isinstance(benchmark.get("docker"), dict) else {}
        launcher_image = manifest.container.image or docker.get("image") or config.launcher_image
        env = _docker_env(task, benchmark, docker, config, manifest)
        image_error = _ensure_launcher_image(config, launcher_image)
        if image_error:
            result_payload = _synthetic_result_payload(task, FAILED_CONTAINER_BUILD, image_error, score=0.0)
            _write_result_payload(task_output_dir / "agent_bench_result.json", result_payload)
            return ExternalBenchmarkResult(
                score=0.0,
                passed=False,
                latency_seconds=time.perf_counter() - started,
                error=image_error,
                details=_benchmark_details(task, manifest, task_output_dir, result_payload),
            )
        asset_cache_error = _prepare_benchmark_asset_cache(task, config, launcher_image=launcher_image)
        if asset_cache_error:
            env["AGENT_BENCH_ASSET_CACHE_WARNING"] = asset_cache_error

        container_name = f"agent-bench-{task.id.lower()}-{uuid.uuid4().hex[:12]}"
        command = _docker_run_command(config, manifest, container_name, launcher_image, env)
        setup_details = _external_setup_details(
            task=task,
            manifest=manifest,
            config=config,
            container_name=container_name,
            launcher_image=launcher_image,
            task_output_dir=task_output_dir,
            env=env,
            asset_cache_error=asset_cache_error,
        )

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
                FAILED_TIMEOUT,
                f"External benchmark timed out after {config.timeout:.1f}s",
                score=0.0,
            )
            _attach_external_setup_details(result_payload, setup_details)
            _write_result_payload(task_output_dir / "agent_bench_result.json", result_payload)
            return ExternalBenchmarkResult(
                score=0.0,
                passed=False,
                latency_seconds=time.perf_counter() - started,
                output=_timeout_output(exc),
                error=f"External benchmark timed out after {config.timeout:.1f}s",
                timed_out=True,
                details=_benchmark_details(task, manifest, task_output_dir, result_payload),
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
        _attach_external_setup_details(payload, setup_details)
        _write_result_payload(result_file, payload)
        details = {
            "suite_id": task.id,
            "benchmark": benchmark.get("name", manifest.display_name),
            "group": benchmark.get("group", task.category),
            "required_capabilities": benchmark.get("capabilities", []),
            "homepage": benchmark.get("homepage", manifest.homepage_url),
            "license": benchmark.get("license", manifest.license),
            "credit": benchmark.get("credit", manifest.credit),
            "citation": benchmark.get("citation", manifest.citation or manifest.homepage_url),
            "official_leaderboard_url": benchmark.get("official_leaderboard_url", manifest.official_leaderboard_url),
            "docker_image": launcher_image,
            "container_name": container_name,
            "network_mode": setup_details["network_mode"],
            "docker_socket_mount": setup_details["docker_socket_mount"],
            "output_dir": str(task_output_dir),
            "output_mount": setup_details["output_mount"],
            "asset_cache_mount": setup_details["asset_cache_mount"],
            "benchmark_checkout_path": setup_details["benchmark_checkout_path"],
            "manifest": manifest.to_dict(),
            "container_command": manifest.container.command,
            "requires_host_docker_socket": manifest.container.requires_host_docker_socket,
            "setup_details": setup_details,
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


def _manifest_validation_payload(
    task: Task,
    manifest: BenchmarkManifest,
    validation: ValidationResult,
) -> dict[str, Any]:
    return {
        "benchmark": manifest.display_name,
        "group": manifest.task_group,
        "status": FAILED_MANIFEST_VALIDATION,
        "score": 0.0,
        "raw_score": None,
        "valid_score": None,
        "error": validation.error_message,
        "included_in_official_score": False,
        "validation": validation.to_dict(),
        "manifest": manifest.to_dict(),
        "required_capabilities": _benchmark_capabilities(task.benchmark),
        "supported_capabilities": [],
        "required_tools": _benchmark_required_tools(task.benchmark),
        "exposed_tools": [],
        "missing_tools": _benchmark_required_tools(task.benchmark),
        "unsupported_capabilities": [],
        "capabilities_verified": False,
        "extracted_task_count": 0,
        "evaluated_task_count": 0,
        "valid_evaluated_task_count": 0,
        "evaluation_passed_count": 0,
        "skipped_task_count": 0,
        "model_evals": [],
        "model_eval": {},
        "status_counts": {FAILED_MANIFEST_VALIDATION: 1},
    }


def _benchmark_details(
    task: Task,
    manifest: BenchmarkManifest,
    output_dir: Path,
    result_payload: dict[str, Any],
) -> dict[str, Any]:
    benchmark = task.benchmark if isinstance(task.benchmark, dict) else {}
    return {
        "suite_id": task.id,
        "benchmark": benchmark.get("name", manifest.display_name),
        "group": benchmark.get("group", manifest.task_group),
        "required_capabilities": benchmark.get("capabilities", manifest.capabilities),
        "homepage": benchmark.get("homepage", manifest.homepage_url),
        "license": benchmark.get("license", manifest.license),
        "credit": benchmark.get("credit", manifest.credit),
        "citation": benchmark.get("citation", manifest.citation or manifest.homepage_url),
        "official_leaderboard_url": benchmark.get("official_leaderboard_url", manifest.official_leaderboard_url),
        "output_dir": str(output_dir),
        "manifest": manifest.to_dict(),
        "result": result_payload,
    }


def _ensure_launcher_image(config: ExternalBenchmarkConfig, image: str | None = None) -> str | None:
    image = image or config.launcher_image
    if _image_is_current(config, image):
        return None
    with _IMAGE_BUILD_LOCK:
        if _image_is_current(config, image):
            return None
        if image != config.launcher_image:
            return _pull_image(config, image)
        return _build_launcher_image(config, image)


def _pull_image(config: ExternalBenchmarkConfig, image: str) -> str | None:
    pull = subprocess.run(
        [config.docker_bin, "pull", image],
        text=True,
        capture_output=True,
        check=False,
    )
    if pull.returncode != 0:
        return (pull.stderr or pull.stdout or f"Unable to pull benchmark image {image}").strip()
    return None


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


def _docker_run_command(
    config: ExternalBenchmarkConfig,
    manifest: BenchmarkManifest,
    container_name: str,
    launcher_image: str,
    env: dict[str, str],
) -> list[str]:
    command = [
        config.docker_bin,
        "run",
        "--name",
        container_name,
        "--add-host",
        "host.docker.internal:host-gateway",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--pids-limit",
        str(manifest.container.pids_limit or 1024),
        "--tmpfs",
        "/tmp:rw,nosuid,nodev,noexec,size=512m,uid=10001,gid=10001",
        "--tmpfs",
        "/workspace:rw,nosuid,nodev,size=4g,uid=10001,gid=10001",
    ]
    if manifest.container.run_as_user:
        command.extend(["--user", manifest.container.run_as_user])
    if manifest.container.memory:
        command.extend(["--memory", manifest.container.memory])
    if manifest.container.cpus is not None:
        command.extend(["--cpus", str(manifest.container.cpus)])
    command.extend(["--network", _docker_network_mode(manifest)])
    command.extend(["--mount", f"type=bind,src={config.asset_root.resolve()},dst={CONTAINER_ASSET_ROOT},readonly"])
    if _docker_socket_mount_enabled(config, manifest):
        command.extend(["-v", "/var/run/docker.sock:/var/run/docker.sock"])
    for key, value in sorted(env.items()):
        command.extend(["-e", f"{key}={value}"])
    command.append(launcher_image)
    return command


def _docker_network_mode(manifest: BenchmarkManifest) -> str:
    if manifest.container.network == "none":
        return "none"
    if manifest.container.network == "host":
        return "host"
    return "bridge"


def _docker_socket_mount_enabled(config: ExternalBenchmarkConfig, manifest: BenchmarkManifest) -> bool:
    return bool(manifest.container.requires_host_docker_socket and config.allow_host_docker_socket)


def _docker_env(
    task: Task,
    benchmark: dict[str, Any],
    docker: dict[str, Any],
    config: ExternalBenchmarkConfig,
    manifest: BenchmarkManifest,
) -> dict[str, str]:
    env = {
        "AGENT_BENCH_TASK_ID": task.id,
        "AGENT_BENCH_BENCHMARK_ID": task.id,
        "AGENT_BENCH_BENCHMARK_NAME": benchmark.get("name", manifest.display_name),
        "AGENT_BENCH_BENCHMARK_GROUP": benchmark.get("group", task.category),
        "AGENT_BENCH_REPOSITORY": benchmark.get("repository", manifest.source.repository_url or benchmark.get("homepage", "")),
        "AGENT_BENCH_REPOSITORY_REF": benchmark.get("ref", manifest.source.commit or manifest.source.dataset_revision),
        "AGENT_BENCH_SUBDIR": benchmark.get("subdir", ""),
        "AGENT_BENCH_DATASET_ID": benchmark.get("dataset_id", manifest.source.dataset_id),
        "AGENT_BENCH_BENCHMARK_HOMEPAGE": benchmark.get("homepage", manifest.homepage_url),
        "AGENT_BENCH_BENCHMARK_LICENSE": benchmark.get("license", manifest.license),
        "AGENT_BENCH_BENCHMARK_CREDIT": benchmark.get("credit", manifest.credit),
        "AGENT_BENCH_BENCHMARK_CITATION": benchmark.get("citation", manifest.citation or manifest.homepage_url),
        "AGENT_BENCH_REQUIRED_CAPABILITIES": ",".join(_benchmark_capabilities(benchmark)),
        "AGENT_BENCH_DOCKER_IMAGE": docker.get("image", config.launcher_image),
        "AGENT_BENCH_SETUP": "\n".join(docker.get("setup", [])),
        "AGENT_BENCH_COMMAND": manifest.container.command or docker.get("command", ""),
        "AGENT_BENCH_PROVIDER": config.provider,
        "AGENT_BENCH_BASE_URL": config.base_url,
        "AGENT_BENCH_MODEL": config.model,
        "AGENT_BENCH_MODEL_REQUEST_TIMEOUT": str(config.model_request_timeout),
        "AGENT_BENCH_MAX_TOKENS": str(config.max_tokens),
        "AGENT_BENCH_TEMPERATURE": str(config.temperature),
        "AGENT_BENCH_TOOL_PARSER": config.tool_parser,
        "AGENT_BENCH_OUTPUT_DIR": CONTAINER_OUTPUT_DIR,
        "AGENT_BENCH_ASSET_ROOT": CONTAINER_ASSET_ROOT,
        "AGENT_BENCH_ASSET_CACHE_KEY": _asset_cache_key(str(benchmark.get("name") or manifest.display_name)),
        "AGENT_BENCH_MANIFEST_JSON": json.dumps(manifest.to_dict(), sort_keys=True),
        "AGENT_BENCH_BENCHMARK_JSON": json.dumps(benchmark, sort_keys=True),
        "AGENT_BENCH_JUDGE_BASE_URL": config.judge_base_url,
        "AGENT_BENCH_JUDGE_MODEL": config.judge_model,
        "AGENT_BENCH_JUDGE_TEMPERATURE": str(config.judge_temperature),
        "AGENT_BENCH_JUDGE_TIMEOUT": str(config.judge_timeout),
        "AGENT_BENCH_JUDGE_MAX_RETRIES": str(config.judge_max_retries),
        "AGENT_BENCH_JUDGE_FALLBACK_USED": "1" if config.judge_fallback_used else "0",
    }
    if config.top_p is not None:
        env["AGENT_BENCH_TOP_P"] = str(config.top_p)
    if config.seed is not None:
        env["AGENT_BENCH_SEED"] = str(config.seed)
    if config.context_window is not None:
        env["AGENT_BENCH_CONTEXT_LIMIT"] = str(config.context_window)
    if config.stop:
        env["AGENT_BENCH_STOP"] = json.dumps(config.stop)
    if "repo_patch" in _benchmark_capabilities(benchmark):
        env.setdefault("AGENT_BENCH_ALLOW_TARGET_CHECKOUT", "1")
    if config.pass_api_key_to_container and config.api_key_env:
        env["AGENT_BENCH_API_KEY_ENV"] = config.api_key_env
        env["AGENT_BENCH_API_KEY"] = os.environ.get(config.api_key_env, "")
    if config.limit is not None:
        env["AGENT_BENCH_LIMIT"] = str(config.limit)
    sample_limit = os.environ.get("AGENT_BENCH_SAMPLE_LIMIT", "").strip()
    if sample_limit:
        env["AGENT_BENCH_SAMPLE_LIMIT"] = sample_limit
    for item in docker.get("environment", []):
        key, _, value = item.partition("=")
        env[key] = value
    return env


def _external_setup_details(
    *,
    task: Task,
    manifest: BenchmarkManifest,
    config: ExternalBenchmarkConfig,
    container_name: str,
    launcher_image: str,
    task_output_dir: Path,
    env: dict[str, str],
    asset_cache_error: str | None,
) -> dict[str, Any]:
    asset_cache_key = env.get("AGENT_BENCH_ASSET_CACHE_KEY", "")
    cache_dir = config.asset_root / asset_cache_key if asset_cache_key else config.asset_root
    copied_asset_paths = _relative_file_sample(cache_dir) if cache_dir.exists() else []
    required_asset_paths = [
        asset.expected_local_path
        for asset in manifest.assets
        if asset.required and asset.expected_local_path
    ]
    validation = _asset_validation_details(cache_dir, required_asset_paths, asset_cache_error)
    checkout_path = "/workspace/repo"
    subdir = manifest.source.subdir or env.get("AGENT_BENCH_SUBDIR", "")
    if subdir:
        checkout_path = f"{checkout_path}/{subdir.strip('/')}"
    return {
        "container_name": container_name,
        "image": launcher_image,
        "network_mode": _docker_network_mode(manifest),
        "docker_socket_mount": {
            "enabled": _docker_socket_mount_enabled(config, manifest),
            "host_path": "/var/run/docker.sock" if _docker_socket_mount_enabled(config, manifest) else "",
            "container_path": "/var/run/docker.sock" if _docker_socket_mount_enabled(config, manifest) else "",
            "descriptor_requires_host_docker_socket": manifest.container.requires_host_docker_socket,
            "global_allow_host_docker_socket": config.allow_host_docker_socket,
        },
        "output_mount": {
            "host_path": str(task_output_dir),
            "container_path": CONTAINER_OUTPUT_DIR,
            "mode": "docker_cp",
        },
        "asset_cache_mount": {
            "host_path": str(config.asset_root.resolve()),
            "container_path": CONTAINER_ASSET_ROOT,
            "readonly": True,
            "cache_key": asset_cache_key,
            "cache_path": str(cache_dir),
        },
        "benchmark_checkout_path": checkout_path,
        "required_asset_paths": required_asset_paths,
        "copied_asset_paths": copied_asset_paths,
        "missing_assets_count": int(validation.get("missing_count", 0)),
        "validation_result": validation,
        "cache_recipe": _asset_cache_recipe(task.benchmark) or {},
    }


def _attach_external_setup_details(payload: dict[str, Any], setup_details: dict[str, Any]) -> None:
    payload["container_name"] = setup_details["container_name"]
    payload["docker_image"] = setup_details["image"]
    payload["network_mode"] = setup_details["network_mode"]
    payload["docker_socket_mount"] = setup_details["docker_socket_mount"]
    payload["output_mount"] = setup_details["output_mount"]
    payload["asset_cache_mount"] = setup_details["asset_cache_mount"]
    payload["benchmark_checkout_path"] = setup_details["benchmark_checkout_path"]
    payload["required_asset_paths"] = setup_details["required_asset_paths"]
    payload["copied_asset_paths"] = setup_details["copied_asset_paths"]
    payload["missing_assets_count"] = setup_details["missing_assets_count"]
    existing = payload.get("setup_details")
    if isinstance(existing, dict):
        merged = dict(existing)
        merged["external_harness"] = setup_details
        payload["setup_details"] = merged
    else:
        payload["setup_details"] = {"external_harness": setup_details}
    if payload.get("capabilities_verified") is False:
        payload["included_in_official_score"] = False


def _relative_file_sample(root: Path, *, limit: int = 100) -> list[str]:
    paths: list[str] = []
    if not root.exists():
        return paths
    for path in sorted(root.rglob("*")):
        if len(paths) >= limit:
            break
        if path.is_file() and path.name != ".agent-bench-assets-ready.json":
            paths.append(str(path.relative_to(root)))
    return paths


def _asset_validation_details(
    cache_dir: Path,
    required_asset_paths: list[str],
    asset_cache_error: str | None,
) -> dict[str, Any]:
    missing = []
    for relative in required_asset_paths:
        if relative in {".", ""}:
            continue
        if not (cache_dir / relative).exists():
            missing.append(relative)
    sentinel = cache_dir / ".agent-bench-assets-ready.json"
    return {
        "ok": not asset_cache_error and not missing,
        "warning": asset_cache_error or "",
        "cache_dir_exists": cache_dir.exists(),
        "ready_sentinel": sentinel.is_file(),
        "missing_required_asset_paths": missing,
        "missing_count": len(missing),
        "materialized_file_count": len(_relative_file_sample(cache_dir, limit=10_000)) if cache_dir.exists() else 0,
    }


def _prepare_benchmark_asset_cache(
    task: Task,
    config: ExternalBenchmarkConfig,
    *,
    launcher_image: str | None = None,
) -> str | None:
    recipe = _asset_cache_recipe(task.benchmark)
    if recipe is None:
        return None
    cache_dir = config.asset_root / recipe["key"]
    sentinel = cache_dir / ".agent-bench-assets-ready.json"
    if sentinel.is_file() and _cache_has_materialized_files(cache_dir):
        return None
    git_bin = shutil.which("git")
    git_lfs_bin = shutil.which("git-lfs")
    requires_lfs = bool(recipe.get("requires_git_lfs", True))
    if (git_bin is None or (requires_lfs and git_lfs_bin is None)) and launcher_image:
        return _prepare_benchmark_asset_cache_with_docker(recipe, config, launcher_image)
    if git_bin is None:
        return "asset cache download skipped: git was not found and Docker fallback was unavailable"
    if requires_lfs and git_lfs_bin is None:
        return "asset cache download skipped: git-lfs was not found and Docker fallback was unavailable"

    download_root = config.asset_root / "_downloads" / recipe["key"]
    clone_dir = download_root / "repo"
    if clone_dir.exists():
        shutil.rmtree(clone_dir)
    download_root.mkdir(parents=True, exist_ok=True)
    repository = str(recipe["repository"])
    ref = str(recipe.get("ref") or "main")
    env = os.environ.copy()
    env["GIT_LFS_SKIP_SMUDGE"] = "1"
    clone_error = _clone_repository_at_ref(git_bin, repository, ref, clone_dir, env=env)
    if clone_error is not None:
        return f"asset cache download skipped: {clone_error}"
    if requires_lfs:
        _run_asset_command([git_bin, "-C", str(clone_dir), "lfs", "install", "--local"], env=env)
        lfs_error = _run_asset_command(
            [
                git_bin,
                "-C",
                str(clone_dir),
                "lfs",
                "pull",
                "--include",
                ",".join(str(item) for item in recipe["includes"]),
                "--exclude",
                "",
            ],
            env=env,
        )
        if lfs_error is not None:
            return f"asset cache download skipped: {lfs_error}"

    copied = 0
    for subpath in recipe["subpaths"]:
        source = clone_dir / str(subpath)
        if not source.exists():
            continue
        target = cache_dir / str(subpath)
        if target.exists():
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, target)
            copied += sum(1 for path in target.rglob("*") if path.is_file())
        else:
            shutil.copy2(source, target)
            copied += 1
    if copied == 0:
        return "asset cache download skipped: no requested asset files were materialized"
    sentinel.write_text(
        json.dumps(
            {
                "benchmark": task.benchmark["name"],
                "repository": repository,
                "ref": ref,
                "copied_file_count": copied,
                "subpaths": list(recipe["subpaths"]),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return None


def _clone_repository_at_ref(
    git_bin: str,
    repository: str,
    ref: str,
    clone_dir: Path,
    *,
    env: dict[str, str],
) -> str | None:
    if clone_dir.exists():
        shutil.rmtree(clone_dir)
    clone_error = _run_asset_command(
        [git_bin, "clone", "--depth", "1", "--branch", ref, repository, str(clone_dir)],
        env=env,
    )
    if clone_error is None:
        return None
    if clone_dir.exists():
        shutil.rmtree(clone_dir)
    clone_error = _run_asset_command([git_bin, "clone", "--depth", "1", repository, str(clone_dir)], env=env)
    if clone_error is not None:
        return clone_error
    fetch_error = _run_asset_command([git_bin, "-C", str(clone_dir), "fetch", "--depth", "1", "origin", ref], env=env)
    if fetch_error is not None:
        return fetch_error
    return _run_asset_command([git_bin, "-C", str(clone_dir), "checkout", "--detach", ref], env=env)


def _prepare_benchmark_asset_cache_with_docker(
    recipe: dict[str, Any],
    config: ExternalBenchmarkConfig,
    launcher_image: str,
) -> str | None:
    if shutil.which(config.docker_bin) is None:
        return "asset cache download skipped: git-lfs was not found and Docker was unavailable"
    key = str(recipe["key"])
    cache_dir = config.asset_root / key
    cache_dir.mkdir(parents=True, exist_ok=True)
    script = _docker_asset_download_script(recipe)
    completed = subprocess.run(
        [
            config.docker_bin,
            "run",
            "--rm",
            "--network",
            "host",
            "-v",
            f"{config.asset_root.resolve()}:{CONTAINER_ASSET_ROOT}",
            "--entrypoint",
            "bash",
            launcher_image,
            "-lc",
            script,
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        output = (completed.stderr or completed.stdout or "").strip()
        return f"asset cache download skipped: {(output[-1000:] or 'Docker downloader failed')}"
    if _cache_has_materialized_files(cache_dir):
        return None
    return "asset cache download skipped: Docker downloader did not materialize requested files"


def _docker_asset_download_script(recipe: dict[str, Any]) -> str:
    key = shlex.quote(str(recipe["key"]))
    repository = shlex.quote(str(recipe["repository"]))
    ref = shlex.quote(str(recipe.get("ref") or "main"))
    includes = shlex.quote(",".join(str(item) for item in recipe["includes"]))
    requires_lfs = "1" if recipe.get("requires_git_lfs", True) else "0"
    copy_commands = []
    for subpath in recipe["subpaths"]:
        quoted = shlex.quote(str(subpath))
        copy_commands.append(
            f'if [[ -e "$tmp/{quoted}" ]]; then '
            f'mkdir -p "$cache/$(dirname {quoted})"; '
            f'rm -rf "$cache/{quoted}"; '
            f'cp -a "$tmp/{quoted}" "$cache/{quoted}"; '
            "fi"
        )
    copy_script = "\n".join(copy_commands)
    return "\n".join(
        [
            "set -euo pipefail",
            f"key={key}",
            f"repository={repository}",
            f"ref={ref}",
            f"includes={includes}",
            f"requires_lfs={requires_lfs}",
            f'cache="{CONTAINER_ASSET_ROOT}/$key"',
            f'tmp="{CONTAINER_ASSET_ROOT}/_downloads/$key/repo"',
            'rm -rf "$tmp"',
            'mkdir -p "$(dirname "$tmp")" "$cache"',
            "export GIT_LFS_SKIP_SMUDGE=1",
            'git clone --depth 1 --branch "$ref" "$repository" "$tmp" || { rm -rf "$tmp"; git clone --depth 1 "$repository" "$tmp"; git -C "$tmp" fetch --depth 1 origin "$ref"; git -C "$tmp" checkout --detach "$ref"; }',
            'if [[ "$requires_lfs" == "1" ]]; then git -C "$tmp" lfs install --local || true; git -C "$tmp" lfs pull --include "$includes" --exclude ""; fi',
            copy_script,
            """printf '{"downloaded":true}\\n' > "$cache/.agent-bench-assets-ready.json" """,
        ]
    )


def _asset_cache_recipe(benchmark: dict[str, Any]) -> dict[str, Any] | None:
    key = _asset_cache_key(str(benchmark.get("name") or ""))
    if key == "gdpval":
        return {
            "key": key,
            "repository": benchmark.get("repository") or "https://huggingface.co/datasets/openai/gdpval",
            "ref": benchmark.get("ref") or "main",
            "includes": ("reference_files/**", "deliverable_files/**"),
            "subpaths": ("reference_files", "deliverable_files"),
        }
    if key == "paperbench":
        return {
            "key": key,
            "repository": benchmark.get("repository") or "https://github.com/openai/frontier-evals.git",
            "ref": benchmark.get("ref") or "main",
            "includes": ("project/paperbench/data/papers/**",),
            "subpaths": ("project/paperbench/data/papers",),
        }
    if key == "exploitbench":
        return {
            "key": key,
            "repository": benchmark.get("repository") or "https://github.com/exploitbench/exploitbench.git",
            "ref": benchmark.get("ref") or "main",
            "requires_git_lfs": False,
            "includes": (),
            "subpaths": (
                "benchmarks",
                "data",
                "docs",
                "exploitbench",
                "scripts",
                "testenvs",
                "pyproject.toml",
                "README.md",
                "Makefile",
            ),
        }
    return None


def _asset_cache_key(value: str) -> str:
    return _safe_slug(value).lower()


def _safe_slug(value: str) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in value.strip())
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-") or "benchmark"


def _cache_has_materialized_files(path: Path) -> bool:
    try:
        return any(child.is_file() for child in path.rglob("*") if child.name != ".agent-bench-assets-ready.json")
    except OSError:
        return False


def _run_asset_command(command: list[str], *, env: dict[str, str]) -> str | None:
    completed = subprocess.run(command, text=True, capture_output=True, check=False, env=env)
    if completed.returncode == 0:
        return None
    output = (completed.stderr or completed.stdout or "").strip()
    return output[-1000:] or f"{command[0]} exited with code {completed.returncode}"


def _benchmark_capabilities(benchmark: dict[str, Any]) -> list[str]:
    capabilities = benchmark.get("capabilities", [])
    if not isinstance(capabilities, list):
        return []
    return [item.strip() for item in capabilities if isinstance(item, str) and item.strip()]


def _benchmark_required_tools(benchmark: dict[str, Any]) -> list[str]:
    tools = benchmark.get("required_tools", [])
    if not isinstance(tools, list):
        return []
    return sorted({item.strip() for item in tools if isinstance(item, str) and item.strip()})


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
    required_tools = _benchmark_required_tools(benchmark)
    capability_contract = _synthetic_capability_contract(required_capabilities)
    return {
        "benchmark": benchmark.get("name", task.id),
        "group": benchmark.get("group", task.category),
        "status": status,
        "score": score,
        "raw_score": score,
        "valid_score": 0.0,
        "error": error,
        "included_in_official_score": False,
        "repository": benchmark.get("repository", benchmark.get("homepage", "")),
        "repository_ref": benchmark.get("ref", ""),
        "required_capabilities": required_capabilities,
        "supported_capabilities": [
            capability
            for capability, support in capability_contract.items()
            if support.get("supported") is True
        ],
        "required_tools": required_tools,
        "exposed_tools": [],
        "missing_tools": required_tools if status in {FAILED_HARNESS_SETUP, FAILED_CONTAINER_RUNTIME} else [],
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
    supported = {
        "browser_or_gui",
        "chat_answer",
        "external_data_required",
        "office_document_editing",
        "tool_call",
    }
    reasons = {
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
