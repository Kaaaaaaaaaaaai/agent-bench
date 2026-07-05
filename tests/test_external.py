import json
from pathlib import Path
from subprocess import CompletedProcess

from agent_bench.external import (
    ExternalBenchmarkConfig,
    ExternalBenchmarkRunner,
    _prepare_benchmark_asset_cache,
)
from agent_bench.models import Task


def _manifest_task(capabilities: list[str] | None = None) -> Task:
    manifest = {
        "id": "PB_001",
        "display_name": "ExampleBench",
        "task_group": "Coding",
        "description": "Run ExampleBench under official conditions.",
        "homepage_url": "https://example.com",
        "official_leaderboard_url": "https://example.com/leaderboard",
        "source": {
            "repository_url": "https://example.com/examplebench.git",
            "commit": "0123456789abcdef0123456789abcdef01234567",
        },
        "license": "MIT",
        "credit": "Example authors",
        "citation": "https://example.com/citation",
        "official_conditions": {
            "official_split": "test",
            "official_scoring_method": "exact official score",
            "official_prompt_format": "official prompt",
            "official_grader_command": "examplebench grade",
            "official_evaluation_config": "official.yaml",
        },
        "assets": [
            {
                "source": "https://example.com/assets.jsonl",
                "revision": "0123456789abcdef0123456789abcdef01234567",
                "checksum": "sha256:abc",
                "expected_local_path": "examplebench/assets.jsonl",
                "required": True,
            }
        ],
        "container": {
            "image": "example-benchmark:local",
            "command": "agent-bench-probe --benchmark ExampleBench",
            "network": "model_proxy",
            "timeout_seconds": 5,
        },
        "adapter": {
            "module": "examplebench.adapter",
            "expected_output_files": ["agent_bench_result.json"],
            "result_parser": "agent_bench_result_json",
        },
        "scoring": {
            "raw_score_field": "score",
            "max_score": 1.0,
            "normalization": "fraction_to_0_100",
            "direction": "higher_is_better",
        },
        "reporting": {
            "category_label": "Coding",
            "display_order": 1,
            "license": "MIT",
            "credit": "Example authors",
            "citation": "https://example.com/citation",
        },
        "capabilities": capabilities or ["repo_patch", "chat_answer"],
    }
    return Task(
        id="PB_001",
        category="Coding",
        type="external_benchmark",
        question="Run benchmark",
        source="public_benchmarks.json",
        benchmark={
            "name": "ExampleBench",
            "group": "Coding",
            "homepage": "https://example.com",
            "license": "MIT",
            "credit": "Example authors",
            "citation": "https://example.com/citation",
            "capabilities": capabilities or ["repo_patch", "chat_answer"],
            "manifest": manifest,
            "docker": {
                "image": "example-benchmark:local",
                "command": "agent-bench-probe --benchmark ExampleBench",
            },
        },
    )


def test_external_runner_uses_descriptor_docker_image(monkeypatch, tmp_path):
    commands: list[list[str]] = []

    def fake_run(command, **kwargs):
        commands.append(command)
        if command[:3] == ["docker", "image", "inspect"]:
            return CompletedProcess(command, 0, "", "")
        if command[:3] == ["docker", "run", "--name"]:
            return CompletedProcess(command, 0, "ready", "")
        if command[:2] == ["docker", "cp"]:
            output_dir = Path(command[3])
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "agent_bench_result.json").write_text(
                json.dumps({"score": 1.0, "ok": True}) + "\n",
                encoding="utf-8",
            )
            return CompletedProcess(command, 0, "", "")
        if command[:3] == ["docker", "rm", "-f"]:
            return CompletedProcess(command, 0, "", "")
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr("agent_bench.external.shutil.which", lambda name: "/usr/bin/docker")
    monkeypatch.setattr("agent_bench.external.subprocess.run", fake_run)
    monkeypatch.setenv("AGENT_BENCH_SAMPLE_LIMIT", "1")

    task = _manifest_task()

    result = ExternalBenchmarkRunner()._run_sync(
        task,
        ExternalBenchmarkConfig(
            provider="openai-compatible",
            base_url="http://localhost:8000/v1",
            model="example-model",
            api_key_env="",
            output_dir=tmp_path,
            timeout=5.0,
            asset_root=tmp_path / "asset-cache",
            allow_host_docker_socket=True,
        ),
    )

    assert result.passed is True
    assert result.details["docker_image"] == "example-benchmark:local"
    assert result.details["network_mode"] == "bridge"
    assert result.details["docker_socket_mount"]["enabled"] is False
    assert result.details["docker_socket_mount"]["global_allow_host_docker_socket"] is True
    assert result.details["asset_cache_mount"]["cache_key"] == "examplebench"
    assert result.details["result"]["network_mode"] == "bridge"
    assert result.details["result"]["docker_socket_mount"]["enabled"] is False
    assert result.details["result"]["asset_cache_mount"]["cache_key"] == "examplebench"
    assert commands[0] == ["docker", "image", "inspect", "example-benchmark:local"]
    assert commands[1][-1] == "example-benchmark:local"
    assert "--cap-drop" in commands[1]
    assert "ALL" in commands[1]
    assert "--security-opt" in commands[1]
    assert "no-new-privileges" in commands[1]
    assert "--user" in commands[1]
    assert "10001:10001" in commands[1]
    assert "--network" in commands[1]
    assert "bridge" in commands[1]
    assert "AGENT_BENCH_DOCKER_IMAGE=example-benchmark:local" in commands[1]
    assert "AGENT_BENCH_REQUIRED_CAPABILITIES=repo_patch,chat_answer" in commands[1]
    assert "AGENT_BENCH_ALLOW_TARGET_CHECKOUT=1" in commands[1]
    assert "AGENT_BENCH_MODEL_REQUEST_TIMEOUT=1800.0" in commands[1]
    assert "AGENT_BENCH_MAX_TOKENS=16384" in commands[1]
    assert "AGENT_BENCH_ASSET_ROOT=/asset-cache" in commands[1]
    assert "AGENT_BENCH_ASSET_CACHE_KEY=examplebench" in commands[1]
    assert "AGENT_BENCH_SAMPLE_LIMIT=1" in commands[1]
    assert f"type=bind,src={(tmp_path / 'asset-cache').resolve()},dst=/asset-cache,readonly" in commands[1]
    assert "/var/run/docker.sock:/var/run/docker.sock" not in commands[1]
    assert not any(item.startswith("AGENT_BENCH_API_KEY=") for item in commands[1])
    assert not any(item.endswith(":/outputs") for item in commands[1])
    assert commands[2][:2] == ["docker", "cp"]
    assert commands[3][:3] == ["docker", "rm", "-f"]


def test_external_runner_mounts_docker_socket_only_when_descriptor_and_config_allow(monkeypatch, tmp_path):
    commands: list[list[str]] = []

    def fake_run(command, **kwargs):
        commands.append(command)
        if command[:3] == ["docker", "image", "inspect"]:
            return CompletedProcess(command, 0, "", "")
        if command[:3] == ["docker", "run", "--name"]:
            return CompletedProcess(command, 0, "ready", "")
        if command[:2] == ["docker", "cp"]:
            output_dir = Path(command[3])
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "agent_bench_result.json").write_text(
                json.dumps({"score": 1.0, "ok": True}) + "\n",
                encoding="utf-8",
            )
            return CompletedProcess(command, 0, "", "")
        if command[:3] == ["docker", "rm", "-f"]:
            return CompletedProcess(command, 0, "", "")
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr("agent_bench.external.shutil.which", lambda name: "/usr/bin/docker")
    monkeypatch.setattr("agent_bench.external.subprocess.run", fake_run)

    task = _manifest_task(["tool_call"])
    task.benchmark["manifest"]["container"]["requires_host_docker_socket"] = True
    task.benchmark["manifest"]["container"]["network"] = "host"

    result = ExternalBenchmarkRunner()._run_sync(
        task,
        ExternalBenchmarkConfig(
            provider="openai-compatible",
            base_url="http://localhost:8000/v1",
            model="example-model",
            api_key_env="",
            output_dir=tmp_path,
            timeout=5.0,
            asset_root=tmp_path / "asset-cache",
            allow_host_docker_socket=True,
        ),
    )

    assert result.passed is True
    docker_run = commands[1]
    assert docker_run[docker_run.index("--network") + 1] == "host"
    assert "/var/run/docker.sock:/var/run/docker.sock" in docker_run
    assert result.details["network_mode"] == "host"
    assert result.details["docker_socket_mount"]["enabled"] is True
    assert result.details["docker_socket_mount"]["descriptor_requires_host_docker_socket"] is True
    assert result.details["docker_socket_mount"]["global_allow_host_docker_socket"] is True
    assert result.details["result"]["network_mode"] == "host"
    assert result.details["result"]["docker_socket_mount"]["enabled"] is True


def test_external_runner_fails_when_result_file_missing(monkeypatch, tmp_path):
    def fake_run(command, **kwargs):
        if command[:3] == ["docker", "image", "inspect"]:
            return CompletedProcess(command, 0, "", "")
        if command[:3] == ["docker", "run", "--name"]:
            return CompletedProcess(command, 0, "completed without result", "")
        if command[:2] == ["docker", "cp"]:
            Path(command[3]).mkdir(parents=True, exist_ok=True)
            return CompletedProcess(command, 0, "", "")
        if command[:3] == ["docker", "rm", "-f"]:
            return CompletedProcess(command, 0, "", "")
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr("agent_bench.external.shutil.which", lambda name: "/usr/bin/docker")
    monkeypatch.setattr("agent_bench.external.subprocess.run", fake_run)

    task = _manifest_task(["chat_answer"])

    result = ExternalBenchmarkRunner()._run_sync(
        task,
        ExternalBenchmarkConfig(
            provider="openai-compatible",
            base_url="http://localhost:8000/v1",
            model="example-model",
            api_key_env="",
            output_dir=tmp_path,
            timeout=5.0,
            asset_root=tmp_path / "asset-cache",
        ),
    )

    assert result.score == 0.0
    assert result.passed is False
    assert result.error == "External benchmark did not produce agent_bench_result.json"
    result_file = tmp_path / "external" / "PB_001" / "agent_bench_result.json"
    payload = json.loads(result_file.read_text(encoding="utf-8"))
    assert payload["status"] == "failed_harness_setup"
    assert payload["error"] == "External benchmark did not produce agent_bench_result.json"
    assert payload["capability_contract"]["chat_answer"]["supported"] is True
    assert result.details["result"]["status"] == "failed_harness_setup"


def test_external_runner_rejects_incomplete_legacy_descriptor(tmp_path):
    task = Task(
        id="PB_001",
        category="public_benchmarks",
        type="external_benchmark",
        question="Run benchmark",
        source="public_benchmarks.json",
        benchmark={
            "name": "ExampleBench",
            "homepage": "https://example.com",
            "license": "MIT",
            "credit": "Example authors",
            "capabilities": ["chat_answer"],
            "docker": {
                "image": "example-benchmark:local",
                "command": "agent-bench-probe --benchmark ExampleBench",
            },
        },
    )

    result = ExternalBenchmarkRunner()._run_sync(
        task,
        ExternalBenchmarkConfig(
            provider="openai-compatible",
            base_url="http://localhost:8000/v1",
            model="example-model",
            api_key_env="",
            output_dir=tmp_path,
            timeout=5.0,
            asset_root=tmp_path / "asset-cache",
        ),
    )

    assert result.passed is False
    assert result.details["result"]["status"] == "failed_manifest_validation"
    assert "official_conditions.official_split" in result.error


def test_external_runner_prepares_paperbench_asset_cache(monkeypatch, tmp_path):
    commands: list[list[str]] = []

    def fake_run(command, **kwargs):
        commands.append(command)
        if "clone" in command:
            clone_dir = Path(command[-1])
            paper_dir = clone_dir / "project" / "paperbench" / "data" / "papers" / "paper-1"
            paper_dir.mkdir(parents=True)
            (paper_dir / "paper.pdf").write_bytes(b"%PDF-1.4\nreal pdf")
            return CompletedProcess(command, 0, "", "")
        return CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("agent_bench.external.shutil.which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr("agent_bench.external.subprocess.run", fake_run)
    task = Task(
        id="PB_003",
        category="public_benchmarks",
        type="external_benchmark",
        question="Run PaperBench",
        source="public_benchmarks.json",
        benchmark={
            "name": "PaperBench",
            "homepage": "https://github.com/openai/frontier-evals/tree/main/project/paperbench",
            "repository": "https://github.com/openai/frontier-evals.git",
            "ref": "main",
            "license": "MIT",
            "credit": "OpenAI",
            "docker": {"command": "agent-bench-probe --benchmark PaperBench"},
        },
    )
    config = ExternalBenchmarkConfig(
        provider="openai-compatible",
        base_url="http://localhost:8000/v1",
        model="example-model",
        api_key_env="",
        output_dir=tmp_path / "runs",
        timeout=5.0,
        asset_root=tmp_path / "agent-bench-assets",
    )

    error = _prepare_benchmark_asset_cache(task, config)

    assert error is None
    assert (
        tmp_path
        / "agent-bench-assets"
        / "paperbench"
        / "project"
        / "paperbench"
        / "data"
        / "papers"
        / "paper-1"
        / "paper.pdf"
    ).is_file()
    assert any(command[:4] == ["/usr/bin/git", "-C", str(tmp_path / "agent-bench-assets" / "_downloads" / "paperbench" / "repo"), "lfs"] for command in commands)


def test_external_runner_prepares_gdpval_asset_cache(monkeypatch, tmp_path):
    def fake_run(command, **kwargs):
        if "clone" in command:
            clone_dir = Path(command[-1])
            (clone_dir / "reference_files" / "case-1").mkdir(parents=True)
            (clone_dir / "reference_files" / "case-1" / "Population.xlsx").write_bytes(b"PK\x03\x04real xlsx")
            (clone_dir / "deliverable_files" / "case-1").mkdir(parents=True)
            (clone_dir / "deliverable_files" / "case-1" / "Sample.xlsx").write_bytes(b"PK\x03\x04real xlsx")
        return CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("agent_bench.external.shutil.which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr("agent_bench.external.subprocess.run", fake_run)
    task = Task(
        id="PB_002",
        category="public_benchmarks",
        type="external_benchmark",
        question="Run GDPval",
        source="public_benchmarks.json",
        benchmark={
            "name": "GDPval",
            "homepage": "https://huggingface.co/datasets/openai/gdpval",
            "repository": "https://huggingface.co/datasets/openai/gdpval",
            "ref": "main",
            "license": "MIT",
            "credit": "OpenAI",
            "docker": {"command": "agent-bench-probe --benchmark GDPval"},
        },
    )
    config = ExternalBenchmarkConfig(
        provider="openai-compatible",
        base_url="http://localhost:8000/v1",
        model="example-model",
        api_key_env="",
        output_dir=tmp_path / "runs",
        timeout=5.0,
        asset_root=tmp_path / "agent-bench-assets",
    )

    error = _prepare_benchmark_asset_cache(task, config)

    assert error is None
    assert (
        tmp_path
        / "agent-bench-assets"
        / "gdpval"
        / "reference_files"
        / "case-1"
        / "Population.xlsx"
    ).is_file()
    assert (
        tmp_path
        / "agent-bench-assets"
        / "gdpval"
        / "deliverable_files"
        / "case-1"
        / "Sample.xlsx"
    ).is_file()


def test_external_runner_uses_docker_asset_downloader_when_git_lfs_missing(monkeypatch, tmp_path):
    commands: list[list[str]] = []

    def fake_which(name):
        if name == "git-lfs":
            return None
        return f"/usr/bin/{name}"

    def fake_run(command, **kwargs):
        commands.append(command)
        if command[:2] == ["docker", "run"]:
            mount = command[command.index("-v") + 1]
            host_asset_root = Path(mount.split(":", 1)[0])
            target = host_asset_root / "gdpval" / "reference_files" / "case-1"
            target.mkdir(parents=True)
            (target / "Population.xlsx").write_bytes(b"PK\x03\x04real xlsx")
            (host_asset_root / "gdpval" / ".agent-bench-assets-ready.json").write_text(
                '{"downloaded":true}\n',
                encoding="utf-8",
            )
            return CompletedProcess(command, 0, "", "")
        return CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("agent_bench.external.shutil.which", fake_which)
    monkeypatch.setattr("agent_bench.external.subprocess.run", fake_run)
    task = Task(
        id="PB_002",
        category="public_benchmarks",
        type="external_benchmark",
        question="Run GDPval",
        source="public_benchmarks.json",
        benchmark={
            "name": "GDPval",
            "homepage": "https://huggingface.co/datasets/openai/gdpval",
            "repository": "https://huggingface.co/datasets/openai/gdpval",
            "license": "MIT",
            "credit": "OpenAI",
            "docker": {"command": "agent-bench-probe --benchmark GDPval"},
        },
    )
    config = ExternalBenchmarkConfig(
        provider="openai-compatible",
        base_url="http://localhost:8000/v1",
        model="example-model",
        api_key_env="",
        output_dir=tmp_path / "runs",
        timeout=5.0,
        asset_root=tmp_path / "agent-bench-assets",
    )

    error = _prepare_benchmark_asset_cache(
        task,
        config,
        launcher_image="agent-bench-external:python3.12",
    )

    assert error is None
    assert commands[0][:2] == ["docker", "run"]
    assert "--entrypoint" in commands[0]
    assert "agent-bench-external:python3.12" in commands[0]
