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
        source="tasks/examplebench/manifest.json",
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


def _write_example_task_folder(root: Path) -> Path:
    task_dir = root / "tasks" / "examplebench"
    (task_dir / "harness").mkdir(parents=True, exist_ok=True)
    (task_dir / "manifest.json").write_text("{}\n", encoding="utf-8")
    (task_dir / "harness" / "run.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    return task_dir


def _write_asset_lock_task_folder(root: Path, slug: str, recipe: dict) -> Path:
    task_dir = root / "tasks" / slug
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "manifest.json").write_text("{}\n", encoding="utf-8")
    (task_dir / "assets.lock.json").write_text(
        json.dumps(
            {
                "schema_version": "agent-bench.assets-lock.v1",
                "benchmark_slug": slug,
                "source": {
                    "repository_url": recipe["repository"],
                    "commit": recipe.get("ref", "0123456789abcdef0123456789abcdef01234567"),
                },
                "materialization": {"cache_recipe": recipe},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return task_dir


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
    monkeypatch.setenv("AGENT_BENCH_CUSTOM_FLAG", "1")

    task = _manifest_task()
    task.benchmark["manifest"]["container"]["environment_allowed"] = ["AGENT_BENCH_CUSTOM_*"]
    task_dir = _write_example_task_folder(tmp_path)

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
            source_root=tmp_path,
            allow_host_docker_socket=True,
        ),
    )

    assert result.passed is True
    assert result.details["docker_image"] == "example-benchmark:local"
    assert result.details["network_mode"] == "bridge"
    assert result.details["docker_socket_mount"]["enabled"] is False
    assert result.details["docker_socket_mount"]["global_allow_host_docker_socket"] is True
    assert result.details["asset_cache_mount"]["cache_key"] == "examplebench"
    assert result.details["benchmark_task_mount"]["enabled"] is True
    assert result.details["benchmark_task_mount"]["host_path"] == str(task_dir.resolve())
    assert result.details["benchmark_task_mount"]["container_path"] == "/benchmark/task"
    assert result.details["benchmark_task_mount"]["readonly"] is True
    assert result.details["benchmark_task_mount"]["packaged_path"] == ""
    assert result.details["result"]["network_mode"] == "bridge"
    assert result.details["result"]["docker_socket_mount"]["enabled"] is False
    assert result.details["result"]["asset_cache_mount"]["cache_key"] == "examplebench"
    assert result.details["result"]["benchmark_task_mount"]["container_path"] == "/benchmark/task"
    assert commands[0] == ["docker", "image", "inspect", "example-benchmark:local"]
    assert commands[1][-1] == "example-benchmark:local"
    assert "--cap-drop" in commands[1]
    assert "ALL" in commands[1]
    assert "--security-opt" in commands[1]
    assert "no-new-privileges" in commands[1]
    tmpfs_mounts = [
        commands[1][index + 1] for index, item in enumerate(commands[1]) if item == "--tmpfs"
    ]
    assert "/tmp:rw,nosuid,nodev,noexec,size=512m,uid=10001,gid=10001" in tmpfs_mounts
    assert "/workspace:rw,nosuid,nodev,exec,size=4g,uid=10001,gid=10001" in tmpfs_mounts
    assert "--user" in commands[1]
    assert "10001:10001" in commands[1]
    assert "--network" in commands[1]
    assert "bridge" in commands[1]
    assert "AGENT_BENCH_DOCKER_IMAGE=example-benchmark:local" in commands[1]
    assert "AGENT_BENCH_REQUIRED_CAPABILITIES=repo_patch,chat_answer" in commands[1]
    assert "AGENT_BENCH_ALLOW_TARGET_CHECKOUT=1" in commands[1]
    assert "AGENT_BENCH_MODEL_REQUEST_TIMEOUT=1800.0" in commands[1]
    assert "AGENT_BENCH_MAX_TOKENS=16384" in commands[1]
    assert "AGENT_BENCH_TASK_DIR=/benchmark/task" in commands[1]
    assert "AGENT_BENCH_ASSET_ROOT=/benchmark/assets" in commands[1]
    assert "AGENT_BENCH_ASSET_CACHE_KEY=examplebench" in commands[1]
    assert "AGENT_BENCH_SAMPLE_LIMIT=1" in commands[1]
    assert "AGENT_BENCH_CUSTOM_FLAG=1" in commands[1]
    assert f"type=bind,src={task_dir.resolve()},dst=/benchmark/task,readonly" in commands[1]
    assert f"type=bind,src={(tmp_path / 'asset-cache' / 'examplebench').resolve()},dst=/benchmark/assets,readonly" in commands[1]
    assert "/var/run/docker.sock:/var/run/docker.sock" not in commands[1]
    assert not any(item.startswith("AGENT_BENCH_API_KEY=") for item in commands[1])
    assert not any(item.endswith(":/outputs") for item in commands[1])
    assert commands[2][:2] == ["docker", "cp"]
    assert commands[3][:3] == ["docker", "rm", "-f"]


def test_external_runner_mounts_docker_socket_when_descriptor_requires_it(monkeypatch, tmp_path):
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
    _write_example_task_folder(tmp_path)
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
            source_root=tmp_path,
            allow_host_docker_socket=True,
        ),
    )

    assert result.passed is True
    docker_run = commands[1]
    assert docker_run[docker_run.index("--network") + 1] == "host"
    assert docker_run[docker_run.index("--user") + 1] == "0:0"
    assert docker_run[docker_run.index("--cap-drop") + 1] == "ALL"
    assert docker_run[docker_run.index("--cap-add") + 1] == "DAC_OVERRIDE"
    assert "/var/run/docker.sock:/var/run/docker.sock" in docker_run
    assert result.details["network_mode"] == "host"
    assert result.details["docker_socket_mount"]["enabled"] is True
    assert result.details["docker_socket_mount"]["descriptor_requires_host_docker_socket"] is True
    assert result.details["docker_socket_mount"]["global_allow_host_docker_socket"] is True
    assert result.details["docker_socket_mount"]["deprecated_allow_host_docker_socket_flag"] is True
    assert result.details["docker_socket_mount"]["enabled_by_manifest"] is True
    assert result.details["result"]["network_mode"] == "host"
    assert result.details["result"]["docker_socket_mount"]["enabled"] is True


def test_external_runner_uses_packaged_task_dir_without_bind_mounts(monkeypatch, tmp_path):
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

    monkeypatch.setenv("AGENT_BENCH_CONTAINERIZED", "1")
    monkeypatch.setattr("agent_bench.external.shutil.which", lambda name: "/usr/bin/docker")
    monkeypatch.setattr("agent_bench.external.subprocess.run", fake_run)

    task = _manifest_task()
    _write_example_task_folder(tmp_path)

    result = ExternalBenchmarkRunner()._run_sync(
        task,
        ExternalBenchmarkConfig(
            provider="openai-compatible",
            base_url="http://proxy.test/v1",
            model="example-model",
            api_key_env="",
            output_dir=tmp_path,
            timeout=5.0,
            asset_root=tmp_path / "asset-cache",
            source_root=tmp_path,
        ),
    )

    docker_run = commands[1]
    assert result.passed is True
    assert "AGENT_BENCH_TASK_DIR=/benchmark/task" in docker_run
    assert "AGENT_BENCH_PACKAGED_TASK_DIR=/opt/agent-bench/tasks/examplebench" in docker_run
    assert not any("/opt/agent-bench/tasks/examplebench" in item for item in docker_run if item.startswith("type=bind"))
    assert not any("/benchmark/task" in item for item in docker_run if item.startswith("type=bind"))
    assert not any("/benchmark/assets" in item for item in docker_run if item.startswith("type=bind"))
    assert result.details["benchmark_task_mount"]["enabled"] is False
    assert result.details["benchmark_task_mount"]["packaged_path"] == "/opt/agent-bench/tasks/examplebench"
    assert result.details["asset_cache_mount"]["enabled"] is False


def test_external_runner_requires_docker_socket_opt_in(monkeypatch, tmp_path):
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
    _write_example_task_folder(tmp_path)

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
            source_root=tmp_path,
            allow_host_docker_socket=False,
        ),
    )

    assert result.passed is False
    assert result.details["result"]["status"] == "failed_manifest_validation"
    assert "explicit operator opt-in" in (result.error or "")
    assert commands == []


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
    _write_example_task_folder(tmp_path)

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
            source_root=tmp_path,
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


def test_external_runner_reports_container_startup_stderr(monkeypatch, tmp_path):
    def fake_run(command, **kwargs):
        if command[:3] == ["docker", "image", "inspect"]:
            return CompletedProcess(command, 0, "", "")
        if command[:3] == ["docker", "run", "--name"]:
            return CompletedProcess(
                command,
                125,
                "",
                'docker: Error response from daemon: invalid mount config for type "bind".\n',
            )
        if command[:2] == ["docker", "cp"]:
            return CompletedProcess(command, 1, "", "No such container\n")
        if command[:3] == ["docker", "rm", "-f"]:
            return CompletedProcess(command, 0, "", "")
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr("agent_bench.external.shutil.which", lambda name: "/usr/bin/docker")
    monkeypatch.setattr("agent_bench.external.subprocess.run", fake_run)

    task = _manifest_task(["chat_answer"])
    _write_example_task_folder(tmp_path)

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
            source_root=tmp_path,
        ),
    )

    assert result.passed is False
    assert "External benchmark exited with code 125" in result.error
    assert "invalid mount config" in result.error
    payload = json.loads((tmp_path / "external" / "PB_001" / "agent_bench_result.json").read_text(encoding="utf-8"))
    assert payload["error"] == result.error


def test_external_runner_rejects_asset_cache_parent_traversal(tmp_path):
    task = _manifest_task()
    task.source = "tasks/unsafe/manifest.json"
    _write_asset_lock_task_folder(
        tmp_path,
        "unsafe",
        {
            "repository": "https://example.com/repo.git",
            "ref": "0123456789abcdef0123456789abcdef01234567",
            "requires_git_lfs": False,
            "subpaths": ["../../outside"],
        },
    )
    outside = tmp_path / "outside"
    outside.mkdir()
    marker = outside / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    config = ExternalBenchmarkConfig(
        provider="openai-compatible",
        base_url="http://localhost:8000/v1",
        model="example-model",
        api_key_env="",
        output_dir=tmp_path / "output",
        timeout=5.0,
        asset_root=tmp_path / "asset-cache",
        source_root=tmp_path,
    )

    error = _prepare_benchmark_asset_cache(task, config)

    assert error is not None
    assert "unsafe subpaths" in error
    assert marker.read_text(encoding="utf-8") == "keep"


def test_external_runner_uses_safe_output_directory_for_invalid_task_id(tmp_path):
    task = _manifest_task()
    task.id = "../../escape"
    task.benchmark["manifest"]["id"] = task.id
    config = ExternalBenchmarkConfig(
        provider="openai-compatible",
        base_url="http://localhost:8000/v1",
        model="example-model",
        api_key_env="",
        output_dir=tmp_path / "output",
        timeout=5.0,
        asset_root=tmp_path / "asset-cache",
        source_root=tmp_path,
    )

    result = ExternalBenchmarkRunner()._run_sync(task, config)

    assert result.passed is False
    result_path = Path(result.details["output_dir"])
    assert result_path.is_relative_to(config.output_dir / "external")
    assert result_path.name.startswith("invalid-")
    assert not (tmp_path / "escape").exists()


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
    task.source = "tasks/paperbench/manifest.json"
    _write_asset_lock_task_folder(
        tmp_path,
        "paperbench",
        {
            "key": "paperbench",
            "repository": "https://github.com/openai/frontier-evals.git",
            "ref": "main",
            "includes": ["project/paperbench/data/papers/**"],
            "subpaths": ["project/paperbench/data/papers"],
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
        source_root=tmp_path,
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


def test_external_runner_prepares_swelancer_asset_cache_without_lfs(monkeypatch, tmp_path):
    commands: list[list[str]] = []

    def fake_run(command, **kwargs):
        commands.append(command)
        if "clone" in command:
            clone_dir = Path(command[-1])
            swelancer_dir = clone_dir / "project" / "swelancer"
            issue_dir = swelancer_dir / "issues" / "16912_4"
            issue_dir.mkdir(parents=True)
            (swelancer_dir / "all_swelancer_tasks.csv").write_text(
                "question_id,title,description,cwd\n"
                "16912_4,Fix zip hint,Patch the failing behavior,/app/expensify\n",
                encoding="utf-8",
            )
            (issue_dir / "commit_id.txt").write_text(
                "2b791c9f3053c1682ddcb50ab036deb3e55a7542\n",
                encoding="utf-8",
            )
        return CompletedProcess(command, 0, "", "")

    def fake_which(name):
        if name == "git":
            return "/usr/bin/git"
        if name == "git-lfs":
            return None
        return f"/usr/bin/{name}"

    monkeypatch.setattr("agent_bench.external.shutil.which", fake_which)
    monkeypatch.setattr("agent_bench.external.subprocess.run", fake_run)
    task = Task(
        id="PB_004",
        category="public_benchmarks",
        type="external_benchmark",
        question="Run SWE-Lancer",
        source="public_benchmarks.json",
        benchmark={
            "name": "SWE-Lancer",
            "homepage": "https://github.com/openai/frontier-evals/tree/main/project/swelancer",
            "repository": "https://github.com/openai/frontier-evals.git",
            "ref": "51052cede8cc608f95bb00346635e03759013e5a",
            "subdir": "project/swelancer",
            "license": "MIT",
            "credit": "OpenAI",
            "docker": {"command": "agent-bench-probe --benchmark SWE-Lancer"},
        },
    )
    task.source = "tasks/swe-lancer/manifest.json"
    _write_asset_lock_task_folder(
        tmp_path,
        "swe-lancer",
        {
            "key": "swe-lancer",
            "repository": "https://github.com/openai/frontier-evals.git",
            "ref": "51052cede8cc608f95bb00346635e03759013e5a",
            "requires_git_lfs": False,
            "includes": [],
            "subpaths": ["project/swelancer"],
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
        source_root=tmp_path,
    )

    error = _prepare_benchmark_asset_cache(task, config)

    assert error is None
    assert (
        tmp_path
        / "agent-bench-assets"
        / "swe-lancer"
        / "project"
        / "swelancer"
        / "all_swelancer_tasks.csv"
    ).is_file()
    assert (
        tmp_path
        / "agent-bench-assets"
        / "swe-lancer"
        / "project"
        / "swelancer"
        / "issues"
        / "16912_4"
        / "commit_id.txt"
    ).is_file()
    assert not any("lfs" in command for command in commands)


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
    task.source = "tasks/gdpval/manifest.json"
    _write_asset_lock_task_folder(
        tmp_path,
        "gdpval",
        {
            "key": "gdpval",
            "repository": "https://huggingface.co/datasets/openai/gdpval",
            "ref": "main",
            "includes": ["reference_files/**", "deliverable_files/**"],
            "subpaths": ["reference_files", "deliverable_files"],
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
        source_root=tmp_path,
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
    task.source = "tasks/gdpval/manifest.json"
    _write_asset_lock_task_folder(
        tmp_path,
        "gdpval",
        {
            "key": "gdpval",
            "repository": "https://huggingface.co/datasets/openai/gdpval",
            "ref": "main",
            "includes": ["reference_files/**", "deliverable_files/**"],
            "subpaths": ["reference_files", "deliverable_files"],
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
        source_root=tmp_path,
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
