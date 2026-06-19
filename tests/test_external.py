import json
from pathlib import Path
from subprocess import CompletedProcess

from agent_bench.external import ExternalBenchmarkConfig, ExternalBenchmarkRunner
from agent_bench.models import Task


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
        ),
    )

    assert result.passed is True
    assert result.details["docker_image"] == "example-benchmark:local"
    assert commands[0] == ["docker", "image", "inspect", "example-benchmark:local"]
    assert commands[1][-1] == "example-benchmark:local"
    assert "AGENT_BENCH_DOCKER_IMAGE=example-benchmark:local" in commands[1]
    assert not any(item.endswith(":/outputs") for item in commands[1])
    assert commands[2][:2] == ["docker", "cp"]
    assert commands[3][:3] == ["docker", "rm", "-f"]
