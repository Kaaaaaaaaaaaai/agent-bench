from pathlib import Path
from subprocess import CompletedProcess

from agent_bench import container_cli


def test_container_cli_builds_missing_docker_sandbox_before_running(monkeypatch, tmp_path):
    source_root = tmp_path / "source"
    docker_dir = source_root / "docker"
    docker_dir.mkdir(parents=True)
    (docker_dir / "sandbox.Dockerfile").write_text("FROM python:3.12-slim\n", encoding="utf-8")
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        if command[:3] == ["docker", "image", "inspect"]:
            return CompletedProcess(command, 1, "", "missing")
        return CompletedProcess(command, 0, "", "")

    monkeypatch.setenv("AGENT_BENCH_SOURCE_ROOT", str(source_root))
    monkeypatch.setattr(container_cli.subprocess, "run", fake_run)
    monkeypatch.setattr(container_cli, "cli_main", lambda arguments: 17)

    exit_code = container_cli.main(["run", "--provider", "mock"])

    assert exit_code == 17
    assert commands == [
        ["docker", "image", "inspect", "agent-bench-python:3.12"],
        [
            "docker",
            "build",
            "-f",
            str(docker_dir / "sandbox.Dockerfile"),
            "-t",
            "agent-bench-python:3.12",
            str(source_root),
        ],
    ]


def test_container_cli_skips_sandbox_bootstrap_for_subprocess(monkeypatch):
    monkeypatch.setattr(
        container_cli.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected docker call")),
    )
    monkeypatch.setattr(container_cli, "cli_main", lambda arguments: 0)

    assert container_cli.main(["run", "--sandbox", "subprocess"]) == 0
