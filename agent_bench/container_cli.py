import os
import subprocess
import sys
from pathlib import Path

from agent_bench.cli import main as cli_main


DEFAULT_SANDBOX_IMAGE = "agent-bench-python:3.12"


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if _uses_docker_sandbox(arguments):
        _ensure_sandbox_image(_option_value(arguments, "--sandbox-image") or DEFAULT_SANDBOX_IMAGE)
    return cli_main(arguments)


def _uses_docker_sandbox(arguments: list[str]) -> bool:
    if not arguments or arguments[0] != "run":
        return False
    return (_option_value(arguments, "--sandbox") or "docker") == "docker"


def _option_value(arguments: list[str], option: str) -> str | None:
    for index, argument in enumerate(arguments):
        if argument == option and index + 1 < len(arguments):
            return arguments[index + 1]
        prefix = option + "="
        if argument.startswith(prefix):
            return argument[len(prefix) :]
    return None


def _ensure_sandbox_image(image: str) -> None:
    docker_bin = os.environ.get("AGENT_BENCH_DOCKER_BIN", "docker")
    inspect = subprocess.run(
        [docker_bin, "image", "inspect", image],
        text=True,
        capture_output=True,
        check=False,
    )
    if inspect.returncode == 0:
        return

    source_root = Path(os.environ.get("AGENT_BENCH_SOURCE_ROOT", Path.cwd()))
    dockerfile = source_root / "docker" / "sandbox.Dockerfile"
    if not dockerfile.is_file():
        raise RuntimeError(f"Sandbox Dockerfile was not found at {dockerfile}")

    print(f"Preparing coding sandbox image: {image}", file=sys.stderr)
    build = subprocess.run(
        [docker_bin, "build", "-f", str(dockerfile), "-t", image, str(source_root)],
        text=True,
        check=False,
    )
    if build.returncode != 0:
        raise RuntimeError(
            "Unable to build the coding sandbox image. "
            "Attach a usable Docker daemon and retry."
        )


if __name__ == "__main__":
    raise SystemExit(main())
