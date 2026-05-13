import json
import asyncio
import subprocess

import pytest

from agent_bench.models import ModelResponse, Task
from agent_bench.sandbox import DockerSandbox
from agent_bench.verifiers import grade_coding


def _docker_image_available():
    completed = subprocess.run(
        ["docker", "image", "inspect", "agent-bench-python:3.12"],
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.returncode == 0


pytestmark = pytest.mark.docker


@pytest.mark.skipif(not _docker_image_available(), reason="agent-bench-python:3.12 image is not built")
def test_docker_sandbox_executes_successful_python_code():
    task = Task(
        id="D_001",
        category="coding",
        type="coding",
        question="Add",
        source="coding.json",
        function_name="add",
        test_cases=[{"input": {"a": 1, "b": 2}, "output": 3}],
    )

    result = asyncio.run(
        grade_coding(
            task,
            ModelResponse(
                task_id="D_001",
                model="test",
                raw_response=json.dumps({"code": "def add(a, b):\n    return a + b"}),
                latency_seconds=0.0,
            ),
            DockerSandbox(),
            timeout_seconds=5.0,
        )
    )

    assert result.passed is True


@pytest.mark.skipif(not _docker_image_available(), reason="agent-bench-python:3.12 image is not built")
def test_docker_sandbox_times_out_infinite_loop():
    task = Task(
        id="D_002",
        category="coding",
        type="coding",
        question="Loop",
        source="coding.json",
        function_name="loop",
        test_cases=[{"input": {}, "output": 1}],
    )

    result = asyncio.run(
        grade_coding(
            task,
            ModelResponse(
                task_id="D_002",
                model="test",
                raw_response=json.dumps({"code": "def loop():\n    while True:\n        pass"}),
                latency_seconds=0.0,
            ),
            DockerSandbox(),
            timeout_seconds=1.0,
        )
    )

    assert result.passed is False
    assert result.timed_out is True
