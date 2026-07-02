import asyncio
import io
import json
from pathlib import Path

from agent_bench.external import ExternalBenchmarkResult, ExternalBenchmarkRunner
from agent_bench.models import ModelResponse, Task
from agent_bench.runner import RunConfig, _process_task


class SequenceClient:
    def __init__(self, raw_responses: list[str]) -> None:
        self.raw_responses = list(raw_responses)
        self.calls = 0

    async def complete(self, task: Task) -> ModelResponse:
        raw_response = self.raw_responses.pop(0)
        self.calls += 1
        return ModelResponse(
            task_id=task.id,
            model="sequence",
            raw_response=raw_response,
            latency_seconds=0.01,
        )


def test_process_task_retries_empty_responses_until_non_empty() -> None:
    task = _multiple_choice_task()
    client = SequenceClient(["", "   ", '{"answer": ["A"], "confidence": 0.9}'])
    raw_handle = io.StringIO()
    graded_handle = io.StringIO()
    responses: list[ModelResponse] = []

    grade = asyncio.run(
        _process_task(
            task=task,
            client=client,
            sandbox=None,
            external_runner=ExternalBenchmarkRunner(),
            config=RunConfig(provider="mock"),
            output_dir=Path("."),
            timeout=5.0,
            request_sem=asyncio.Semaphore(1),
            eval_sem=asyncio.Semaphore(1),
            raw_lock=asyncio.Lock(),
            graded_lock=asyncio.Lock(),
            raw_handle=raw_handle,
            graded_handle=graded_handle,
            responses=responses,
        )
    )

    raw_attempts = [json.loads(line) for line in raw_handle.getvalue().splitlines()]
    assert client.calls == 3
    assert [attempt["raw_response"] for attempt in raw_attempts] == ['{"answer": ["A"], "confidence": 0.9}']
    assert len(responses) == 1
    assert responses[0].raw_response == '{"answer": ["A"], "confidence": 0.9}'
    assert grade.passed is True


def test_process_task_stops_after_three_empty_responses() -> None:
    task = _multiple_choice_task()
    client = SequenceClient(["", "\n", "\t", '{"answer": ["A"]}'])
    raw_handle = io.StringIO()
    graded_handle = io.StringIO()
    responses: list[ModelResponse] = []

    grade = asyncio.run(
        _process_task(
            task=task,
            client=client,
            sandbox=None,
            external_runner=ExternalBenchmarkRunner(),
            config=RunConfig(provider="mock"),
            output_dir=Path("."),
            timeout=5.0,
            request_sem=asyncio.Semaphore(1),
            eval_sem=asyncio.Semaphore(1),
            raw_lock=asyncio.Lock(),
            graded_lock=asyncio.Lock(),
            raw_handle=raw_handle,
            graded_handle=graded_handle,
            responses=responses,
        )
    )

    raw_attempts = [json.loads(line) for line in raw_handle.getvalue().splitlines()]
    assert client.calls == 3
    assert [attempt["raw_response"] for attempt in raw_attempts] == ["\t"]
    assert len(responses) == 1
    assert responses[0].raw_response == "\t"
    assert grade.passed is False
    assert grade.error == "empty response"


def test_external_process_task_uses_request_semaphore(tmp_path) -> None:
    tasks = [_external_task("PB_001"), _external_task("PB_002")]
    runner = TrackingExternalRunner()
    raw_handle = io.StringIO()
    graded_handle = io.StringIO()
    responses: list[ModelResponse] = []
    request_sem = asyncio.Semaphore(1)
    eval_sem = asyncio.Semaphore(2)
    raw_lock = asyncio.Lock()
    graded_lock = asyncio.Lock()

    async def run_all():
        return await asyncio.gather(
            *[
                _process_task(
                    task=task,
                    client=None,
                    sandbox=None,
                    external_runner=runner,
                    config=RunConfig(provider="openai-compatible", base_url="http://model.test/v1", model="model"),
                    output_dir=tmp_path,
                    timeout=5.0,
                    request_sem=request_sem,
                    eval_sem=eval_sem,
                    raw_lock=raw_lock,
                    graded_lock=graded_lock,
                    raw_handle=raw_handle,
                    graded_handle=graded_handle,
                    responses=responses,
                )
                for task in tasks
            ]
        )

    grades = asyncio.run(run_all())

    assert runner.max_active == 1
    assert len(responses) == 2
    assert all(grade.passed for grade in grades)


class TrackingExternalRunner:
    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0

    async def run(self, task: Task, config) -> ExternalBenchmarkResult:
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0.01)
        self.active -= 1
        return ExternalBenchmarkResult(
            score=1.0,
            passed=True,
            latency_seconds=0.01,
            details={
                "benchmark": task.benchmark["name"],
                "group": task.benchmark["group"],
                "homepage": task.benchmark["homepage"],
                "license": task.benchmark["license"],
                "credit": task.benchmark["credit"],
                "citation": task.benchmark["citation"],
                "result": {
                    "status": "passed",
                    "score": 1.0,
                    "status_counts": {"passed": 1},
                    "evaluated_task_count": 1,
                    "valid_evaluated_task_count": 1,
                },
            },
        )


def _multiple_choice_task() -> Task:
    return Task(
        id="T_001",
        category="logic",
        type="multiple_choice",
        question="Pick A",
        source="logic.json",
        choices=["A", "B"],
        answer=["A"],
    )


def _external_task(task_id: str) -> Task:
    return Task(
        id=task_id,
        category="Benchmarks",
        type="external_benchmark",
        question="Run benchmark",
        source="public_benchmarks.json",
        benchmark={
            "name": f"ExampleBench {task_id}",
            "group": "Benchmarks",
            "homepage": "https://example.com",
            "license": "MIT",
            "credit": "Example authors",
            "citation": "https://example.com/citation",
            "docker": {"image": "example:local", "command": "agent-bench-probe"},
        },
    )
