import asyncio
import io
import json

from agent_bench.models import ModelResponse, Task
from agent_bench.runner import _process_task


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
