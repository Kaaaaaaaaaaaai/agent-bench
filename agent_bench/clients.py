import asyncio
import json
import os
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any

import httpx

from agent_bench.models import ModelResponse, Task
from agent_bench.prompts import build_messages


class ModelClient(ABC):
    model: str

    @abstractmethod
    async def complete(self, task: Task) -> ModelResponse:
        raise NotImplementedError

    async def aclose(self) -> None:
        return None


class OpenAICompatibleClient(ModelClient):
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key_env: str | None = None,
        timeout: float = 60.0,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        json_mode: str = "auto",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.json_mode = json_mode
        headers: dict[str, str] = {}
        if api_key_env:
            api_key = os.environ.get(api_key_env)
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.AsyncClient(timeout=timeout, headers=headers)

    async def complete(self, task: Task) -> ModelResponse:
        started = time.perf_counter()
        payload = {
            "model": self.model,
            "messages": build_messages(task),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if self.json_mode in {"auto", "on"}:
            payload["response_format"] = {"type": "json_object"}
        try:
            content, usage, time_to_first_token_seconds = await self._stream_completion(payload, started)
            latency_seconds = time.perf_counter() - started
            output_token_count = _completion_tokens_from_usage(usage)
            return ModelResponse(
                task_id=task.id,
                model=self.model,
                raw_response=content,
                latency_seconds=latency_seconds,
                time_to_first_token_seconds=time_to_first_token_seconds,
                tokens_per_second=_tokens_per_second(
                    output_token_count,
                    latency_seconds,
                    time_to_first_token_seconds,
                ),
                output_token_count=output_token_count,
                usage=usage,
            )
        except Exception as exc:
            return ModelResponse(
                task_id=task.id,
                model=self.model,
                raw_response="",
                latency_seconds=time.perf_counter() - started,
                error=str(exc),
            )

    async def _stream_completion(
        self,
        payload: dict[str, Any],
        started: float,
    ) -> tuple[str, dict[str, Any], float | None]:
        last_error: httpx.HTTPStatusError | None = None
        for variant in _openai_stream_payload_variants(payload, self.json_mode):
            try:
                return await self._stream_once(variant, started)
            except httpx.HTTPStatusError as exc:
                last_error = exc
                if exc.response.status_code not in {400, 404, 422}:
                    raise
        if last_error is not None:
            raise last_error
        return "", {}, None

    async def _stream_once(
        self,
        payload: dict[str, Any],
        started: float,
    ) -> tuple[str, dict[str, Any], float | None]:
        endpoint = f"{self.base_url}/chat/completions"
        fragments: list[str] = []
        usage: dict[str, Any] = {}
        time_to_first_token_seconds: float | None = None
        async with self._client.stream("POST", endpoint, json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                event = _parse_sse_json(line)
                if event is None:
                    continue
                usage_payload = event.get("usage")
                if isinstance(usage_payload, dict):
                    usage = usage_payload
                content = _extract_openai_stream_content(event)
                if content:
                    if time_to_first_token_seconds is None:
                        time_to_first_token_seconds = time.perf_counter() - started
                    fragments.append(content)
        return "".join(fragments), usage, time_to_first_token_seconds

    async def aclose(self) -> None:
        await self._client.aclose()


class OllamaNativeClient(ModelClient):
    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: float = 60.0,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = httpx.AsyncClient(timeout=timeout)

    async def complete(self, task: Task) -> ModelResponse:
        started = time.perf_counter()
        payload = {
            "model": self.model,
            "messages": build_messages(task),
            "stream": True,
            "format": "json",
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }
        try:
            fragments: list[str] = []
            usage: dict[str, Any] = {}
            time_to_first_token_seconds: float | None = None
            async with self._client.stream("POST", f"{self.base_url}/api/chat", json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    message = data.get("message", {})
                    content = ""
                    if isinstance(message, dict):
                        content = message.get("content") or ""
                    if not content:
                        content = data.get("response") or ""
                    if isinstance(content, str) and content:
                        if time_to_first_token_seconds is None:
                            time_to_first_token_seconds = time.perf_counter() - started
                        fragments.append(content)
                    usage.update({key: value for key, value in data.items() if key.endswith("_count")})
            latency_seconds = time.perf_counter() - started
            output_token_count = _completion_tokens_from_usage(usage)
            return ModelResponse(
                task_id=task.id,
                model=self.model,
                raw_response="".join(fragments),
                latency_seconds=latency_seconds,
                time_to_first_token_seconds=time_to_first_token_seconds,
                tokens_per_second=_tokens_per_second(
                    output_token_count,
                    latency_seconds,
                    time_to_first_token_seconds,
                ),
                output_token_count=output_token_count,
                usage=usage,
            )
        except Exception as exc:
            return ModelResponse(
                task_id=task.id,
                model=self.model,
                raw_response="",
                latency_seconds=time.perf_counter() - started,
                error=str(exc),
            )

    async def aclose(self) -> None:
        await self._client.aclose()


class MockClient(ModelClient):
    def __init__(self, model: str = "mock-perfect", latency_seconds: float = 0.0) -> None:
        self.model = model
        self.latency_seconds = latency_seconds

    async def complete(self, task: Task) -> ModelResponse:
        started = time.perf_counter()
        if self.latency_seconds:
            await asyncio.sleep(self.latency_seconds)
        if task.is_multiple_choice:
            payload: dict[str, Any] = {"answer": task.answer, "confidence": 1.0}
        elif task.is_coding:
            payload = {"code": _mock_code_for_task(task), "confidence": 1.0}
        elif task.is_text_recall:
            payload = {"answer": task.expected_text or "", "confidence": 1.0}
        else:
            payload = {}
        return ModelResponse(
            task_id=task.id,
            model=self.model,
            raw_response=json.dumps(payload),
            latency_seconds=time.perf_counter() - started,
        )


def make_client(
    provider: str,
    base_url: str | None,
    model: str | None,
    api_key_env: str | None,
    timeout: float,
    temperature: float,
    max_tokens: int,
    json_mode: str,
) -> ModelClient:
    if provider == "mock":
        return MockClient(model or "mock-perfect")
    if not model:
        raise ValueError("--model is required for remote providers")
    if provider == "openai-compatible":
        return OpenAICompatibleClient(
            base_url=base_url or "http://localhost:8000/v1",
            model=model,
            api_key_env=api_key_env,
            timeout=timeout,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
        )
    if provider == "ollama-native":
        return OllamaNativeClient(
            base_url=base_url or "http://localhost:11434",
            model=model,
            timeout=timeout,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    raise ValueError(f"Unsupported provider: {provider}")


def _openai_stream_payload_variants(
    payload: dict[str, Any],
    json_mode: str,
) -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = [payload]
    if json_mode == "auto" and "response_format" in payload:
        without_response_format = dict(payload)
        without_response_format.pop("response_format", None)
        variants.append(without_response_format)
    if "stream_options" in payload:
        without_stream_options = dict(payload)
        without_stream_options.pop("stream_options", None)
        variants.append(without_stream_options)
    if json_mode == "auto" and "response_format" in payload and "stream_options" in payload:
        minimal_payload = dict(payload)
        minimal_payload.pop("response_format", None)
        minimal_payload.pop("stream_options", None)
        variants.append(minimal_payload)
    return variants


def _parse_sse_json(line: str) -> dict[str, Any] | None:
    text = line.strip()
    if not text or text.startswith(":"):
        return None
    if text.startswith("data:"):
        text = text[5:].strip()
    elif not text.startswith("{"):
        return None
    if text == "[DONE]":
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _extract_openai_stream_content(data: dict[str, Any]) -> str:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    delta = first.get("delta")
    if isinstance(delta, dict):
        content = delta.get("content")
        if isinstance(content, str):
            return content
    message = first.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content
    text = first.get("text")
    if isinstance(text, str):
        return text
    return ""


def _completion_tokens_from_usage(usage: dict[str, Any]) -> int | None:
    for key in ("completion_tokens", "output_tokens", "eval_count"):
        value = usage.get(key)
        if isinstance(value, int) and value >= 0:
            return value
    return None


def _tokens_per_second(
    output_token_count: int | None,
    latency_seconds: float,
    time_to_first_token_seconds: float | None,
) -> float | None:
    if output_token_count is None or output_token_count <= 0:
        return None
    generation_seconds = latency_seconds
    if time_to_first_token_seconds is not None:
        generation_seconds -= time_to_first_token_seconds
    if generation_seconds <= 0:
        return None
    return output_token_count / generation_seconds


def _mock_code_for_task(task: Task) -> str:
    if task.function_name in {"LRUCache", "MinStack", "Trie"}:
        return _mock_class_code(task.function_name)

    cases = OrderedDict()
    for case in task.test_cases:
        key = json.dumps(case["input"], sort_keys=True, separators=(",", ":"))
        cases[key] = case["output"]

    cases_json = json.dumps(cases)
    return (
        "import json\n"
        f"_CASES = json.loads({cases_json!r})\n\n"
        f"def {task.function_name}(**kwargs):\n"
        "    key = json.dumps(kwargs, sort_keys=True, separators=(',', ':'))\n"
        "    return _CASES.get(key)\n"
    )


def _mock_class_code(function_name: str | None) -> str:
    if function_name == "LRUCache":
        return """
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.values = OrderedDict()
    def get(self, key):
        if key not in self.values:
            return -1
        self.values.move_to_end(key)
        return self.values[key]
    def put(self, key, value):
        if key in self.values:
            self.values.move_to_end(key)
        self.values[key] = value
        if len(self.values) > self.capacity:
            self.values.popitem(last=False)
"""
    if function_name == "MinStack":
        return """
class MinStack:
    def __init__(self):
        self.stack = []
        self.mins = []
    def push(self, val):
        self.stack.append(val)
        if not self.mins or val <= self.mins[-1]:
            self.mins.append(val)
    def pop(self):
        value = self.stack.pop()
        if value == self.mins[-1]:
            self.mins.pop()
    def top(self):
        return self.stack[-1]
    def getMin(self):
        return self.mins[-1]
"""
    if function_name == "Trie":
        return """
class Trie:
    def __init__(self):
        self.root = {}
    def insert(self, word):
        node = self.root
        for char in word:
            node = node.setdefault(char, {})
        node['$'] = True
    def search(self, word):
        node = self.root
        for char in word:
            if char not in node:
                return False
            node = node[char]
        return '$' in node
    def startsWith(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node:
                return False
            node = node[char]
        return True
"""
    return ""
