from __future__ import annotations

import json
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from agent_bench.tool_parsers import parse_tool_calls


REDACTED = "<redacted>"
SECRET_KEYS = {
    "authorization",
    "api-key",
    "api_key",
    "access_token",
    "token",
    "password",
    "secret",
    "openai_api_key",
}
_CURL_STATUS_MARKER = "\n__AGENT_BENCH_CURL_STATUS__"


@dataclass(slots=True)
class JsonlRecorder:
    path: Path
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _handle: Any = field(default=None, init=False)

    def __enter__(self) -> "JsonlRecorder":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w", encoding="utf-8")
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def write(self, payload: dict[str, Any]) -> None:
        line = json.dumps(redact_secrets(payload), ensure_ascii=False, sort_keys=True) + "\n"
        with self._lock:
            if self._handle is None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(line)
                return
            self._handle.write(line)
            self._handle.flush()


@dataclass(slots=True)
class OpenAIProxyConfig:
    upstream_base_url: str
    model: str
    api_key: str = ""
    label: str = "target"
    timeout_seconds: float = 1800.0
    tool_parser: str = "auto"
    default_benchmark_id: str = ""


class OpenAIRecordingProxy:
    def __init__(self, config: OpenAIProxyConfig, recorder: JsonlRecorder) -> None:
        self.config = config
        self.recorder = recorder
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._client = httpx.Client(timeout=config.timeout_seconds)

    @property
    def base_url(self) -> str:
        if self._server is None:
            raise RuntimeError("Proxy has not been started")
        host, port = self._server.server_address
        if host in {"0.0.0.0", "::"}:
            host = "127.0.0.1"
        return f"http://{host}:{port}/v1"

    @property
    def container_base_url(self) -> str:
        if self._server is None:
            raise RuntimeError("Proxy has not been started")
        return f"http://host.docker.internal:{self._server.server_address[1]}/v1"

    def __enter__(self) -> "OpenAIRecordingProxy":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.stop()

    def start(self) -> None:
        if self._server is not None:
            return
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "AgentBenchOpenAIProxy/1.0"

            def do_POST(self) -> None:  # noqa: N802
                proxy._handle_post(self)

            def do_GET(self) -> None:  # noqa: N802
                proxy._handle_get(self)

            def log_message(self, format: str, *args: Any) -> None:
                return

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        server = self._server
        self._server = None
        if server is not None:
            server.shutdown()
            server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._client.close()

    def _handle_get(self, handler: BaseHTTPRequestHandler) -> None:
        if handler.path.rstrip("/") in {"/health", "/v1/health"}:
            self._send_json(handler, 200, {"ok": True, "proxy": self.config.label})
            return
        self._send_json(handler, 404, {"error": "not found"})

    def _handle_post(self, handler: BaseHTTPRequestHandler) -> None:
        request_id = handler.headers.get("X-Request-Id") or f"req_{uuid.uuid4().hex}"
        benchmark_id = (
            handler.headers.get("X-Agent-Bench-Benchmark-Id")
            or self.config.default_benchmark_id
            or "unknown"
        )
        task_id = handler.headers.get("X-Agent-Bench-Task-Id")
        started = time.perf_counter()
        request_body = self._read_json(handler)
        if request_body is None:
            self._record(
                request_id=request_id,
                benchmark_id=benchmark_id,
                task_id=task_id,
                request={},
                response=None,
                started=started,
                status_code=400,
                error="request body must be a JSON object",
            )
            self._send_json(handler, 400, {"error": {"message": "request body must be a JSON object"}})
            return

        target_url = _join_upstream_path(self.config.upstream_base_url, handler.path)
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        try:
            response = self._client.post(target_url, json=request_body, headers=headers)
            status_code = response.status_code
            content = response.content
            content_type = response.headers.get("content-type", "application/json")
        except Exception as exc:
            curl_response = self._post_with_curl(target_url, request_body, headers)
            if curl_response is not None:
                status_code, content, content_type = curl_response
            else:
                error_message = str(exc)
                if shutil.which("curl"):
                    error_message = f"{error_message}; curl fallback failed or returned no response"
                self._record(
                    request_id=request_id,
                    benchmark_id=benchmark_id,
                    task_id=task_id,
                    request=request_body,
                    response=None,
                    started=started,
                    status_code=502,
                    error=error_message,
                )
                self._send_json(handler, 502, {"error": {"message": error_message}})
                return

        text = content.decode("utf-8", errors="replace")
        response_payload = _json_or_text(text)
        parser_result = (
            parse_tool_calls(self.config.tool_parser, response_payload)
            if isinstance(response_payload, dict)
            else None
        )
        self._record(
            request_id=request_id,
            benchmark_id=benchmark_id,
            task_id=task_id,
            request=request_body,
            response=response_payload,
            started=started,
            status_code=status_code,
            parser=parser_result.to_dict() if parser_result is not None else {},
        )
        handler.send_response(status_code)
        handler.send_header("Content-Type", content_type)
        handler.send_header("Content-Length", str(len(content)))
        handler.end_headers()
        handler.wfile.write(content)

    def _post_with_curl(
        self,
        target_url: str,
        request_body: dict[str, Any],
        headers: dict[str, str],
    ) -> tuple[int, bytes, str] | None:
        if shutil.which("curl") is None:
            return None
        command = [
            "curl",
            "-sS",
            "-m",
            str(max(1, int(self.config.timeout_seconds))),
            "-X",
            "POST",
            target_url,
            "-H",
            "Content-Type: application/json",
            "--data-binary",
            "@-",
            "-w",
            f"{_CURL_STATUS_MARKER}%{{http_code}} %{{content_type}}",
        ]
        if "Authorization" in headers:
            command.extend(["-H", f"Authorization: {headers['Authorization']}"])
        completed = subprocess.run(
            command,
            input=json.dumps(request_body).encode("utf-8"),
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            return None
        body, marker, trailer = completed.stdout.rpartition(_CURL_STATUS_MARKER.encode("utf-8"))
        if not marker:
            return None
        status_text, _, content_type = trailer.partition(b" ")
        try:
            status_code = int(status_text.decode("ascii"))
        except ValueError:
            return None
        return status_code, body, content_type.decode("utf-8", errors="replace") or "application/json"

    def _read_json(self, handler: BaseHTTPRequestHandler) -> dict[str, Any] | None:
        try:
            length = int(handler.headers.get("Content-Length") or "0")
        except ValueError:
            return None
        raw = handler.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
        return payload if isinstance(payload, dict) else None

    def _record(
        self,
        *,
        request_id: str,
        benchmark_id: str,
        task_id: str | None,
        request: dict[str, Any],
        response: dict[str, Any] | str | None,
        started: float,
        status_code: int,
        parser: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        usage = response.get("usage") if isinstance(response, dict) and isinstance(response.get("usage"), dict) else {}
        self.recorder.write(
            {
                "record_type": "model_proxy_response",
                "proxy": self.config.label,
                "benchmark_id": benchmark_id,
                "task_id": task_id,
                "request_id": request_id,
                "timestamp": datetime.now(UTC).isoformat(),
                "target_model": self.config.model,
                "status_code": status_code,
                "request": _redacted_request_metadata(request),
                "raw_response": response,
                "latency_seconds": time.perf_counter() - started,
                "usage": usage,
                "parser": parser or {},
                "error": error,
            }
        )

    def _send_json(self, handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)


def redact_secrets(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            lowered = str(key).lower()
            if any(secret in lowered for secret in SECRET_KEYS):
                redacted[key] = REDACTED
            else:
                redacted[key] = redact_secrets(item)
        return redacted
    if isinstance(value, list):
        return [redact_secrets(item) for item in value]
    if isinstance(value, str) and value.startswith("Bearer "):
        return REDACTED
    return value


def redact_url(url: str) -> str:
    if not url:
        return ""
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return url
    host = parsed.hostname or parsed.netloc
    port = f":{parsed.port}" if parsed.port else ""
    return f"{parsed.scheme}://{host}{port}{parsed.path.rstrip('/')}"


def _redacted_request_metadata(request: dict[str, Any]) -> dict[str, Any]:
    payload = redact_secrets(request)
    messages = payload.get("messages")
    if isinstance(messages, list):
        payload["message_count"] = len(messages)
    return payload


def _json_or_text(text: str) -> dict[str, Any] | str:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return text
    return payload if isinstance(payload, dict) else text


def _join_upstream_path(base_url: str, request_path: str) -> str:
    base = base_url.rstrip("/")
    path = request_path.split("?", 1)[0]
    if base.endswith("/v1") and path.startswith("/v1/"):
        return f"{base}{path[3:]}"
    return f"{base}{path}"
