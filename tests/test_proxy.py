import io
import json
from subprocess import CompletedProcess

from agent_bench.proxy import JsonlRecorder, OpenAIProxyConfig, OpenAIRecordingProxy, _CURL_STATUS_MARKER


def test_openai_recording_proxy_forwards_and_records_redacted_jsonl(tmp_path):
    raw_path = tmp_path / "raw_responses.jsonl"
    with JsonlRecorder(raw_path) as recorder:
        proxy = OpenAIRecordingProxy(
            OpenAIProxyConfig(
                upstream_base_url="http://upstream.test/v1",
                model="example-model",
                api_key="secret-key",
                tool_parser="openai-native",
            ),
            recorder,
        )
        fake_client = _FakeClient()
        proxy._client = fake_client
        handler = _FakeHandler(
            path="/v1/chat/completions",
            headers={"Content-Length": "106", "X-Agent-Bench-Benchmark-Id": "bench_1"},
            body={
                "model": "example-model",
                "messages": [{"role": "user", "content": "hello"}],
                "api_key": "request-secret",
            },
        )

        proxy._handle_post(handler)

    assert handler.status == 200
    assert json.loads(handler.wfile.getvalue().decode("utf-8"))["choices"][0]["message"]["content"] == "ok"
    assert fake_client.headers["Authorization"] == "Bearer secret-key"
    assert fake_client.url == "http://upstream.test/v1/chat/completions"
    records = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines()]
    assert records[0]["benchmark_id"] == "bench_1"
    assert records[0]["request"]["api_key"] == "<redacted>"
    assert records[0]["raw_response"]["choices"][0]["message"]["content"] == "ok"


def test_openai_recording_proxy_uses_curl_fallback(tmp_path, monkeypatch):
    raw_path = tmp_path / "raw_responses.jsonl"

    def fake_run(command, *, input, capture_output, check):
        assert command[:3] == ["curl", "-sS", "-m"]
        assert b'"model": "example-model"' in input
        body = json.dumps(
            {
                "choices": [{"message": {"role": "assistant", "content": "curl-ok"}}],
                "usage": {"completion_tokens": 1},
            }
        ).encode("utf-8")
        return CompletedProcess(
            command,
            0,
            body + f"{_CURL_STATUS_MARKER}200 application/json".encode("utf-8"),
            b"",
        )

    monkeypatch.setattr("agent_bench.proxy.shutil.which", lambda name: "/usr/bin/curl")
    monkeypatch.setattr("agent_bench.proxy.subprocess.run", fake_run)

    with JsonlRecorder(raw_path) as recorder:
        proxy = OpenAIRecordingProxy(
            OpenAIProxyConfig(
                upstream_base_url="http://upstream.test/v1",
                model="example-model",
            ),
            recorder,
        )
        proxy._client = _FailingClient()
        handler = _FakeHandler(
            path="/v1/chat/completions",
            headers={"Content-Length": "84", "X-Agent-Bench-Benchmark-Id": "bench_1"},
            body={"model": "example-model", "messages": [{"role": "user", "content": "hello"}]},
        )

        proxy._handle_post(handler)

    assert handler.status == 200
    assert json.loads(handler.wfile.getvalue().decode("utf-8"))["choices"][0]["message"]["content"] == "curl-ok"
    records = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines()]
    assert records[0]["benchmark_id"] == "bench_1"
    assert records[0]["raw_response"]["choices"][0]["message"]["content"] == "curl-ok"


def test_openai_recording_proxy_uses_container_ip_for_containerized_runs(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_BENCH_CONTAINERIZED", "1")
    monkeypatch.setattr("agent_bench.proxy._container_ip_address", lambda: "172.18.0.5")

    with JsonlRecorder(tmp_path / "raw_responses.jsonl") as recorder:
        proxy = OpenAIRecordingProxy(
            OpenAIProxyConfig(upstream_base_url="http://upstream.test/v1", model="example-model"),
            recorder,
        )
        proxy._server = _FakeServer()
        assert proxy.base_url == "http://127.0.0.1:43210/v1"
        assert proxy.container_base_url == "http://172.18.0.5:43210/v1"


class _FakeResponse:
    status_code = 200
    headers = {"content-type": "application/json"}

    def __init__(self) -> None:
        self.text = json.dumps(
            {
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                "usage": {"completion_tokens": 1},
            }
        )
        self.content = self.text.encode("utf-8")


class _FakeServer:
    server_address = ("0.0.0.0", 43210)


class _FakeClient:
    def __init__(self) -> None:
        self.url = ""
        self.payload = {}
        self.headers = {}

    def post(self, url, json, headers):
        self.url = url
        self.payload = json
        self.headers = headers
        return _FakeResponse()

    def close(self):
        return None


class _FailingClient:
    def post(self, url, json, headers):
        raise OSError("No route to host")

    def close(self):
        return None


class _FakeHandler:
    def __init__(self, *, path: str, headers: dict[str, str], body: dict) -> None:
        raw = json.dumps(body).encode("utf-8")
        self.path = path
        self.headers = dict(headers)
        self.headers["Content-Length"] = str(len(raw))
        self.rfile = io.BytesIO(raw)
        self.wfile = io.BytesIO()
        self.status = 0
        self.response_headers: list[tuple[str, str]] = []

    def send_response(self, status: int) -> None:
        self.status = status

    def send_header(self, key: str, value: str) -> None:
        self.response_headers.append((key, value))

    def end_headers(self) -> None:
        return None
