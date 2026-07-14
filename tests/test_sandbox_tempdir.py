import io

from agent_bench import sandbox


def test_docker_temp_root_uses_configured_directory(monkeypatch, tmp_path):
    configured = tmp_path / "sandboxes"
    monkeypatch.setenv("AGENT_BENCH_SANDBOX_TMPDIR", str(configured))

    assert sandbox._docker_temp_root() == str(configured)
    assert configured.is_dir()


def test_sandbox_output_reader_keeps_only_bounded_tail():
    output = io.BytesIO(b"prefix" + b"x" * 100 + b"tail")

    assert sandbox._read_output_tail(output, limit=8) == "xxxxtail"
