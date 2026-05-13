from agent_bench import sandbox


def test_docker_temp_root_uses_configured_directory(monkeypatch, tmp_path):
    configured = tmp_path / "sandboxes"
    monkeypatch.setenv("AGENT_BENCH_SANDBOX_TMPDIR", str(configured))

    assert sandbox._docker_temp_root() == str(configured)
    assert configured.is_dir()
