import json

from agent_bench.manifest import BenchmarkManifest, load_manifest_tasks
from agent_bench.models import Task


def _manifest_payload() -> dict:
    return {
        "id": "official_example",
        "display_name": "Official Example",
        "task_group": "Reasoning",
        "description": "Run the official example benchmark.",
        "homepage_url": "https://example.com",
        "source": {
            "repository_url": "https://example.com/bench.git",
            "commit": "0123456789abcdef0123456789abcdef01234567",
        },
        "license": "MIT",
        "credit": "Example authors",
        "citation": "https://example.com/cite",
        "official_conditions": {
            "official_split": "test",
            "official_scoring_method": "official exact score",
            "official_prompt_format": "official prompt format",
            "official_grader_command": "bench grade",
            "official_evaluation_config": "official.yaml",
        },
        "assets": [
            {
                "source": "https://example.com/data.jsonl",
                "revision": "0123456789abcdef0123456789abcdef01234567",
                "checksum": "sha256:abc",
                "expected_local_path": "official_example/data.jsonl",
            }
        ],
        "container": {
            "image": "official-example:1",
            "command": "bench run --official",
            "timeout_seconds": 60,
        },
        "adapter": {
            "module": "official_example.adapter",
            "expected_output_files": ["agent_bench_result.json"],
            "result_parser": "agent_bench_result_json",
        },
        "scoring": {
            "raw_score_field": "score",
            "max_score": 1.0,
            "direction": "higher_is_better",
        },
        "reporting": {
            "category_label": "Reasoning",
            "display_order": 1,
        },
    }


def test_manifest_validation_requires_pinned_official_metadata():
    manifest = BenchmarkManifest.from_mapping(_manifest_payload(), source_path="benchmarks/example/manifest.yaml")

    result = manifest.validate()

    assert result.ok is True


def test_manifest_validation_accepts_declared_host_docker_socket_without_cli_flag():
    payload = _manifest_payload()
    payload["container"]["requires_host_docker_socket"] = True
    manifest = BenchmarkManifest.from_mapping(payload, source_path="benchmarks/example/manifest.yaml")

    result = manifest.validate(allow_host_docker_socket=False)

    assert result.ok is True


def test_legacy_descriptor_without_official_fields_fails_validation():
    task = Task(
        id="PB_001",
        category="Coding",
        type="external_benchmark",
        question="Run legacy descriptor",
        source="public_benchmarks.json",
        benchmark={
            "name": "LegacyBench",
            "homepage": "https://example.com",
            "license": "MIT",
            "credit": "Legacy authors",
            "repository": "https://example.com/legacy.git",
            "ref": "main",
            "docker": {"image": "legacy:latest", "command": "legacy run"},
        },
    )

    result = BenchmarkManifest.from_task(task).validate()

    assert result.ok is False
    assert any(issue.field == "source.commit" for issue in result.issues)
    assert any(issue.field == "official_conditions.official_split" for issue in result.issues)


def test_load_manifest_tasks_discovers_benchmark_manifests(tmp_path):
    manifest_dir = tmp_path / "benchmarks" / "official_example"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "manifest.yaml").write_text(json.dumps(_manifest_payload()), encoding="utf-8")

    tasks = load_manifest_tasks(tmp_path / "benchmarks")

    assert len(tasks) == 1
    assert tasks[0].id == "official_example"
    assert tasks[0].benchmark["manifest"]["display_name"] == "Official Example"
