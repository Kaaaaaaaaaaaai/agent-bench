import json
import shutil
from pathlib import Path

import pytest

from agent_bench.tasks import TaskLoadError, load_task_registry, load_tasks


REPO_TASKS_DIR = Path(__file__).resolve().parents[1] / "tasks"
ACTIVE_BENCHMARKS = [
    "SWE-bench Verified",
    "codeneedle",
    "StockBench",
    "InvestorBench",
    "QuantCode-Bench",
    "FinMCP-Bench",
    "Finance Agent v2",
    "FinanceMath",
    "Terminal-Bench 2.0",
    "NL2RepoBench",
    "DeepSWE",
    "ProgramBench",
    "MCP Atlas",
    "Toolathlon",
    "Humanity's Last Exam",
    "LongBench",
    "BigCodeBench",
    "MathArena",
    "CLAW-Eval",
    "MMLU-Pro",
    "AIME25 (no tools)",
    "HMMT Feb25 (no tools)",
    "HMMT Feb25 (with tools)",
    "GPQA Diamond (no tools)",
    "GPQA Diamond (with tools)",
    "LiveCodeBench v5 (2024-07 to 2024-12)",
    "SciCode (subtask)",
    "Terminal-Bench Hard (NVIDIA 48-task subset)",
    "TauBench V2 Airline",
    "TauBench V2 Retail",
    "TauBench V2 Telecom",
    "IFBench (prompt)",
    "Scale AI MultiChallenge",
    "Arena-Hard-V2",
    "AA-LCR",
    "RULER 256k",
    "RULER 512k",
    "RULER 1M",
    "MMLU-ProX",
    "WMT24++ (en-to-xx)",
]

def _manifest_payload(name: str, display_order: int = 1) -> dict:
    return {
        "display_name": name,
        "task_group": "Coding",
        "description": f"Run {name}.",
        "homepage_url": "https://example.com",
        "source": {
            "repository_url": "https://example.com/repo.git",
            "commit": "0123456789abcdef0123456789abcdef01234567",
        },
        "license": "MIT",
        "credit": "Example authors",
        "citation": "https://example.com/cite",
        "official_conditions": {
            "official_split": "test",
            "official_scoring_method": "official score",
            "official_prompt_format": "official prompt",
            "official_grader_command": "bash /benchmark/task/harness/run.sh",
            "official_evaluation_config": f"tasks/{name.lower()}/configs/official.json",
        },
        "assets": [
            {
                "source": "https://example.com/data.jsonl",
                "revision": "0123456789abcdef0123456789abcdef01234567",
                "checksum": "sha256:abc",
                "expected_local_path": ".",
            }
        ],
        "container": {
            "image": "agent-bench-external:python3.12",
            "command": "bash /benchmark/task/harness/run.sh",
        },
        "adapter": {
            "entry_point": "harness/run.sh",
            "expected_output_files": ["agent_bench_result.json"],
            "result_parser": "agent_bench_result_json",
        },
        "scoring": {"raw_score_field": "score", "max_score": 1.0},
        "reporting": {"category_label": "Coding", "display_order": display_order},
        "capabilities": ["chat_answer"],
    }


def test_load_tasks_discovers_json_files_and_derives_category(tmp_path):
    task_dir = tmp_path / "tasks"
    task_dir.mkdir()
    (task_dir / "extra_reasoning.json").write_text(
        json.dumps(
            [
                {
                    "id": "EX_001",
                    "type": "multiple_choice",
                    "question": "Pick A",
                    "choices": ["yes", "no"],
                    "answer": ["A"],
                }
            ]
        ),
        encoding="utf-8",
    )

    tasks = load_tasks(task_dir)

    assert len(tasks) == 1
    assert tasks[0].id == "EX_001"
    assert tasks[0].category == "extra_reasoning"
    assert tasks[0].source == "extra_reasoning.json"


def test_load_tasks_filters_by_category_and_limit(tmp_path):
    task_dir = tmp_path / "tasks"
    task_dir.mkdir()
    for name in ("alpha", "beta"):
        (task_dir / f"{name}.json").write_text(
            json.dumps(
                [
                    {
                        "id": f"{name}_001",
                        "type": "multiple_choice",
                        "question": "Pick A",
                        "choices": ["yes", "no"],
                        "answer": ["A"],
                    }
                ]
            ),
            encoding="utf-8",
        )

    tasks = load_tasks(task_dir, include={"beta"}, limit=1)

    assert [task.category for task in tasks] == ["beta"]


def test_load_tasks_rejects_missing_required_fields(tmp_path):
    task_dir = tmp_path / "tasks"
    task_dir.mkdir()
    (task_dir / "bad.json").write_text(json.dumps([{"id": "BAD"}]), encoding="utf-8")

    with pytest.raises(TaskLoadError):
        load_tasks(task_dir)


def test_load_tasks_supports_text_recall(tmp_path):
    task_dir = tmp_path / "tasks"
    task_dir.mkdir()
    ref_dir = task_dir / "ref"
    ref_dir.mkdir()
    (ref_dir / "example.py").write_text("def target(value):\n    return value\n", encoding="utf-8")
    (task_dir / "code_recall.json").write_text(
        json.dumps(
            [
                {
                    "id": "CR_001",
                    "type": "text_recall",
                    "question": "Recall the requested function header.",
                    "expected_text": "def target(value):\n    return value",
                    "reference_path": "ref/example.py",
                }
            ]
        ),
        encoding="utf-8",
    )

    tasks = load_tasks(task_dir)

    assert len(tasks) == 1
    assert tasks[0].is_text_recall is True
    assert tasks[0].expected_text == "def target(value):\n    return value"
    assert tasks[0].reference_path == "ref/example.py"
    assert tasks[0].reference_text == "def target(value):\n    return value\n"




def test_bundled_benchmark_folders_are_external_tasks_with_credits():
    legacy_tasks = json.loads((REPO_TASKS_DIR / "public_benchmarks.json").read_text(encoding="utf-8"))
    registry = load_task_registry(REPO_TASKS_DIR)
    loaded = load_tasks(REPO_TASKS_DIR)
    loaded_ids = [task.id for task in loaded]

    assert len(legacy_tasks) == 40
    assert [task["benchmark"]["name"] for task in legacy_tasks] == ACTIVE_BENCHMARKS
    assert len(registry) == 40
    assert len(loaded) == 40
    assert loaded_ids == ACTIVE_BENCHMARKS
    assert [task.benchmark["name"] for task in loaded] == ACTIVE_BENCHMARKS
    assert all(task.id == task.benchmark["name"] for task in loaded)
    assert all("id" not in task for task in legacy_tasks)
    assert all(Path(task.source).name == "manifest.json" for task in loaded)
    assert all(task["type"] == "external_benchmark" for task in legacy_tasks)
    assert all(task.benchmark.get("license") for task in loaded)
    assert all(task.benchmark.get("credit") for task in loaded)
    assert all(task.benchmark.get("citation") for task in loaded)
    assert "public_benchmarks" not in {task.category for task in loaded}
    assert {task.category for task in loaded} == {
        "Agentic",
        "Chat",
        "Coding",
        "Finance",
        "General Knowledge",
        "Instruction Following",
        "Knowledge",
        "Long Context",
        "Math",
        "Multilingual",
        "Reasoning",
        "Terminal",
        "Tool Use",
    }

    for manifest_path in REPO_TASKS_DIR.glob("*/manifest.json"):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert "id" not in manifest
        task_dir = manifest_path.parent
        asset_lock = json.loads((task_dir / "assets.lock.json").read_text(encoding="utf-8"))
        official_config = json.loads(
            (task_dir / "configs" / "official.json").read_text(encoding="utf-8")
        )
        assert asset_lock["benchmark_name"] == manifest["display_name"]
        assert official_config["benchmark_name"] == manifest["display_name"]


def test_full_active_profile_selects_all_current_public_benchmarks():
    loaded = load_tasks(REPO_TASKS_DIR, profile="full_active")

    assert len(loaded) == 40
    assert [task.id for task in loaded] == ACTIVE_BENCHMARKS


def test_legacy_descriptor_flag_does_not_reparse_manifest_duplicates(monkeypatch):
    monkeypatch.setenv("AGENT_BENCH_ENABLE_LEGACY_PUBLIC_BENCHMARKS", "1")

    loaded = load_task_registry(REPO_TASKS_DIR)

    assert [task.id for task in loaded] == ACTIVE_BENCHMARKS


def test_nemotron_model_card_leaf_conditions_match_registered_tasks():
    descriptor = json.loads(
        (REPO_TASKS_DIR / "_metadata" / "nemotron-3-puzzle-benchmarks.json").read_text(
            encoding="utf-8"
        )
    )
    registered = {task.id for task in load_task_registry(REPO_TASKS_DIR)}
    expected = set(descriptor["tasks"])

    assert descriptor["source"]["revision"] == "1d370e47fbc56d1019a471c2339663cdbbb5236f"
    assert len(descriptor["tasks"]) == 22
    assert len(expected) == 22
    assert expected <= registered


def test_nemotron_leaf_assets_use_original_benchmark_publishers():
    manifests = {
        payload["display_name"]: payload
        for path in REPO_TASKS_DIR.glob("*/manifest.json")
        for payload in [json.loads(path.read_text(encoding="utf-8"))]
    }
    expected_sources = {
        "MMLU-Pro": ("https://github.com/TIGER-AI-Lab/MMLU-Pro.git", "TIGER-Lab/MMLU-Pro"),
        "AIME25 (no tools)": ("https://github.com/eth-sri/matharena.git", "MathArena/aime_2025"),
        "HMMT Feb25 (no tools)": ("https://github.com/eth-sri/matharena.git", "MathArena/hmmt_feb_2025"),
        "HMMT Feb25 (with tools)": ("https://github.com/eth-sri/matharena.git", "MathArena/hmmt_feb_2025"),
        "GPQA Diamond (no tools)": ("https://github.com/idavidrein/gpqa.git", "Idavidrein/gpqa"),
        "GPQA Diamond (with tools)": ("https://github.com/idavidrein/gpqa.git", "Idavidrein/gpqa"),
        "Scale AI MultiChallenge": ("", "ScaleAI/MultiChallenge"),
        "AA-LCR": ("", "ArtificialAnalysis/AA-LCR"),
        "MMLU-ProX": ("https://github.com/weihao1115/MMLU-ProX.git", "li-lab/MMLU-ProX"),
        "WMT24++ (en-to-xx)": ("https://github.com/google-research/mt-metrics-eval.git", "google/wmt24pp"),
    }

    for benchmark_name, (repository, dataset_id) in expected_sources.items():
        manifest = manifests[benchmark_name]
        assert manifest["source"]["repository_url"] == repository
        assert manifest["source"]["dataset_id"] == dataset_id
        assert all(
            "github.com/NVIDIA-NeMo/" not in asset["source"]
            for asset in manifest["assets"]
        )

    source_files = [
        path
        for task_dir in REPO_TASKS_DIR.glob("nemotron-*")
        for path in (task_dir / "manifest.json", task_dir / "assets.lock.json")
        if path.is_file()
    ]
    assert all(
        "github.com/NVIDIA-NeMo/Evaluator" not in path.read_text(encoding="utf-8")
        and "github.com/NVIDIA-NeMo/Skills" not in path.read_text(encoding="utf-8")
        for path in source_files
    )


def test_nemotron_source_adapter_harnesses_cannot_claim_official_scores():
    manifests = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in REPO_TASKS_DIR.glob("*/manifest.json")
    ]
    nemo_manifests = [
        manifest
        for manifest in manifests
        if str(manifest.get("official_conditions", {}).get("official_grader_command", ""))
        .startswith("nemo-evaluator-launcher ")
    ]

    assert len(nemo_manifests) == 22
    for manifest in nemo_manifests:
        task_dir = next(
            path.parent
            for path in REPO_TASKS_DIR.glob("*/manifest.json")
            if json.loads(path.read_text(encoding="utf-8"))["display_name"]
            == manifest["display_name"]
        )
        harness = (task_dir / "harness" / "run.sh").read_text(encoding="utf-8")
        assert "export AGENT_BENCH_EVALUATION_CONTRACT=source_adapter_smoke" in harness


def test_load_tasks_discovers_benchmark_folders_without_central_registry(tmp_path):
    task_dir = tmp_path / "tasks"
    first = task_dir / "example_bench"
    second = task_dir / "second_bench"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    (first / "manifest.json").write_text(json.dumps(_manifest_payload("ExampleBench", 2)), encoding="utf-8")
    (second / "manifest.json").write_text(json.dumps(_manifest_payload("SecondBench", 1)), encoding="utf-8")
    (task_dir / "public_benchmarks.json").write_text(json.dumps([{"id": "BROKEN"}]), encoding="utf-8")

    loaded = load_tasks(task_dir)

    assert [task.id for task in loaded] == ["SecondBench", "ExampleBench"]
    assert {Path(task.source).parent.name for task in loaded} == {"example_bench", "second_bench"}

    shutil.rmtree(second)
    loaded_after_remove = load_tasks(task_dir)

    assert [task.id for task in loaded_after_remove] == ["ExampleBench"]


def test_core_runner_execution_code_does_not_branch_on_bundled_benchmark_names():
    repo_root = Path(__file__).resolve().parents[1]
    core_files = [
        repo_root / "agent_bench" / "external.py",
        repo_root / "agent_bench" / "manifest.py",
        repo_root / "agent_bench" / "runner.py",
        repo_root / "agent_bench" / "tasks.py",
    ]
    forbidden = {item.lower() for item in ACTIVE_BENCHMARKS + ["FinToolBench"]}
    hits = []
    for path in core_files:
        text = path.read_text(encoding="utf-8").lower()
        hits.extend(f"{path.name}:{name}" for name in sorted(forbidden) if name in text)

    assert hits == []


def test_finmcp_descriptor_is_static_not_live_tool_call():
    tasks = json.loads((REPO_TASKS_DIR / "public_benchmarks.json").read_text(encoding="utf-8"))
    descriptor = next(task for task in tasks if task["benchmark"]["name"] == "FinMCP-Bench")

    assert "tool_call" not in descriptor["benchmark"]["capabilities"]
    assert descriptor["benchmark"]["adapter"] == "static_transcript_reasoning"


def test_load_tasks_supports_external_benchmark(tmp_path):
    task_dir = tmp_path / "tasks"
    task_dir.mkdir()
    (task_dir / "benchmarks.json").write_text(
        json.dumps(
            [
                {
                    "type": "external_benchmark",
                    "question": "Run benchmark",
                    "benchmark": {
                        "name": "ExampleBench",
                        "group": "Coding",
                        "homepage": "https://example.com",
                        "repository": "https://example.com/repo.git",
                        "license": "MIT",
                        "credit": "Example authors",
                        "docker": {"image": "example", "command": "echo ok"},
                    },
                }
            ]
        ),
        encoding="utf-8",
    )

    tasks = load_tasks(task_dir)

    assert len(tasks) == 1
    assert tasks[0].is_external_benchmark is True
    assert tasks[0].category == "Coding"
    assert tasks[0].benchmark["name"] == "ExampleBench"


def test_load_tasks_keeps_non_mit_license_as_metadata(tmp_path):
    task_dir = tmp_path / "tasks"
    task_dir.mkdir()
    (task_dir / "benchmarks.json").write_text(
        json.dumps(
            [
                {
                    "type": "external_benchmark",
                    "question": "Run benchmark",
                    "benchmark": {
                        "name": "ExampleBench",
                        "homepage": "https://example.com",
                        "license": "Apache-2.0",
                        "credit": "Example authors",
                        "docker": {"image": "example", "command": "echo ok"},
                    },
                }
            ]
        ),
        encoding="utf-8",
    )

    tasks = load_tasks(task_dir)

    assert tasks[0].benchmark["license"] == "Apache-2.0"
