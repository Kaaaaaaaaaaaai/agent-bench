from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from agent_bench.models import Task


MANIFEST_VERSION = "agent-bench.manifest.v1"
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$", re.IGNORECASE)
MOVING_REFS = {"", "main", "master", "latest", "head", "dev", "develop", "trunk"}


@dataclass(slots=True)
class ValidationIssue:
    field: str
    message: str
    suggestion: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ValidationResult:
    ok: bool
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def error_message(self) -> str:
        return "; ".join(f"{issue.field}: {issue.message}" for issue in self.issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "issues": [issue.to_dict() for issue in self.issues],
        }


@dataclass(slots=True)
class SourceSpec:
    repository_url: str = ""
    commit: str = ""
    dataset_id: str = ""
    dataset_revision: str = ""
    subdir: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AssetSpec:
    source: str
    expected_local_path: str
    revision: str = ""
    checksum: str = ""
    required: bool = True
    validation_command: str = ""
    validation_rules: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ContainerSpec:
    image: str = ""
    build_context: str = ""
    dockerfile: str = ""
    command: str = ""
    environment_allowed: list[str] = field(default_factory=list)
    mounts: list[dict[str, Any]] = field(default_factory=list)
    network: str = "model_proxy"
    cpus: float | None = None
    memory: str = ""
    pids_limit: int | None = 1024
    timeout_seconds: float = 21600.0
    requires_nested_docker: bool = False
    requires_host_docker_socket: bool = False
    run_as_user: str = "10001:10001"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AdapterSpec:
    module: str = ""
    entry_point: str = ""
    expected_output_files: list[str] = field(default_factory=list)
    result_parser: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ScoringSpec:
    raw_score_field: str = "score"
    max_score: float | None = None
    normalization: str = "fraction_to_0_100"
    direction: str = "higher_is_better"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ReportingSpec:
    category_label: str = "Other"
    display_order: int = 1000
    credit: str = ""
    citation: str = ""
    license: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class OfficialConditions:
    official_split: str = ""
    official_scoring_method: str = ""
    official_prompt_format: str = ""
    official_grader_command: str = ""
    official_evaluation_config: str = ""
    hardware_requirements: str = ""
    container_requirements: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ModelMetadata:
    provider_type: str = "openai-compatible"
    base_url: str = ""
    model: str = ""
    temperature: float = 0.0
    top_p: float | None = None
    max_tokens: int | None = None
    seed: int | None = None
    request_timeout_seconds: float | None = None
    max_retries: int | None = None
    concurrency: int | None = None
    tool_parser: str = "auto"
    context_window: int | None = None
    stop: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class JudgeMetadata:
    provider: str = "none"
    base_url: str = ""
    model: str = ""
    temperature: float = 0.0
    timeout_seconds: float | None = None
    max_retries: int | None = None
    prompt_version: str = ""
    fallback_used: bool = False
    fallback_policy: str = "fail"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class BenchmarkRunContext:
    run_id: str
    output_dir: Path
    asset_root: Path
    manifest: "BenchmarkManifest"
    model: ModelMetadata
    judge: JudgeMetadata
    allow_host_docker_socket: bool = False


@dataclass(slots=True)
class BenchmarkRunResult:
    benchmark_id: str
    status: str
    output_dir: Path
    duration_seconds: float
    exit_code: int | None = None
    stdout_tail: str = ""
    stderr_tail: str = ""
    error: str = ""
    artifacts: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["output_dir"] = str(self.output_dir)
        return payload


@dataclass(slots=True)
class NormalizedBenchmarkResult:
    benchmark_id: str
    display_name: str
    task_group: str
    status: str
    raw_score: float | None
    normalized_score: float | None
    included_in_official_score: bool
    duration_seconds: float
    error: str = ""
    artifacts: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RawResponseRecord:
    benchmark_id: str
    task_id: str | None
    request_id: str
    timestamp: str
    target_model: str
    request: dict[str, Any]
    response: dict[str, Any] | str | None
    latency_seconds: float
    usage: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    parser: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class GradedResultRecord:
    benchmark_id: str
    task_id: str | None
    raw_score: float | None
    normalized_score: float | None
    status: str
    grader_metadata: dict[str, Any] = field(default_factory=dict)
    judge_metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SummaryReport:
    run_id: str
    timestamp: str
    target_model: ModelMetadata
    judge: JudgeMetadata
    benchmarks: list[NormalizedBenchmarkResult]
    coverage: dict[str, Any]
    category_scores: dict[str, float]
    overall_score: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "target_model": self.target_model.to_dict(),
            "judge": self.judge.to_dict(),
            "benchmarks": [benchmark.to_dict() for benchmark in self.benchmarks],
            "coverage": self.coverage,
            "category_scores": self.category_scores,
            "overall_score": self.overall_score,
        }


@dataclass(slots=True)
class BenchmarkManifest:
    id: str
    display_name: str
    task_group: str
    description: str
    homepage_url: str
    source: SourceSpec
    license: str
    credit: str
    citation: str
    official_conditions: OfficialConditions
    assets: list[AssetSpec]
    container: ContainerSpec
    adapter: AdapterSpec
    scoring: ScoringSpec
    reporting: ReportingSpec
    official_leaderboard_url: str = ""
    version: str = MANIFEST_VERSION
    source_path: str = ""
    capabilities: list[str] = field(default_factory=list)
    required_tools: list[str] = field(default_factory=list)

    @classmethod
    def from_task(cls, task: Task) -> "BenchmarkManifest":
        benchmark = task.benchmark if isinstance(task.benchmark, dict) else {}
        docker = benchmark.get("docker") if isinstance(benchmark.get("docker"), dict) else {}
        official = _dict(benchmark.get("official_conditions"))
        assets = benchmark.get("assets") if isinstance(benchmark.get("assets"), list) else []
        name = str(benchmark.get("name") or task.id)
        group = str(benchmark.get("group") or task.category)
        return cls(
            id=task.id,
            display_name=name,
            task_group=group,
            description=str(task.question or f"Run {name}"),
            homepage_url=str(benchmark.get("homepage") or ""),
            official_leaderboard_url=str(benchmark.get("official_leaderboard_url") or ""),
            source=SourceSpec(
                repository_url=str(benchmark.get("repository") or ""),
                commit=str(benchmark.get("ref") or ""),
                dataset_id=str(benchmark.get("dataset_id") or ""),
                dataset_revision=str(benchmark.get("dataset_revision") or benchmark.get("ref") or ""),
                subdir=str(benchmark.get("subdir") or ""),
            ),
            license=str(benchmark.get("license") or ""),
            credit=str(benchmark.get("credit") or ""),
            citation=str(benchmark.get("citation") or benchmark.get("homepage") or ""),
            official_conditions=OfficialConditions(
                official_split=str(official.get("official_split") or ""),
                official_scoring_method=str(official.get("official_scoring_method") or ""),
                official_prompt_format=str(official.get("official_prompt_format") or ""),
                official_grader_command=str(official.get("official_grader_command") or ""),
                official_evaluation_config=str(official.get("official_evaluation_config") or ""),
                hardware_requirements=str(official.get("hardware_requirements") or ""),
                container_requirements=str(official.get("container_requirements") or ""),
            ),
            assets=[
                AssetSpec(
                    source=str(_dict(asset).get("source") or ""),
                    revision=str(_dict(asset).get("revision") or _dict(asset).get("ref") or ""),
                    checksum=str(_dict(asset).get("checksum") or ""),
                    expected_local_path=str(_dict(asset).get("expected_local_path") or _dict(asset).get("path") or ""),
                    required=bool(_dict(asset).get("required", True)),
                    validation_command=str(_dict(asset).get("validation_command") or ""),
                    validation_rules=_dict(_dict(asset).get("validation_rules")),
                )
                for asset in assets
                if isinstance(asset, dict)
            ],
            container=ContainerSpec(
                image=str(docker.get("image") or ""),
                command=str(docker.get("command") or ""),
                environment_allowed=_str_list(docker.get("environment")),
                mounts=[],
            ),
            adapter=AdapterSpec(
                module=str(benchmark.get("adapter_module") or ""),
                entry_point=str(benchmark.get("adapter") or ""),
                expected_output_files=["agent_bench_result.json"],
                result_parser="agent_bench_result_json" if docker.get("command") else "",
            ),
            scoring=ScoringSpec(),
            reporting=ReportingSpec(
                category_label=group,
                credit=str(benchmark.get("credit") or ""),
                citation=str(benchmark.get("citation") or benchmark.get("homepage") or ""),
                license=str(benchmark.get("license") or ""),
            ),
            source_path=task.source,
            capabilities=_str_list(benchmark.get("capabilities")),
            required_tools=_str_list(benchmark.get("required_tools")),
        )

    @classmethod
    def from_mapping(cls, raw: dict[str, Any], source_path: str = "") -> "BenchmarkManifest":
        source = _dict(raw.get("source"))
        container = _dict(raw.get("container"))
        adapter = _dict(raw.get("adapter"))
        scoring = _dict(raw.get("scoring"))
        reporting = _dict(raw.get("reporting"))
        official = _dict(raw.get("official_conditions"))
        assets = raw.get("assets") if isinstance(raw.get("assets"), list) else []
        return cls(
            id=str(raw.get("id") or ""),
            display_name=str(raw.get("display_name") or raw.get("name") or ""),
            task_group=str(raw.get("task_group") or reporting.get("category_label") or "Other"),
            description=str(raw.get("description") or ""),
            homepage_url=str(raw.get("homepage_url") or raw.get("homepage") or ""),
            official_leaderboard_url=str(raw.get("official_leaderboard_url") or ""),
            source=SourceSpec(
                repository_url=str(source.get("repository_url") or source.get("repository") or ""),
                commit=str(source.get("commit") or source.get("ref") or ""),
                dataset_id=str(source.get("dataset_id") or ""),
                dataset_revision=str(source.get("dataset_revision") or source.get("revision") or ""),
                subdir=str(source.get("subdir") or ""),
            ),
            license=str(raw.get("license") or reporting.get("license") or ""),
            credit=str(raw.get("credit") or reporting.get("credit") or ""),
            citation=str(raw.get("citation") or reporting.get("citation") or ""),
            official_conditions=OfficialConditions(
                official_split=str(official.get("official_split") or ""),
                official_scoring_method=str(official.get("official_scoring_method") or ""),
                official_prompt_format=str(official.get("official_prompt_format") or ""),
                official_grader_command=str(official.get("official_grader_command") or ""),
                official_evaluation_config=str(official.get("official_evaluation_config") or ""),
                hardware_requirements=str(official.get("hardware_requirements") or ""),
                container_requirements=str(official.get("container_requirements") or ""),
            ),
            assets=[
                AssetSpec(
                    source=str(_dict(asset).get("source") or ""),
                    revision=str(_dict(asset).get("revision") or _dict(asset).get("ref") or ""),
                    checksum=str(_dict(asset).get("checksum") or ""),
                    expected_local_path=str(_dict(asset).get("expected_local_path") or _dict(asset).get("path") or ""),
                    required=bool(_dict(asset).get("required", True)),
                    validation_command=str(_dict(asset).get("validation_command") or ""),
                    validation_rules=_dict(_dict(asset).get("validation_rules")),
                )
                for asset in assets
                if isinstance(asset, dict)
            ],
            container=ContainerSpec(
                image=str(container.get("image") or ""),
                build_context=str(container.get("build_context") or ""),
                dockerfile=str(container.get("dockerfile") or container.get("dockerfile_path") or ""),
                command=str(container.get("command") or ""),
                environment_allowed=_str_list(container.get("environment_allowed")),
                mounts=_dict_list(container.get("mounts")),
                network=str(container.get("network") or "model_proxy"),
                cpus=_optional_float(container.get("cpus")),
                memory=str(container.get("memory") or ""),
                pids_limit=_optional_int(container.get("pids_limit"), 1024),
                timeout_seconds=float(container.get("timeout") or container.get("timeout_seconds") or 21600.0),
                requires_nested_docker=bool(container.get("requires_nested_docker", False)),
                requires_host_docker_socket=bool(container.get("requires_host_docker_socket", False)),
                run_as_user=str(container.get("run_as_user") or "10001:10001"),
            ),
            adapter=AdapterSpec(
                module=str(adapter.get("module") or adapter.get("module_path") or ""),
                entry_point=str(adapter.get("entry_point") or ""),
                expected_output_files=_str_list(adapter.get("expected_output_files")),
                result_parser=str(adapter.get("result_parser") or ""),
            ),
            scoring=ScoringSpec(
                raw_score_field=str(scoring.get("raw_score_field") or "score"),
                max_score=_optional_float(scoring.get("max_score")),
                normalization=str(scoring.get("normalization") or "fraction_to_0_100"),
                direction=str(scoring.get("direction") or "higher_is_better"),
            ),
            reporting=ReportingSpec(
                category_label=str(reporting.get("category_label") or raw.get("task_group") or "Other"),
                display_order=int(reporting.get("display_order") or 1000),
                credit=str(reporting.get("credit") or raw.get("credit") or ""),
                citation=str(reporting.get("citation") or raw.get("citation") or ""),
                license=str(reporting.get("license") or raw.get("license") or ""),
            ),
            version=str(raw.get("version") or MANIFEST_VERSION),
            source_path=source_path,
            capabilities=_str_list(raw.get("capabilities")),
            required_tools=_str_list(raw.get("required_tools")),
        )

    def validate(self, *, allow_host_docker_socket: bool = False) -> ValidationResult:
        issues: list[ValidationIssue] = []
        _require_text(issues, "id", self.id)
        _require_text(issues, "display_name", self.display_name)
        _require_text(issues, "task_group", self.task_group)
        _require_text(issues, "description", self.description)
        _require_text(issues, "homepage_url", self.homepage_url)
        _require_text(issues, "license", self.license)
        _require_text(issues, "credit", self.credit)
        _require_text(issues, "citation", self.citation)

        has_repo = bool(self.source.repository_url.strip())
        has_dataset = bool(self.source.dataset_id.strip())
        if not has_repo and not has_dataset:
            issues.append(
                ValidationIssue(
                    "source",
                    "repository_url or dataset_id is required",
                    "Declare the upstream repository commit or dataset revision used by the official run.",
                )
            )
        if has_repo and not _is_pinned_commit(self.source.commit):
            issues.append(
                ValidationIssue(
                    "source.commit",
                    "repository source must be pinned to a 40-character commit hash",
                    "Resolve the official upstream ref once and record the exact commit hash.",
                )
            )
        if has_dataset and _is_moving_ref(self.source.dataset_revision):
            issues.append(
                ValidationIssue(
                    "source.dataset_revision",
                    "dataset source must be pinned to a non-moving revision",
                    "Use a dataset revision, commit, or immutable release tag rather than main/master/latest.",
                )
            )

        for field_name, value in (
            ("official_conditions.official_split", self.official_conditions.official_split),
            ("official_conditions.official_scoring_method", self.official_conditions.official_scoring_method),
            ("official_conditions.official_prompt_format", self.official_conditions.official_prompt_format),
            ("official_conditions.official_grader_command", self.official_conditions.official_grader_command),
            ("official_conditions.official_evaluation_config", self.official_conditions.official_evaluation_config),
        ):
            _require_text(issues, field_name, value)

        if not self.assets:
            issues.append(
                ValidationIssue(
                    "assets",
                    "at least one official asset declaration is required",
                    "Declare required datasets, repositories, checksums, and validation rules.",
                )
            )
        for index, asset in enumerate(self.assets):
            prefix = f"assets[{index}]"
            _require_text(issues, f"{prefix}.source", asset.source)
            _require_text(issues, f"{prefix}.expected_local_path", asset.expected_local_path)
            if asset.required and not asset.checksum and not asset.validation_command and not asset.validation_rules:
                issues.append(
                    ValidationIssue(
                        f"{prefix}.checksum",
                        "required assets need a checksum or validation rule",
                        "Record a checksum when available, otherwise add validation_command or validation_rules.",
                    )
                )
            if _is_moving_ref(asset.revision):
                issues.append(
                    ValidationIssue(
                        f"{prefix}.revision",
                        "asset revision must not be a moving ref",
                        "Pin the asset to an immutable revision or commit.",
                    )
                )

        if not self.container.image and not self.container.build_context:
            issues.append(
                ValidationIssue(
                    "container.image",
                    "container.image or container.build_context is required",
                    "Declare the exact official benchmark image or local build context.",
                )
            )
        _require_text(issues, "container.command", self.container.command)
        if self.container.timeout_seconds <= 0:
            issues.append(ValidationIssue("container.timeout", "timeout must be positive"))
        if not self.adapter.module and not self.adapter.entry_point:
            issues.append(
                ValidationIssue(
                    "adapter",
                    "adapter.module or adapter.entry_point is required",
                    "Declare a benchmark adapter path; do not put benchmark-specific logic in the core runner.",
                )
            )
        if not self.adapter.expected_output_files:
            issues.append(
                ValidationIssue(
                    "adapter.expected_output_files",
                    "expected output files are required",
                    "At minimum declare agent_bench_result.json or the official normalized output file.",
                )
            )
        _require_text(issues, "adapter.result_parser", self.adapter.result_parser)
        _require_text(issues, "scoring.raw_score_field", self.scoring.raw_score_field)
        if self.scoring.direction not in {"higher_is_better", "lower_is_better"}:
            issues.append(
                ValidationIssue(
                    "scoring.direction",
                    "direction must be higher_is_better or lower_is_better",
                )
            )
        _require_text(issues, "reporting.category_label", self.reporting.category_label)
        return ValidationResult(ok=not issues, issues=issues)

    def to_task(self) -> Task:
        return Task(
            id=self.id,
            category=self.task_group,
            type="external_benchmark",
            question=self.description,
            source=self.source_path or f"tasks/{self.id}/manifest.json",
            benchmark=self.to_legacy_benchmark(),
        )

    def to_legacy_benchmark(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.display_name,
            "group": self.task_group,
            "capabilities": list(self.capabilities),
            "required_tools": list(self.required_tools),
            "homepage": self.homepage_url,
            "official_leaderboard_url": self.official_leaderboard_url,
            "repository": self.source.repository_url,
            "dataset_id": self.source.dataset_id,
            "ref": self.source.commit or self.source.dataset_revision,
            "subdir": self.source.subdir,
            "license": self.license,
            "credit": self.credit,
            "citation": self.citation,
            "manifest": self.to_dict(),
            "docker": {
                "image": self.container.image,
                "command": self.container.command,
                "environment": [],
                "volumes": [],
                "requires_host_docker_socket": self.container.requires_host_docker_socket,
                "requires_nested_docker": self.container.requires_nested_docker,
                "run_as_user": self.container.run_as_user,
                "timeout_seconds": self.container.timeout_seconds,
            },
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "id": self.id,
            "display_name": self.display_name,
            "task_group": self.task_group,
            "description": self.description,
            "homepage_url": self.homepage_url,
            "official_leaderboard_url": self.official_leaderboard_url,
            "source": self.source.to_dict(),
            "license": self.license,
            "credit": self.credit,
            "citation": self.citation,
            "official_conditions": self.official_conditions.to_dict(),
            "assets": [asset.to_dict() for asset in self.assets],
            "container": self.container.to_dict(),
            "adapter": self.adapter.to_dict(),
            "scoring": self.scoring.to_dict(),
            "reporting": self.reporting.to_dict(),
            "source_path": self.source_path,
            "capabilities": list(self.capabilities),
            "required_tools": list(self.required_tools),
        }


def manifest_from_task(task: Task) -> BenchmarkManifest:
    manifest = task.benchmark.get("manifest") if isinstance(task.benchmark, dict) else None
    if isinstance(manifest, dict):
        return BenchmarkManifest.from_mapping(manifest, source_path=task.source)
    return BenchmarkManifest.from_task(task)


def load_manifest_tasks(benchmark_root: str | Path) -> list[Task]:
    root = Path(benchmark_root)
    if not root.exists():
        return []
    tasks: list[Task] = []
    manifest_paths = (
        sorted(root.glob("*/manifest.json"))
        + sorted(root.glob("*/manifest.yaml"))
        + sorted(root.glob("*/manifest.yml"))
    )
    manifests: list[BenchmarkManifest] = []
    for path in manifest_paths:
        raw = load_manifest_mapping(path)
        manifests.append(BenchmarkManifest.from_mapping(raw, source_path=str(path)))
    for manifest in sorted(manifests, key=lambda item: (item.reporting.display_order, item.id)):
        tasks.append(manifest.to_task())
    return tasks


def load_manifest_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = _load_yaml_with_optional_dependency(text, path)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: manifest must be an object")
    return payload


def _load_yaml_with_optional_dependency(text: str, path: Path) -> Any:
    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ValueError(
            f"{path}: YAML manifests require PyYAML at runtime, or the file must use JSON-compatible YAML"
        ) from exc
    return yaml.safe_load(text)


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _dict_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def _optional_int(value: Any, default: int | None = None) -> int | None:
    if value is None:
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return default


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _require_text(issues: list[ValidationIssue], field_name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        issues.append(ValidationIssue(field_name, "non-empty string is required"))


def _is_pinned_commit(value: str) -> bool:
    return bool(COMMIT_RE.fullmatch((value or "").strip()))


def _is_moving_ref(value: str) -> bool:
    return (value or "").strip().lower() in MOVING_REFS
