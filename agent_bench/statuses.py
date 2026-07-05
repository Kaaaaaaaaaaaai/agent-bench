from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


SUCCESS = "success"
SUCCESS_WITH_WARNINGS = "success_with_warnings"
PASSED = "passed"
FAILED_MODEL_ANSWER = "failed_model_answer"
FAILED_MODEL_FORMAT = "failed_model_format"
FAILED_MODEL_TOOL_USE = "failed_model_tool_use"
FAILED_MANIFEST_VALIDATION = "failed_manifest_validation"
FAILED_ASSET_DOWNLOAD = "failed_asset_download"
FAILED_MISSING_ASSET = "failed_missing_asset"
FAILED_ASSET_CHECKSUM = "failed_asset_checksum"
FAILED_ASSET_VALIDATION = "failed_asset_validation"
FAILED_CONTAINER_BUILD = "failed_container_build"
FAILED_CONTAINER_START = "failed_container_start"
FAILED_CONTAINER_RUNTIME = "failed_container_runtime"
FAILED_TIMEOUT = "failed_timeout"
FAILED_OOM = "failed_oom"
FAILED_SECURITY_POLICY = "failed_security_policy"
FAILED_MODEL_ENDPOINT = "failed_model_endpoint"
FAILED_MODEL_RESPONSE = "failed_model_response"
FAILED_MODEL_OUTPUT_PARSE = "failed_model_output_parse"
FAILED_MODEL_MISSING_ARTIFACT = "failed_model_missing_artifact"
FAILED_REQUIRED_TOOL = "failed_required_tool"
FAILED_JUDGE_UNAVAILABLE = "failed_judge_unavailable"
FAILED_JUDGE_PARSE = "failed_judge_parse"
FAILED_OFFICIAL_REPRO_MISMATCH = "failed_official_repro_mismatch"
SKIPPED_BY_USER = "skipped_by_user"
SKIPPED_UNSUPPORTED = "skipped_unsupported"
CANCELLED = "cancelled"
FAILED_HARNESS_SETUP = "failed_harness_setup"
FAILED_DATASET_EXTRACTION = "failed_dataset_extraction"
FAILED_MISSING_ASSETS = "failed_missing_assets"
FAILED_UNSUPPORTED_CAPABILITY = "failed_unsupported_capability"
SKIPPED_UNSUPPORTED_CAPABILITY = "skipped_unsupported_capability"
FAILED_GRADER = "failed_grader"
FAILED_TOKEN_BUDGET = "failed_token_budget"
FAILED_MISSING_REQUIRED_TOOL = "failed_missing_required_tool"
FAILED_INVALID_TASK_CONTEXT = "failed_invalid_task_context"
TIMED_OUT = "timed_out"

RUN_COMPLETED = "completed"
RUN_SKIPPED = "skipped"
RUN_EXECUTION_ERROR = "execution_error"
RUN_INFRASTRUCTURE_ERROR = "infrastructure_error"

SCORE_PASSED = "passed"
SCORE_FAILED_MODEL_ANSWER = "failed_model_answer"
SCORE_PARTIALLY_CORRECT = "partially_correct"
SCORE_NOT_APPLICABLE = "not_applicable"
SCORE_UNGRADED = "ungraded"

BLOCKER_UNSUPPORTED_CAPABILITY = "unsupported_capability"
BLOCKER_MISSING_ASSET = "missing_asset"
BLOCKER_MISSING_GRADER = "missing_grader"
BLOCKER_DISABLED_SCORING = "disabled_scoring"
BLOCKER_EXTERNAL_PLATFORM_UNAVAILABLE = "external_platform_unavailable"
BLOCKER_GIT_LFS_POINTER_STUB = "git_lfs_pointer_stub"
BLOCKER_INVALID_TASK_CONTEXT = "invalid_task_context"
BLOCKER_JUDGE_PARSE_ERROR = "judge_parse_error"
BLOCKER_MISSING_REFERENCE_DATASET = "missing_reference_dataset"
BLOCKER_MISSING_REFERENCE_DOCUMENTS = "missing_reference_documents"
BLOCKER_MISSING_REQUIRED_TOOL = "missing_required_tool"
BLOCKER_MISSING_TASK_INSTANCES = "missing_task_instances"
BLOCKER_OUTPUT_PARSE_ERROR = "output_parse_error"
BLOCKER_REPO_PATCH_HARNESS_SETUP = "repo_patch_harness_setup"

STATUS_ALIASES = {
    "failed_unsupported_capability": SKIPPED_UNSUPPORTED_CAPABILITY,
    "skipped_missing_assets": FAILED_MISSING_ASSETS,
}

STRICT_STATUSES = {
    SUCCESS,
    SUCCESS_WITH_WARNINGS,
    PASSED,
    FAILED_MODEL_ANSWER,
    FAILED_MODEL_FORMAT,
    FAILED_MODEL_TOOL_USE,
    FAILED_MANIFEST_VALIDATION,
    FAILED_ASSET_DOWNLOAD,
    FAILED_MISSING_ASSET,
    FAILED_ASSET_CHECKSUM,
    FAILED_ASSET_VALIDATION,
    FAILED_CONTAINER_BUILD,
    FAILED_CONTAINER_START,
    FAILED_CONTAINER_RUNTIME,
    FAILED_TIMEOUT,
    FAILED_OOM,
    FAILED_SECURITY_POLICY,
    FAILED_MODEL_ENDPOINT,
    FAILED_MODEL_RESPONSE,
    FAILED_MODEL_OUTPUT_PARSE,
    FAILED_MODEL_MISSING_ARTIFACT,
    FAILED_REQUIRED_TOOL,
    FAILED_JUDGE_UNAVAILABLE,
    FAILED_JUDGE_PARSE,
    FAILED_OFFICIAL_REPRO_MISMATCH,
    SKIPPED_BY_USER,
    SKIPPED_UNSUPPORTED,
    CANCELLED,
    FAILED_HARNESS_SETUP,
    FAILED_DATASET_EXTRACTION,
    FAILED_MISSING_ASSETS,
    FAILED_UNSUPPORTED_CAPABILITY,
    SKIPPED_UNSUPPORTED_CAPABILITY,
    FAILED_GRADER,
    FAILED_TOKEN_BUDGET,
    FAILED_MISSING_REQUIRED_TOOL,
    FAILED_INVALID_TASK_CONTEXT,
    TIMED_OUT,
}

INVALID_EVALUATION_STATUSES = {
    FAILED_MANIFEST_VALIDATION,
    FAILED_ASSET_DOWNLOAD,
    FAILED_MISSING_ASSET,
    FAILED_ASSET_CHECKSUM,
    FAILED_ASSET_VALIDATION,
    FAILED_CONTAINER_BUILD,
    FAILED_CONTAINER_START,
    FAILED_CONTAINER_RUNTIME,
    FAILED_TIMEOUT,
    FAILED_OOM,
    FAILED_SECURITY_POLICY,
    FAILED_MODEL_ENDPOINT,
    FAILED_REQUIRED_TOOL,
    FAILED_JUDGE_UNAVAILABLE,
    FAILED_JUDGE_PARSE,
    FAILED_OFFICIAL_REPRO_MISMATCH,
    SKIPPED_BY_USER,
    SKIPPED_UNSUPPORTED,
    CANCELLED,
    FAILED_HARNESS_SETUP,
    FAILED_DATASET_EXTRACTION,
    FAILED_MISSING_ASSETS,
    FAILED_UNSUPPORTED_CAPABILITY,
    SKIPPED_UNSUPPORTED_CAPABILITY,
    FAILED_GRADER,
    FAILED_TOKEN_BUDGET,
    FAILED_MISSING_REQUIRED_TOOL,
    FAILED_INVALID_TASK_CONTEXT,
    TIMED_OUT,
}

MODEL_FAILURE_STATUSES = {
    FAILED_MODEL_RESPONSE,
    FAILED_MODEL_OUTPUT_PARSE,
    FAILED_MODEL_ANSWER,
    FAILED_MODEL_FORMAT,
    FAILED_MODEL_TOOL_USE,
    FAILED_MODEL_MISSING_ARTIFACT,
}


@dataclass(frozen=True, slots=True)
class StatusInfo:
    code: str
    counts_toward_official_score: bool
    counts_toward_coverage_denominator: bool
    failure_class: str
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _status_info(
    code: str,
    *,
    score: bool,
    coverage: bool = True,
    failure_class: str,
    explanation: str,
) -> StatusInfo:
    return StatusInfo(
        code=code,
        counts_toward_official_score=score,
        counts_toward_coverage_denominator=coverage,
        failure_class=failure_class,
        explanation=explanation,
    )


STATUS_CATALOG: dict[str, StatusInfo] = {
    SUCCESS: _status_info(
        SUCCESS,
        score=True,
        failure_class="none",
        explanation="Benchmark completed and produced a valid official score.",
    ),
    SUCCESS_WITH_WARNINGS: _status_info(
        SUCCESS_WITH_WARNINGS,
        score=True,
        failure_class="none",
        explanation="Benchmark produced a valid score but emitted non-fatal warnings.",
    ),
    PASSED: _status_info(
        PASSED,
        score=True,
        failure_class="none",
        explanation="Legacy success status; benchmark completed and produced a valid score.",
    ),
    FAILED_MODEL_ANSWER: _status_info(
        FAILED_MODEL_ANSWER,
        score=True,
        failure_class="model",
        explanation="The model response was valid but did not satisfy the grader.",
    ),
    FAILED_MODEL_FORMAT: _status_info(
        FAILED_MODEL_FORMAT,
        score=True,
        failure_class="model",
        explanation="The model response could not be parsed in the required format.",
    ),
    FAILED_MODEL_TOOL_USE: _status_info(
        FAILED_MODEL_TOOL_USE,
        score=True,
        failure_class="model",
        explanation="The model used tools incorrectly or requested unavailable tool behavior.",
    ),
    FAILED_MODEL_MISSING_ARTIFACT: _status_info(
        FAILED_MODEL_MISSING_ARTIFACT,
        score=True,
        failure_class="model",
        explanation="The model did not produce a required file, patch, or other artifact.",
    ),
    FAILED_MANIFEST_VALIDATION: _status_info(
        FAILED_MANIFEST_VALIDATION,
        score=False,
        failure_class="benchmark_setup",
        explanation="The benchmark manifest is incomplete or not official-run equivalent.",
    ),
    FAILED_ASSET_DOWNLOAD: _status_info(
        FAILED_ASSET_DOWNLOAD,
        score=False,
        failure_class="benchmark_setup",
        explanation="Required benchmark assets could not be downloaded.",
    ),
    FAILED_MISSING_ASSET: _status_info(
        FAILED_MISSING_ASSET,
        score=False,
        failure_class="benchmark_setup",
        explanation="Required benchmark assets are missing.",
    ),
    FAILED_ASSET_CHECKSUM: _status_info(
        FAILED_ASSET_CHECKSUM,
        score=False,
        failure_class="benchmark_setup",
        explanation="A required benchmark asset did not match its declared checksum.",
    ),
    FAILED_ASSET_VALIDATION: _status_info(
        FAILED_ASSET_VALIDATION,
        score=False,
        failure_class="benchmark_setup",
        explanation="A required benchmark asset failed schema or content validation.",
    ),
    FAILED_CONTAINER_BUILD: _status_info(
        FAILED_CONTAINER_BUILD,
        score=False,
        failure_class="infrastructure",
        explanation="The benchmark container image could not be built or pulled.",
    ),
    FAILED_CONTAINER_START: _status_info(
        FAILED_CONTAINER_START,
        score=False,
        failure_class="infrastructure",
        explanation="The benchmark container could not be started.",
    ),
    FAILED_CONTAINER_RUNTIME: _status_info(
        FAILED_CONTAINER_RUNTIME,
        score=False,
        failure_class="infrastructure",
        explanation="The benchmark container exited before producing a valid result.",
    ),
    FAILED_TIMEOUT: _status_info(
        FAILED_TIMEOUT,
        score=False,
        failure_class="infrastructure",
        explanation="The benchmark exceeded its configured timeout.",
    ),
    FAILED_OOM: _status_info(
        FAILED_OOM,
        score=False,
        failure_class="infrastructure",
        explanation="The benchmark container exceeded its memory limit.",
    ),
    FAILED_SECURITY_POLICY: _status_info(
        FAILED_SECURITY_POLICY,
        score=False,
        failure_class="infrastructure",
        explanation="The benchmark requested permissions disallowed by the manifest or CLI policy.",
    ),
    FAILED_MODEL_ENDPOINT: _status_info(
        FAILED_MODEL_ENDPOINT,
        score=False,
        failure_class="infrastructure",
        explanation="The configured model endpoint was unavailable or rejected requests.",
    ),
    FAILED_MODEL_RESPONSE: _status_info(
        FAILED_MODEL_RESPONSE,
        score=True,
        failure_class="model",
        explanation="The model endpoint returned a malformed or unusable response.",
    ),
    FAILED_MODEL_OUTPUT_PARSE: _status_info(
        FAILED_MODEL_OUTPUT_PARSE,
        score=True,
        failure_class="model",
        explanation="The benchmark could not parse the model output.",
    ),
    FAILED_REQUIRED_TOOL: _status_info(
        FAILED_REQUIRED_TOOL,
        score=False,
        failure_class="benchmark_setup",
        explanation="A benchmark-required external tool is unavailable.",
    ),
    FAILED_MISSING_REQUIRED_TOOL: _status_info(
        FAILED_MISSING_REQUIRED_TOOL,
        score=False,
        failure_class="benchmark_setup",
        explanation="A benchmark-required tool schema, backend, or fixture is unavailable.",
    ),
    FAILED_JUDGE_UNAVAILABLE: _status_info(
        FAILED_JUDGE_UNAVAILABLE,
        score=False,
        failure_class="infrastructure",
        explanation="The required judge model or official judge endpoint was unavailable.",
    ),
    FAILED_JUDGE_PARSE: _status_info(
        FAILED_JUDGE_PARSE,
        score=False,
        failure_class="infrastructure",
        explanation="The judge response was not strict machine-readable output.",
    ),
    FAILED_OFFICIAL_REPRO_MISMATCH: _status_info(
        FAILED_OFFICIAL_REPRO_MISMATCH,
        score=False,
        failure_class="benchmark_setup",
        explanation="The configured run does not match official leaderboard conditions.",
    ),
    SKIPPED_BY_USER: _status_info(
        SKIPPED_BY_USER,
        score=False,
        failure_class="user_skip",
        explanation="The benchmark was skipped by explicit user selection.",
    ),
    SKIPPED_UNSUPPORTED: _status_info(
        SKIPPED_UNSUPPORTED,
        score=False,
        failure_class="user_skip",
        explanation="The benchmark is unsupported in the current runner configuration.",
    ),
    CANCELLED: _status_info(
        CANCELLED,
        score=False,
        failure_class="user_skip",
        explanation="The run was cancelled before benchmark completion.",
    ),
}


def normalize_status(status: Any) -> str:
    if not isinstance(status, str):
        return ""
    status = status.strip()
    return STATUS_ALIASES.get(status, status)


def status_info(status: Any) -> StatusInfo:
    normalized = normalize_status(status)
    if normalized in STATUS_CATALOG:
        return STATUS_CATALOG[normalized]
    if normalized in INVALID_EVALUATION_STATUSES:
        return _status_info(
            normalized,
            score=False,
            failure_class="infrastructure",
            explanation=normalized.replace("_", " "),
        )
    if normalized in MODEL_FAILURE_STATUSES:
        return _status_info(
            normalized,
            score=True,
            failure_class="model",
            explanation=normalized.replace("_", " "),
        )
    return _status_info(
        normalized or "unknown",
        score=False,
        failure_class="infrastructure",
        explanation=(normalized or "unknown").replace("_", " "),
    )


def status_catalog_dict() -> dict[str, dict[str, Any]]:
    return {code: info.to_dict() for code, info in sorted(STATUS_CATALOG.items())}


def is_invalid_evaluation_status(status: Any) -> bool:
    normalized = normalize_status(status)
    if normalized in STATUS_CATALOG:
        return not STATUS_CATALOG[normalized].counts_toward_official_score
    return normalized in INVALID_EVALUATION_STATUSES


def is_skipped_like_status(status: Any) -> bool:
    return normalize_status(status) in {
        FAILED_MISSING_ASSETS,
        SKIPPED_UNSUPPORTED_CAPABILITY,
        FAILED_MISSING_REQUIRED_TOOL,
    }
