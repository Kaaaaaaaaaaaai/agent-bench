from __future__ import annotations

from typing import Any


PASSED = "passed"
FAILED_MODEL_ANSWER = "failed_model_answer"
FAILED_MODEL_FORMAT = "failed_model_format"
FAILED_MODEL_TOOL_USE = "failed_model_tool_use"
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
    "failed_judge_parse": FAILED_GRADER,
}

STRICT_STATUSES = {
    PASSED,
    FAILED_MODEL_ANSWER,
    FAILED_MODEL_FORMAT,
    FAILED_MODEL_TOOL_USE,
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
    FAILED_MODEL_ANSWER,
    FAILED_MODEL_FORMAT,
    FAILED_MODEL_TOOL_USE,
}


def normalize_status(status: Any) -> str:
    if not isinstance(status, str):
        return ""
    status = status.strip()
    return STATUS_ALIASES.get(status, status)


def is_invalid_evaluation_status(status: Any) -> bool:
    return normalize_status(status) in INVALID_EVALUATION_STATUSES


def is_skipped_like_status(status: Any) -> bool:
    return normalize_status(status) in {
        FAILED_MISSING_ASSETS,
        SKIPPED_UNSUPPORTED_CAPABILITY,
        FAILED_MISSING_REQUIRED_TOOL,
    }
