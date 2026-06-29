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
TIMED_OUT = "timed_out"

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
    }
