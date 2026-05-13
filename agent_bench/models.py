from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class Task:
    id: str
    category: str
    type: str
    question: str
    source: str
    choices: list[str] = field(default_factory=list)
    answer: list[str] = field(default_factory=list)
    expected_text: str | None = None
    reference_path: str | None = None
    reference_text: str | None = None
    title: str | None = None
    function_name: str | None = None
    test_cases: list[dict[str, Any]] = field(default_factory=list)
    comparison: str | None = None

    @property
    def is_coding(self) -> bool:
        return self.type == "coding"

    @property
    def is_multiple_choice(self) -> bool:
        return self.type == "multiple_choice"

    @property
    def is_text_recall(self) -> bool:
        return self.type == "text_recall"


@dataclass(slots=True)
class ModelResponse:
    task_id: str
    model: str
    raw_response: str
    latency_seconds: float
    time_to_first_token_seconds: float | None = None
    tokens_per_second: float | None = None
    output_token_count: int | None = None
    error: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class GradeResult:
    task_id: str
    category: str
    kind: str
    score: float
    max_score: float
    passed: bool
    json_valid: bool
    latency_seconds: float
    time_to_first_token_seconds: float | None = None
    tokens_per_second: float | None = None
    output_token_count: int | None = None
    task_duration_seconds: float | None = None
    answer: Any = None
    confidence: float | None = None
    error: str | None = None
    timed_out: bool = False
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
