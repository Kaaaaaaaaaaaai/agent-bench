#!/usr/bin/env python3
import argparse
import ast
import csv
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib import error, request


QUESTION_KEYS = (
    "question",
    "prompt",
    "instruction",
    "reformulated_task",
    "requirements",
    "problem",
    "task",
    "query",
    "input",
    "goal",
    "issue",
    "issue_text",
    "description",
    "statement",
    "problem_statement",
    "input_text",
    "query_text",
    "prompt_text",
    "text",
)
ANSWER_KEYS = (
    "answer",
    "answers",
    "rubric",
    "criteria",
    "grading",
    "evaluation_criteria",
    "correct_answer",
    "correct",
    "label",
    "target",
    "output",
    "expected",
    "expected_answer",
    "solution",
    "gold",
    "reference",
    "ground_truth",
    "final_answer",
    "answer_key",
    "correct_option",
    "expected_output",
    "final_response",
    "patch",
    "gold_patch",
)
CHOICE_KEYS = ("choices", "options", "candidates", "multiple_choice_targets")
RUBRIC_KEYS = ("rubric", "criteria", "grading", "evaluation_criteria")
SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules", ".mypy_cache", ".pytest_cache"}
MAX_FIELD_CHARS = 6000
MAX_RECORDS_PER_FILE = 500


@dataclass(slots=True)
class BenchmarkItem:
    question: str
    expected: str
    source: str
    choices: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run local benchmark records from a cloned public benchmark.")
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--kind", default="repository")
    args = parser.parse_args()

    cwd = Path.cwd()
    files = sorted(str(path.relative_to(cwd)) for path in cwd.rglob("*") if path.is_file())[:200]
    markers = [
        name
        for name in (
            "README.md",
            "README.rst",
            "pyproject.toml",
            "requirements.txt",
            "Dockerfile",
            "LICENSE",
            "LICENSE.md",
        )
        if (cwd / name).exists()
    ]
    output_dir = Path(os.environ.get("AGENT_BENCH_OUTPUT_DIR", "/outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_limit = _sample_limit()
    items, extraction_errors = extract_benchmark_items(cwd, sample_limit)
    evaluations = run_model_evaluations(args.benchmark, items)
    evaluated_count = len(evaluations)
    score = _average_score(evaluations)
    error_message = ""
    if not items:
        error_message = "No benchmark task records were found in the cloned repository"
    elif not evaluations and _is_remote_provider():
        error_message = "No model evaluations completed"

    payload = {
        "score": score,
        "benchmark": args.benchmark,
        "group": os.environ.get("AGENT_BENCH_BENCHMARK_GROUP") or "Benchmarks",
        "kind": args.kind,
        "repository": os.environ.get("AGENT_BENCH_REPOSITORY", ""),
        "repository_ref": os.environ.get("AGENT_BENCH_REPOSITORY_REF", ""),
        "subdir": os.environ.get("AGENT_BENCH_SUBDIR", ""),
        "model": os.environ.get("AGENT_BENCH_MODEL", ""),
        "repository_ready": bool(files),
        "file_count_sampled": len(files),
        "markers": markers,
        "sample_files": files,
        "sample_limit": sample_limit,
        "extracted_task_count": len(items),
        "evaluated_task_count": evaluated_count,
        "evaluation_passed_count": sum(1 for item in evaluations if item.get("passed")),
        "extraction_errors": extraction_errors[:20],
        "extraction_sources": sorted({item.source for item in items}),
        "model_evals": evaluations,
        "model_eval": summarize_evaluations(evaluations),
    }
    if error_message:
        payload["error"] = error_message

    (output_dir / "agent_bench_result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0 if evaluated_count > 0 else 2


def extract_benchmark_items(root: Path, limit: int) -> tuple[list[BenchmarkItem], list[str]]:
    items: list[BenchmarkItem] = []
    errors: list[str] = []
    hf_items, hf_errors = extract_huggingface_items(limit)
    items.extend(hf_items)
    errors.extend(hf_errors)
    specialized_items, specialized_errors = extract_specialized_items(root, limit - len(items))
    items.extend(specialized_items)
    errors.extend(specialized_errors)
    for path in _candidate_files(root):
        if len(items) >= limit:
            break
        try:
            new_items = extract_items_from_file(path, root, limit - len(items))
        except Exception as exc:
            errors.append(f"{path.relative_to(root)}: {exc}")
            continue
        items.extend(new_items)
    return items, errors


def _candidate_files(root: Path) -> list[Path]:
    suffix_priority = {
        ".jsonl": 0,
        ".json": 1,
        ".csv": 2,
        ".parquet": 3,
        ".py": 4,
        ".yaml": 5,
        ".yml": 5,
        ".md": 6,
        ".txt": 7,
    }
    candidates: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in suffix_priority:
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if _skip_candidate_file(path):
            continue
        try:
            if path.stat().st_size == 0:
                continue
        except OSError:
            continue
        candidates.append(path)
    return sorted(
        candidates,
        key=lambda path: (
            _path_relevance(path),
            suffix_priority[path.suffix.lower()],
            len(path.parts),
            str(path),
        ),
    )


def _skip_candidate_file(path: Path) -> bool:
    text = str(path).lower()
    filename = path.name.lower()
    if filename in {"package-lock.json", "pnpm-lock.yaml", "uv.lock", "poetry.lock"}:
        return True
    if path.suffix.lower() == ".py" and filename != "tasks.py":
        return True
    if path.suffix.lower() in {".yaml", ".yml"} and "/prompt/" not in text and "prompt" not in filename:
        return True
    if path.suffix.lower() in {".md", ".txt", ".yaml", ".yml"} and not _is_prompt_text_file(path):
        return True
    return any(
        marker in text
        for marker in (
            "model_pricing",
            "pricing.json",
            "cache",
            ".pytest_cache",
            "node_modules",
            "package.json",
            "tsconfig",
        )
    )


def _is_prompt_text_file(path: Path) -> bool:
    text = str(path).lower()
    filename = path.name.lower()
    if filename in {"readme.md", "license", "security.md", "contributing.md", "requirements.txt"}:
        return False
    return any(
        marker in text
        for marker in (
            "description.md",
            "description_obfuscated.md",
            "spec.md",
            "benchmark_plan.md",
            "/prompts/",
            "/prompt/",
            "research_problem.txt",
            "instructions.txt",
            "task",
            "question",
            "problem",
            "benchmark",
        )
    )


def _path_relevance(path: Path) -> int:
    text = str(path).lower()
    if any(word in text for word in ("test", "validation", "eval", "benchmark", "question", "problem", "task", "prompt")):
        return 0
    if any(word in text for word in ("description.md", "spec.md", "instructions.txt")):
        return 1
    if any(word in text for word in ("config", "pricing", "cache", "knowledge", "example")):
        return 2
    return 1


def extract_huggingface_items(limit: int) -> tuple[list[BenchmarkItem], list[str]]:
    repository = os.environ.get("AGENT_BENCH_REPOSITORY", "")
    marker = "huggingface.co/datasets/"
    if marker not in repository:
        return [], []
    dataset_id = repository.split(marker, 1)[1].strip("/")
    if not dataset_id:
        return [], []
    try:
        from datasets import load_dataset
    except ImportError as exc:
        return [], [f"{dataset_id}: datasets package is required for Hugging Face streaming ({exc})"]

    items: list[BenchmarkItem] = []
    errors: list[str] = []
    for split in ("test", "validation", "dev", "train"):
        if len(items) >= limit:
            break
        try:
            dataset = load_dataset(dataset_id, split=split, streaming=True)
        except Exception as exc:
            errors.append(f"{dataset_id}/{split}: {exc}")
            continue
        for index, row in enumerate(dataset):
            if len(items) >= limit or index >= MAX_RECORDS_PER_FILE:
                break
            item = item_from_record(row, f"huggingface:{dataset_id}/{split}:{index + 1}")
            if item is not None:
                items.append(item)
    return items, errors


def extract_specialized_items(root: Path, limit: int) -> tuple[list[BenchmarkItem], list[str]]:
    benchmark = os.environ.get("AGENT_BENCH_BENCHMARK_NAME", "")
    if limit <= 0:
        return [], []
    if benchmark == "InvestorBench":
        return extract_investorbench_items(root, limit)
    if benchmark == "Humanity's Last Exam":
        return extract_hle_items(root, limit)
    return [], []


def extract_investorbench_items(root: Path, limit: int) -> tuple[list[BenchmarkItem], list[str]]:
    data_dir = root / "data"
    if not data_dir.is_dir():
        return [], ["InvestorBench data directory was not found"]
    items: list[BenchmarkItem] = []
    errors: list[str] = []
    for path in sorted(data_dir.glob("*.json")):
        if len(items) >= limit:
            break
        try:
            series = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        except Exception as exc:
            errors.append(f"{path.relative_to(root)}: {exc}")
            continue
        if not isinstance(series, dict):
            continue
        dates = sorted(series)
        for index, current_date in enumerate(dates[:-1]):
            if len(items) >= limit:
                break
            current = series.get(current_date)
            future = series.get(dates[index + 1])
            if not isinstance(current, dict) or not isinstance(future, dict):
                continue
            current_price = _as_float(current.get("prices"))
            future_price = _as_float(future.get("prices"))
            if current_price is None or future_price is None:
                continue
            symbol = path.stem.upper()
            movement = (future_price - current_price) / current_price
            if movement > 0.0025:
                expected = "buy"
            elif movement < -0.0025:
                expected = "sell"
            else:
                expected = "hold"
            news = current.get("news") if isinstance(current.get("news"), list) else []
            news_text = "\n".join(str(item) for item in news[:5])
            question = (
                f"InvestorBench trading decision for {symbol} on {current_date}.\n"
                f"Current price: {current_price:.4f}. Recent news:\n{news_text}\n\n"
                "Choose one action for the next trading step: buy, sell, or hold. "
                "Return only the action and a short rationale."
            )
            items.append(
                BenchmarkItem(
                    question=truncate(question),
                    expected=expected,
                    source=f"{path.relative_to(root)}:{current_date}",
                    choices={"A": "buy", "B": "sell", "C": "hold"},
                    metadata={"grading": "exact", "expected_key": "next_price_direction"},
                )
            )
    return items, errors


def extract_hle_items(root: Path, limit: int) -> tuple[list[BenchmarkItem], list[str]]:
    readme = root / "README.md"
    runner = root / "hle_eval" / "run_model_predictions.py"
    if not readme.is_file():
        return [], ["HLE README.md was not found"]
    readme_text = readme.read_text(encoding="utf-8", errors="replace")
    runner_text = runner.read_text(encoding="utf-8", errors="replace") if runner.is_file() else ""
    match = re.search(r"SYSTEM_PROMPT\s*=\s*([\"'])(.*?)\1", runner_text, flags=re.DOTALL)
    system_prompt = match.group(2) if match else "Provide an explanation, answer, and confidence."
    question = (
        "Humanity's Last Exam evaluation-format task from the public benchmark repository.\n"
        f"{system_prompt}\n\n"
        "Summarize the required answer format and then answer a closed-ended academic benchmark item "
        "using that format if a question is supplied."
    )
    item = BenchmarkItem(
        question=truncate(question),
        expected=(
            "Response should follow the HLE public evaluation format with Explanation, Answer, "
            "and Confidence fields for closed-ended academic questions."
        ),
        source="hle_eval/run_model_predictions.py",
        metadata={"grading": "rubric", "expected_key": "public_evaluation_format"},
    )
    return [item][:limit], []


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_items_from_file(path: Path, root: Path, limit: int) -> list[BenchmarkItem]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        return _items_from_json_payload(payload, path.relative_to(root), limit)
    if suffix == ".jsonl":
        return _items_from_jsonl(path, root, limit)
    if suffix == ".csv":
        return _items_from_csv(path, root, limit)
    if suffix == ".parquet":
        return _items_from_parquet(path, root, limit)
    if suffix == ".py":
        return _items_from_python(path, root, limit)
    if suffix in {".md", ".txt", ".yaml", ".yml"}:
        return _items_from_text_file(path, root, limit)
    return []


def _items_from_python(path: Path, root: Path, limit: int) -> list[BenchmarkItem]:
    source = path.read_text(encoding="utf-8", errors="replace")
    tree = ast.parse(source)
    constants = _module_constants(tree)
    items: list[BenchmarkItem] = []
    for node in ast.walk(tree):
        if len(items) >= limit:
            break
        if not isinstance(node, ast.Return) or node.value is None:
            continue
        try:
            record = _literal_from_ast(node.value, constants)
        except ValueError:
            continue
        for candidate in _records_from_python_literal(record):
            if len(items) >= limit:
                break
            item = item_from_record(candidate, f"{path.relative_to(root)}:{node.lineno}")
            if item is not None:
                items.append(item)
    return items


def _module_constants(tree: ast.Module) -> dict[str, Any]:
    constants: dict[str, Any] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        try:
            constants[target.id] = _literal_from_ast(node.value, constants)
        except ValueError:
            continue
    return constants


def _literal_from_ast(node: ast.AST, constants: dict[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in constants:
            return constants[node.id]
        raise ValueError(node.id)
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_literal_from_ast(item, constants) for item in node.elts]
    if isinstance(node, ast.Dict):
        return {
            _literal_from_ast(key, constants): _literal_from_ast(value, constants)
            for key, value in zip(node.keys, node.values)
            if key is not None
        }
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = _literal_from_ast(node.operand, constants)
        if isinstance(value, (int, float)):
            return -value
    raise ValueError(type(node).__name__)


def _records_from_python_literal(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def _items_from_text_file(path: Path, root: Path, limit: int) -> list[BenchmarkItem]:
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if len(text) < 80:
        return []
    record = {
        "prompt": truncate(text),
        "rubric": "Candidate answer should satisfy the benchmark task or prompt described in this source file.",
    }
    item = item_from_record(record, str(path.relative_to(root)))
    return [item] if item is not None and limit > 0 else []


def _items_from_jsonl(path: Path, root: Path, limit: int) -> list[BenchmarkItem]:
    items: list[BenchmarkItem] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for index, line in enumerate(handle):
            if len(items) >= limit or index >= MAX_RECORDS_PER_FILE:
                break
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            item = item_from_record(record, f"{path.relative_to(root)}:{index + 1}")
            if item is not None:
                items.append(item)
    return items


def _items_from_csv(path: Path, root: Path, limit: int) -> list[BenchmarkItem]:
    items: list[BenchmarkItem] = []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader):
            if len(items) >= limit or index >= MAX_RECORDS_PER_FILE:
                break
            item = item_from_record(row, f"{path.relative_to(root)}:{index + 2}")
            if item is not None:
                items.append(item)
    return items


def _items_from_parquet(path: Path, root: Path, limit: int) -> list[BenchmarkItem]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required to read parquet benchmark data") from exc
    table = pq.read_table(path)
    items: list[BenchmarkItem] = []
    for index, row in enumerate(table.slice(0, min(MAX_RECORDS_PER_FILE, table.num_rows)).to_pylist()):
        if len(items) >= limit:
            break
        item = item_from_record(row, f"{path.relative_to(root)}:{index + 1}")
        if item is not None:
            items.append(item)
    return items


def _items_from_json_payload(payload: Any, source: Path, limit: int) -> list[BenchmarkItem]:
    items: list[BenchmarkItem] = []
    for record in walk_records(payload):
        if len(items) >= limit:
            break
        item = item_from_record(record, str(source))
        if item is not None:
            items.append(item)
    return items


def walk_records(payload: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    def visit(value: Any) -> None:
        if len(records) >= MAX_RECORDS_PER_FILE:
            return
        if isinstance(value, dict):
            records.append(value)
            for nested in value.values():
                if isinstance(nested, (dict, list)):
                    visit(nested)
        elif isinstance(value, list):
            for nested in value[:MAX_RECORDS_PER_FILE]:
                visit(nested)

    visit(payload)
    return records


def item_from_record(record: Any, source: str) -> BenchmarkItem | None:
    if not isinstance(record, dict):
        return None
    question = _first_text(record, QUESTION_KEYS)
    expected_key, expected_value = _first_value_with_key(record, ANSWER_KEYS)
    if not question:
        return None
    grading = "exact"
    if expected_value is None:
        if not _is_task_like_record(record, source, question):
            return None
        expected = "Candidate answer should satisfy the task requirements from the benchmark prompt."
        grading = "task_compliance"
    else:
        grading = "rubric" if expected_key in RUBRIC_KEYS else "exact"
        expected = stringify_expected(expected_value, preserve_rubric=grading == "rubric")
        if not expected and _is_task_like_record(record, source, question):
            grading = "task_compliance"
            expected = "Candidate answer should satisfy the task requirements from the benchmark prompt."
        if grading == "exact" and _looks_like_reference_file(expected):
            grading = "task_compliance"
            expected = "Candidate answer should satisfy the task requirements using the referenced benchmark files."
    if not expected:
        return None
    choices = _choices_from_record(record)
    return BenchmarkItem(
        question=truncate(question),
        expected=truncate(expected, 3000 if grading == "rubric" else 1000),
        choices=choices,
        source=source,
        metadata={
            "keys": sorted(str(key) for key in record.keys())[:30],
            "grading": grading,
            "expected_key": expected_key or "",
        },
    )


def _is_task_like_record(record: dict[str, Any], source: str, question: str) -> bool:
    if len(question.strip()) < 40:
        return False
    source_text = source.lower()
    key_text = " ".join(str(key).lower() for key in record.keys())
    if any(word in source_text for word in ("task", "question", "problem", "benchmark", "eval", "test", "validation")):
        return True
    return any(word in key_text for word in ("task", "question", "problem", "prompt", "requirement"))


def _first_text(record: dict[str, Any], keys: tuple[str, ...]) -> str:
    _, value = _first_value_with_key(record, keys)
    text = _text_from_value(value)
    if text:
        return text
    return ""


def _text_from_value(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict):
                content = item.get("content")
                if isinstance(content, str):
                    parts.append(content.strip())
            elif isinstance(item, str):
                parts.append(item.strip())
        return "\n".join(part for part in parts if part).strip()
    if isinstance(value, dict):
        for key in ("content", "text", "prompt", "instruction", "question"):
            if isinstance(value.get(key), str):
                return value[key].strip()
    return ""


def _first_value(record: dict[str, Any], keys: tuple[str, ...]) -> Any:
    _, value = _first_value_with_key(record, keys)
    return value


def _first_value_with_key(record: dict[str, Any], keys: tuple[str, ...]) -> tuple[str, Any]:
    lowered = {str(key).lower(): key for key in record}
    for candidate in keys:
        if candidate in lowered:
            return candidate, record[lowered[candidate]]
    for key, value in record.items():
        key_text = str(key).lower()
        if any(_key_matches(candidate, key_text) for candidate in keys):
            return key_text, value
    return "", None


def _key_matches(candidate: str, key_text: str) -> bool:
    if len(candidate) <= 5:
        return False
    return key_text.endswith(candidate) or key_text.startswith(candidate + "_")


def _choices_from_record(record: dict[str, Any]) -> dict[str, str]:
    value = _first_value(record, CHOICE_KEYS)
    choices: dict[str, str] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            label = normalize_choice_label(str(key), len(choices))
            choices[label] = truncate(str(item), 1000)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            label = normalize_choice_label("", index)
            choices[label] = truncate(str(item), 1000)
    else:
        for label in ("A", "B", "C", "D", "E"):
            if label in record:
                choices[label] = truncate(str(record[label]), 1000)
            elif label.lower() in record:
                choices[label] = truncate(str(record[label.lower()]), 1000)
    return choices


def normalize_choice_label(value: str, index: int) -> str:
    cleaned = value.strip().upper()
    if re.fullmatch(r"[A-Z]", cleaned):
        return cleaned
    if cleaned.isdigit():
        numeric = int(cleaned)
        if 0 <= numeric <= 25:
            return chr(ord("A") + numeric)
        if 1 <= numeric <= 26:
            return chr(ord("A") + numeric - 1)
    return chr(ord("A") + index)


def stringify_expected(value: Any, preserve_rubric: bool = False) -> str:
    if isinstance(value, str):
        value = value.strip()
        return value if preserve_rubric else extract_expected_from_rubric(value)
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        if not value:
            return ""
        if preserve_rubric:
            return stringify_rubric(value)
        return str(value[0]).strip()
    if isinstance(value, dict):
        if preserve_rubric:
            return stringify_rubric(value)
        for key in ("answer", "text", "label", "value"):
            if key in value:
                return stringify_expected(value[key])
    return ""


def stringify_rubric(value: Any) -> str:
    if isinstance(value, str):
        text = value.strip()
        parsed = parse_json_object(text)
        if parsed is not None:
            return stringify_rubric(parsed)
        return text
    if isinstance(value, list):
        parts = [stringify_rubric(item) for item in value]
        return "\n".join(f"- {part}" for part in parts if part)
    if isinstance(value, dict):
        if "criteria" in value:
            return str(value["criteria"]).strip()
        if "requirements" in value:
            return str(value["requirements"]).strip()
        return "; ".join(f"{key}: {item}" for key, item in value.items() if item not in (None, ""))
    return str(value).strip()


def _looks_like_reference_file(value: str) -> bool:
    text = value.strip().lower()
    if not text:
        return False
    if re.search(r"\.(csv|json|jsonl|xlsx|xls|pdf|txt|png|jpg|jpeg|parquet|zip)(?:$|[\s#?])", text):
        return True
    return text.startswith(("reference_files/", "references/", "attachments/", "files/"))


def extract_expected_from_rubric(value: str) -> str:
    for pattern in (
        r"(?:expected answer|answer)\s+is\s+([^.\n]+)",
        r"(?:expected|gold|target)\s*[:\-]\s*([^.\n]+)",
    ):
        match = re.search(pattern, value, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return value


def run_model_evaluations(benchmark: str, items: list[BenchmarkItem]) -> list[dict[str, Any]]:
    if not _is_remote_provider():
        return []
    evaluations: list[dict[str, Any]] = []
    for item in items:
        evaluations.append(run_model_on_item(benchmark, item))
    return evaluations


def run_model_on_item(benchmark: str, item: BenchmarkItem) -> dict[str, Any]:
    base_url = os.environ.get("AGENT_BENCH_BASE_URL", "").rstrip("/")
    model = os.environ.get("AGENT_BENCH_MODEL", "")
    choice_text = ""
    if item.choices:
        choice_text = "\nChoices:\n" + "\n".join(f"{key}. {value}" for key, value in sorted(item.choices.items()))
        response_instruction = 'Return JSON exactly like {"answer":"A","confidence":0.0}.'
    else:
        response_instruction = 'Return JSON exactly like {"answer":"your final answer","confidence":0.0}.'
    body = json.dumps(
        {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Answer this benchmark task. Do not include hidden reasoning or scratch work. "
                        "Return only a compact JSON object."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "/no_think\n"
                        f"Benchmark: {benchmark}\n"
                        f"Question/task from benchmark data:\n{item.question}"
                        f"{choice_text}\n"
                        f"{response_instruction}"
                    ),
                },
            ],
            "temperature": 0,
            "max_tokens": _max_answer_tokens(item),
            "stream": False,
        }
    ).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get("AGENT_BENCH_API_KEY", "")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    probe = request.Request(f"{base_url}/chat/completions", data=body, headers=headers, method="POST")
    try:
        with request.urlopen(probe, timeout=120) as response:
            response_body = response.read().decode("utf-8", errors="replace")
    except error.HTTPError as exc:
        response_body = exc.read().decode("utf-8", errors="replace")
        return evaluation_payload(item, "", 0.0, response_body[-2000:] or str(exc), status_code=exc.code)
    except Exception as exc:
        return evaluation_payload(item, "", 0.0, str(exc))

    try:
        parsed = json.loads(response_body)
    except json.JSONDecodeError:
        return evaluation_payload(item, "", 0.0, "model response was not JSON", response=response_body[-2000:])

    content = extract_openai_content(parsed)
    answer = extract_answer(content)
    score, grade = grade_answer(benchmark, item, answer)
    error_message = ""
    if score < 1.0:
        if grade.get("reason"):
            error_message = str(grade["reason"])
        else:
            error_message = f"expected {item.expected!r}, got {answer!r}"
    return evaluation_payload(
        item,
        answer,
        score,
        error_message,
        status_code=200,
        content_sample=content[:500],
        usage=parsed.get("usage") if isinstance(parsed.get("usage"), dict) else {},
        grade=grade,
    )


def _max_answer_tokens(item: BenchmarkItem) -> int:
    grading = item.metadata.get("grading")
    if grading in {"rubric", "task_compliance"}:
        return 1600
    return 512


def grade_answer(benchmark: str, item: BenchmarkItem, answer: str) -> tuple[float, dict[str, Any]]:
    grading = item.metadata.get("grading", "exact")
    if grading == "exact":
        score = score_answer(answer, item.expected, item.choices)
        return score, {"method": "exact", "score": score}
    return judge_answer(benchmark, item, answer, str(grading))


def judge_answer(benchmark: str, item: BenchmarkItem, answer: str, method: str) -> tuple[float, dict[str, Any]]:
    if not answer:
        return 0.0, {"method": method, "score": 0.0, "reason": "empty candidate answer"}
    base_url = os.environ.get("AGENT_BENCH_BASE_URL", "").rstrip("/")
    model = os.environ.get("AGENT_BENCH_MODEL", "")
    rubric = item.expected
    if method == "task_compliance":
        rubric = "Grade whether the candidate answer directly satisfies the benchmark task prompt. Award partial credit only for concrete, correct, task-relevant work."
    body = json.dumps(
        {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are grading a benchmark response. Do not include hidden reasoning or scratch work. "
                        "Return only compact JSON with "
                        '{"score":0.0,"passed":false,"reason":"short reason"}. '
                        "The score must be between 0 and 1."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "/no_think\n"
                        f"Benchmark: {benchmark}\n"
                        f"Grading method: {method}\n\n"
                        f"Task prompt:\n{item.question}\n\n"
                        f"Rubric or expected behavior:\n{rubric}\n\n"
                        f"Candidate answer:\n{answer}\n"
                    ),
                },
            ],
            "temperature": 0,
            "max_tokens": 300,
            "stream": False,
        }
    ).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get("AGENT_BENCH_API_KEY", "")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    probe = request.Request(f"{base_url}/chat/completions", data=body, headers=headers, method="POST")
    try:
        with request.urlopen(probe, timeout=120) as response:
            response_body = response.read().decode("utf-8", errors="replace")
    except Exception as exc:
        return 0.0, {"method": method, "score": 0.0, "reason": f"judge request failed: {exc}"}
    try:
        parsed = json.loads(response_body)
    except json.JSONDecodeError:
        return 0.0, {"method": method, "score": 0.0, "reason": "judge response was not JSON"}
    content = extract_openai_content(parsed)
    grade = parse_json_object(content)
    if not isinstance(grade, dict):
        return 0.0, {"method": method, "score": 0.0, "reason": "judge content was not a JSON object", "judge_sample": content[:300]}
    score = coerce_unit_score(grade.get("score"))
    passed = bool(grade.get("passed")) if "passed" in grade else score >= 1.0
    return score, {
        "method": method,
        "score": score,
        "passed": passed,
        "reason": str(grade.get("reason", ""))[:500],
        "judge_sample": content[:500],
        "usage": parsed.get("usage") if isinstance(parsed.get("usage"), dict) else {},
    }


def extract_openai_content(parsed: Any) -> str:
    choices = parsed.get("choices") if isinstance(parsed, dict) else None
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    message = first.get("message")
    if isinstance(message, dict) and isinstance(message.get("content"), str):
        return message["content"]
    text = first.get("text")
    return text if isinstance(text, str) else ""


def evaluation_payload(
    item: BenchmarkItem,
    answer: str,
    score: float,
    error_message: str,
    **extra: Any,
) -> dict[str, Any]:
    payload = {
        "source": item.source,
        "question": item.question,
        "answer": answer,
        "expected": item.expected,
        "choices": item.choices,
        "metadata": item.metadata,
        "score": score,
        "passed": score >= 1.0,
        "error": error_message,
    }
    payload.update(extra)
    return payload


def summarize_evaluations(evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    if not evaluations:
        return {}
    passed = sum(1 for item in evaluations if item.get("passed"))
    total = len(evaluations)
    first = evaluations[0]
    grading_methods = sorted(
        {
            str(item.get("metadata", {}).get("grading") or item.get("grade", {}).get("method"))
            for item in evaluations
            if isinstance(item.get("metadata"), dict) or isinstance(item.get("grade"), dict)
        }
        - {""}
    )
    return {
        "ok": passed == total,
        "score": _average_score(evaluations),
        "answer": f"{passed}/{total}",
        "expected": f"{total}/{total}",
        "question": first.get("question", ""),
        "content_sample": first.get("content_sample", ""),
        "grading_methods": grading_methods,
        "usage": first.get("usage", {}),
        "error": "" if passed == total else f"{passed}/{total} benchmark records passed",
    }


def score_answer(answer: str, expected: str, choices: dict[str, str]) -> float:
    if not answer:
        return 0.0
    if choices:
        answer_label = normalize_answer_label(answer, choices)
        expected_label = normalize_answer_label(expected, choices)
        if answer_label and expected_label and answer_label == expected_label:
            return 1.0
    return 1.0 if normalize_text(answer) == normalize_text(expected) else 0.0


def coerce_unit_score(value: Any) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))
    if isinstance(value, str):
        match = re.search(r"-?\d+(?:\.\d+)?", value)
        if match:
            numeric = float(match.group(0))
            if numeric > 1.0:
                numeric /= 100.0
            return max(0.0, min(1.0, numeric))
    return 0.0


def normalize_answer_label(value: str, choices: dict[str, str]) -> str:
    stripped = value.strip()
    upper = stripped.upper()
    if upper in choices:
        return upper
    match = re.search(r"\b([A-Z])\b", upper)
    if match and match.group(1) in choices:
        return match.group(1)
    normalized = normalize_text(stripped)
    for label, choice in choices.items():
        if normalized == normalize_text(choice):
            return label
    if stripped.isdigit():
        return normalize_choice_label(stripped, 0)
    return ""


def extract_answer(content: str) -> str:
    text = content.strip()
    if not text:
        return ""
    payload = parse_json_object(text)
    if isinstance(payload, dict):
        value = payload.get("answer")
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, (int, float, bool)):
            return str(value)
    match = re.search(r"\b(?:answer|final)\s*[:\-]\s*([A-Za-z0-9_.+-]+)", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text[:500]


def parse_json_object(text: str) -> object | None:
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9.+-]+", " ", value.lower())).strip()


def truncate(value: str, limit: int = MAX_FIELD_CHARS) -> str:
    value = value.strip()
    if len(value) <= limit:
        return value
    return value[: limit - 20].rstrip() + " ...[truncated]"


def _sample_limit() -> int:
    raw = os.environ.get("AGENT_BENCH_SAMPLE_LIMIT") or os.environ.get("AGENT_BENCH_LIMIT") or "3"
    try:
        return max(1, min(20, int(raw)))
    except ValueError:
        return 3


def _average_score(evaluations: list[dict[str, Any]]) -> float:
    if not evaluations:
        return 0.0
    return sum(float(item.get("score", 0.0)) for item in evaluations) / len(evaluations)


def _is_remote_provider() -> bool:
    return (
        os.environ.get("AGENT_BENCH_PROVIDER") == "openai-compatible"
        and bool(os.environ.get("AGENT_BENCH_BASE_URL", "").strip())
        and bool(os.environ.get("AGENT_BENCH_MODEL", "").strip())
    )


if __name__ == "__main__":
    raise SystemExit(main())
