import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent_bench.models import Task


@dataclass(slots=True)
class SandboxResult:
    passed_cases: int
    total_cases: int
    case_results: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    timed_out: bool = False


class BaseSandbox:
    async def run(self, task: Task, code: str, timeout_seconds: float) -> SandboxResult:
        raise NotImplementedError


class DockerSandbox(BaseSandbox):
    def __init__(
        self,
        image: str = "agent-bench-python:3.12",
        docker_bin: str = "docker",
        memory: str = "256m",
        cpus: str = "1.0",
    ) -> None:
        self.image = image
        self.docker_bin = docker_bin
        self.memory = memory
        self.cpus = cpus

    async def run(self, task: Task, code: str, timeout_seconds: float) -> SandboxResult:
        return await asyncio.to_thread(self._run_sync, task, code, timeout_seconds)

    def _run_sync(self, task: Task, code: str, timeout_seconds: float) -> SandboxResult:
        if shutil.which(self.docker_bin) is None:
            return SandboxResult(
                passed_cases=0,
                total_cases=len(task.test_cases),
                error="Docker is required for coding evaluation but was not found",
            )

        try:
            temp_root = _docker_temp_root()
        except OSError as exc:
            return SandboxResult(
                passed_cases=0,
                total_cases=len(task.test_cases),
                error=f"Unable to prepare Docker sandbox temp directory: {exc}",
            )

        with tempfile.TemporaryDirectory(prefix="agent-bench-", dir=temp_root) as tmp:
            workdir = Path(tmp)
            _write_sandbox_files(workdir, task, code)
            command = [
                self.docker_bin,
                "run",
                "--rm",
                "--network",
                "none",
                "--cpus",
                self.cpus,
                "--memory",
                self.memory,
                "--pids-limit",
                "64",
                "--read-only",
                "--tmpfs",
                "/tmp:rw,nosuid,nodev,size=64m",
                "--user",
                "65534:65534",
                "-v",
                f"{workdir}:/work:ro",
                "-w",
                "/work",
                self.image,
                "python",
                "/work/harness.py",
            ]
            return _run_command(command, timeout_seconds + 2.0, len(task.test_cases))


class SubprocessSandbox(BaseSandbox):
    """Local sandbox for tests and explicit non-Docker runs."""

    async def run(self, task: Task, code: str, timeout_seconds: float) -> SandboxResult:
        return await asyncio.to_thread(self._run_sync, task, code, timeout_seconds)

    def _run_sync(self, task: Task, code: str, timeout_seconds: float) -> SandboxResult:
        with tempfile.TemporaryDirectory(prefix="agent-bench-local-") as tmp:
            workdir = Path(tmp)
            _write_sandbox_files(workdir, task, code)
            command = [sys.executable, str(workdir / "harness.py")]
            return _run_command(command, timeout_seconds, len(task.test_cases), cwd=workdir)


def make_sandbox(kind: str, image: str) -> BaseSandbox:
    if kind == "docker":
        return DockerSandbox(image=image)
    if kind == "subprocess":
        return SubprocessSandbox()
    raise ValueError(f"Unsupported sandbox: {kind}")


def _docker_temp_root() -> str | None:
    configured = os.environ.get("AGENT_BENCH_SANDBOX_TMPDIR")
    if not configured:
        return None
    root = Path(configured)
    root.mkdir(parents=True, exist_ok=True)
    return str(root)


def _write_sandbox_files(workdir: Path, task: Task, code: str) -> None:
    workdir.chmod(0o755)
    prelude = (
        "from typing import *\n"
        "from collections import *\n"
        "import bisect, functools, heapq, itertools, math, re\n\n"
    )
    (workdir / "candidate.py").write_text(prelude + code + "\n", encoding="utf-8")
    (workdir / "task.json").write_text(
        json.dumps(
            {
                "id": task.id,
                "function_name": task.function_name,
                "test_cases": task.test_cases,
                "comparison": task.comparison,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (workdir / "harness.py").write_text(HARNESS_CODE, encoding="utf-8")


def _run_command(
    command: list[str],
    timeout_seconds: float,
    total_cases: int,
    cwd: Path | None = None,
) -> SandboxResult:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return SandboxResult(
            passed_cases=0,
            total_cases=total_cases,
            error=f"Execution timed out after {timeout_seconds:.1f}s",
            timed_out=True,
        )

    stdout = completed.stdout.strip()
    if stdout:
        try:
            payload = json.loads(stdout.splitlines()[-1])
            return SandboxResult(
                passed_cases=int(payload.get("passed_cases", 0)),
                total_cases=int(payload.get("total_cases", total_cases)),
                case_results=list(payload.get("case_results", [])),
                error=payload.get("error"),
                timed_out=bool(payload.get("timed_out", False)),
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            pass

    stderr = completed.stderr.strip()
    error = stderr or stdout or f"Sandbox process exited with code {completed.returncode}"
    return SandboxResult(passed_cases=0, total_cases=total_cases, error=error)


HARNESS_CODE = r'''
import copy
import importlib.util
import json
import traceback
from collections import Counter
from pathlib import Path


FUNCTION_COMPARISONS = {
    "twoSum": "two_sum",
    "longestPalindrome": "longest_palindrome",
    "topKFrequent": "top_k_frequent",
    "subsets": "unordered_nested_sorted",
    "letterCombinations": "unordered_list",
    "generateParenthesis": "unordered_list",
    "combinationSum": "unordered_nested_sorted",
    "permute": "unordered_nested_exact",
}


def main():
    task = json.loads(Path("task.json").read_text(encoding="utf-8"))
    try:
        module = load_candidate()
    except BaseException as exc:
        print(json.dumps({
            "passed_cases": 0,
            "total_cases": len(task["test_cases"]),
            "case_results": [],
            "error": "candidate import failed: " + format_error(exc),
        }))
        return

    results = []
    passed = 0
    for index, case in enumerate(task["test_cases"]):
        try:
            actual = run_case(module, task, case)
            ok = compare(task, case, actual)
            if ok:
                passed += 1
            results.append({
                "index": index,
                "passed": ok,
                "actual": make_json_safe(actual),
                "expected": case.get("output"),
                "error": None,
            })
        except BaseException as exc:
            results.append({
                "index": index,
                "passed": False,
                "actual": None,
                "expected": case.get("output"),
                "error": format_error(exc),
            })

    print(json.dumps({
        "passed_cases": passed,
        "total_cases": len(task["test_cases"]),
        "case_results": results,
        "error": None,
    }, ensure_ascii=False))


def load_candidate():
    spec = importlib.util.spec_from_file_location("candidate", "candidate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_case(module, task, case):
    name = task["function_name"]
    if is_class_case(name, case["input"]):
        return run_class_case(module, name, case["input"])
    func = getattr(module, name)
    kwargs = copy.deepcopy(case["input"])
    original = copy.deepcopy(kwargs)
    result = func(**kwargs)
    if result is None and case.get("output") is not None:
        result = infer_mutated_result(kwargs, original)
    return result


def is_class_case(class_name, input_payload):
    operations = input_payload.get("operations")
    arguments = input_payload.get("arguments")
    return (
        isinstance(operations, list)
        and isinstance(arguments, list)
        and bool(operations)
        and operations[0] == class_name
    )


def run_class_case(module, class_name, input_payload):
    cls = getattr(module, class_name)
    operations = input_payload["operations"]
    arguments = input_payload["arguments"]
    instance = None
    outputs = []
    for operation, args in zip(operations, arguments):
        if operation == class_name:
            instance = cls(*args)
            outputs.append(None)
        else:
            method = getattr(instance, operation)
            outputs.append(method(*args))
    return outputs


def infer_mutated_result(kwargs, original):
    for key in ("nums1", "nums", "matrix", "image"):
        if key in kwargs and kwargs[key] != original.get(key):
            return kwargs[key]
    if len(kwargs) == 1:
        return next(iter(kwargs.values()))
    for key in ("nums1", "nums", "matrix", "image"):
        if key in kwargs:
            return kwargs[key]
    return None


def compare(task, case, actual):
    expected = case.get("output")
    mode = task.get("comparison") or FUNCTION_COMPARISONS.get(task["function_name"], "exact")
    if mode == "exact":
        return actual == expected
    if mode == "unordered_list":
        return sorted(serialized_items(actual)) == sorted(serialized_items(expected))
    if mode == "unordered_nested_exact":
        return sorted(serialized_items(actual)) == sorted(serialized_items(expected))
    if mode == "unordered_nested_sorted":
        return sorted(serialized_items(sort_inner_lists(actual))) == sorted(serialized_items(sort_inner_lists(expected)))
    if mode == "two_sum":
        return valid_two_sum(case["input"], actual)
    if mode == "top_k_frequent":
        return valid_top_k_frequent(case["input"], actual)
    if mode == "longest_palindrome":
        return valid_longest_palindrome(case["input"], expected, actual)
    return actual == expected


def serialized_items(value):
    if not isinstance(value, list):
        return []
    return [json.dumps(item, sort_keys=True, separators=(",", ":")) for item in value]


def sort_inner_lists(value):
    if not isinstance(value, list):
        return value
    normalized = []
    for item in value:
        if isinstance(item, list):
            normalized.append(sorted(item))
        else:
            normalized.append(item)
    return normalized


def valid_two_sum(input_payload, actual):
    nums = input_payload["nums"]
    target = input_payload["target"]
    if not isinstance(actual, list) or len(actual) != 2:
        return False
    if not all(isinstance(index, int) for index in actual):
        return False
    i, j = actual
    if i == j or i < 0 or j < 0 or i >= len(nums) or j >= len(nums):
        return False
    return nums[i] + nums[j] == target


def valid_top_k_frequent(input_payload, actual):
    nums = input_payload["nums"]
    k = input_payload["k"]
    if not isinstance(actual, list) or len(actual) != k:
        return False
    if len(set(actual)) != len(actual):
        return False
    counts = Counter(nums)
    if k == 0:
        return actual == []
    if any(item not in counts for item in actual):
        return False
    selected_min = min(counts[item] for item in actual)
    for item, count in counts.items():
        if item not in actual and count > selected_min:
            return False
    return True


def valid_longest_palindrome(input_payload, expected, actual):
    source = input_payload["s"]
    if not isinstance(actual, str):
        return False
    return actual == actual[::-1] and actual in source and len(actual) == len(expected)


def make_json_safe(value):
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)


def format_error(exc):
    message = "".join(traceback.format_exception_only(type(exc), exc)).strip()
    return message[-500:]


if __name__ == "__main__":
    main()
'''
