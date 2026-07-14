import argparse
import asyncio
from pathlib import Path

from agent_bench.runner import (
    DEFAULT_EXTERNAL_ASSET_ROOT,
    DEFAULT_EXTERNAL_TIMEOUT_SECONDS,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL_REQUEST_TIMEOUT_SECONDS,
    DEFAULT_TASK_TIMEOUT_SECONDS,
    RunConfig,
    run_benchmark,
)
from agent_bench.tool_parsers import TOOL_PARSER_CANONICAL_NAMES, normalize_parser_name


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agent-bench")
    subparsers = parser.add_subparsers(dest="command")

    run = subparsers.add_parser("run", help="Run benchmark tasks against a model")
    run.add_argument(
        "--provider",
        choices=["openai-compatible", "mock"],
        default="mock",
        help="Target interface. Production runs support OpenAI-compatible endpoints only; mock is for offline smoke tests.",
    )
    run.add_argument("--base-url", default=None)
    run.add_argument("--model", default=None)
    run.add_argument("--api-key-env", default="OPENAI_API_KEY")
    run.add_argument("--tasks", default="tasks")
    run.add_argument("--benchmark-root", default="benchmarks")
    run.add_argument("--out", default="runs/latest")
    run.add_argument("--request-concurrency", "--concurrency", type=int, default=8)
    run.add_argument("--eval-concurrency", type=int, default=4)
    run.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TASK_TIMEOUT_SECONDS,
        help="Per-task timeout for non-external tasks, in seconds",
    )
    run.add_argument(
        "--external-timeout",
        type=float,
        default=DEFAULT_EXTERNAL_TIMEOUT_SECONDS,
        help="Wall-clock timeout for each Docker-backed external benchmark row, in seconds",
    )
    run.add_argument(
        "--model-request-timeout",
        type=float,
        default=DEFAULT_MODEL_REQUEST_TIMEOUT_SECONDS,
        help="Timeout for each model or judge request made inside an external benchmark, in seconds",
    )
    run.add_argument("--limit", type=int, default=None)
    run.add_argument("--include", default=None, help="Comma-separated category, source file, or task IDs")
    run.add_argument("--profile", default="full_active", help="Benchmark profile to run; default: full_active")
    run.add_argument(
        "--suite",
        action="append",
        default=None,
        help="Suite ID or benchmark name to run; may be passed multiple times or comma-separated",
    )
    run.add_argument("--temperature", type=float, default=0.0)
    run.add_argument("--top-p", type=float, default=None)
    run.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    run.add_argument("--seed", type=int, default=None)
    run.add_argument("--max-retries", type=int, default=2)
    run.add_argument(
        "--tool-parser",
        "--tool-call-parser",
        choices=TOOL_PARSER_CANONICAL_NAMES,
        default="auto",
        dest="tool_parser",
        type=normalize_parser_name,
        help="Tool-call parser for OpenAI-compatible responses; accepts vLLM-style names such as hermes, longcat, xlam, functiongemma, pythonic, and olmo3.",
    )
    run.add_argument("--context-window", type=int, default=None)
    run.add_argument("--stop", action="append", default=None)
    run.add_argument("--json-mode", choices=["auto", "on", "off"], default="auto")
    run.add_argument(
        "--sandbox",
        choices=["docker", "subprocess"],
        default="docker",
        help=(
            "Coding evaluator isolation. Docker is the secure default; subprocess executes "
            "model-generated code directly on the host and is intended only for trusted tests."
        ),
    )
    run.add_argument("--sandbox-image", default="agent-bench-python:3.12")
    run.add_argument("--external-launcher-image", default="agent-bench-external:python3.12")
    run.add_argument(
        "--asset-root",
        default=str(DEFAULT_EXTERNAL_ASSET_ROOT),
        help="Host directory for external benchmark asset cache; mount this path into Docker runs",
    )
    run.add_argument(
        "--allow-host-docker-socket",
        action="store_true",
        help=(
            "Allow trusted benchmark manifests to mount the host Docker socket. This grants "
            "root-equivalent host access and is disabled by default."
        ),
    )
    run.add_argument("--judge-provider", choices=["openai-compatible", "same-as-target", "none"], default="none")
    run.add_argument("--judge-base-url", default=None)
    run.add_argument("--judge-model", default=None)
    run.add_argument("--judge-api-key-env", default=None)
    run.add_argument("--judge-temperature", type=float, default=0.0)
    run.add_argument("--judge-timeout", type=float, default=DEFAULT_MODEL_REQUEST_TIMEOUT_SECONDS)
    run.add_argument("--judge-max-retries", type=int, default=2)
    run.add_argument("--judge-fallback", choices=["same-as-target", "fail"], default="same-as-target")
    run.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress logs; the final summary is still printed.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 2
    if args.command == "run":
        if args.provider == "openai-compatible" and (not args.base_url or not args.model):
            parser.error("--base-url and --model are required for --provider openai-compatible")
        _validate_cli_arguments(parser, args)
        _validate_cli_runtime_path(parser, Path(args.out), Path("runs"), "--out")
        _validate_cli_runtime_path(parser, Path(args.asset_root), Path("agent-bench-assets"), "--asset-root")
        include = _split_selectors(args.include)
        suite_ids = _split_selectors(args.suite)
        config = RunConfig(
            provider=args.provider,
            base_url=args.base_url,
            model=args.model,
            api_key_env=args.api_key_env,
            tasks_dir=Path(args.tasks),
            benchmark_root=Path(args.benchmark_root),
            out=Path(args.out),
            request_concurrency=args.request_concurrency,
            eval_concurrency=args.eval_concurrency,
            timeout=args.timeout,
            external_timeout=args.external_timeout,
            model_request_timeout=args.model_request_timeout,
            limit=args.limit,
            include=include,
            profile=args.profile,
            suite_ids=suite_ids,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            seed=args.seed,
            max_retries=args.max_retries,
            tool_parser=args.tool_parser,
            context_window=args.context_window,
            stop=args.stop,
            json_mode=args.json_mode,
            sandbox=args.sandbox,
            sandbox_image=args.sandbox_image,
            external_launcher_image=args.external_launcher_image,
            asset_root=Path(args.asset_root),
            judge_provider=args.judge_provider,
            judge_base_url=args.judge_base_url,
            judge_model=args.judge_model,
            judge_api_key_env=args.judge_api_key_env,
            judge_temperature=args.judge_temperature,
            judge_timeout=args.judge_timeout,
            judge_max_retries=args.judge_max_retries,
            judge_fallback=args.judge_fallback,
            allow_host_docker_socket=args.allow_host_docker_socket,
            cli_args=list(argv) if argv is not None else None,
            log_to_terminal=not args.quiet,
        )
        summary = asyncio.run(run_benchmark(config))
        print(f"Tasks: {summary['passed_count']} / {summary['task_count']} passed")
        print(
            "Scored-suite score: "
            f"{summary['valid_judged_score'] * 100.0:.2f}% "
            f"(suite coverage {summary['valid_task_count']}/{summary['task_count']}, "
            f"item coverage {summary.get('valid_judged_item_count', 0)}/{summary.get('item_count', 0)})"
        )
        print(f"Conservative selected-suite score: {summary['conservative_all_suite_score'] * 100.0:.2f}%")
        if "model_score_valid_tasks_only" in summary:
            print(f"Model-valid score: {summary['model_score_valid_tasks_only']:.2f}%")
        if summary.get("raw_score_all_tasks") != summary.get("score_valid_tasks_only") or summary.get("skipped_count"):
            print(f"Raw score: {summary['raw_score_all_tasks']:.2f}%")
            print(
                "Coverage/setup/grader: "
                f"{summary.get('excluded_suite_count', summary.get('skipped_count', 0))} score exclusions, "
                f"{summary.get('setup_failed_count', 0)} setup failed, "
                f"{summary.get('grader_failure_count', summary.get('judge_parse_failed_count', 0))} "
                "grader failed"
            )
        print(f"Run time: {summary['total_run_duration_seconds']:.3f}s")
        if summary["average_time_to_first_token_seconds"] is not None:
            print(f"Avg TTFT: {summary['average_time_to_first_token_seconds']:.3f}s")
        if summary["average_tokens_per_second"] is not None:
            print(f"Avg tokens/s: {summary['average_tokens_per_second']:.2f}")
        print(f"Results: {summary['metadata']['output_dir']}")
        return 0
    parser.error(f"Unknown command: {args.command}")
    return 2


def _split_selectors(value: str | list[str] | None) -> set[str] | None:
    if value is None:
        return None
    items = value if isinstance(value, list) else [value]
    parsed = {
        part.strip()
        for item in items
        for part in item.split(",")
        if part.strip()
    }
    return parsed or None


def _validate_cli_runtime_path(
    parser: argparse.ArgumentParser,
    path: Path,
    allowed_relative_root: Path,
    flag: str,
) -> None:
    cwd = Path.cwd().resolve()
    root = (cwd / allowed_relative_root).resolve()
    candidate = (cwd / path).resolve() if not path.is_absolute() else path.resolve()
    if candidate == root or candidate.is_relative_to(root):
        return
    parser.error(f"{flag} must be under ./{allowed_relative_root.as_posix()}/")


def _validate_cli_arguments(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    positive_values = {
        "--request-concurrency": args.request_concurrency,
        "--eval-concurrency": args.eval_concurrency,
        "--timeout": args.timeout,
        "--external-timeout": args.external_timeout,
        "--model-request-timeout": args.model_request_timeout,
        "--max-tokens": args.max_tokens,
        "--judge-timeout": args.judge_timeout,
    }
    if args.context_window is not None:
        positive_values["--context-window"] = args.context_window
    for flag, value in positive_values.items():
        if value <= 0:
            parser.error(f"{flag} must be positive")
    for flag, value in {
        "--max-retries": args.max_retries,
        "--judge-max-retries": args.judge_max_retries,
    }.items():
        if value < 0:
            parser.error(f"{flag} must be non-negative")
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be non-negative")
    if args.top_p is not None and not 0 < args.top_p <= 1:
        parser.error("--top-p must be greater than 0 and at most 1")


if __name__ == "__main__":
    raise SystemExit(main())
