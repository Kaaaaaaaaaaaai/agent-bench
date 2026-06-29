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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agent-bench")
    subparsers = parser.add_subparsers(dest="command")

    run = subparsers.add_parser("run", help="Run benchmark tasks against a model")
    run.add_argument("--provider", choices=["openai-compatible", "ollama-native", "mock"], default="mock")
    run.add_argument("--base-url", default=None)
    run.add_argument("--model", default=None)
    run.add_argument("--api-key-env", default=None)
    run.add_argument("--tasks", default="tasks")
    run.add_argument("--out", default="runs/latest")
    run.add_argument("--request-concurrency", type=int, default=8)
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
    run.add_argument("--temperature", type=float, default=0.0)
    run.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    run.add_argument("--json-mode", choices=["auto", "on", "off"], default="auto")
    run.add_argument("--sandbox", choices=["docker", "subprocess"], default="docker")
    run.add_argument("--sandbox-image", default="agent-bench-python:3.12")
    run.add_argument("--external-launcher-image", default="agent-bench-external:python3.12")
    run.add_argument(
        "--asset-root",
        default=str(DEFAULT_EXTERNAL_ASSET_ROOT),
        help="Host directory for external benchmark asset cache; mount this path into Docker runs",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 2
    if args.command == "run":
        include = {item.strip() for item in args.include.split(",")} if args.include else None
        config = RunConfig(
            provider=args.provider,
            base_url=args.base_url,
            model=args.model,
            api_key_env=args.api_key_env,
            tasks_dir=Path(args.tasks),
            out=Path(args.out),
            request_concurrency=args.request_concurrency,
            eval_concurrency=args.eval_concurrency,
            timeout=args.timeout,
            external_timeout=args.external_timeout,
            model_request_timeout=args.model_request_timeout,
            limit=args.limit,
            include=include,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            json_mode=args.json_mode,
            sandbox=args.sandbox,
            sandbox_image=args.sandbox_image,
            external_launcher_image=args.external_launcher_image,
            asset_root=Path(args.asset_root),
        )
        summary = asyncio.run(run_benchmark(config))
        print(f"Tasks: {summary['passed_count']} / {summary['task_count']} passed")
        print(f"Valid score: {summary['score_valid_tasks_only']:.2f}%")
        if "model_score_valid_tasks_only" in summary:
            print(f"Model-valid score: {summary['model_score_valid_tasks_only']:.2f}%")
        if summary.get("raw_score_all_tasks") != summary.get("score_valid_tasks_only") or summary.get("skipped_count"):
            print(f"Raw score: {summary['raw_score_all_tasks']:.2f}%")
            print(
                "Coverage/setup/grader: "
                f"{summary.get('skipped_count', 0)} coverage gaps, "
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


if __name__ == "__main__":
    raise SystemExit(main())
