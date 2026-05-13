import argparse
import asyncio
from pathlib import Path

from agent_bench.runner import RunConfig, run_benchmark


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
    run.add_argument("--timeout", type=float, default=60.0)
    run.add_argument("--limit", type=int, default=None)
    run.add_argument("--include", default=None, help="Comma-separated category, source file, or task IDs")
    run.add_argument("--temperature", type=float, default=0.0)
    run.add_argument("--max-tokens", type=int, default=4096)
    run.add_argument("--json-mode", choices=["auto", "on", "off"], default="auto")
    run.add_argument("--sandbox", choices=["docker", "subprocess"], default="docker")
    run.add_argument("--sandbox-image", default="agent-bench-python:3.12")

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
            limit=args.limit,
            include=include,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            json_mode=args.json_mode,
            sandbox=args.sandbox,
            sandbox_image=args.sandbox_image,
        )
        summary = asyncio.run(run_benchmark(config))
        print(f"Tasks: {summary['passed_count']} / {summary['task_count']} passed")
        print(f"Total score: {summary['total_score']:.2f}%")
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
