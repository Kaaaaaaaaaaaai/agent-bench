#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Record a lightweight local Docker benchmark readiness result.")
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--kind", default="repository")
    args = parser.parse_args()

    cwd = Path.cwd()
    files = sorted(str(path.relative_to(cwd)) for path in cwd.rglob("*") if path.is_file())[:200]
    markers = [name for name in ("README.md", "README.rst", "pyproject.toml", "requirements.txt", "Dockerfile", "LICENSE", "LICENSE.md") if (cwd / name).exists()]
    output_dir = Path(os.environ.get("AGENT_BENCH_OUTPUT_DIR", "/outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "score": 1.0 if files else 0.0,
        "benchmark": args.benchmark,
        "kind": args.kind,
        "repository": os.environ.get("AGENT_BENCH_REPOSITORY", ""),
        "repository_ref": os.environ.get("AGENT_BENCH_REPOSITORY_REF", ""),
        "subdir": os.environ.get("AGENT_BENCH_SUBDIR", ""),
        "model": os.environ.get("AGENT_BENCH_MODEL", ""),
        "file_count_sampled": len(files),
        "markers": markers,
        "sample_files": files,
    }
    (output_dir / "agent_bench_result.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if files else 2


if __name__ == "__main__":
    raise SystemExit(main())
