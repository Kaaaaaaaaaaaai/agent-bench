#!/usr/bin/env bash
set -euo pipefail

exec agent-bench-probe --benchmark "${AGENT_BENCH_BENCHMARK_NAME:?}" --kind public-benchmark
