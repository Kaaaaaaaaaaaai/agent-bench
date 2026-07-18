#!/usr/bin/env bash
set -euo pipefail
export AGENT_BENCH_EVALUATION_CONTRACT=source_adapter_smoke

exec agent-bench-probe --benchmark "${AGENT_BENCH_BENCHMARK_NAME:?}" --kind public-benchmark
