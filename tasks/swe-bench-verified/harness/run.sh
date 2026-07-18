#!/usr/bin/env bash
set -euo pipefail

# The outer runner supplies the public display name; no endpoint or secret is stored here.

exec agent-bench-probe --benchmark "${AGENT_BENCH_BENCHMARK_NAME:?}" --kind public-benchmark
