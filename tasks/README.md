# Public Benchmark Tasks

The old seven hand-written task categories have been removed. `public_benchmarks.json` now contains Docker-backed `external_benchmark` descriptors for all requested public benchmarks.

Each descriptor credits the upstream benchmark, records the observed license as metadata, and points at the upstream repository or Hugging Face dataset. The runner builds `docker/external-benchmark.Dockerfile`, clones the upstream source inside that container, and runs the descriptor command with neutral model environment variables such as `AGENT_BENCH_BASE_URL`, `AGENT_BENCH_MODEL`, and `AGENT_BENCH_OUTPUT_DIR`. These credits are also copied into external benchmark result details.

## Included

- SWE-bench
- GDPval
- PaperBench
- SWE-Lancer
- MLE-bench
- SWE-bench Verified
- AutomationBench
- OSWorld
- Humanity's Last Exam
- BioMystery Bench
- ExploitBench
- codeneedle
- StockBench
- InvestorBench
- QuantCode-Bench
- FinMCP-Bench
- FinToolBench
- Finance Agent v2
- FinanceMath
- EDINET-Bench

## Notes

The default command for each descriptor performs a local Docker readiness run: clone the public benchmark source or dataset, sample the files present, and write normalized metadata to `agent_bench_result.json`. This keeps every requested benchmark runnable through the same Agent Bench Docker path while preserving a clear place to replace the probe command with a full upstream harness invocation for benchmarks that require credentials, GPUs, API subscriptions, or large datasets.

Benchmark data is not vendored into this repository; upstream projects are credited in the task metadata.
