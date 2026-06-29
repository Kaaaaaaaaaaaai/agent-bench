# Public Benchmark Tasks

The old hand-written task categories have been removed. `public_benchmarks.json` contains Docker-backed `external_benchmark` descriptors for the requested public benchmarks.

Each descriptor records upstream credit, license metadata, and a citation URL. The runner builds `docker/external-benchmark.Dockerfile`, clones the upstream repository or Hugging Face dataset inside Docker, and runs the descriptor command with neutral model environment variables such as `AGENT_BENCH_BASE_URL`, `AGENT_BENCH_MODEL`, and `AGENT_BENCH_OUTPUT_DIR`.

## Included Benchmarks

| ID | Group | Benchmark | Citation |
| --- | --- | --- | --- |
| `PB_001` | Coding | SWE-bench | <https://github.com/SWE-bench/SWE-bench> |
| `PB_002` | Work | GDPval | <https://huggingface.co/datasets/openai/gdpval> |
| `PB_003` | Research | PaperBench | <https://github.com/openai/frontier-evals/tree/main/project/paperbench> |
| `PB_004` | Coding | SWE-Lancer | <https://github.com/openai/frontier-evals/tree/main/project/swelancer> |
| `PB_005` | Machine Learning | MLE-bench | <https://github.com/openai/mle-bench> |
| `PB_006` | Coding | SWE-bench Verified | <https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified> |
| `PB_007` | Work | AutomationBench | <https://github.com/zapier/AutomationBench> |
| `PB_008` | GUI | OSWorld | <https://github.com/xlang-ai/OSWorld> |
| `PB_009` | Biosecurity | BioMystery Bench | <https://huggingface.co/datasets/Anthropic/BioMysteryBench-preview> |
| `PB_010` | Security | ExploitBench | <https://github.com/exploitbench/exploitbench> |
| `PB_011` | Long Context | codeneedle | <https://github.com/alexziskind1/codeneedle> |
| `PB_012` | Finance | StockBench | <https://github.com/ChenYXxxx/stockbench> |
| `PB_013` | Finance | InvestorBench | <https://github.com/felis33/INVESTOR-BENCH> |
| `PB_014` | Finance | QuantCode-Bench | <https://github.com/LimexAILab/QuantCode-Bench> |
| `PB_015` | Finance | FinMCP-Bench | <https://huggingface.co/datasets/DianJin/FinMCP-Bench> |
| `PB_016` | Finance | FinToolBench | <https://github.com/Double-wk/FinToolBench> |
| `PB_017` | Finance | Finance Agent v2 | <https://github.com/vals-ai/finance-agent-v2> |
| `PB_018` | Finance | FinanceMath | <https://github.com/yale-nlp/FinanceMath> |
| `PB_019` | Finance | EDINET-Bench | <https://github.com/SakanaAI/EDINET-Bench> |

## Notes

The default command for each descriptor performs a local Docker run: clone the public benchmark source or dataset, sample the files present, query the configured model endpoint for sampled tasks when the adapter can provide the required capability, and write normalized metadata to `agent_bench_result.json`.

The probe uses an explicit adapter contract for workspace preparation, tool schemas, agent execution, output collection, and grading. Generic repository tools do not satisfy benchmark-native `tool_call` requirements. Repo-patch rows require target repository checkout metadata, a checkout/patch/diff/test canary, and an official patch/test grader, not exact reference-diff matching. File-artifact rows require declared input assets, a read/write/list/collect canary, isolated per-item workspaces, and generated output files. Unsupported benchmark-native capabilities are reported as `skipped_unsupported_capability`; missing assets and harness defects remain strict invalid statuses such as `failed_missing_assets` and `failed_harness_setup`.

Because each row represents a whole benchmark, external benchmark descriptors use a longer wall-clock timeout than ordinary single-task prompts. The CLI default is `--external-timeout 21600` per benchmark row and `--model-request-timeout 1800` per model or judge request inside the probe. For local model servers, use low benchmark-row concurrency first, usually `--request-concurrency 1` or `2`.

Some upstream full leaderboard harnesses require large downloads, GPUs, VM/KVM support, paid APIs, external accounts, or gated services. For those, the descriptor still runs locally in Docker against the public source, but it is marked skipped/setup-failed unless the adapter can materialize the required assets and grader.

Benchmark data is not vendored into this repository.
