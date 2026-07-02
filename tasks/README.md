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
| `PB_005` | Coding | SWE-bench Verified | <https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified> |
| `PB_008` | Biosecurity | BioMystery Bench | <https://huggingface.co/datasets/Anthropic/BioMysteryBench-preview> |
| `PB_009` | Security | ExploitBench | <https://github.com/exploitbench/exploitbench> |
| `PB_010` | Long Context | codeneedle | <https://github.com/alexziskind1/codeneedle> |
| `PB_011` | Finance | StockBench | <https://github.com/ChenYXxxx/stockbench> |
| `PB_012` | Finance | InvestorBench | <https://github.com/felis33/INVESTOR-BENCH> |
| `PB_013` | Finance | QuantCode-Bench | <https://github.com/LimexAILab/QuantCode-Bench> |
| `PB_014` | Finance | FinMCP-Bench | <https://huggingface.co/datasets/DianJin/FinMCP-Bench> |
| `PB_015` | Finance | FinToolBench | <https://github.com/Double-wk/FinToolBench> |
| `PB_016` | Finance | Finance Agent v2 | <https://github.com/vals-ai/finance-agent-v2> |
| `PB_017` | Finance | FinanceMath | <https://github.com/yale-nlp/FinanceMath> |

## Notes

The default command for each descriptor performs a local Docker run: clone the public benchmark source or dataset, sample the files present, query the configured model endpoint for sampled tasks when the adapter can provide the required capability, and write normalized metadata to `agent_bench_result.json`.

The probe uses an explicit adapter contract for workspace preparation, tool schemas, agent execution, output collection, and grading. `tool_call` rows use the stateful agent loop with native calls or text-call parsing fallbacks and fail preflight as `failed_missing_required_tool` when benchmark metadata names a tool the adapter does not expose. Repo-patch rows require target repository checkout metadata and a checkout/patch/diff canary; an official patch/test grader is used when configured, with a model-judge task-compliance fallback otherwise. File-artifact and office-document rows require declared input assets, a read/write/list/collect canary, isolated per-item workspaces, and generated output files. Unsupported benchmark-native capabilities are reported as `skipped_unsupported_capability`; missing assets, Git LFS pointer stubs, invalid task context, and harness defects remain strict invalid statuses such as `failed_missing_assets`, `failed_invalid_task_context`, and `failed_harness_setup`.

GDPval and PaperBench use the git-ignored `agent-bench-assets/` cache for best-effort Git LFS asset materialization. Cached assets are mounted into Docker at `/asset-cache` and copied into the benchmark checkout before the probe runs.

Because each row represents a whole benchmark, external benchmark descriptors use a longer wall-clock timeout than ordinary single-task prompts. The CLI default is `--external-timeout 21600` per benchmark row and `--model-request-timeout 1800` per model or judge request inside the probe. For local model servers, use low benchmark-row concurrency first, usually `--request-concurrency 1` or `2`.

Some upstream full leaderboard harnesses require large downloads, GPUs, VM/KVM support, paid APIs, external accounts, or gated services. For those, the descriptor still runs locally in Docker against the public source, but it is marked skipped/setup-failed unless the adapter can materialize the required assets and grader.

Benchmark data is not vendored into this repository.
