# Public Benchmark Tasks

The old hand-written task categories have been removed. `public_benchmarks.json` contains legacy Docker-backed `external_benchmark` descriptors for public benchmark discovery. New production benchmark registrations should live in `benchmarks/<benchmark_name>/manifest.yaml`.

Each descriptor records upstream credit, license metadata, and a citation URL. Production execution now requires strict manifest metadata: pinned source revision, official conditions, assets, container policy, adapter, scoring, and reporting fields. Descriptors that cannot prove official-run equivalence fail validation as `failed_manifest_validation` instead of being silently treated as supported official benchmarks.

## Active Benchmarks

| ID | Group | Benchmark | Citation |
| --- | --- | --- | --- |
| `PB_001` | Coding | SWE-bench | <https://github.com/SWE-bench/SWE-bench> |
| `PB_004` | Coding | SWE-Lancer | <https://github.com/openai/frontier-evals/tree/main/project/swelancer> |
| `PB_005` | Coding | SWE-bench Verified | <https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified> |
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

The legacy default command for each descriptor performs a local Docker run: clone the public benchmark source or dataset, sample the files present, query the configured model endpoint for sampled tasks when the adapter can provide the required capability, and write normalized metadata to `agent_bench_result.json`. That legacy probe path is useful for discovery and compatibility tests, but production benchmark entries must use strict manifests and official conditions.

The probe uses an explicit adapter contract for workspace preparation, tool schemas, agent execution, output collection, and grading. `tool_call` rows use the stateful agent loop with native calls or text-call parsing fallbacks and fail preflight as `failed_missing_required_tool` when benchmark metadata names a tool the adapter does not expose. FinMCP-Bench is converted to static transcript reasoning and does not require live Qieman MCP tools. Repo-patch rows require target repository checkout metadata and a checkout/patch/diff canary; an official patch/test grader is used when configured, with `task_compliance_fallback` otherwise. File-artifact and office-document rows require declared input assets, a read/write/list/collect canary, isolated per-item workspaces, and generated output files. Unsupported benchmark-native capabilities are reported as `skipped_unsupported_capability`; missing assets, Git LFS pointer stubs, invalid task context, and harness defects remain strict invalid statuses such as `failed_missing_assets`, `failed_invalid_task_context`, and `failed_harness_setup`.

Benchmarks with explicit cache recipes use the git-ignored `agent-bench-assets/` cache for upstream data materialization. Cached assets are mounted into Docker at `/asset-cache` and copied into the benchmark checkout before the probe runs.

Because each row represents a whole benchmark, external benchmark descriptors use a longer wall-clock timeout than ordinary single-task prompts. The CLI default is `--external-timeout 21600` per benchmark row and `--model-request-timeout 1800` per model or judge request inside the probe. For local model servers, use low benchmark-row concurrency first, usually `--request-concurrency 1` or `2`.

Some upstream full leaderboard harnesses require large downloads, GPUs, VM/KVM support, paid APIs, external accounts, or gated services. For those, the descriptor or manifest is marked failed/skipped unless the adapter can materialize the required assets and grader under official-equivalent conditions.

Benchmark data is not vendored into this repository.
