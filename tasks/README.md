# Public Benchmark Tasks

Production benchmark registrations live in dedicated `tasks/<benchmark_id>/` folders. Each folder owns its manifest, harness, configs, small data, and asset lock file. `public_benchmarks.json` is retained only as a legacy compatibility descriptor file.

Each descriptor records upstream credit, license metadata, and a citation URL. Production execution now requires strict manifest metadata: pinned source revision, official conditions, assets, container policy, adapter, scoring, and reporting fields. Descriptors that cannot prove official-run equivalence fail validation as `failed_manifest_validation` instead of being silently treated as supported official benchmarks.

## Active Benchmarks

| ID | Group | Benchmark | Citation |
| --- | --- | --- | --- |
| `PB_001` | Coding | SWE-bench | <https://github.com/SWE-bench/SWE-bench> |
| `PB_004` | Coding | SWE-Lancer | <https://github.com/openai/frontier-evals/tree/main/project/swelancer> |
| `PB_009` | Long Context | codeneedle | <https://github.com/alexziskind1/codeneedle> |
| `PB_010` | Finance | StockBench | <https://github.com/ChenYXxxx/stockbench> |
| `PB_011` | Finance | InvestorBench | <https://github.com/felis33/INVESTOR-BENCH> |
| `PB_012` | Finance | QuantCode-Bench | <https://github.com/LimexAILab/QuantCode-Bench> |
| `PB_013` | Finance | FinMCP-Bench | <https://huggingface.co/datasets/DianJin/FinMCP-Bench> |
| `PB_015` | Finance | Finance Agent v2 | <https://github.com/vals-ai/finance-agent-v2> |
| `PB_016` | Finance | FinanceMath | <https://github.com/yale-nlp/FinanceMath> |
| `PB_017` | Terminal | Terminal-Bench 2.0 | <https://www.harborframework.com/docs/tutorials/running-terminal-bench> |
| `PB_018` | Coding | NL2RepoBench | <https://github.com/multimodal-art-projection/NL2RepoBench> |
| `PB_019` | Coding | DeepSWE | <https://github.com/datacurve-ai/deep-swe> |
| `PB_020` | Coding | ProgramBench | <https://github.com/facebookresearch/ProgramBench> |
| `PB_021` | Tool Use | MCP Atlas | <https://github.com/scaleapi/mcp-atlas> |
| `PB_022` | Tool Use | Toolathlon | <https://github.com/hkust-nlp/Toolathlon> |
| `PB_023` | Knowledge | Humanity's Last Exam | <https://github.com/centerforaisafety/hle> |
| `PB_024` | Long Context | LongBench | <https://github.com/THUDM/LongBench> |
| `PB_025` | Coding | BigCodeBench | <https://github.com/bigcode-project/bigcodebench> |
| `PB_026` | Math | MathArena | <https://github.com/eth-sri/matharena> |
| `PB_027` | Agentic | CLAW-Eval | <https://github.com/claw-eval/claw-eval> |

## Notes

The orchestrator discovers active production benchmarks from `tasks/*/manifest.json` and runs one outer container per benchmark. The legacy descriptor probe path is useful for compatibility tests, but production benchmark entries use strict manifests and official conditions.

The probe uses an explicit adapter contract for workspace preparation, tool schemas, agent execution, output collection, and grading. `tool_call` rows use the stateful agent loop with native calls or text-call parsing fallbacks and fail preflight as `failed_missing_required_tool` when benchmark metadata names a tool the adapter does not expose. FinMCP-Bench is converted to static transcript reasoning and does not require live Qieman MCP tools. Repo-patch rows require target repository checkout metadata and a checkout/patch/diff canary; an official patch/test grader is used when configured, with `task_compliance_fallback` otherwise. File-artifact and office-document rows require declared input assets, a read/write/list/collect canary, isolated per-item workspaces, and generated output files. Unsupported benchmark-native capabilities are reported as `skipped_unsupported_capability`; missing assets, Git LFS pointer stubs, invalid task context, and harness defects remain strict invalid statuses such as `failed_missing_assets`, `failed_invalid_task_context`, and `failed_harness_setup`.

Benchmarks with explicit cache recipes use the git-ignored `agent-bench-assets/` cache for upstream data materialization. Each benchmark container sees only its own task folder at `/benchmark/task`, its own materialized asset root at `/benchmark/assets`, writable outputs at `/outputs`, and a tmpfs workspace.

Because each row represents a whole benchmark, external benchmark descriptors use a longer wall-clock timeout than ordinary single-task prompts. The CLI default is `--external-timeout 21600` per benchmark row and `--model-request-timeout 1800` per model or judge request inside the probe. For local model servers, use low benchmark-row concurrency first, usually `--request-concurrency 1` or `2`.

Some upstream full leaderboard harnesses require large downloads, GPUs, VM/KVM support, paid APIs, external accounts, or gated services. For those, the descriptor or manifest is marked failed/skipped unless the adapter can materialize the required assets and grader under official-equivalent conditions.

Benchmark data is not vendored into this repository.
