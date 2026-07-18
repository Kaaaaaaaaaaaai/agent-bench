# Public Benchmark Tasks

Production benchmark registrations live in dedicated `tasks/<benchmark-slug>/` folders. Each folder owns its manifest, harness, configs, small data, and asset lock file. The manifest's `display_name` is the canonical benchmark name; no separate synthetic code is stored. `public_benchmarks.json` is retained only as a legacy compatibility descriptor file.

Each descriptor records upstream credit, license metadata, and a citation URL. Production execution now requires strict manifest metadata: pinned source revision, official conditions, assets, container policy, adapter, scoring, and reporting fields. Descriptors that cannot prove official-run equivalence fail validation as `failed_manifest_validation` instead of being silently treated as supported official benchmarks.

## Active Benchmarks

| Group | Benchmark | Citation |
| --- | --- | --- |
| Coding | SWE-bench Verified | <https://www.swebench.com/verified.html> |
| Long Context | codeneedle | <https://github.com/alexziskind1/codeneedle> |
| Finance | StockBench | <https://github.com/ChenYXxxx/stockbench> |
| Finance | InvestorBench | <https://github.com/felis33/INVESTOR-BENCH> |
| Finance | QuantCode-Bench | <https://github.com/LimexAILab/QuantCode-Bench> |
| Finance | FinMCP-Bench | <https://huggingface.co/datasets/DianJin/FinMCP-Bench> |
| Finance | Finance Agent v2 | <https://github.com/vals-ai/finance-agent-v2> |
| Finance | FinanceMath | <https://github.com/yale-nlp/FinanceMath> |
| Terminal | Terminal-Bench 2.0 | <https://www.harborframework.com/docs/tutorials/running-terminal-bench> |
| Coding | NL2RepoBench | <https://github.com/multimodal-art-projection/NL2RepoBench> |
| Coding | DeepSWE | <https://github.com/datacurve-ai/deep-swe> |
| Coding | ProgramBench | <https://github.com/facebookresearch/ProgramBench> |
| Tool Use | MCP Atlas | <https://github.com/scaleapi/mcp-atlas> |
| Tool Use | Toolathlon | <https://github.com/hkust-nlp/Toolathlon> |
| Knowledge | Humanity's Last Exam | <https://github.com/centerforaisafety/hle> |
| Long Context | LongBench | <https://github.com/THUDM/LongBench> |
| Coding | BigCodeBench | <https://github.com/bigcode-project/bigcodebench> |
| Math | MathArena | <https://github.com/eth-sri/matharena> |
| Agentic | CLAW-Eval | <https://github.com/claw-eval/claw-eval> |
| General Knowledge | MMLU-Pro | <https://github.com/TIGER-AI-Lab/MMLU-Pro> |
| Math | AIME25 (no tools) | <https://github.com/eth-sri/matharena> |
| Math | HMMT Feb25 (no tools) | <https://github.com/eth-sri/matharena> |
| Math | HMMT Feb25 (with tools) | <https://github.com/eth-sri/matharena> |
| Reasoning | GPQA Diamond (no tools) | <https://github.com/idavidrein/gpqa> |
| Reasoning | GPQA Diamond (with tools) | <https://github.com/idavidrein/gpqa> |
| Coding | LiveCodeBench v5 (2024-07 to 2024-12) | <https://github.com/LiveCodeBench/LiveCodeBench> |
| Coding | SciCode (subtask) | <https://github.com/scicode-bench/SciCode> |
| Agentic | Terminal-Bench Hard (NVIDIA 48-task subset) | <https://github.com/laude-institute/terminal-bench> |
| Agentic | TauBench V2 Airline | <https://github.com/sierra-research/tau2-bench> |
| Agentic | TauBench V2 Retail | <https://github.com/sierra-research/tau2-bench> |
| Agentic | TauBench V2 Telecom | <https://github.com/sierra-research/tau2-bench> |
| Instruction Following | IFBench (prompt) | <https://github.com/allenai/IFBench> |
| Instruction Following | Scale AI MultiChallenge | <https://huggingface.co/datasets/ScaleAI/MultiChallenge> |
| Chat | Arena-Hard-V2 | <https://github.com/lmarena/arena-hard-auto> |
| Long Context | AA-LCR | <https://huggingface.co/datasets/ArtificialAnalysis/AA-LCR> |
| Long Context | RULER 256k | <https://github.com/NVIDIA/RULER> |
| Long Context | RULER 512k | <https://github.com/NVIDIA/RULER> |
| Long Context | RULER 1M | <https://github.com/NVIDIA/RULER> |
| Multilingual | MMLU-ProX | <https://github.com/weihao1115/MMLU-ProX> |
| Multilingual | WMT24++ (en-to-xx) | <https://github.com/google-research/mt-metrics-eval> |

## Notes

The orchestrator discovers active production benchmarks from `tasks/*/manifest.json` and runs one outer container per benchmark. The legacy descriptor probe path is useful for compatibility tests, but production benchmark entries use strict manifests and official conditions.

For the NVIDIA Puzzle model-card suite, the model card determines benchmark membership and reported conditions only. Repository and dataset provenance comes from each benchmark's original publisher; NVIDIA integration repositories are not treated as benchmark assets unless NVIDIA is itself the original benchmark publisher, as with RULER.

Each of those leaf harnesses runs its generic pinned-source integration check under the fixed `source_adapter_smoke` contract. That contract can validate delegated asset materialization, extraction, endpoint calls, and adapter scoring, but it always reports `official_equivalent: false` and `included_in_official_score: false`. The declared NeMo evaluator command remains the separate official contract and continues to fail closed when its native component, grader, simulator, or other required capability is unavailable.

The probe uses an explicit adapter contract for workspace preparation, tool schemas, agent execution, output collection, and grading. `tool_call` rows use the stateful agent loop with native calls or text-call parsing fallbacks and fail preflight as `failed_missing_required_tool` when benchmark metadata names a tool the adapter does not expose. FinMCP-Bench is converted to static transcript reasoning and does not require live Qieman MCP tools. Repo-patch rows require target repository checkout metadata and a checkout/patch/diff canary; an official patch/test grader is used when configured, with `task_compliance_fallback` otherwise. File-artifact and office-document rows require declared input assets, a read/write/list/collect canary, isolated per-item workspaces, and generated output files. Unsupported benchmark-native capabilities are reported as `skipped_unsupported_capability`; missing assets, Git LFS pointer stubs, invalid task context, and harness defects remain strict invalid statuses such as `failed_missing_assets`, `failed_invalid_task_context`, and `failed_harness_setup`.

Benchmarks with explicit cache recipes use the git-ignored `agent-bench-assets/` cache for upstream data materialization. Each benchmark container sees only its own task folder at `/benchmark/task`, its own materialized asset root at `/benchmark/assets`, writable outputs at `/outputs`, and a tmpfs workspace.

Because each row represents a whole benchmark, external benchmark descriptors use a longer wall-clock timeout than ordinary single-task prompts. The CLI default is `--external-timeout 21600` per benchmark row and `--model-request-timeout 1800` per model or judge request inside the probe. For local model servers, use low benchmark-row concurrency first, usually `--request-concurrency 1` or `2`.

Some upstream full leaderboard harnesses require large downloads, GPUs, VM/KVM support, paid APIs, external accounts, or gated services. For those, the descriptor or manifest is marked failed/skipped unless the adapter can materialize the required assets and grader under official-equivalent conditions.

The NVIDIA Puzzle model-card conditions are first-class leaf registrations (`MMLU-Pro` through `WMT24++ (en-to-xx)`) rather than a synthetic aggregate score. Existing `Humanity's Last Exam` supplies the HLE no-tools leaf. Tool/no-tool, TauBench domain, and RULER context conditions remain independently selectable and reportable. Component-native results require the declared datasets, containers, context capacity, graders, and user simulators and fail closed when those requirements are absent. The model card's exact 48-task Terminal-Bench package is not public; the registration records that provenance blocker and does not substitute the current 47-task Evaluator definition.

Benchmark data is not vendored into this repository.
