# Agent Bench

Agent Bench is a Python 3.12+ benchmark runner for evaluating local or remote language models against JSON task files and manifest-registered public benchmark suites. Production target models must expose an OpenAI-compatible chat-completions endpoint. vLLM and Ollama are supported through their OpenAI-compatible endpoints only; native Ollama APIs are not a supported target interface.

## Build

```bash
docker build -t agent-bench .
```

The runner container prepares the coding-evaluation image automatically in the attached Docker daemon on first use. Create persistent host directories for benchmark outputs and evaluator scratch files:

```bash
mkdir -p runs /tmp/agent-bench-sandboxes
```

## Run

Run the offline mock provider:

```bash
docker run --rm -it \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v /tmp/agent-bench-sandboxes:/tmp/agent-bench-sandboxes \
  -v "$PWD/runs:/opt/agent-bench/runs" \
  agent-bench \
  bench run --provider mock --limit 10 \
  --judge-provider "same-as-target"
```

Run against vLLM:

```bash
docker run --rm -it \
  --add-host host.docker.internal:host-gateway \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v /tmp/agent-bench-sandboxes:/tmp/agent-bench-sandboxes \
  -v "$PWD/runs:/opt/agent-bench/runs" \
  agent-bench \
  bench run \
  --provider openai-compatible \
  --base-url http://host.docker.internal:8000/v1 \
  --model <model> \
  --request-concurrency 2 \
  --judge-provider "same-as-target"
```

Run directly from a local checkout against an OpenAI-compatible server:

```bash
curl http://<model-server>:<port>/v1/models

UV_CACHE_DIR=/private/tmp/uv-cache-agent-bench \
uv run agent-bench run \
  --provider openai-compatible \
  --base-url http://<model-server>:<port>/v1 \
  --model <model-id-from-/v1/models> \
  --request-concurrency 2 \
  --eval-concurrency 2 \
  --timeout 180 \
  --sandbox subprocess \
  --out runs/local-full
```

Use `http://`, not `http;/`. The `--model` value should match the model ID returned by `/v1/models`. When running the Docker image against a model server on the same host, use `http://host.docker.internal:<port>/v1`; when running from the host checkout, use the server's reachable host or LAN address directly. Keep private endpoint addresses and credentials in local shell arguments or environment variables, never in repository files.

Run against Ollama's OpenAI-compatible endpoint:

```bash
docker run --rm -it \
  --add-host host.docker.internal:host-gateway \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v /tmp/agent-bench-sandboxes:/tmp/agent-bench-sandboxes \
  -v "$PWD/runs:/opt/agent-bench/runs" \
  agent-bench \
  bench run \
  --provider openai-compatible \
  --base-url http://host.docker.internal:11434/v1 \
  --model llama3.1 \
  --judge-provider "same-as-target"
```

By default, results are written to a timestamped directory under `runs/` and copied to `runs/latest/`.
Runs print timestamped configuration, progress, and error logs to stderr; pass `--quiet` to suppress those logs while keeping the final summary.

## Tool-Call Parsers

Agent Bench records and executes native OpenAI-compatible `tool_calls` first. Some local model servers instead emit tool calls as text. For those cases, pass `--tool-call-parser <name>` or the older `--tool-parser <name>`; both flags write the same `AGENT_BENCH_TOOL_PARSER` setting. The default `auto` parser tries known output shapes in order, so most runs should not need a model-specific parser unless a text format is ambiguous.

Supported local parser names include:

- `openai-native`: OpenAI-compatible `message.tool_calls`.
- `vllm-compatible`: OpenAI-style `tool_calls` plus legacy `message.function_call`.
- `hermes` / `qwen3.5`: Hermes/Qwen tagged JSON such as `<tool_call>{"name":"...","arguments":{...}}</tool_call>`.
- `longcat`: LongCat tagged JSON using `<longcat_tool_call>...</longcat_tool_call>`.
- `xlam`: JSON-array tool calls, including code blocks, `[TOOL_CALLS]`, `<tool_call>`, and post-`</think>` JSON.
- `functiongemma`: FunctionGemma calls such as `<start_function_call>call:name{arg:<escape>value<escape>}<end_function_call>`.
- `pythonic`: Python-call lists such as `[tool_name(arg='value')]`.
- `olmo3`: Olmo 3 `<function_calls>...</function_calls>` blocks with newline-separated Pythonic calls.
- `json-in-content`: fallback JSON object or array parsing from assistant content.
- `none`: disable text-call parsing.

Several vLLM parser names normalize to these local parsers for convenience, including `openai`, `qwen`, `qwen2.5`, `qwen3_xml`, `llama3_json`, `llama4_pythonic`, `granite`, `granite4`, `granite-20b-fc`, `internlm`, `jamba`, `mistral`, `deepseek_v3`, `deepseek_v31`, `kimi_k2`, `hunyuan_a13b`, `cohere_command3`, `glm45`, `glm47`, `gigachat3`, and `apertus`. This is intentionally narrower than vLLM's server-side parser system: Agent Bench parses completed OpenAI-compatible responses and text fallbacks, but it does not install vLLM chat templates, reasoning parsers, or parser plugins. If vLLM itself is launched with `--enable-auto-tool-choice --tool-call-parser ...`, it will usually return native `tool_calls`, and Agent Bench will record them without needing a matching local parser. See the vLLM tool-calling reference for server-side flags and model-specific chat templates: https://docs.vllm.ai/en/stable/features/tool_calling/

## Timeouts For Local Models

The bundled rows are whole benchmark descriptors, not single quiz questions. For remote providers, each `external_benchmark` row runs a Docker launcher that may clone a repository, extract benchmark records, call the configured model several times, and run judge calls. The default external benchmark timeout is therefore 21,600 seconds (6 hours) per benchmark row.

The normal `--timeout` option still controls non-external single-task requests and defaults to 60 seconds. External benchmark rows use:

- `--external-timeout`: wall-clock timeout for each benchmark Docker container; default `21600`.
- `--model-request-timeout`: timeout for each model or judge request made inside the benchmark probe; default `1800`.
- `--max-tokens`: generation budget for normal tasks and each agent-loop model turn inside external benchmarks; default `16384`.

For a single local model server, start with `--request-concurrency 1` or `--request-concurrency 2`. Higher values can reduce elapsed wall time only when the local serving stack has enough GPU/CPU capacity for concurrent benchmark rows.


## Task Files

The bundled `tasks/` directory contains one production folder per benchmark. New production benchmark suites should be registered with `tasks/<benchmark-slug>/manifest.json` plus the benchmark-owned harness, configs, data, and asset lock files. The manifest's `display_name` is the canonical name used by the runner; there is no separate synthetic code. The runner keeps `tasks/public_benchmarks.json` only for backward-compatible descriptor discovery.

Strict production manifests must declare official leaderboard-equivalent conditions: pinned source commit or dataset revision, official split, official scoring method, official prompt format, official grader command/config, required assets with validation, container settings, adapter path, scoring normalization, and reporting metadata. Incomplete or moving-ref descriptors fail as `failed_manifest_validation` and are shown in `summary.json` and `summary.html`; the runner does not silently execute sample-only or approximate benchmark variants as supported production entries.

External benchmark folder shape:

```text
tasks/<benchmark-slug>/
  manifest.json
  README.md
  harness/
    Dockerfile
    run.sh
    normalize.py
  configs/
    official.json
  data/
  assets.lock.json
```

The active suite currently records 40 public benchmarks by their canonical names. Upstream credits, license notes, and citation URLs are recorded in each task folder manifest and summarized in `tasks/README.md`; license metadata is not used as a loader gate.

Relevant selection controls:

- `--profile full_active`: all active configured suites.
- `--suite "SWE-bench Verified"`: run a specific active suite by benchmark name.

When running a remote provider, Agent Bench starts a main-process OpenAI-compatible recording proxy and points the benchmark container at that proxy. The proxy forwards requests to the configured target endpoint, records raw requests/responses into `raw_responses.jsonl`, and keeps upstream API secrets out of the benchmark container. The launcher receives neutral model settings through environment variables:

- `AGENT_BENCH_BASE_URL`
- `AGENT_BENCH_MODEL`
- `AGENT_BENCH_PROVIDER`
- `AGENT_BENCH_OUTPUT_DIR`
- parser/generation settings such as `AGENT_BENCH_TOOL_PARSER` (`--tool-call-parser`), `AGENT_BENCH_MAX_TOKENS`, and `AGENT_BENCH_CONTEXT_LIMIT`

Most benchmark folders currently use `agent-bench-probe` from their benchmark-owned `harness/run.sh` to normalize public benchmark formats through an explicit adapter contract:

- `prepare_task(task) -> TaskWorkspace`
- `available_tools(task) -> ToolSchema[]`
- `run_agent_loop(task, workspace, tools) -> AgentRun`
- `collect_outputs(run) -> OutputBundle`
- `grade(task, outputs) -> GradeResult`

Capabilities are reported only when an adapter can provide the required workspace, tools, output collection, and grader. `tool_call` rows use the stateful agent tool loop, including native OpenAI-compatible tool calls and text tool-call fallbacks for models that emit tagged JSON, JSON arrays, FunctionGemma calls, or Pythonic calls; rows that name a required tool fail preflight as `failed_missing_required_tool` if that tool is not exposed. FinMCP-Bench is evaluated as static transcript reasoning and does not expose live MCP tools. Finance Agent v2 uses a deterministic CRWD fixture backend for smoke coverage; `web_search`, `edgar_search`, `parse_html_page`, `retrieve_information`, and `price_history` are exposed only when fixture checksums and semantic canaries pass. Browser/GUI rows are evaluated from extracted task data and repository files when no live display is available. Repo-patch rows require target repository metadata and a checkout/patch/diff canary; when `AGENT_BENCH_REPO_PATCH_GRADER` is set, that official patch/test grader is used. Otherwise repo-patch generation is reported only as a non-official integration smoke result. File-artifact and office-document rows run a read/write/list/collect canary and use isolated per-item workspaces populated only with declared task inputs. Missing, corrupt, or Git LFS pointer-stub assets are marked `failed_missing_assets`.

The default external asset cache is the git-ignored `agent-bench-assets/` directory. Benchmarks with cache recipes download upstream data into that cache before Docker starts; each benchmark container receives only that benchmark's materialized assets at `/benchmark/assets`.

Extracted chat-answer records are graded with deterministic methods when possible and LLM judging only when no deterministic grader exists. LLM judges must return strict JSON; invalid judge output is retried with a repair prompt and then marked `failed_grader` with `judge_parse_error` rather than counted as a model-answer failure.

- `exact`: deterministic answer, patch, label, or multiple-choice matching.
- `numeric`: deterministic numeric matching with unit normalization for FinanceMath-style records.
- `rubric`: a published rubric or benchmark prompt is used for model-based grading.
- `task_compliance`: the benchmark exposes a real task prompt without a standalone answer key, so a grading call scores whether the response satisfies the task requirements.

Each descriptor or manifest records the adapter, capability contract, extracted source paths, item counts, grading method, strict statuses, normalized score, and whether each row is included in official scoring in `${AGENT_BENCH_OUTPUT_DIR}/agent_bench_result.json`.

## Outputs

Each run writes:

- `raw_responses.jsonl`
- `graded_results.jsonl`
- `results.csv`
- `summary.json`
- `summary.html`

The HTML report is fully static and includes run metadata, score cards, a category radar chart, benchmark score table, target/judge/container/asset metadata, failure/status details, and benchmark credits/citations/licenses. It is generated from `summary.json` and relative artifact paths and has no CDN dependency.

Scores are reported over selected suites.

Scores are reported three ways:

- `raw_score_all_tasks`: includes every configured task for audit.
- `score_valid_tasks_only`: excludes coverage-gap, setup, dataset-extraction, missing-asset, missing-tool, invalid-context, grader-invalid, and timed-out task rows.
- `model_score_valid_tasks_only`: includes only valid rows where the adapter fully provided the required benchmark capabilities.

The `Benchmark Scores` table is grouped by benchmark group such as `Coding`, `Finance`, `GUI`, `Research`, `Security`, `Work`, and `Reasoning`. It includes only:

- benchmark name
- model score
- passed/evaluated item count
- grading method

The `Task Results` table uses benchmark names rather than internal task IDs where benchmark metadata is available. Detailed diagnostics, including strict statuses, capability contracts, raw judge text, parsed judge JSON, and judge usage, remain available in `summary.json`, `graded_results.jsonl`, and `results.csv`.

Remote providers are queried with streaming responses so the runner can record:

- time to first token (TTFT)
- output tokens per second when the provider exposes output token counts
- benchmark wall-clock run time
- accumulated per-task end-to-end time

Those values are written into `raw_responses.jsonl`, `graded_results.jsonl`, `results.csv`, and the top-level summary metrics where applicable. Providers that do not return output-token usage still report TTFT and timing data, while tokens-per-second remains `n/a`.

## Docker Sandboxing

The default coding evaluator runs generated Python inside Docker with no network, memory and process limits, a read-only `/work` mount, and a timeout. The benchmark runner container needs the Docker socket mount so it can launch those evaluator containers, and it needs the shared `/tmp/agent-bench-sandboxes` mount so the host Docker daemon can read generated harness files.

External benchmark suites run in separate disposable containers. The runner drops capabilities, uses `no-new-privileges`, applies pids/memory/CPU/timeout options where configured, mounts the asset cache read-only, and copies outputs back before removing the container. Host Docker socket access is mounted only when a benchmark manifest declares `requires_host_docker_socket`. The older `--allow-host-docker-socket` flag is deprecated and accepted only for script compatibility.
