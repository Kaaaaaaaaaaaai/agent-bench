# Agent Bench

Agent Bench is a Python 3.12+ benchmark runner for evaluating local or remote language models against JSON task files and cloned public benchmark sources. It supports OpenAI-compatible servers such as vLLM and Ollama's OpenAI endpoint, plus Ollama's native chat API.

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
  bench run --provider mock --limit 10
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
  --request-concurrency 2
```

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
  --model llama3.1
```

Run against Ollama's native API:

```bash
docker run --rm -it \
  --add-host host.docker.internal:host-gateway \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v /tmp/agent-bench-sandboxes:/tmp/agent-bench-sandboxes \
  -v "$PWD/runs:/opt/agent-bench/runs" \
  agent-bench \
  bench run \
  --provider ollama-native \
  --base-url http://host.docker.internal:11434 \
  --model llama3.1
```

By default, results are written to a timestamped directory under `runs/` and copied to `runs/latest/`.

## Timeouts For Local Models

The bundled rows are whole benchmark descriptors, not single quiz questions. For remote providers, each `external_benchmark` row runs a Docker launcher that may clone a repository, extract benchmark records, call the configured model several times, and run judge calls. The default external benchmark timeout is therefore 21,600 seconds (6 hours) per benchmark row.

The normal `--timeout` option still controls non-external single-task requests and defaults to 60 seconds. External benchmark rows use:

- `--external-timeout`: wall-clock timeout for each benchmark Docker container; default `21600`.
- `--model-request-timeout`: timeout for each model or judge request made inside the benchmark probe; default `1800`.
- `--max-tokens`: generation budget for normal tasks and each agent-loop model turn inside external benchmarks; default `16384`.

For a single local model server, start with `--request-concurrency 1` or `--request-concurrency 2`. Higher values can reduce elapsed wall time only when the local serving stack has enough GPU/CPU capacity for concurrent benchmark rows.


## Task Files

The bundled `tasks/` directory contains public benchmark descriptors only. Each bundled task is an `external_benchmark` entry that runs its descriptor command locally in Docker through `docker/external-benchmark.Dockerfile`.

External benchmark task shape:

```json
{
  "id": "PB_001",
  "type": "external_benchmark",
  "question": "Run SWE-bench locally in Docker using its public benchmark harness.",
  "benchmark": {
    "name": "SWE-bench",
    "group": "Coding",
    "homepage": "https://github.com/SWE-bench/SWE-bench",
    "repository": "https://github.com/SWE-bench/SWE-bench.git",
    "dataset_id": "princeton-nlp/SWE-bench",
    "ref": "main",
    "license": "MIT",
    "credit": "SWE-bench authors",
    "citation": "https://github.com/SWE-bench/SWE-bench#citation--license",
    "docker": {
      "image": "agent-bench-external:python3.12",
      "setup": [],
      "command": "agent-bench-probe --benchmark \"SWE-bench\" --kind public-benchmark",
      "environment": [],
      "volumes": []
    }
  }
}
```

The descriptor file records 12 active public benchmark IDs. Upstream credits, license notes, and citation URLs are recorded in `tasks/public_benchmarks.json` and summarized in `tasks/README.md`; license metadata is not used as a loader gate.

Relevant selection controls:

- `--profile full_active`: all active configured suites.
- `--suite PB_009`: run a specific active suite by ID or benchmark name.

When running a remote provider, Agent Bench starts the benchmark launcher container, clones the upstream benchmark source or dataset inside Docker, extracts real benchmark task records, calls the configured model endpoint for each sampled task, and grades the model's answer. The launcher receives neutral model settings through environment variables:

- `AGENT_BENCH_BASE_URL`
- `AGENT_BENCH_MODEL`
- `AGENT_BENCH_PROVIDER`
- `AGENT_BENCH_OUTPUT_DIR`
- `AGENT_BENCH_API_KEY`, when `--api-key-env` is provided

The bundled descriptors use `agent-bench-probe` to normalize public benchmark formats through an explicit adapter contract:

- `prepare_task(task) -> TaskWorkspace`
- `available_tools(task) -> ToolSchema[]`
- `run_agent_loop(task, workspace, tools) -> AgentRun`
- `collect_outputs(run) -> OutputBundle`
- `grade(task, outputs) -> GradeResult`

Capabilities are reported only when an adapter can provide the required workspace, tools, output collection, and grader. `tool_call` rows use the stateful agent tool loop, including native OpenAI-compatible tool calls and text tool-call fallbacks for models that emit tagged JSON; rows that name a required tool fail preflight as `failed_missing_required_tool` if that tool is not exposed. FinMCP-Bench is evaluated as static transcript reasoning and does not expose live MCP tools. Browser/GUI rows are evaluated from extracted task data and repository files when no live display is available. Repo-patch rows require target repository metadata and a checkout/patch/diff canary; when `AGENT_BENCH_REPO_PATCH_GRADER` is set, the official patch/test grader is used, otherwise a `task_compliance_fallback` grades the produced diff. File-artifact and office-document rows run a read/write/list/collect canary and use isolated per-item workspaces populated only with declared task inputs. Missing, corrupt, or Git LFS pointer-stub assets are marked `failed_missing_assets`.

The default external asset cache is the git-ignored `agent-bench-assets/` directory. Benchmarks with cache recipes, such as ExploitBench, download upstream data into that cache before Docker starts; the launcher then copies cached, repository-relative assets into the container checkout before probing.

Extracted chat-answer records are graded with deterministic methods when possible and LLM judging only when no deterministic grader exists. LLM judges must return strict JSON; invalid judge output is retried with a repair prompt and then marked `failed_grader` with `judge_parse_error` rather than counted as a model-answer failure.

- `exact`: deterministic answer, patch, label, or multiple-choice matching.
- `numeric`: deterministic numeric matching with unit normalization for FinanceMath-style records.
- `rubric`: a published rubric or benchmark prompt is used for model-based grading.
- `task_compliance`: the benchmark exposes a real task prompt without a standalone answer key, so a grading call scores whether the response satisfies the task requirements.

Each descriptor records the adapter, capability contract, extracted source paths, item counts, grading method, strict statuses, normalized score, and whether each row is included in official scoring in `${AGENT_BENCH_OUTPUT_DIR}/agent_bench_result.json`.

## Outputs

Each run writes:

- `raw_responses.jsonl`
- `graded_results.jsonl`
- `results.csv`
- `summary.json`
- `summary.html`

The HTML report is fully static and includes metric cards, an average-score radar chart, grouped benchmark scores, category summaries, status distributions, detailed task rows, and benchmark citations.

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
