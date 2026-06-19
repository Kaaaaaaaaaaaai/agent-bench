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
  --model <model>
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


## Task Files

The bundled `tasks/` directory contains the screenshot-style benchmark rows as external benchmark descriptors. Each bundled task is an `external_benchmark` entry that runs its descriptor command locally in Docker through `docker/external-benchmark.Dockerfile` when the descriptor points at a public cloneable source.

External benchmark task shape:

```json
{
  "id": "PB_001",
  "type": "external_benchmark",
  "question": "Run SWE-bench locally in Docker using its public benchmark harness.",
  "benchmark": {
    "name": "SWE-bench",
    "group": "Coding",
    "homepage": "https://huggingface.co/datasets/princeton-nlp/SWE-bench",
    "repository": "https://huggingface.co/datasets/princeton-nlp/SWE-bench",
    "ref": "main",
    "license": "MIT",
    "credit": "SWE-bench authors",
    "docker": {
      "image": "agent-bench-external:python3.12",
      "setup": ["python -m pip install --upgrade pip uv"],
      "command": "agent-bench-probe --benchmark \"SWE-bench\" --kind public-benchmark",
      "environment": [],
      "volumes": []
    }
  }
}
```

The default descriptor file now covers 32 benchmark rows across `Coding`, `Cowork`, `GUI`, `Multimodal`, and `Reasoning`, including SWE-Bench Verified, Terminal Bench 2.1, BrowseComp, MCP Atlas, OSWorld-Verified, OmniDocBench, VideoMME, IMO 2025, and USAMO 2026. Upstream credits and license notes are recorded as metadata in `tasks/public_benchmarks.json`; license metadata is not used as a loader gate. Some leaderboard rows do not currently expose a public cloneable task repository, so those descriptors are present for reporting/smoke coverage and should be replaced with a cloneable source as soon as the benchmark publisher releases one.

When running a remote provider, Agent Bench starts the benchmark launcher container, clones the upstream benchmark source or dataset inside Docker, extracts real benchmark task records, calls the configured model endpoint for each sampled task, and grades the model's answer. The launcher receives neutral model settings through environment variables:

- `AGENT_BENCH_BASE_URL`
- `AGENT_BENCH_MODEL`
- `AGENT_BENCH_PROVIDER`
- `AGENT_BENCH_OUTPUT_DIR`
- `AGENT_BENCH_API_KEY`, when `--api-key-env` is provided

The bundled descriptors use `agent-bench-probe` to normalize different public benchmark formats into a common sample-and-grade flow. It supports JSON, JSONL, CSV, Parquet, selected Python task dictionaries, prompt/description text files, Hugging Face datasets, and benchmark-specific adapters for repositories that expose tasks through code or market data rather than a single task file. Extracted records are graded with one of three methods:

- `exact`: deterministic answer, patch, label, or multiple-choice matching.
- `rubric`: a published rubric or benchmark prompt is used for model-based grading.
- `task_compliance`: the benchmark exposes a real task prompt without a standalone answer key, so a grading call scores whether the response satisfies the task requirements.

Each descriptor records the extracted source paths, item counts, grading method, and normalized score in `${AGENT_BENCH_OUTPUT_DIR}/agent_bench_result.json`.

## Outputs

Each run writes:

- `raw_responses.jsonl`
- `graded_results.jsonl`
- `results.csv`
- `summary.json`
- `summary.html`

The HTML report is fully static and includes metric cards, an average-score radar chart, grouped benchmark scores, category summaries, detailed task rows, and timing tables for categories plus individual problems.

The `Benchmark Scores` table is grouped by benchmark group such as `Coding`, `Cowork`, `Finance`, `GUI`, and `Reasoning`. It includes only:

- benchmark name
- model score
- passed/evaluated item count
- grading method

The report intentionally does not include `Status`, `Answer`, `Expected`, or `Evaluation Methodology` columns/sections in `summary.html`. Detailed diagnostics remain available in `summary.json`, `graded_results.jsonl`, and `results.csv`.

Remote providers are queried with streaming responses so the runner can record:

- time to first token (TTFT)
- output tokens per second when the provider exposes output token counts
- benchmark wall-clock run time
- accumulated per-task end-to-end time, summarized by category and by problem

Those values are written into `raw_responses.jsonl`, `graded_results.jsonl`, `results.csv`, `summary.json`, and `summary.html`. Providers that do not return output-token usage still report TTFT and timing data, while tokens-per-second remains `n/a`.

## Docker Sandboxing

The default coding evaluator runs generated Python inside Docker with no network, memory and process limits, a read-only `/work` mount, and a timeout. The benchmark runner container needs the Docker socket mount so it can launch those evaluator containers, and it needs the shared `/tmp/agent-bench-sandboxes` mount so the host Docker daemon can read generated harness files.
