# Agent Bench

Agent Bench is a Python 3.12+ benchmark runner for evaluating local or remote language models against JSON task files. It supports OpenAI-compatible servers such as vLLM and Ollama's OpenAI endpoint, plus Ollama's native chat API.

## Build

```bash
docker build -t agent .
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

The bundled `tasks/` directory now contains public benchmark descriptors instead of the previous seven hand-written task categories. Each bundled task is an `external_benchmark` entry that runs its descriptor command locally in Docker through `docker/external-benchmark.Dockerfile`.

External benchmark task shape:

```json
{
  "id": "PB_001",
  "type": "external_benchmark",
  "question": "Run SWE-bench locally in Docker using its public benchmark harness.",
  "benchmark": {
    "name": "SWE-bench",
    "homepage": "https://github.com/SWE-bench/SWE-bench",
    "repository": "https://github.com/SWE-bench/SWE-bench.git",
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

All requested benchmarks are loaded as runnable external benchmark descriptors. Upstream credits and license notes are recorded as metadata in `tasks/public_benchmarks.json`; license metadata is not used as a loader gate.

When running a remote provider, `external_benchmark` tasks do not call the model directly through Agent Bench prompts. Instead, Agent Bench starts the benchmark launcher container, clones the upstream benchmark inside Docker, and passes neutral model settings to the benchmark process through environment variables:

- `AGENT_BENCH_BASE_URL`
- `AGENT_BENCH_MODEL`
- `AGENT_BENCH_PROVIDER`
- `AGENT_BENCH_OUTPUT_DIR`
- `AGENT_BENCH_API_KEY`, when `--api-key-env` is provided

The bundled descriptors use `agent-bench-probe` as a lightweight local Docker readiness run: it clones the public benchmark source or dataset, samples the files present, and writes normalized metadata to `${AGENT_BENCH_OUTPUT_DIR}/agent_bench_result.json`. For a full upstream benchmark run, replace the descriptor command with that benchmark's canonical local invocation; the launcher will still pass the same model and output environment variables.

## Outputs

Each run writes:

- `raw_responses.jsonl`
- `graded_results.jsonl`
- `results.csv`
- `summary.json`
- `summary.html`

The HTML report is fully static and includes metric cards, category summaries, detailed task rows, an inline SVG radar chart, and timing tables for categories plus individual problems.

Remote providers are queried with streaming responses so the runner can record:

- time to first token (TTFT)
- output tokens per second when the provider exposes output token counts
- benchmark wall-clock run time
- accumulated per-task end-to-end time, summarized by category and by problem

Those values are written into `raw_responses.jsonl`, `graded_results.jsonl`, `results.csv`, `summary.json`, and `summary.html`. Providers that do not return output-token usage still report TTFT and timing data, while tokens-per-second remains `n/a`.

## Docker Sandboxing

The default coding evaluator runs generated Python inside Docker with no network, memory and process limits, a read-only `/work` mount, and a timeout. The benchmark runner container needs the Docker socket mount so it can launch those evaluator containers, and it needs the shared `/tmp/agent-bench-sandboxes` mount so the host Docker daemon can read generated harness files.
