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

Add new evaluations by dropping another `*.json` file into `tasks/`. The filename becomes the category name in summaries and the radar chart.

Multiple-choice task shape:

```json
{
  "id": "EX_001",
  "type": "multiple_choice",
  "question": "Which option is correct?",
  "choices": ["First", "Second"],
  "answer": ["A"]
}
```

Coding task shape:

```json
{
  "id": "CODE_001",
  "type": "coding",
  "title": "Add",
  "function_name": "add",
  "question": "Return a + b.",
  "test_cases": [
    {"input": {"a": 1, "b": 2}, "output": 3}
  ]
}
```

Coding tasks are scored by passed test cases, so a solution that passes 3 of 5 cases receives `0.6`. Optional `comparison` metadata can be added for order-insensitive results; the bundled LeetCode-style tasks also have built-in fair comparators for cases such as `twoSum`, `topKFrequent`, permutations, subsets, and longest palindrome.

Text recall task shape:

```json
{
  "id": "CR_001",
  "type": "text_recall",
  "question": "Use this source code as the context window:\n<source>\n{{REFERENCE_CODE}}\n</source>\nReproduce the requested function lines verbatim.",
  "reference_path": "ref/example.py",
  "expected_text": "def target(value):\n    return value"
}
```

`text_recall` tasks are scored by whitespace-token F1 after normalizing line endings and ignoring only outer blank lines for both the ground truth and model answer. Missing expected tokens count as false negatives, and extra model tokens count as false positives. `reference_path` is resolved relative to the task file, must stay inside that task directory, and `{{REFERENCE_CODE}}` inside `question` is replaced with the full referenced file contents when the prompt is built. The bundled `tasks/code_recall.json` category follows the CodeNeedle-style pattern: load a large source file from `tasks/ref`, splice it into the model-facing question, then ask the model to reproduce the opening lines of a named function exactly.

If a model returns an empty response for a task, the runner resends that task until it receives a non-empty response or reaches three empty responses for the same task. Only the final response for the task is written to `raw_responses.jsonl`; if all attempts were empty, that final empty response is recorded and graded.

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
