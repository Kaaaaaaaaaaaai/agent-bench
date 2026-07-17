# SWE-bench Verified

This folder owns the Agent Bench integration for the 500-instance expert-validated `SWE-bench Verified` test split. The orchestrator mounts this directory read-only at `/benchmark/task`, mounts only this benchmark's materialized assets at `/benchmark/assets`, and expects `harness/run.sh` to write `/outputs/agent_bench_result.json`.

Harness source: https://github.com/SWE-bench/SWE-bench.git at `f7bbbb2ccdf479001d6467c9e34af59e44a840f9`.

Dataset source: `princeton-nlp/SWE-bench_Verified` at `c104f840cc67f8b6eec6f759ebc8b2693d585d4a`. The test split must contain exactly 500 records. Gold patches and test-oracle fields are grader-only and must not be included in model prompts or published run artifacts.

The ordinary adapter run is a safe generation/integration smoke test. An official resolved-rate score additionally requires the upstream per-instance evaluator images in a dedicated Docker environment. Agent Bench does not silently replace that grader with an LLM judge or call a smoke score leaderboard-equivalent.
