# SWE-Lancer

This folder owns the Agent Bench integration for `PB_004`. The orchestrator mounts this directory read-only at `/benchmark/task`, mounts only this benchmark's materialized assets at `/benchmark/assets`, and expects `harness/run.sh` to write `/outputs/agent_bench_result.json`.

Source: https://github.com/openai/frontier-evals.git at `51052cede8cc608f95bb00346635e03759013e5a`.
