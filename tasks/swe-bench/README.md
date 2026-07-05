# SWE-bench

This folder owns the Agent Bench integration for `PB_001`. The orchestrator mounts this directory read-only at `/benchmark/task`, mounts only this benchmark's materialized assets at `/benchmark/assets`, and expects `harness/run.sh` to write `/outputs/agent_bench_result.json`.

Source: https://github.com/SWE-bench/SWE-bench.git at `f7bbbb2ccdf479001d6467c9e34af59e44a840f9`.
