# codeneedle

This folder owns the Agent Bench integration for `codeneedle`. The orchestrator mounts this directory read-only at `/benchmark/task`, mounts only this benchmark's materialized assets at `/benchmark/assets`, and expects `harness/run.sh` to write `/outputs/agent_bench_result.json`.

Source: https://github.com/alexziskind1/codeneedle.git at `a6ccc03145ca700ed4a556846ac87e0488ad5437`.
