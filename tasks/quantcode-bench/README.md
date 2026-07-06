# QuantCode-Bench

This folder owns the Agent Bench integration for `PB_012`. The orchestrator mounts this directory read-only at `/benchmark/task`, mounts only this benchmark's materialized assets at `/benchmark/assets`, and expects `harness/run.sh` to write `/outputs/agent_bench_result.json`.

Source: https://github.com/LimexAILab/QuantCode-Bench.git at `f8bda951addb409a81aa316c00401dbde60774ae`.
