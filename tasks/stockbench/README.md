# StockBench

This folder owns the Agent Bench integration for `StockBench`. The orchestrator mounts this directory read-only at `/benchmark/task`, mounts only this benchmark's materialized assets at `/benchmark/assets`, and expects `harness/run.sh` to write `/outputs/agent_bench_result.json`.

Source: https://github.com/ChenYXxxx/stockbench.git at `ce8b2b3483590646ad3b650ac8221f43f76fd091`.
