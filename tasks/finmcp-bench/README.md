# FinMCP-Bench

This folder owns the Agent Bench integration for `PB_013`. The orchestrator mounts this directory read-only at `/benchmark/task`, mounts only this benchmark's materialized assets at `/benchmark/assets`, and expects `harness/run.sh` to write `/outputs/agent_bench_result.json`.

Source: https://huggingface.co/datasets/DianJin/FinMCP-Bench at `fa3ffa6939ee29eb78576cfe0d31888de6085202`.
