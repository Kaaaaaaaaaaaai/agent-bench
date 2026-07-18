# Finance Agent v2

This folder owns the Agent Bench integration for `Finance Agent v2`. The orchestrator mounts this directory read-only at `/benchmark/task`, mounts only this benchmark's materialized assets at `/benchmark/assets`, and expects `harness/run.sh` to write `/outputs/agent_bench_result.json`.

Source: https://github.com/vals-ai/finance-agent-v2.git at `abae841eadd1467f2865f3699a1dfc0774d44d76`.
