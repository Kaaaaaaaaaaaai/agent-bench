# InvestorBench

This folder owns the Agent Bench integration for `InvestorBench`. The orchestrator mounts this directory read-only at `/benchmark/task`, mounts only this benchmark's materialized assets at `/benchmark/assets`, and expects `harness/run.sh` to write `/outputs/agent_bench_result.json`.

Source: https://github.com/felis33/INVESTOR-BENCH.git at `87e0f7bc83baf7d89fbdc1a51f92458f8d4ee11a`.
