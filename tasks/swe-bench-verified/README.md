# SWE-bench Verified

This folder owns the Agent Bench integration for `PB_005`. The orchestrator mounts this directory read-only at `/benchmark/task`, mounts only this benchmark's materialized assets at `/benchmark/assets`, and expects `harness/run.sh` to write `/outputs/agent_bench_result.json`.

Source: https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified at `c104f840cc67f8b6eec6f759ebc8b2693d585d4a`.
