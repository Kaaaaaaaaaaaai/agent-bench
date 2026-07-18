# Humanity's Last Exam (no tools)

This folder owns the Agent Bench integration for `Humanity's Last Exam`. The manifest pins upstream assets, official conditions, container policy, and reporting metadata used by the public benchmark adapter.

Harness source: https://github.com/centerforaisafety/hle.git at `26dca2e253b405105b4c3d8c2f5af06f86f90c66`.

NVIDIA condition: gated `cais/hle` test data at `5a81a4c7271a2a2a312b9a690f0c2fde837e4c29`, no tools, and an independent GPT-4o judge. The gated token belongs only in the asset-fetch stage and benchmark records must not enter model-visible logs or repository artifacts.
