FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    AGENT_BENCH_SOURCE_ROOT=/opt/agent-bench \
    AGENT_BENCH_SANDBOX_TMPDIR=/tmp/agent-bench-sandboxes

WORKDIR /opt/agent-bench

RUN apt-get update \
    && apt-get install -y --no-install-recommends docker-cli \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md main.py ./
COPY agent_bench ./agent_bench
COPY docker ./docker
COPY tasks ./tasks

RUN python -m pip install --no-cache-dir .

CMD ["bench"]
