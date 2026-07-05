FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends bash ca-certificates curl docker-cli git git-lfs make patch \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip install --no-cache-dir datasets pyarrow \
    && useradd --create-home --uid 10001 agentbench \
    && mkdir -p /workspace /outputs \
    && chown -R agentbench:agentbench /workspace /outputs

COPY docker/external_launcher.sh /usr/local/bin/agent-bench-external-launcher
COPY docker/benchmark_probe.py /usr/local/bin/agent-bench-probe
COPY tasks/finance-agent-v2/fixtures/finance_agent_v2 /opt/agent-bench/tasks/finance-agent-v2/fixtures/finance_agent_v2
RUN chmod -R a+rX /opt/agent-bench/tasks/finance-agent-v2/fixtures
RUN chmod 0755 /usr/local/bin/agent-bench-external-launcher /usr/local/bin/agent-bench-probe

ENTRYPOINT ["agent-bench-external-launcher"]
