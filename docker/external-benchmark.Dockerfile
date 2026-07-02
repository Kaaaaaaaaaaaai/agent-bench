FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends bash ca-certificates curl docker-cli git git-lfs patch \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip install --no-cache-dir datasets pyarrow

COPY docker/external_launcher.sh /usr/local/bin/agent-bench-external-launcher
COPY docker/benchmark_probe.py /usr/local/bin/agent-bench-probe
RUN chmod +x /usr/local/bin/agent-bench-external-launcher /usr/local/bin/agent-bench-probe

ENTRYPOINT ["agent-bench-external-launcher"]
