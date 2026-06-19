#!/usr/bin/env bash
set -euo pipefail

: "${AGENT_BENCH_REPOSITORY:?AGENT_BENCH_REPOSITORY is required}"
: "${AGENT_BENCH_COMMAND:?AGENT_BENCH_COMMAND is required}"
: "${AGENT_BENCH_OUTPUT_DIR:=/outputs}"

mkdir -p "${AGENT_BENCH_OUTPUT_DIR}"
repo_dir="/workspace/repo"
export GIT_LFS_SKIP_SMUDGE="${AGENT_BENCH_GIT_LFS_SKIP_SMUDGE:-1}"

if [[ ! -d "${repo_dir}/.git" ]]; then
  repository_ref="${AGENT_BENCH_REPOSITORY_REF:-main}"
  if ! git clone --depth 1 --branch "${repository_ref}" "${AGENT_BENCH_REPOSITORY}" "${repo_dir}"; then
    rm -rf "${repo_dir}"
    echo "Unable to clone ${AGENT_BENCH_REPOSITORY} at ${repository_ref}; retrying repository default branch." >&2
    git clone --depth 1 "${AGENT_BENCH_REPOSITORY}" "${repo_dir}"
  fi
fi

cd "${repo_dir}"
if [[ "${AGENT_BENCH_GIT_LFS_PULL:-0}" == "1" ]] && command -v git-lfs >/dev/null 2>&1; then
  unset GIT_LFS_SKIP_SMUDGE
  git lfs pull || true
fi
if [[ -n "${AGENT_BENCH_SUBDIR:-}" ]]; then
  cd "${AGENT_BENCH_SUBDIR}"
fi

export OPENAI_BASE_URL="${AGENT_BENCH_BASE_URL:-}"
export OPENAI_API_BASE="${AGENT_BENCH_BASE_URL:-}"
export OPENAI_API_KEY="${AGENT_BENCH_API_KEY:-dummy}"
export MODEL="${AGENT_BENCH_MODEL:-}"
export BENCHMARK_OUTPUT_DIR="${AGENT_BENCH_OUTPUT_DIR}"

if [[ -n "${AGENT_BENCH_SETUP:-}" ]]; then
  bash -lc "${AGENT_BENCH_SETUP}"
fi

set +e
bash -lc "${AGENT_BENCH_COMMAND}"
status=$?
set -e

result_file="${AGENT_BENCH_OUTPUT_DIR}/agent_bench_result.json"
if [[ ! -f "${result_file}" ]]; then
  python - <<PY
import json
from pathlib import Path
out = Path('${AGENT_BENCH_OUTPUT_DIR}')
out.mkdir(parents=True, exist_ok=True)
(out / 'agent_bench_result.json').write_text(json.dumps({
    'score': 1.0 if ${status} == 0 else 0.0,
    'exit_code': ${status},
}) + '\n')
PY
fi

exit "${status}"
