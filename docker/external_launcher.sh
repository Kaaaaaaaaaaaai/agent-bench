#!/usr/bin/env bash
set -euo pipefail

: "${AGENT_BENCH_REPOSITORY:?AGENT_BENCH_REPOSITORY is required}"
: "${AGENT_BENCH_COMMAND:?AGENT_BENCH_COMMAND is required}"
: "${AGENT_BENCH_OUTPUT_DIR:=/outputs}"

mkdir -p "${AGENT_BENCH_OUTPUT_DIR}"
repo_dir="/workspace/repo"
export GIT_LFS_SKIP_SMUDGE="${AGENT_BENCH_GIT_LFS_SKIP_SMUDGE:-1}"

task_dir="${AGENT_BENCH_TASK_DIR:-/benchmark/task}"
packaged_task_dir="${AGENT_BENCH_PACKAGED_TASK_DIR:-}"
if [[ -n "${packaged_task_dir}" && ! -e "${task_dir}" ]]; then
  mkdir -p "$(dirname "${task_dir}")"
  ln -s "${packaged_task_dir}" "${task_dir}"
fi

asset_cache_key="${AGENT_BENCH_ASSET_CACHE_KEY:-}"
asset_cache_root="${AGENT_BENCH_ASSET_ROOT:-/benchmark/assets}"
copy_cached_assets() {
  local source_dir=""
  if [[ -n "${asset_cache_key}" && -d "${asset_cache_root}/${asset_cache_key}" ]]; then
    source_dir="${asset_cache_root}/${asset_cache_key}"
  elif [[ -d "${asset_cache_root}" ]]; then
    source_dir="${asset_cache_root}"
  fi
  if [[ -z "${source_dir}" ]] || [[ -z "$(find "${source_dir}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    return 1
  fi
  mkdir -p "${repo_dir}"
  cp -a "${source_dir}/." "${repo_dir}/"
  return 0
}

if [[ ! -d "${repo_dir}/.git" ]]; then
  if ! copy_cached_assets; then
    repository_ref="${AGENT_BENCH_REPOSITORY_REF:-main}"
    if ! git clone --depth 1 --branch "${repository_ref}" "${AGENT_BENCH_REPOSITORY}" "${repo_dir}"; then
      rm -rf "${repo_dir}"
      echo "Unable to clone ${AGENT_BENCH_REPOSITORY} branch/tag ${repository_ref}; retrying default branch plus exact checkout." >&2
      git clone --depth 1 "${AGENT_BENCH_REPOSITORY}" "${repo_dir}"
      git -C "${repo_dir}" fetch --depth 1 origin "${repository_ref}"
      git -C "${repo_dir}" checkout --detach "${repository_ref}"
    fi
  fi
fi

copy_cached_assets || true

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
import os
from pathlib import Path
out = Path('${AGENT_BENCH_OUTPUT_DIR}')
out.mkdir(parents=True, exist_ok=True)
status = ${status}
benchmark = os.environ.get('AGENT_BENCH_BENCHMARK_NAME', '')
group = os.environ.get('AGENT_BENCH_BENCHMARK_GROUP', '')
required = [item for item in os.environ.get('AGENT_BENCH_REQUIRED_CAPABILITIES', '').split(',') if item]
error = f"External benchmark command exited with code {status} before writing agent_bench_result.json"
(out / 'agent_bench_result.json').write_text(json.dumps({
    'benchmark': benchmark,
    'group': group,
    'status': 'completed' if status == 0 else 'failed_harness_setup',
    'score': 1.0 if status == 0 else 0.0,
    'raw_score': 1.0 if status == 0 else 0.0,
    'valid_score': 1.0 if status == 0 else 0.0,
    'error': '' if status == 0 else error,
    'exit_code': status,
    'required_capabilities': required,
    'supported_capabilities': [],
    'unsupported_capabilities': [],
    'capabilities_verified': status == 0,
    'extracted_task_count': 0,
    'evaluated_task_count': 0,
    'valid_evaluated_task_count': 0,
    'evaluation_passed_count': 0,
    'skipped_task_count': 0,
    'model_evals': [],
    'model_eval': {},
    'status_counts': {'passed': 1} if status == 0 else {'failed_harness_setup': 1},
}) + '\n')
PY
fi

exit "${status}"
