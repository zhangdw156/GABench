#!/usr/bin/env bash
set -euo pipefail

# Parallel GABench wrapper for OpenAI-compatible endpoints such as vLLM.
# GABench itself writes tool outputs under a shared output_dir, so this script
# shards task IDs across isolated worker copies and merges JSONL/output artifacts
# after all workers finish.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

usage() {
  cat <<'USAGE'
Usage:
  scripts/run_parallel_vllm_eval.sh [options]

Common options:
  --model MODEL_KEY          Model key passed to runners/run_benchmark.py.
                             Default: MODEL or qwen3-4b-instruct-2507.
  --model-id MODEL_ID        Model id sent to the vLLM/OpenAI endpoint.
                             Default: MODEL_ID or MODEL_KEY.
  --base-url URL             Endpoint base URL. Default: BASE_URL,
                             OPENAI_BASE_URL, or http://127.0.0.1:8000/v1.
  --api-key KEY              API key. Default: API_KEY, OPENAI_API_KEY, or EMPTY.
  --agent AGENT              react, plan_and_react, plan_and_solve, or base.
                             Default: AGENT or react.
  --jobs N                   Number of parallel worker processes. Default: JOBS or 4.
  --max-tokens N             Override config.yaml max_tokens inside worker copies.
  --tasks IDS                Comma-separated task IDs. Default: all benchmark IDs.
  --tasks-file FILE          Read task IDs from a file; comments/blank lines ignored.
  --workspace DIR            Worker workspace. Default: .parallel_eval/<run_id>.
  --run-id RUN_ID            Merged run id. Default: all_tools_<timestamp>_parallel.
  --copy-data                Copy benchmark/ and dataset/ into each worker instead of
                             symlinking them from the source checkout.
  --dry-run                  Print shards and planned commands without running.
  -h, --help                 Show this help.

Examples:
  scripts/run_parallel_vllm_eval.sh \
    --model qwen3-4b-instruct-2507 \
    --base-url http://127.0.0.1:8000/v1 \
    --api-key EMPTY \
    --agent react \
    --jobs 8

  JOBS=4 scripts/run_parallel_vllm_eval.sh --tasks 1,2,5 --dry-run
USAGE
}

MODEL_KEY="${MODEL:-qwen3-4b-instruct-2507}"
MODEL_ID="${MODEL_ID:-}"
BASE_URL_VALUE="${BASE_URL:-${OPENAI_BASE_URL:-http://127.0.0.1:8000/v1}}"
API_KEY_VALUE="${API_KEY:-${OPENAI_API_KEY:-EMPTY}}"
AGENT="${AGENT:-react}"
JOBS="${JOBS:-4}"
MAX_TOKENS="${MAX_TOKENS:-}"
TASK_IDS="${TASKS:-}"
TASKS_FILE="${TASKS_FILE:-}"
RUN_ID="${RUN_ID:-}"
WORKSPACE="${WORKSPACE:-}"
COPY_DATA=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model|--model-key)
      MODEL_KEY="${2:?missing value for $1}"
      shift 2
      ;;
    --model-id)
      MODEL_ID="${2:?missing value for --model-id}"
      shift 2
      ;;
    --base-url)
      BASE_URL_VALUE="${2:?missing value for --base-url}"
      shift 2
      ;;
    --api-key)
      API_KEY_VALUE="${2:?missing value for --api-key}"
      shift 2
      ;;
    --agent)
      AGENT="${2:?missing value for --agent}"
      shift 2
      ;;
    --jobs)
      JOBS="${2:?missing value for --jobs}"
      shift 2
      ;;
    --max-tokens)
      MAX_TOKENS="${2:?missing value for --max-tokens}"
      shift 2
      ;;
    --tasks|--ids)
      TASK_IDS="${2:?missing value for $1}"
      shift 2
      ;;
    --tasks-file)
      TASKS_FILE="${2:?missing value for --tasks-file}"
      shift 2
      ;;
    --workspace)
      WORKSPACE="${2:?missing value for --workspace}"
      shift 2
      ;;
    --run-id)
      RUN_ID="${2:?missing value for --run-id}"
      shift 2
      ;;
    --copy-data)
      COPY_DATA=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

MODEL_ID="${MODEL_ID:-$MODEL_KEY}"

case "$AGENT" in
  react|plan_and_react|plan_and_solve|base) ;;
  *) echo "ERROR: --agent must be one of: react, plan_and_react, plan_and_solve, base" >&2; exit 2 ;;
esac
if ! [[ "$JOBS" =~ ^[0-9]+$ ]] || [[ "$JOBS" -lt 1 ]]; then
  echo "ERROR: --jobs must be a positive integer, got: $JOBS" >&2
  exit 2
fi
if [[ -n "$MAX_TOKENS" ]] && (! [[ "$MAX_TOKENS" =~ ^[0-9]+$ ]] || [[ "$MAX_TOKENS" -lt 1 ]]); then
  echo "ERROR: --max-tokens must be a positive integer, got: $MAX_TOKENS" >&2
  exit 2
fi
if [[ -n "$TASKS_FILE" && ! -f "$TASKS_FILE" ]]; then
  echo "ERROR: tasks file not found: $TASKS_FILE" >&2
  exit 2
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv is required. Install uv, then run: uv sync" >&2
  exit 2
fi
if ! command -v rsync >/dev/null 2>&1; then
  echo "ERROR: rsync is required for worker workspace preparation" >&2
  exit 2
fi
if [[ "$DRY_RUN" -eq 0 ]]; then
  if [[ ! -f "$REPO_ROOT/benchmark/benchmark.csv" || ! -d "$REPO_ROOT/dataset" ]]; then
    cat >&2 <<EOF_MISSING
ERROR: benchmark data is missing under $REPO_ROOT.
Download it first with:
  uv run scripts/download_data.py
EOF_MISSING
    exit 2
  fi
elif [[ -z "$TASK_IDS" && -z "$TASKS_FILE" && ! -f "$REPO_ROOT/benchmark/benchmark.csv" ]]; then
  cat >&2 <<EOF_MISSING
ERROR: benchmark/benchmark.csv is required to dry-run all tasks.
Download data first or pass --tasks/--tasks-file.
EOF_MISSING
  exit 2
fi

if [[ -z "$RUN_ID" ]]; then
  RUN_ID="all_tools_$(date '+%Y%m%d_%H%M%S')_parallel"
fi
if [[ -z "$WORKSPACE" ]]; then
  WORKSPACE="$REPO_ROOT/.parallel_eval/$RUN_ID"
elif [[ "$WORKSPACE" != /* ]]; then
  WORKSPACE="$REPO_ROOT/$WORKSPACE"
fi
LOG_ROOT="$REPO_ROOT/logs/parallel_eval/$RUN_ID"
SHARDS_FILE="$LOG_ROOT/shards.tsv"
mkdir -p "$LOG_ROOT" "$WORKSPACE"

python3 - "$REPO_ROOT/benchmark/benchmark.csv" "$JOBS" "$TASK_IDS" "$TASKS_FILE" > "$SHARDS_FILE" <<'PYSHARD'
from __future__ import annotations

import csv
import sys
from pathlib import Path

benchmark_path = Path(sys.argv[1])
jobs = int(sys.argv[2])
raw_ids = sys.argv[3].strip()
tasks_file = sys.argv[4].strip()

if raw_ids and tasks_file:
    raise SystemExit("ERROR: use either --tasks or --tasks-file, not both")

if raw_ids:
    ids = [item.strip() for item in raw_ids.split(",") if item.strip()]
elif tasks_file:
    ids = []
    with Path(tasks_file).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            ids.extend(item.strip() for item in line.split(",") if item.strip())
else:
    ids = []
    with benchmark_path.open("r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            task_id = row.get("ID") or row.get("id") or row.get("任务ID")
            if task_id is not None and str(task_id).strip():
                ids.append(str(task_id).strip())

# Keep the first occurrence of duplicate ids while preserving order.
seen = set()
unique_ids = []
for task_id in ids:
    if task_id in seen:
        continue
    seen.add(task_id)
    unique_ids.append(task_id)
ids = unique_ids

if not ids:
    raise SystemExit("ERROR: no task ids selected")

for worker_idx in range(jobs):
    shard = ids[worker_idx::jobs]
    if shard:
        print(f"{worker_idx}\t{','.join(shard)}")
PYSHARD

if [[ ! -s "$SHARDS_FILE" ]]; then
  echo "ERROR: no non-empty shards generated" >&2
  exit 2
fi

model_safe="${MODEL_KEY//\//_}"
model_safe="${model_safe//\\/_}"

print_plan() {
  echo "==> GABench parallel vLLM evaluation"
  echo "Model key:   $MODEL_KEY"
  echo "Model id:    $MODEL_ID"
  echo "Base URL:    $BASE_URL_VALUE"
  echo "Agent:       $AGENT"
  echo "Jobs:        $JOBS"
  echo "Run ID:      $RUN_ID"
  echo "Workspace:   $WORKSPACE"
  echo "Logs:        $LOG_ROOT"
  [[ -n "$MAX_TOKENS" ]] && echo "Max tokens:  $MAX_TOKENS" || true
  echo "Data mode:   $([[ "$COPY_DATA" -eq 1 ]] && echo copy || echo symlink)"
  echo "Shards:"
  sed 's/^/  /' "$SHARDS_FILE"
}

configure_worker() {
  local worker_dir="$1"
  uv run --project "$REPO_ROOT" python - "$worker_dir/config.yaml" "$MODEL_KEY" "$MODEL_ID" "$BASE_URL_VALUE" "$API_KEY_VALUE" "$MAX_TOKENS" <<'PYCONF'
from __future__ import annotations

import sys
from pathlib import Path

import yaml

config_path = Path(sys.argv[1])
model_key = sys.argv[2]
model_id = sys.argv[3]
base_url = sys.argv[4]
api_key = sys.argv[5]
max_tokens = sys.argv[6]

with config_path.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}

cfg.setdefault("llm", {})[model_key] = {
    "model": model_id,
    "base_url": base_url,
    "api_key": api_key,
}
cfg["output_dir"] = "./output"
if max_tokens:
    cfg["max_tokens"] = int(max_tokens)

with config_path.open("w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
PYCONF
}

prepare_worker() {
  local worker_idx="$1"
  local worker_dir="$WORKSPACE/w${worker_idx}"
  mkdir -p "$worker_dir"

  rsync -a --delete \
    --exclude '.git' \
    --exclude '.venv' \
    --exclude '.omx' \
    --exclude '.parallel_eval' \
    --exclude 'benchmark' \
    --exclude 'dataset' \
    --exclude 'results' \
    --exclude 'output' \
    --exclude 'output_results' \
    --exclude 'logs' \
    "$REPO_ROOT/" "$worker_dir/"

  rm -rf "$worker_dir/benchmark" "$worker_dir/dataset"
  if [[ "$COPY_DATA" -eq 1 ]]; then
    rsync -a --delete "$REPO_ROOT/benchmark" "$worker_dir/"
    rsync -a --delete "$REPO_ROOT/dataset" "$worker_dir/"
  else
    ln -s "$REPO_ROOT/benchmark" "$worker_dir/benchmark"
    ln -s "$REPO_ROOT/dataset" "$worker_dir/dataset"
  fi

  configure_worker "$worker_dir"
}

run_worker() {
  local worker_idx="$1"
  local ids="$2"
  local worker_dir="$WORKSPACE/w${worker_idx}"
  local log_file="$LOG_ROOT/worker-${worker_idx}.log"

  prepare_worker "$worker_idx"
  echo "==> Worker $worker_idx: $(awk -F, '{print NF}' <<< "$ids") tasks -> $log_file"
  (
    cd "$worker_dir"
    uv run --project "$REPO_ROOT" python runners/run_benchmark.py \
      --model "$MODEL_KEY" \
      --agent "$AGENT" \
      --id "$ids"
  ) > "$log_file" 2>&1
}

print_plan
if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "Dry run only; no workers launched."
  exit 0
fi

pids=()
worker_indices=()
while IFS=$'\t' read -r worker_idx ids; do
  [[ -n "$worker_idx" && -n "$ids" ]] || continue
  run_worker "$worker_idx" "$ids" &
  pids+=("$!")
  worker_indices+=("$worker_idx")
done < "$SHARDS_FILE"

failed=0
for i in "${!pids[@]}"; do
  pid="${pids[$i]}"
  worker_idx="${worker_indices[$i]}"
  if wait "$pid"; then
    echo "==> Worker $worker_idx completed"
  else
    echo "ERROR: worker $worker_idx failed; inspect $LOG_ROOT/worker-${worker_idx}.log" >&2
    failed=1
  fi
done

if [[ "$failed" -ne 0 ]]; then
  echo "ERROR: at least one worker failed; not merging partial outputs" >&2
  exit 1
fi

merged_dir="$REPO_ROOT/results/$MODEL_KEY/$AGENT"
merged_result="$merged_dir/$RUN_ID.jsonl"
mkdir -p "$merged_dir"
: > "$merged_result"

merged_records=0
while IFS=$'\t' read -r worker_idx _ids; do
  worker_dir="$WORKSPACE/w${worker_idx}"
  debug_dir="$worker_dir/results/debug"
  if [[ -d "$debug_dir" ]]; then
    while IFS= read -r result_file; do
      cat "$result_file" >> "$merged_result"
      count=$(wc -l < "$result_file" | tr -d '[:space:]')
      merged_records=$((merged_records + count))
    done < <(find "$debug_dir" -maxdepth 1 -type f -name "${model_safe}_${AGENT}_*.jsonl" | sort)
  fi

done < "$SHARDS_FILE"

output_dest="$REPO_ROOT/output_results/$MODEL_KEY/$AGENT/output"
mkdir -p "$output_dest"
while IFS=$'\t' read -r worker_idx _ids; do
  worker_output="$WORKSPACE/w${worker_idx}/output"
  if [[ -d "$worker_output" ]]; then
    rsync -a "$worker_output/" "$output_dest/"
  fi

done < "$SHARDS_FILE"

cat <<EOF_DONE
==> Parallel evaluation completed
Merged records: $merged_records
Result JSONL:   $merged_result
Output dir:     $output_dest
Worker logs:    $LOG_ROOT
Workspace:      $WORKSPACE

Next checks:
  uv run python scripts/estimate_eval_progress.py --run $merged_result
  uv run evaluation/step_by_step.py --result $merged_result
EOF_DONE
