#!/usr/bin/env bash
set -u

ROOT="/root/autodl-tmp/former-based+gnn-based_three"
PY="/root/miniconda3/envs/former-based-gnn-based-py311/bin/python"
LOG_DIR="$ROOT/logs/pretrain_4exp"

cd "$ROOT" || exit 1
mkdir -p "$LOG_DIR" weights results analysis analysis_packs

export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

gpu_count="$($PY - <<'PY'
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
)"
if [ "$gpu_count" -lt 2 ]; then
  PARALLEL=0
else
  PARALLEL=1
fi

echo "resume started: $(date)"
echo "gpu_count=$gpu_count"

wait_existing() {
  local name="$1"
  local pattern="$2"
  while pgrep -f "$pattern" >/dev/null 2>&1; do
    echo "WAIT existing $name time=$(date)"
    sleep 60
  done
  echo "EXISTING_DONE $name time=$(date)"
}

start_job() {
  local name="$1"
  local gpu="$2"
  local logfile="$3"
  shift 3
  echo "START $name gpu=$gpu log=$logfile time=$(date)"
  CUDA_VISIBLE_DEVICES="$gpu" nohup "$@" > "$logfile" 2>&1 &
  LAST_PID=$!
}

wait_job() {
  local name="$1"
  local pid="$2"
  if wait "$pid"; then
    echo "DONE $name time=$(date)"
    return 0
  fi
  local code=$?
  echo "FAILED $name exit=$code time=$(date)"
  return "$code"
}

run_pair_or_single() {
  local name_a="$1"; local gpu_a="$2"; local log_a="$3"; shift 3
  local cmd_a=()
  while [ "$1" != "::" ]; do cmd_a+=("$1"); shift; done
  shift
  local name_b="$1"; local gpu_b="$2"; local log_b="$3"; shift 3
  local cmd_b=("$@")

  if [ "$PARALLEL" -eq 1 ]; then
    start_job "$name_a" "$gpu_a" "$log_a" "${cmd_a[@]}"
    pid_a=$LAST_PID
    start_job "$name_b" "$gpu_b" "$log_b" "${cmd_b[@]}"
    pid_b=$LAST_PID
    wait_job "$name_a" "$pid_a"; status_a=$?
    wait_job "$name_b" "$pid_b"; status_b=$?
    return $(( status_a + status_b ))
  fi

  start_job "$name_a" 0 "$log_a" "${cmd_a[@]}"
  wait_job "$name_a" "$LAST_PID" || return 1
  start_job "$name_b" 0 "$log_b" "${cmd_b[@]}"
  wait_job "$name_b" "$LAST_PID"
}

wait_existing "PT0_no_pretrain_40ep" "fine_tune.py .*PT0_no_pretrain_40ep"
wait_existing "PT1_current_pretrain_40ep" "run_pretrain_pt1_current.py .*PT1_current_pretrain_40ep"

if [ ! -f weights/PT1_current_masked_pretrain.pt ]; then
  echo "MISSING weights/PT1_current_masked_pretrain.pt"
else
  start_job "PT1_current_finetune_40ep" 1 "$LOG_DIR/PT1_finetune_40ep.log" \
    "$PY" fine_tune.py --use_pretrain true --pretrained_path weights/PT1_current_masked_pretrain.pt \
    --run_name PT1_current_finetune_40ep --skip_test true --epochs 40
  wait_job "PT1_current_finetune_40ep" "$LAST_PID"
fi

run_pair_or_single \
  "PT2_edge_weight_pretrain_40ep" 0 "$LOG_DIR/PT2_pretrain_40ep.log" \
  "$PY" scripts/run_pretrain_pt2_edge_weight.py --epochs 40 --run_name PT2_edge_weight_pretrain_40ep \
  :: \
  "PT3_edge_type_weight_pretrain_40ep" 1 "$LOG_DIR/PT3_pretrain_40ep.log" \
  "$PY" scripts/run_pretrain_pt3_edge_type_weight.py --epochs 40 --run_name PT3_edge_type_weight_pretrain_40ep

if [ ! -f weights/PT2_edge_weight_pretrain.pt ]; then
  echo "MISSING weights/PT2_edge_weight_pretrain.pt"
fi
if [ ! -f weights/PT3_edge_type_weight_pretrain.pt ]; then
  echo "MISSING weights/PT3_edge_type_weight_pretrain.pt"
fi

if [ -f weights/PT2_edge_weight_pretrain.pt ] && [ -f weights/PT3_edge_type_weight_pretrain.pt ]; then
  run_pair_or_single \
    "PT2_edge_weight_finetune_40ep" 0 "$LOG_DIR/PT2_finetune_40ep.log" \
    "$PY" fine_tune.py --use_pretrain true --pretrained_path weights/PT2_edge_weight_pretrain.pt \
      --run_name PT2_edge_weight_finetune_40ep --skip_test true --epochs 40 \
    :: \
    "PT3_edge_type_weight_finetune_40ep" 1 "$LOG_DIR/PT3_finetune_40ep.log" \
    "$PY" fine_tune.py --use_pretrain true --pretrained_path weights/PT3_edge_type_weight_pretrain.pt \
      --run_name PT3_edge_type_weight_finetune_40ep --skip_test true --epochs 40
fi

"$PY" scripts/summarize_pretrain_4exp_40ep.py > "$LOG_DIR/summarize_40ep.log" 2>&1
echo "resume finished: $(date)"
