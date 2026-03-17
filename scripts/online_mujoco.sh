#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/cql_online_finetune_12envs_parallel4.sh <GPU_ID> [PROJECT_NAME] [SEED] [OFFLINE_CKPT_ROOT] [ONLINE_LOG_ROOT]
# Example:
#   bash scripts/cql_online_finetune_12envs_parallel4.sh 0 rlpd-cql-online 42 /data/wuhao666/project/codex/checkpoints/cql_offline /data/wuhao666/project/codex/logs/cql_online

GPU_ID="${1:-0}"
PROJECT_NAME="${2:-rlpd-cql-online}"
SEED="${3:-42}"
OFFLINE_CKPT_ROOT="${4:-/data/wuhao666/project/codex/checkpoints/cql_offline}"
ONLINE_LOG_ROOT="${5:-/data/wuhao666/project/codex/logs/cql_online}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/data/wuhao666/anaconda3/envs/jaxrl

envs=(
  "halfcheetah-medium-expert-v2"
  "hopper-medium-expert-v2"
  "walker2d-medium-expert-v2"
  "walker2d-medium-v2"
  "hopper-medium-v2"
  "halfcheetah-medium-v2"
  "walker2d-medium-replay-v2"
  "hopper-medium-replay-v2"
  "halfcheetah-medium-replay-v2"
  "halfcheetah-random-v2"
  "walker2d-random-v2"
  "hopper-random-v2"
)

MAX_PARALLEL=4
running=0

for env_name in "${envs[@]}"; do
  offline_ckpt_dir="${OFFLINE_CKPT_ROOT}/${env_name}/seed${SEED}"
  log_dir="${ONLINE_LOG_ROOT}/${env_name}/seed${SEED}"
  mkdir -p "${log_dir}"

  echo "[online] launch env=${env_name}, seed=${SEED}, gpu=${GPU_ID}, load=${offline_ckpt_dir}"
  python cql_online_finetuning.py \
    --env_name="${env_name}" \
    --seed="${SEED}" \
    --project_name="${PROJECT_NAME}" \
    --config=configs/cql_config.py \
    --offline_checkpoint_dir="${offline_ckpt_dir}" \
    --log_dir="${log_dir}" \
    --max_steps=1000000 \
    --start_training=10000 \
    --batch_size=256 \
    --online_utd_ratio=5 \
    --offline_ratio=0.5 \
    --pretrain_steps=1000000 \
    --log_interval=1000 \
    --eval_interval=5000 \
    --eval_episodes=10 &

  running=$((running + 1))
  if (( running >= MAX_PARALLEL )); then
    wait -n
    running=$((running - 1))
  fi
done

wait
echo "[online] all jobs finished."
