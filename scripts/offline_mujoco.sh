#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/cql_offline_pretrain_12envs_parallel4.sh <GPU_ID> [PROJECT_NAME] [SEED] [CKPT_ROOT]
# Example:
#   bash scripts/cql_offline_pretrain_12envs_parallel4.sh 0 rlpd-cql-offline 42 /data/wuhao666/project/codex/checkpoints/cql_offline

GPU_ID="${1:-0}"
PROJECT_NAME="${2:-rlpd-cql-offline}"
SEED="${3:-42}"
CKPT_ROOT="${4:-/data/wuhao666/project/codex/checkpoints/cql_offline}"

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
  ckpt_dir="${CKPT_ROOT}/${env_name}/seed${SEED}"
  mkdir -p "${ckpt_dir}"

  echo "[offline] launch env=${env_name}, seed=${SEED}, gpu=${GPU_ID}"
  python cql_offline_training.py \
    --env_name="${env_name}" \
    --seed="${SEED}" \
    --project_name="${PROJECT_NAME}" \
    --config=configs/cql_config.py \
    --pretrain_steps=1000000 \
    --batch_size=256 \
    --offline_utd_ratio=1 \
    --log_interval=1000 \
    --eval_interval=5000 \
    --eval_episodes=10 \
    --offline_checkpoint_dir="${ckpt_dir}" &

  running=$((running + 1))
  if (( running >= MAX_PARALLEL )); then
    wait -n
    running=$((running - 1))
  fi
done

wait
echo "[offline] all jobs finished."
