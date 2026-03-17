#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/cql_single_env_offline200k_online200k.sh <GPU_ID> [ENV_NAME] [SEED] [PROJECT_PREFIX] [OFFLINE_CKPT_ROOT] [ONLINE_LOG_ROOT]
# Example:
#   bash scripts/cql_single_env_offline200k_online200k.sh 0 halfcheetah-medium-v2 42 rlpd-cql-smoke /data/wuhao666/project/codex/checkpoints/cql_offline_test /data/wuhao666/project/codex/logs/cql_online_test

GPU_ID="${1:-0}"
ENV_NAME="${2:-halfcheetah-medium-v2}"
SEED="${3:-42}"
PROJECT_PREFIX="${4:-rlpd-cql-smoke}"
OFFLINE_CKPT_ROOT="${5:-/data/wuhao666/project/codex/checkpoints/cql_offline_test}"
ONLINE_LOG_ROOT="${6:-/data/wuhao666/project/codex/logs/cql_online_test}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/data/wuhao666/anaconda3/envs/jaxrl

offline_ckpt_dir="${OFFLINE_CKPT_ROOT}/${ENV_NAME}/seed${SEED}"
online_log_dir="${ONLINE_LOG_ROOT}/${ENV_NAME}/seed${SEED}"
mkdir -p "${offline_ckpt_dir}" "${online_log_dir}"

echo "[smoke] offline pretrain 200k: env=${ENV_NAME}, seed=${SEED}"
python cql_offline_training.py \
  --env_name="${ENV_NAME}" \
  --seed="${SEED}" \
  --project_name="${PROJECT_PREFIX}-offline" \
  --config=configs/cql_config.py \
  --pretrain_steps=200000 \
  --batch_size=256 \
  --offline_utd_ratio=1 \
  --log_interval=1000 \
  --eval_interval=5000 \
  --eval_episodes=10 \
  --offline_checkpoint_dir="${offline_ckpt_dir}"

echo "[smoke] online finetune 200k: env=${ENV_NAME}, seed=${SEED}"
python cql_online_finetuning.py \
  --env_name="${ENV_NAME}" \
  --seed="${SEED}" \
  --project_name="${PROJECT_PREFIX}-online" \
  --config=configs/cql_config.py \
  --offline_checkpoint_dir="${offline_ckpt_dir}" \
  --log_dir="${online_log_dir}" \
  --max_steps=200000 \
  --start_training=10000 \
  --batch_size=256 \
  --online_utd_ratio=5 \
  --offline_ratio=0.5 \
  --pretrain_steps=200000 \
  --log_interval=1000 \
  --eval_interval=5000 \
  --eval_episodes=10

echo "[smoke] done."
