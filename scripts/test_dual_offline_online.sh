#!/usr/bin/env bash
set -euo pipefail

# 用法：
#   bash scripts/test_dual_offline_online.sh /ABS/CKPT_DIR
# 示例：
#   bash scripts/test_dual_offline_online.sh /tmp/dual_ckpt_test

if [ "$#" -lt 1 ]; then
  echo "Usage: bash scripts/test_dual_offline_online.sh /ABS/CHECKPOINT_DIR"
  exit 1
fi

OFFLINE_CKPT_DIR="$1"
if [[ "$OFFLINE_CKPT_DIR" != /* ]]; then
  echo "[ERROR] CHECKPOINT_DIR must be an absolute path: $OFFLINE_CKPT_DIR"
  exit 1
fi

mkdir -p "$OFFLINE_CKPT_DIR"

# 可按需改环境名
ENV_NAME="halfcheetah-expert-v2"
SEED=42

# 快速 smoke 参数（可直接跑通流程）
OFFLINE_PRETRAIN_STEPS=50
ONLINE_MAX_STEPS=100
START_TRAINING=10
BATCH_SIZE=64
LOG_INTERVAL=10
EVAL_INTERVAL=50

# 避免 XLA 预分配占满显存
export XLA_PYTHON_CLIENT_PREALLOCATE=false

echo "[1/2] Running dual offline pretraining..."
python dual_offline_training.py \
  --env_name="$ENV_NAME" \
  --seed="$SEED" \
  --config=configs/cql_config.py \
  --offline_checkpoint_dir="$OFFLINE_CKPT_DIR" \
  --pretrain_steps="$OFFLINE_PRETRAIN_STEPS" \
  --batch_size="$BATCH_SIZE" \
  --offline_utd_ratio=1 \
  --offline_mask_prob=0.9 \
  --log_interval="$LOG_INTERVAL" \
  --eval_interval="$EVAL_INTERVAL" \
  --tqdm=False

echo "[2/2] Running dual online finetuning from offline checkpoint..."
python dual_online_finetuning.py \
  --env_name="$ENV_NAME" \
  --seed="$SEED" \
  --config=configs/dual_adaptive_config.py \
  --offline_checkpoint_dir="$OFFLINE_CKPT_DIR" \
  --max_steps="$ONLINE_MAX_STEPS" \
  --start_training="$START_TRAINING" \
  --batch_size="$BATCH_SIZE" \
  --online_utd_ratio=1 \
  --offline_ratio=0.5 \
  --log_interval="$LOG_INTERVAL" \
  --eval_interval="$EVAL_INTERVAL" \
  --tqdm=False

echo "Done. Checkpoints should exist under: $OFFLINE_CKPT_DIR"
