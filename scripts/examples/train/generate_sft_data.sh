#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd -P)
cd "${repo_dir}"

# Sample 50 base-model responses per DeepScaleR question and build CalibSFT targets (lambda=0.5).
DATASETS="DeepScaleR" \
MODEL_CONFIGS="Qwen3_8B" \
CONFIDENCE_FORMATS="probability" \
K=50 \
TARGET_LAMBDA=0.5 \
bash scripts/common/generate_sft_data.sh "${1:-0,1,2,3}"

# Questions solved by all 50 rollouts; the RL examples skip them.
python data_processing/rl_filters/generate_all_correct_ids.py \
    --audit data/sft_data/base/deepscaler/qwen3_8b/probability/k50_lambda0.5/train_rollout_audit.jsonl \
    --output data/rl_filters/deep_scale_r/qwen3_8b_k50_all_correct.txt
