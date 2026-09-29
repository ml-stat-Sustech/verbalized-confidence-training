#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd -P)
cd "${repo_dir}"

# Base -> RLCR. Skips all-correct questions when generate_sft_data.sh has been run.
exclude_ids=data/rl_filters/deep_scale_r/qwen3_8b_k50_all_correct.txt
train_args=()
if [[ -f "${exclude_ids}" ]]; then
    train_args+=(--train-exclude-ids-file "${exclude_ids}")
fi

DATASETS="DeepScaleRRL" \
METHODS="RLCR" \
MODEL_CONFIGS="Qwen3_8B" \
CONFIDENCE_FORMATS="probability" \
bash scripts/common/train.sh "${1:-0,1,2,3}" "${train_args[@]}"
