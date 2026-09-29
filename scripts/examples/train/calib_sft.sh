#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd -P)
cd "${repo_dir}"

# Base -> CalibSFT. Run generate_sft_data.sh first, or download the released data (see docs/RUN.md).
DATASETS="DeepScaleRSFT" \
METHODS="CalibSFT" \
MODEL_CONFIGS="Qwen3_8B" \
CONFIDENCE_FORMATS="probability" \
SFT_DATA_DIRS="data/sft_data/base/deepscaler/qwen3_8b/probability/k50_lambda0.5" \
bash scripts/common/train.sh "${1:-0,1,2,3}"
