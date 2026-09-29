#!/usr/bin/env bash
# Usage: paper_protocol.sh MODEL_PATH [GPU_IDS]
# Paper protocol: temperature 0.6, 4 responses per question (32 on AIME), up to 16,384 tokens.
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd -P)
cd "${repo_dir}"

model_path=${1:?Usage: paper_protocol.sh MODEL_PATH [GPU_IDS]}
gpu_ids=${2:-0,1,2,3}

# In-distribution: DeepScaleR_Eval, Math500, MinervaMath, OlympiadBench, GSM8K, AIME2024-2026.
# Out-of-distribution: HotpotVanilla, TriviaQA, DROP, MuSiQue, LiveBenchReasoning, NQOpen, PopQA, WebQuestions.
MODEL_PATHS="${model_path}" \
DATASETS="DeepScaleR_Eval Math500 MinervaMath OlympiadBench GSM8K \
HotpotVanilla TriviaQA DROP MuSiQue LiveBenchReasoning NQOpen PopQA WebQuestions" \
EVAL_ARGS="--temperature 0.6 --num-generations 4" \
bash scripts/common/eval.sh "${gpu_ids}"

MODEL_PATHS="${model_path}" \
DATASETS="AIME2024 AIME2025 AIME2026" \
EVAL_ARGS="--temperature 0.6 --num-generations 32" \
bash scripts/common/eval.sh "${gpu_ids}"
