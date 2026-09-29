#!/usr/bin/env bash
set -euo pipefail

# Base -> RLCR on all 16 benchmarks. Set CKPT to a trained RLCR checkpoint.
model_path="${CKPT:?Set CKPT, e.g. logs/train/.../global_step_160/actor/huggingface}"
bash "$(dirname "${BASH_SOURCE[0]}")/paper_protocol.sh" "${model_path}" "${1:-0,1,2,3}"
