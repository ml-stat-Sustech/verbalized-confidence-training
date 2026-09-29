#!/usr/bin/env bash
set -euo pipefail

# Base -> CalibSFT on all 16 benchmarks. Defaults to the released model; set CKPT to evaluate your own checkpoint.
model_path="${CKPT:-SUSTech/Qwen3-8B-CalibSFT}"
bash "$(dirname "${BASH_SOURCE[0]}")/paper_protocol.sh" "${model_path}" "${1:-0,1,2,3}"
