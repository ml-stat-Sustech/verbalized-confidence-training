#!/usr/bin/env bash
set -euo pipefail

# Base Qwen3-8B on all 16 benchmarks.
model_path="${CKPT:-Qwen/Qwen3-8B}"
bash "$(dirname "${BASH_SOURCE[0]}")/paper_protocol.sh" "${model_path}" "${1:-0,1,2,3}"
