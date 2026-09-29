#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
cd "${repo_dir}"

python_bin=${PYTHON_BIN:-${CONDA_PREFIX:+${CONDA_PREFIX}/bin/python}}
python_bin=${python_bin:-python}
gpu_list=${1:-${CUDA_VISIBLE_DEVICES:-0}}
IFS=',' read -r -a gpus <<< "${gpu_list}"
read -r -a datasets <<< "${DATASETS:-DeepScaleR}"
read -r -a models <<< "${MODEL_CONFIGS:-Qwen3_8B}"
read -r -a formats <<< "${CONFIDENCE_FORMATS:-probability}"
read -r -a extra_args <<< "${GENERATION_ARGS:-}"

stage=${STAGE:-all}
splits=${SPLITS:-train,validation}
output_root=${SFT_DATA_ROOT:-data/sft_data/base}
k=${K:-50}
target_lambda=${TARGET_LAMBDA:-0.5}
target_source=${CONFIDENCE_TARGET_SOURCE:-mixed}
sft_method=${SFT_METHOD:-calibsft}
validation_size=${VALIDATION_SIZE:-200}

if [[ ! -x "$(command -v "${python_bin}" 2>/dev/null || true)" ]]; then
    echo "Python interpreter not found: ${python_bin}" >&2
    exit 127
fi
if (( ${#formats[@]} > ${#gpus[@]} )); then
    echo "Need at least one GPU per confidence format: formats=${#formats[@]}, GPUs=${#gpus[@]}" >&2
    exit 2
fi
if (( ${#gpus[@]} % ${#formats[@]} != 0 )); then
    echo "GPU count must be divisible by confidence format count" >&2
    exit 2
fi
if [[ "${OFFLINE:-0}" == 1 ]]; then
    export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
fi

terminal_log_dir=${LOG_DIR:-logs/terminal/sft_generation}
mkdir -p "${terminal_log_dir}"

generate_format() {
    local confidence_format=$1
    local first_gpu_index=$2
    local worker_count=$3
    local assigned_gpus=("${gpus[@]:first_gpu_index:worker_count}")
    local -a common_args shard_pids
    local shard_index shard_log pid shard_status

    for dataset in "${datasets[@]}"; do
        for model in "${models[@]}"; do
            local dataset_slug=${dataset,,}
            local model_slug=${model,,}
            local output_dir=${output_root}/${dataset_slug}/${model_slug}/${confidence_format}/k${k}_lambda${target_lambda}
            if [[ "${target_source}" != mixed ]]; then
                output_dir+=_${target_source}
            fi
            local log_file=${terminal_log_dir}/${dataset_slug}-${model_slug}-${confidence_format}.log

            if [[ "${SKIP_EXISTING:-1}" == 1 && -f "${output_dir}/train.jsonl" && -f "${output_dir}/validation.jsonl" ]]; then
                echo "Skip existing ${output_dir}"
                continue
            fi

            common_args=(
                --dataset "${dataset}"
                --model "${model}"
                --confidence_format "${confidence_format}"
                --output_root "${output_root}"
                --splits "${splits}"
                --validation-size "${validation_size}"
                --k "${k}"
                --target_lambda "${target_lambda}"
                --confidence_target_source "${target_source}"
                --sft-method "${sft_method}"
                --prompt_batch_size "${PROMPT_BATCH_SIZE:-200}"
                --fill_batch_size "${FILL_BATCH_SIZE:-200}"
                --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION:-0.8}"
                "${extra_args[@]}"
            )
            if [[ -n "${ROLLOUT_DIR:-}" ]]; then
                common_args+=(--rollout-dir "${ROLLOUT_DIR}")
            fi

            if [[ "${stage}" == stage2 ]]; then
                echo "Build ${dataset}/${model}/${confidence_format} from ${worker_count} shards"
                CUDA_VISIBLE_DEVICES= "${python_bin}" data_processing/sft_generation/generate_sft_data.py \
                    --stage stage2 --num-shards "${worker_count}" "${common_args[@]}" \
                    > "${log_file}" 2>&1
                continue
            fi

            if (( worker_count == 1 )); then
                echo "Generate ${dataset}/${model}/${confidence_format} on GPU ${assigned_gpus[0]}"
                CUDA_VISIBLE_DEVICES=${assigned_gpus[0]} "${python_bin}" data_processing/sft_generation/generate_sft_data.py \
                    --stage "${stage}" "${common_args[@]}" \
                    > "${log_file}" 2>&1
                continue
            fi

            echo "Generate ${dataset}/${model}/${confidence_format} with ${worker_count} data-parallel GPUs"
            shard_pids=()
            for shard_index in "${!assigned_gpus[@]}"; do
                shard_log=${log_file%.log}.shard-${shard_index}.log
                CUDA_VISIBLE_DEVICES=${assigned_gpus[shard_index]} \
                    "${python_bin}" data_processing/sft_generation/generate_sft_data.py \
                    --stage stage1 \
                    --num-shards "${worker_count}" \
                    --shard-index "${shard_index}" \
                    "${common_args[@]}" \
                    > "${shard_log}" 2>&1 &
                shard_pids+=("$!")
            done
            shard_status=0
            for pid in "${shard_pids[@]}"; do
                wait "${pid}" || shard_status=$?
            done
            if (( shard_status != 0 )); then
                return "${shard_status}"
            fi

            if [[ "${stage}" == all ]]; then
                CUDA_VISIBLE_DEVICES= "${python_bin}" data_processing/sft_generation/generate_sft_data.py \
                    --stage stage2 --num-shards "${worker_count}" "${common_args[@]}" \
                    > "${log_file}" 2>&1
            fi
        done
    done
}

pids=()
workers_per_format=$(( ${#gpus[@]} / ${#formats[@]} ))
for index in "${!formats[@]}"; do
    first_gpu_index=$(( index * workers_per_format ))
    generate_format "${formats[index]}" "${first_gpu_index}" "${workers_per_format}" &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    wait "${pid}" || status=$?
done
exit "${status}"
