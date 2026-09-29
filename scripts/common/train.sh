#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
verl_root=${VERL_ROOT:-${repo_dir}/verl-main}

if [[ ! -f "${verl_root}/verl/__init__.py" ]]; then
    echo "verl was not found under ${verl_root}; set VERL_ROOT to a verl checkout" >&2
    exit 1
fi
if [[ -n "${1:-}" && "${1}" != --* ]]; then
    export CUDA_VISIBLE_DEVICES=$1
    shift
fi

python_bin=${PYTHON_BIN:-${CONDA_PREFIX:+${CONDA_PREFIX}/bin/python}}
python_bin=${python_bin:-python}
if [[ ! -x "${python_bin}" ]]; then
    echo "Python interpreter not found: ${python_bin}" >&2
    exit 127
fi

read -r -a dataset_configs <<< "${DATASETS:-DeepScaleRRL}"
read -r -a method_configs <<< "${METHODS:-RLCR}"
read -r -a model_configs <<< "${MODEL_CONFIGS:-Qwen3_8B}"
read -r -a format_configs <<< "${CONFIDENCE_FORMATS:-probability}"
read -r -a format_model_paths <<< "${FORMAT_MODEL_PATHS:-}"
if [[ ${#format_model_paths[@]} -gt 0 && ${#format_model_paths[@]} -ne ${#format_configs[@]} ]]; then
    echo "FORMAT_MODEL_PATHS must contain one checkpoint per CONFIDENCE_FORMATS entry" >&2
    exit 2
fi
read -r -a sft_data_dirs <<< "${SFT_DATA_DIRS:-}"
read -r -a configured_train_args <<< "${TRAIN_ARGS:-}"
read -r -a configured_seeds <<< "${SEEDS:-}"
if [[ "${#configured_seeds[@]}" -eq 0 ]]; then
    configured_seeds=("")
fi

if [[ -n "${N_GPUS_PER_NODE:-}" ]]; then
    n_gpus_per_node=${N_GPUS_PER_NODE}
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a visible_devices <<< "${CUDA_VISIBLE_DEVICES}"
    n_gpus_per_node=${#visible_devices[@]}
else
    n_gpus_per_node=${GPU_PER_NODE:-1}
fi

cd "${repo_dir}"
export PYTHONPATH="${repo_dir}:${verl_root}${PYTHONPATH:+:${PYTHONPATH}}"
terminal_log_dir=${LOG_DIR:-logs/terminal/${TERMINAL_CONTEXT:-local}}
mkdir -p "${terminal_log_dir}"

for dataset in "${dataset_configs[@]}"; do
    for method_index in "${!method_configs[@]}"; do
        method=${method_configs[method_index]}
        method_sft_data_dir=""
        if [[ "${#sft_data_dirs[@]}" -gt 0 ]]; then
            if [[ "${#sft_data_dirs[@]}" -ne "${#method_configs[@]}" ]]; then
                echo "SFT_DATA_DIRS must contain one directory per method in METHODS" >&2
                exit 2
            fi
            method_sft_data_dir=${sft_data_dirs[method_index]}
        fi
        for model in "${model_configs[@]}"; do
            for format_index in "${!format_configs[@]}"; do
                confidence_format=${format_configs[format_index]}
                for seed in "${configured_seeds[@]}"; do
                    command=(
                        "${python_bin}" -m src.train.train_main
                        --dataset "${dataset}"
                        --method "${method}"
                        --model "${model}"
                        --confidence-format "${confidence_format}"
                        --n-gpus-per-node "${n_gpus_per_node}"
                        --nnodes 1
                        --rollout-workers "${ROLLOUT_WORKERS:-${n_gpus_per_node}}"
                        --reward-workers "${REWARD_WORKERS:-${n_gpus_per_node}}"
                    )
                    command+=("${configured_train_args[@]}" "$@")
                    if [[ ${#format_model_paths[@]} -gt 0 ]]; then
                        command+=(--model-name-or-path "${format_model_paths[format_index]}")
                    fi
                    if [[ -n "${method_sft_data_dir}" ]]; then
                        command+=(--sft-data-dir "${method_sft_data_dir}")
                    fi
                    if [[ -n "${seed}" ]]; then
                        command+=(--seed "${seed}")
                    fi

                    printf 'training_command='
                    printf '%q ' "${command[@]}"
                    printf '\n'
                    log_suffix=${seed:+-seed-${seed}}
                    log_name=$(printf '%s' "${dataset}-${method}-${model}-${confidence_format}${log_suffix}" \
                        | tr '/:[:space:]' '_' \
                        | sed -E 's/[^A-Za-z0-9._-]+/_/g;s/_+/_/g;s/^_//;s/_$//')
                    terminal_log=${terminal_log_dir}/${log_name}.log
                    echo "Terminal log: ${terminal_log}"
                    set +e
                    "${command[@]}" 2>&1 | tee -a "${terminal_log}"
                    train_status=${PIPESTATUS[0]}
                    set -e
                    if [[ "${train_status}" -ne 0 ]]; then
                        exit "${train_status}"
                    fi
                    if [[ "${POST_TRAIN_EVAL:-0}" == 1 ]]; then
                        output_dir=$(sed -n 's/^TRAIN_OUTPUT_DIR=//p' "${terminal_log}" | tail -n 1)
                        checkpoint=$(find "${output_dir}" -type d -path '*/global_step_*/actor/huggingface' \
                            -print | sort -V | tail -n 1)
                        if [[ -z "${checkpoint}" ]]; then
                            checkpoint=$(find "${output_dir}" -type d -path '*/global_step_*/huggingface' \
                                -print | sort -V | tail -n 1)
                        fi
                        if [[ -z "${checkpoint}" ]]; then
                            echo "Checkpoint not found under ${output_dir}" >&2
                            exit 1
                        fi
                        echo "Post-train evaluation checkpoint: ${checkpoint}"
                        MODEL_PATHS="${checkpoint}" \
                            DATASETS="${POST_TRAIN_EVAL_DATASETS:-DeepScaleR_Eval}" \
                            CONFIDENCE_FORMATS="${confidence_format}" \
                            "${repo_dir}/scripts/common/eval.sh"
                    fi
                done
            done
        done
    done
done
