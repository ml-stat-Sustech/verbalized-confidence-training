#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
repo_dir=$(cd "${script_dir}/../.." && pwd -P)

gpu_list=${1:-}
if [[ -z "${gpu_list}" || "${gpu_list}" == --* ]]; then
    if [[ "${N_GPUS_PER_NODE:-1}" =~ ^[1-9][0-9]*$ ]] &&
       (( ${N_GPUS_PER_NODE:-1} > 1 )) && [[ "${TENSOR_PARALLEL_SIZE:-1}" == 1 ]]; then
        gpu_list=$(seq -s, 0 "$((N_GPUS_PER_NODE - 1))")
    fi
fi

if [[ "${gpu_list}" == *,* ]]; then
    if [[ -n "${1:-}" && "${1}" != --* ]]; then
        shift
    fi
    if [[ "${TENSOR_PARALLEL_SIZE:-1}" != 1 ]]; then
        echo "Parallel dataset evaluation requires TENSOR_PARALLEL_SIZE=1" >&2
        exit 2
    fi

    IFS=',' read -r -a gpus <<< "${gpu_list}"
    read -r -a datasets <<< "${DATASETS:-DeepScaleR_Eval}"
    for gpu in "${gpus[@]}"; do
        if [[ ! "${gpu}" =~ ^[0-9]+$ ]]; then
            echo "Invalid GPU ID: ${gpu}" >&2
            exit 2
        fi
    done

    queue_file=$(mktemp)
    printf '0\n' > "${queue_file}"
    trap 'rm -f "${queue_file}"' EXIT

    pids=()
    for gpu in "${gpus[@]}"; do
        (
            exec {queue_fd}<>"${queue_file}"
            worker_status=0
            while true; do
                flock -x "${queue_fd}"
                read -r dataset_index < "${queue_file}"
                if (( dataset_index >= ${#datasets[@]} )); then
                    flock -u "${queue_fd}"
                    break
                fi
                printf '%s\n' "$((dataset_index + 1))" > "${queue_file}"
                flock -u "${queue_fd}"

                dataset=${datasets[dataset_index]}
                echo "GPU ${gpu}: ${dataset}"
                DATASETS="${dataset}" bash "${script_dir}/eval.sh" "${gpu}" "$@" || worker_status=1
            done
            exit "${worker_status}"
        ) &
        pids+=("$!")
    done

    status=0
    for pid in "${pids[@]}"; do
        wait "${pid}" || status=1
    done
    exit "${status}"
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

read -r -a dataset_configs <<< "${DATASETS:-DeepScaleR_Eval}"
read -r -a confidence_formats <<< "${CONFIDENCE_FORMATS:-probability}"
read -r -a configured_eval_args <<< "${EVAL_ARGS:-}"
read -r -a model_paths <<< "${MODEL_PATHS:-Qwen/Qwen3-8B}"

cd "${repo_dir}"
export PYTHONPATH="${repo_dir}${PYTHONPATH:+:${PYTHONPATH}}"
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/tmp/${USER:-root}/triton_autotune}
mkdir -p "${TRITON_CACHE_DIR}"
terminal_log_dir=${LOG_DIR:-logs/terminal/${TERMINAL_CONTEXT:-local}}
mkdir -p "${terminal_log_dir}"
run_timestamp=${RUN_TIMESTAMP:-$(date +%Y-%m%d-%H%M%S)}

for dataset in "${dataset_configs[@]}"; do
    for model_path in "${model_paths[@]}"; do
        for confidence_format in "${confidence_formats[@]}"; do
            eval_args=(
                --dataset "${dataset}"
                --model-name-or-path "${model_path}"
                --inferencer "${INFERENCER:-verbalized_confidence}"
                --confidence_format "${confidence_format}"
                --tensor_parallel_size "${TENSOR_PARALLEL_SIZE:-1}"
            )
            if [[ -n "${CONFIDENCE_MODE:-}" ]]; then
                eval_args+=(--confidence_mode "${CONFIDENCE_MODE}")
            fi
            eval_args+=("${configured_eval_args[@]}" "$@")

            model_label=$(basename "${model_path}")-$(printf '%s' "${model_path}" | sha256sum | cut -c1-10)
            log_name=$(printf '%s' "eval-${dataset}-${model_label}-${confidence_format}-${run_timestamp}" \
                | tr '/:[:space:]' '_' \
                | sed -E 's/[^A-Za-z0-9._-]+/_/g;s/_+/_/g;s/^_//;s/_$//')
            terminal_log=${terminal_log_dir}/${log_name}.log
            echo "Terminal log: ${terminal_log}"
            printf 'eval_command='
            printf '%q ' "${python_bin}" -m src.eval.eval_main "${eval_args[@]}"
            printf '\n'

            set +e
            "${python_bin}" -m src.eval.eval_main "${eval_args[@]}" 2>&1 | tee -a "${terminal_log}"
            eval_status=${PIPESTATUS[0]}
            set -e
            if [[ "${eval_status}" -ne 0 ]]; then
                exit "${eval_status}"
            fi
        done
    done
done
