import importlib
import json
import os
import re
from dataclasses import fields
from pathlib import Path

from src.common.confidence_formats import normalize_confidence_format
from src.eval.configs.base import EvalBaseConfig
from src.eval.configs.models import BaseModelConfig
from src.eval.models.gemma4_vllm_compat import prepare_gemma4_vllm_checkpoint
from transformers import AutoConfig


MODEL_CONFIG_BY_SIGNATURE = {
    ("qwen3", 4096, 36): "Qwen3_8B",
    ("gemma4", 1536, 35): "Gemma4_E2B_Instruct",
}


def first_present(*values):
    for value in values:
        if value is not None:
            return value
    return None


def detect_tensor_parallel_size() -> int:
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible_devices is not None:
        visible = [device.strip() for device in cuda_visible_devices.split(",") if device.strip()]
        if visible:
            return len(visible)

    try:
        import torch

        count = torch.cuda.device_count()
        if count > 0:
            return count
    except Exception:
        pass

    return 1


def load_config_class(module_suffix: str, class_name: str):
    module_name = module_suffix if module_suffix.startswith("src.") else f"src.eval.configs.{module_suffix}"
    module = importlib.import_module(module_name)
    config_cls = getattr(module, class_name)
    return config_cls()


def dataset_name_to_slug(dataset_name: str) -> str:
    return dataset_name.rstrip("/").split("/")[-1]


def load_model_signature(model_name_or_path: str, *, local_files_only: bool = True):
    config_path = os.path.join(model_name_or_path, "config.json")
    if os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            model_config = json.load(f)
    else:
        model_config = AutoConfig.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            local_files_only=local_files_only,
        ).to_dict()

    text_config = model_config.get("text_config") or model_config
    return {
        "model_type": model_config.get("model_type"),
        "hidden_size": text_config.get("hidden_size"),
        "num_hidden_layers": text_config.get("num_hidden_layers"),
        "num_attention_heads": text_config.get("num_attention_heads"),
    }


def infer_model_config(model_name_or_path: str, run_config: dict | None = None):
    configured_name = (run_config or {}).get("model_config_class")
    if configured_name:
        try:
            model_config = load_config_class("models", configured_name)
            original_model_name_or_path = model_config.model_name_or_path
            model_config.model_name_or_path = model_name_or_path
            return configured_name, model_config, original_model_name_or_path
        except (AttributeError, TypeError):
            pass

    signature = load_model_signature(model_name_or_path, local_files_only=False)
    signature_key = (
        signature.get("model_type"),
        signature.get("hidden_size"),
        signature.get("num_hidden_layers"),
    )
    inferred_name = MODEL_CONFIG_BY_SIGNATURE.get(signature_key)
    if inferred_name is not None:
        model_config = load_config_class("models", inferred_name)
        original_model_name_or_path = model_config.model_name_or_path
        model_config.model_name_or_path = model_name_or_path
        return inferred_name, model_config, original_model_name_or_path

    model_type = signature.get("model_type") or "model"
    inferred_name = f"Auto_{re.sub(r'[^A-Za-z0-9]+', '_', str(model_type)).strip('_')}"
    return inferred_name, BaseModelConfig(model_name_or_path=model_name_or_path), None


def _find_train_run_dir(model_name_or_path: str | None) -> str | None:
    if model_name_or_path is None:
        return None
    candidate = Path(model_name_or_path).expanduser()
    if not candidate.exists():
        return None
    current = candidate if candidate.is_dir() else candidate.parent
    current = current.resolve()
    for directory in (current, *current.parents):
        if (directory / "run_config.json").is_file() or (directory / "lineage.json").is_file():
            return str(directory)
    return None


def resolve_checkpoint_run_dir(checkpoint_name: str | None) -> str | None:
    if checkpoint_name is None:
        return None
    discovered = _find_train_run_dir(checkpoint_name)
    if discovered is not None:
        return discovered
    normalized = checkpoint_name.rstrip("/")
    basename = os.path.basename(normalized)
    if basename.startswith("checkpoint-"):
        return os.path.dirname(normalized)
    return normalized


def checkpoint_step(checkpoint_name: str | None) -> int | None:
    if checkpoint_name is None:
        return None
    for part in reversed(Path(checkpoint_name.rstrip("/")).parts):
        match = re.fullmatch(r"(?:checkpoint-|global_step_)(\d+)", part)
        if match is not None:
            return int(match.group(1))
    return None


def _read_json_if_exists(path: str | None):
    if path is None or not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def attach_checkpoint_metadata(config):
    run_dir = resolve_checkpoint_run_dir(getattr(config, "checkpoint_name", None))
    config.train_run_dir = run_dir
    config.checkpoint_step = checkpoint_step(getattr(config, "checkpoint_name", None))
    if run_dir is None:
        return config

    lineage_path = os.path.join(run_dir, "lineage.json")
    run_config_path = os.path.join(run_dir, "run_config.json")
    lineage = _read_json_if_exists(lineage_path) or {}
    run_config = _read_json_if_exists(run_config_path) or {}

    config.lineage_path = lineage_path if lineage else None
    config.run_config_path = run_config_path if run_config else None
    config.train_task = first_present(lineage.get("task"), run_config.get("task_name"))
    config.train_stage = first_present(lineage.get("stage"), run_config.get("stage"))
    config.train_method = first_present(lineage.get("method"), run_config.get("method_config_class"))
    config.trainer_name = first_present(lineage.get("trainer_name"), run_config.get("trainer_name"))
    config.source_stage = first_present(lineage.get("source_stage"), run_config.get("source_stage"))
    config.source_method = first_present(lineage.get("source_method"), run_config.get("source_method"))
    config.source_checkpoint = first_present(lineage.get("source_checkpoint"), run_config.get("source_checkpoint"))
    config.base_model = first_present(
        lineage.get("base_model"),
        run_config.get("base_model"),
        run_config.get("model_name_or_path"),
    )
    config.train_confidence_format = first_present(lineage.get("confidence_format"), run_config.get("confidence_format"))
    config.train_format_pattern = run_config.get("format_pattern")
    config.sft_data_dir = first_present(lineage.get("sft_data_dir"), run_config.get("sft_data_dir"))
    config.sft_k = first_present(lineage.get("sft_k"), run_config.get("sft_k"))
    config.sft_target_lambda = first_present(lineage.get("sft_target_lambda"), run_config.get("sft_target_lambda"))
    config.kl_loss_coef = run_config.get("kl_loss_coef")
    config.sft_loss_mask = run_config.get("sft_loss_mask")
    return config


def apply_config_overrides(target_config: EvalBaseConfig, source_config):
    target_field_names = {field.name for field in fields(target_config)}
    for field in fields(source_config):
        if field.name in target_field_names:
            setattr(target_config, field.name, getattr(source_config, field.name))
    return target_config


def set_eval_output_paths(
    config,
    dataset_store_name: str,
    model_name: str,
    run_name: str,
    checkpoint_parts: list[str] | None = None,
):
    confidence_output_part = config.confidence_mode
    if config.confidence_format not in (None, ""):
        confidence_output_part = os.path.join(str(config.confidence_mode), str(config.confidence_format))
    output_path = os.path.join(
        model_name,
        run_name,
        *(checkpoint_parts or []),
        config.inferencer_name,
        confidence_output_part,
    )
    config.name = run_name
    config.log_path = os.path.join(config.logs_root, dataset_store_name, output_path)
    return config


def update_config(
    base_config: EvalBaseConfig,
    dataset_config,
    model_config,
    model_name: str,
    inferencer_name: str | None = None,
    confidence_format: str | None = None,
    confidence_mode: str | None = None,
    tensor_parallel_size: int | None = None,
):
    config = EvalBaseConfig()
    apply_config_overrides(config, base_config)
    apply_config_overrides(config, dataset_config)
    apply_config_overrides(config, model_config)

    config.dataset_config_name = type(dataset_config).__name__
    config.dataset_cls = type(dataset_config).__name__
    config.model_config_name = model_name
    if inferencer_name is not None:
        config.inferencer_name = inferencer_name
    if confidence_format is not None:
        config.confidence_format = confidence_format
    if confidence_mode is not None:
        config.confidence_mode = confidence_mode
    config.confidence_format = normalize_confidence_format(config.confidence_format)
    config.tensor_parallel_size = tensor_parallel_size if tensor_parallel_size is not None else detect_tensor_parallel_size()
    return config


def _auto_checkpoint_output_parts(model_name_or_path: str, run_dir: str | None) -> list[str]:
    if run_dir is None:
        return ["models", dataset_name_to_slug(model_name_or_path)]
    parts = ["checkpoints", Path(run_dir).name]
    step = checkpoint_step(model_name_or_path)
    if step is not None:
        parts.append(f"step_{step}")
    return parts


def build_eval_config_from_model_path(
    dataset_name: str,
    model_name_or_path: str,
    inferencer_name: str | None = None,
    confidence_format: str | None = None,
    confidence_mode: str | None = None,
    tensor_parallel_size: int | None = None,
):
    if not model_name_or_path:
        raise ValueError("Evaluation requires --model-name-or-path <path-or-repo>.")
    if tensor_parallel_size is not None and tensor_parallel_size < 1:
        raise ValueError("--tensor_parallel_size must be >= 1.")

    run_dir = _find_train_run_dir(model_name_or_path)
    run_config = _read_json_if_exists(os.path.join(run_dir, "run_config.json")) if run_dir else None
    model_config_name, model_config, base_model_name_or_path = infer_model_config(
        model_name_or_path,
        run_config,
    )
    base_config = EvalBaseConfig()
    dataset_config = load_config_class("datasets", dataset_name)
    config = update_config(
        base_config,
        dataset_config,
        model_config,
        model_name=model_config_name,
        inferencer_name=inferencer_name,
        confidence_format=confidence_format,
        confidence_mode=confidence_mode,
        tensor_parallel_size=tensor_parallel_size,
    )
    config.model_name_or_path = model_name_or_path
    config.model_config_name = model_config_name
    config.checkpoint_name = model_name_or_path if run_dir is not None else None

    if run_dir is not None:
        attach_checkpoint_metadata(config)
        train_task = first_present(config.train_task, (run_config or {}).get("dataset_cls"), "unknown")
        train_method = first_present(config.train_method, config.trainer_name, "unknown")
        run_name = f"{train_task}_{train_method}"
    elif base_model_name_or_path and model_name_or_path.rstrip("/") == base_model_name_or_path.rstrip("/"):
        run_name = "Baseline"
    else:
        run_name = "unknown"

    config.policy_name = run_name
    checkpoint_parts = [] if run_name == "Baseline" else _auto_checkpoint_output_parts(model_name_or_path, run_dir)
    set_eval_output_paths(config, type(dataset_config).__name__, model_config_name, run_name, checkpoint_parts)

    if base_model_name_or_path and model_config_name.startswith("Gemma4_"):
        config.model_name_or_path = prepare_gemma4_vllm_checkpoint(
            model_name_or_path,
            base_model_name_or_path,
        )
    return config
