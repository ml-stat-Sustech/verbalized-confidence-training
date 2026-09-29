import importlib
import os
import re
from dataclasses import asdict

from src.common.confidence_formats import normalize_confidence_format


CONFIG_NAME_SLUGS = {
    "CalibSFT": "calib_sft",
}

def load_config_class(module_suffix, class_name):
    module_name = module_suffix if module_suffix.startswith("src.") else f"src.train.configs.{module_suffix}"
    module = importlib.import_module(module_name)
    config_cls = getattr(module, class_name)
    return config_cls()


def config_to_dict(config_obj):
    if hasattr(config_obj, "to_config_dict"):
        return config_obj.to_config_dict()
    return {key: value for key, value in asdict(config_obj).items() if value is not None}


def config_name_slug(name: str) -> str:
    if name in CONFIG_NAME_SLUGS:
        return CONFIG_NAME_SLUGS[name]
    slug = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name)
    slug = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", "_", slug)
    return slug.lower()


def update_config(config_dict, dataset_name, model_name):
    config_dict["dataset_cls"] = config_dict.get("dataset_cls") or dataset_name
    config_dict["_model_dir"] = model_name.lower()
    config_dict["logs_root"] = os.environ.get("TRAIN_LOGS_ROOT") or config_dict.get("logs_root", "logs/train")
    return config_dict


def resolve_confidence_sft_data_config(config_dict, cli_overrides=None):
    if not config_dict.get("sft_rollout_dataset"):
        return config_dict

    from data_processing.sft_generation.sft_data_utils import resolve_output_dir

    cli_overrides = cli_overrides or {}
    confidence_format = normalize_confidence_format(config_dict.get("confidence_format"))
    config_dict["confidence_format"] = confidence_format
    source_checkpoints = config_dict.get("sft_source_checkpoints") or {}
    rollout_model = config_dict.get("sft_rollout_model")
    model_overridden = "model_name_or_path" in cli_overrides
    source_overridden = "source_checkpoint" in cli_overrides
    if model_overridden and "sft_rollout_model" not in cli_overrides:
        rollout_model = config_dict.get("model_name_or_path")
        config_dict["sft_rollout_model"] = rollout_model
    elif source_checkpoints and not rollout_model:
        rollout_model = source_checkpoints.get(confidence_format)
        if rollout_model is None:
            available = ", ".join(sorted(source_checkpoints))
            raise ValueError(
                f"No confidence SFT source checkpoint for confidence_format={confidence_format!r}. "
                f"Available formats: {available}"
            )
        config_dict["sft_rollout_model"] = rollout_model

    if not rollout_model:
        rollout_model = config_dict.get("_model_dir") or config_dict.get("model_name_or_path")
        config_dict["sft_rollout_model"] = rollout_model

    if source_checkpoints and not model_overridden:
        config_dict["model_name_or_path"] = rollout_model
    if source_checkpoints and not source_overridden:
        config_dict["source_checkpoint"] = rollout_model

    if config_dict.get("sft_k") is None:
        config_dict["sft_k"] = 50
    if config_dict.get("sft_target_lambda") is None:
        config_dict["sft_target_lambda"] = 0.5
    output_root = config_dict.get("sft_output_root") or os.path.join(
        "data/sft_data",
        config_name_slug(config_dict.get("source_stage") or "base"),
    )
    data_dir = str(
        resolve_output_dir(
            output_root,
            config_dict["sft_rollout_dataset"],
            rollout_model,
            confidence_format,
            k=int(config_dict["sft_k"]),
            target_lambda=float(config_dict["sft_target_lambda"]),
            target_source=config_dict.get("sft_confidence_target_source", "mixed"),
        )
    )
    config_dict["dataset_name"] = config_dict.get("sft_data_dir") or data_dir
    config_dict["sft_data_dir"] = config_dict["dataset_name"]
    return config_dict


def build_train_config(dataset_name, method_name, model_name):
    dataset_obj = load_config_class("datasets", dataset_name)
    method_obj = load_config_class("algorithms", method_name)
    model_obj = load_config_class("models", model_name)

    config_dict = config_to_dict(dataset_obj)
    config_dict.update(config_to_dict(method_obj))
    config_dict.update(config_to_dict(model_obj))
    return update_config(config_dict, dataset_name, model_name)
