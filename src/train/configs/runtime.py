import json
import math
import os
from datetime import datetime
from pathlib import Path

from src.common.confidence_formats import normalize_confidence_format
from src.common.system_prompts import get_format_pattern
from src.train.configs.config_utils import (
    build_train_config,
    config_name_slug,
    resolve_confidence_sft_data_config,
)


INT_FIELDS = {
    "eval_subset_size",
    "generation_eval_freq",
    "generation_eval_max_new_tokens",
    "generation_eval_sample_size",
    "max_prompt_length",
    "max_response_length",
    "max_sequence_length",
    "total_training_steps",
    "rollout_n",
    "lr_warmup_steps",
    "save_freq",
    "sft_balance_confidence_target_count",
    "test_freq",
    "train_subset_size",
}
FLOAT_FIELDS = {
    "clip_ratio_high",
    "clip_ratio_low",
    "kl_loss_coef",
    "learning_rate",
    "min_lr_ratio",
    "temperature",
    "rollout_gpu_memory_utilization",
    "lr_warmup_steps_ratio",
    "weight_decay",
}
TRAIN_DIR = Path(__file__).resolve().parents[1]
RL_RECIPES = {"dapo", "grpo"}
LEGACY_FIELD_ALIASES = {
    "epsilon": "clip_ratio_low",
    "epsilon_high": "clip_ratio_high",
    "eval_on_start": "val_before_train",
    "eval_steps": "test_freq",
    "gradient_checkpointing": "enable_gradient_checkpointing",
    "max_completion_length": "max_response_length",
    "max_steps": "total_training_steps",
    "num_generations": "rollout_n",
    "num_iterations": "ppo_epochs",
    "report_to": "logger",
    "save_steps": "save_freq",
    "save_total_limit": "max_actor_ckpt_to_keep",
    "scale_rewards": "norm_adv_by_std_in_grpo",
    "vllm_gpu_memory_utilization": "rollout_gpu_memory_utilization",
    "warmup_ratio": "lr_warmup_steps_ratio",
    "wandb_project": "project_name",
}


def normalize_legacy_overrides(overrides: dict | None, config: dict | None = None) -> dict:
    normalized = {}
    for key, value in (overrides or {}).items():
        if key == "save_strategy":
            if value not in {"no", "steps"}:
                raise ValueError("verl compatibility only supports save_strategy=no or steps")
            normalized["save_freq"] = -1 if value == "no" else (config or {}).get("save_freq", 20)
            continue
        if key == "eval_strategy":
            if value not in {"no", "steps"}:
                raise ValueError("verl compatibility only supports eval_strategy=no or steps")
            normalized["test_freq"] = -1 if value == "no" else (config or {}).get("test_freq", 20)
            continue
        normalized[LEGACY_FIELD_ALIASES.get(key, key)] = value
    return normalized


def verl_scheduler_type(value: str) -> str:
    scheduler = str(value).lower()
    aliases = {
        "constant": "constant",
        "constant_with_warmup": "constant",
        "cosine": "cosine",
        "cosine_with_restarts": "cosine",
    }
    if scheduler not in aliases:
        raise ValueError(f"verl FSDP supports constant or cosine LR schedules, got {value!r}")
    return aliases[scheduler]


def verl_model_dtype(config: dict) -> str:
    dtype = str(config.get("torch_dtype", "bfloat16")).lower()
    aliases = {
        "bf16": "bfloat16",
        "bfloat16": "bfloat16",
        "fp16": "float16",
        "float16": "float16",
        "fp32": "fp32",
        "float32": "fp32",
    }
    if dtype not in aliases:
        raise ValueError(f"Unsupported verl model dtype: {dtype}")
    return aliases[dtype]


def build_runtime_config(
    dataset_name: str,
    method_name: str,
    model_name: str,
    overrides=None,
    *,
    validate: bool = True,
) -> dict:
    config = build_train_config(dataset_name, method_name, model_name)
    overrides = normalize_legacy_overrides(overrides, config)

    legacy_recipe = overrides.pop("rl_recipe", None)
    if legacy_recipe is not None:
        legacy_recipe = str(legacy_recipe).lower()
        if legacy_recipe not in RL_RECIPES:
            raise ValueError(f"Unsupported RL recipe {legacy_recipe!r}; choose one of {sorted(RL_RECIPES)}")
        if "enable_filter_groups" not in overrides:
            overrides["enable_filter_groups"] = legacy_recipe == "dapo"
        if "clip_ratio_high" not in overrides:
            overrides["clip_ratio_high"] = 0.28 if legacy_recipe == "dapo" else config.get("clip_ratio_low", 0.2)

    config.update(overrides)
    if config.get("sft_confidence_targets_prebalanced"):
        config["sft_balance_confidence_targets"] = False

    config["confidence_format"] = normalize_confidence_format(config.get("confidence_format"))
    config["format_pattern"] = config.get("format_pattern") or get_format_pattern(config["confidence_format"])
    if config["trainer_name"] == "sft_ce":
        if overrides and "dataset_name" in overrides:
            config["sft_data_dir"] = config["dataset_name"]
        else:
            resolve_confidence_sft_data_config(config, overrides or {})
    for key in [name for name in config if name.startswith("_")]:
        config.pop(key)

    if not validate:
        return config
    supported = {
        "rlvr",
        "rlcr",
        "redoubt",
        "coca",
        "dcpo",
        "sft_ce",
    }
    if config["trainer_name"] not in supported:
        raise ValueError(f"Unsupported verl trainer: {config['trainer_name']}")
    confidence_trainers = {
        "rlcr",
        "redoubt",
        "coca",
        "dcpo",
        "sft_ce",
    }
    if config["trainer_name"] in confidence_trainers and config["confidence_format"] == "none":
        raise ValueError(f"{method_name} requires a non-none confidence_format")
    if config["trainer_name"] == "coca" and config["confidence_format"] != "probability":
        raise ValueError("CoCA requires confidence_format=probability")
    return config


def cast_override(key: str, raw_value: str, current_value):
    if isinstance(current_value, bool):
        normalized = raw_value.lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
        raise ValueError(f"Cannot parse boolean value: {raw_value}")
    if isinstance(current_value, int) and not isinstance(current_value, bool):
        return int(raw_value)
    if isinstance(current_value, float):
        return float(raw_value)
    if isinstance(current_value, dict):
        return json.loads(raw_value)
    if isinstance(current_value, list):
        if raw_value.lower() in {"none", "null"}:
            return []
        if raw_value.startswith("["):
            values = json.loads(raw_value)
        else:
            values = [value for value in raw_value.replace(";", ",").split(",") if value]
        if key == "checkpoint_steps":
            return [int(value) for value in values]
        return values
    if key in INT_FIELDS:
        return int(raw_value)
    if key in FLOAT_FIELDS:
        return float(raw_value)
    if raw_value.lower() in {"none", "null"}:
        return None
    return raw_value


def parse_config_overrides(arguments: list[str], config: dict) -> dict:
    overrides = {}
    index = 0
    while index < len(arguments):
        token = arguments[index]
        if not token.startswith("--"):
            raise ValueError(f"Unexpected positional argument: {token}")
        key = token[2:].replace("-", "_")
        if "=" in key:
            key, raw_value = key.split("=", 1)
        elif index + 1 < len(arguments) and not arguments[index + 1].startswith("--"):
            raw_value = arguments[index + 1]
            index += 1
        else:
            raw_value = "true"
        if key in {"save_strategy", "eval_strategy"}:
            normalized = normalize_legacy_overrides({key: raw_value}, config)
            config.update(normalized)
            overrides.update(normalized)
            index += 1
            continue
        key = LEGACY_FIELD_ALIASES.get(key, key)
        if key not in config:
            raise ValueError(f"Unknown training override: {key}")
        value = cast_override(key, raw_value, config[key])
        config[key] = value
        overrides[key] = value
        index += 1
    return overrides


def build_runtime_config_from_cli(
    dataset_name: str,
    method_name: str,
    model_name: str,
    override_arguments: list[str],
    initial_overrides: dict | None = None,
) -> tuple[dict, dict]:
    initial_overrides = initial_overrides or {}
    config = build_runtime_config(dataset_name, method_name, model_name, initial_overrides, validate=False)
    cli_overrides = {**initial_overrides, **parse_config_overrides(override_arguments, config)}
    return build_runtime_config(dataset_name, method_name, model_name, cli_overrides), cli_overrides


def resolve_run_timestamp() -> str:
    return os.environ.get("TRAIN_RUN_TIMESTAMP") or datetime.now().strftime("%Y-%m%d-%H%M%S")


def _run_tag(config: dict, dataset_name: str, method_name: str, model_name: str) -> str:
    return f"{dataset_name}_{method_name}_{model_name}_{config['confidence_format']}"


def _plain_timestamp(timestamp: str, run_tag: str) -> str:
    suffix = f"_{run_tag}"
    return timestamp[: -len(suffix)] if timestamp.endswith(suffix) else timestamp


def _method_output_slug(config: dict, method_name: str) -> str:
    method_slug = config_name_slug(method_name)
    checkpoint = config.get("model_name_or_path")
    if not checkpoint:
        return method_slug
    if not Path(checkpoint).exists():
        # Hugging Face repository IDs, e.g. SUSTech/Qwen3-8B-CalibSFT.
        return f"calib_sft_{method_slug}" if "calibsft" in checkpoint.lower() else method_slug
    for directory in (Path(checkpoint), *Path(checkpoint).parents):
        metadata_path = directory / "run_config.json"
        if metadata_path.is_file():
            with metadata_path.open(encoding="utf-8") as handle:
                source_method = json.load(handle).get("method_config_class")
            if source_method == "CalibSFT":
                return f"calib_sft_{method_slug}"
            break
    return method_slug


def resolve_output_dir(
    config: dict,
    dataset_name: str,
    method_name: str,
    model_name: str,
    *,
    timestamp: str | None = None,
) -> str:
    if config.get("output_dir"):
        return str(Path(config["output_dir"]).resolve())
    timestamp = timestamp or resolve_run_timestamp()
    task = (config.get("task_name") or dataset_name).lower()
    confidence_format = config["confidence_format"]
    run_dir = _plain_timestamp(timestamp, _run_tag(config, dataset_name, method_name, model_name))
    return str(
        Path(
            config.get("logs_root", "logs/train"),
            task,
            _method_output_slug(config, method_name),
            model_name.lower(),
            confidence_format,
            run_dir,
        ).resolve()
    )


def resolve_tracking_run_name(
    config: dict,
    dataset_name: str,
    method_name: str,
    model_name: str,
    *,
    timestamp: str | None = None,
) -> str:
    if config.get("run_name"):
        return str(config["run_name"])

    timestamp = timestamp or resolve_run_timestamp()
    task_slug = str(config.get("task_name") or dataset_name).lower()
    method_slug = _method_output_slug(config, method_name)
    model_slug = model_name.replace("_", "-").lower()
    confidence_format = str(config["confidence_format"])
    timestamp = _plain_timestamp(
        timestamp,
        _run_tag(config, dataset_name, method_name, model_name),
    )
    return f"{task_slug}-{method_slug}-{model_slug}-{confidence_format}-{timestamp}"


def load_verl_base_config(config_path: str | None = None):
    from hydra import compose, initialize_config_dir

    if config_path is None:
        import verl

        config_path = str(Path(verl.__file__).resolve().parent / "trainer" / "config" / "ppo_trainer.yaml")
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"verl PPO config not found: {path}")
    with initialize_config_dir(config_dir=str(path.parent.resolve()), version_base=None):
        return compose(config_name=path.stem)


def load_verl_sft_base_config(config_path: str | None = None):
    from hydra import compose, initialize_config_dir

    if config_path is None:
        import verl

        config_path = str(Path(verl.__file__).resolve().parent / "trainer" / "config" / "sft_trainer_engine.yaml")
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"verl SFT config not found: {path}")
    with initialize_config_dir(config_dir=str(path.parent.resolve()), version_base=None):
        return compose(config_name=path.stem)


def model_hf_overrides(config: dict) -> dict:
    return dict(config.get("hf_overrides", {}))


def compose_verl_sft_config(config: dict, paths: dict[str, str], runtime) -> object:
    from omegaconf import OmegaConf, open_dict

    if config.get("kl_loss_coef", 0.0) != 0.0:
        raise ValueError("verl SFT does not use a frozen-reference KL term; set --kl-loss-coef 0")
    verl_config = load_verl_sft_base_config(runtime.verl_config)
    train_split = config["dataset_train_split"]
    val_split = config["dataset_test_split"]
    if train_split not in paths:
        raise KeyError(f"Prepared data has no train split {train_split!r}")

    global_batch_size = runtime.train_batch_size or config["train_batch_size"]
    micro_batch_size = runtime.micro_batch_size_per_gpu or config["micro_batch_size_per_gpu"]
    generation_eval_freq = int(config.get("generation_eval_freq", -1))
    generation_eval_enabled = bool(config.get("generation_eval", False))
    generation_eval_datasets = config.get("generation_eval_datasets") or []
    checkpoint_steps = sorted({int(step) for step in config.get("checkpoint_steps", [])})
    checkpoint_trigger_freq = config["save_freq"]
    if checkpoint_steps:
        checkpoint_trigger_freq = math.gcd(*checkpoint_steps)
    if not generation_eval_datasets and config.get("generation_eval_dataset"):
        generation_eval_datasets = [config["generation_eval_dataset"]]
    if generation_eval_enabled:
        if not generation_eval_datasets:
            raise ValueError("SFT generation_eval requires at least one generation_eval dataset")
        if generation_eval_freq == 0 or generation_eval_freq < -1:
            raise ValueError("SFT generation_eval_freq must be positive or -1 for final-only evaluation")
        if generation_eval_freq > 0 and (
            config.get("save_freq", -1) <= 0 or generation_eval_freq % config["save_freq"] != 0
        ):
            raise ValueError(
                "SFT generation_eval requires save_freq > 0 and generation_eval_freq to be divisible by save_freq"
            )
    logger_names = ["console"]
    for logger_name in config.get("logger", []):
        if logger_name in {"wandb", "swanlab"} and logger_name not in logger_names:
            logger_names.append(logger_name)

    val_files = None
    if config.get("test_freq", -1) > 0 and val_split in paths:
        val_files = [paths[val_split]]
    scheduler = verl_scheduler_type(config.get("lr_scheduler_type", "constant"))
    custom_dataset_path = str((TRAIN_DIR / "trainers" / "sft.py").resolve())
    overrides = {
        "data": {
            "train_files": [paths[train_split]],
            "val_files": val_files,
            "train_batch_size": global_batch_size,
            "micro_batch_size_per_gpu": micro_batch_size,
            "train_max_samples": config.get("train_subset_size") or -1,
            "val_max_samples": config.get("eval_subset_size") or -1,
            "max_token_len_per_gpu": config["max_prompt_length"] + config["max_response_length"],
            "max_length": config["max_sequence_length"],
            "truncation": "error",
            # The custom dataset handles Qwen3 inline reasoning; other template mismatches remain warnings.
            "ignore_input_ids_mismatch": True,
            "num_workers": 0,
            "custom_cls": {"path": custom_dataset_path, "name": "VerbalizedSFTDataset"},
            "vc": {
                "algorithm": config["trainer_name"],
                "sft_loss_mask": config["sft_loss_mask"],
            },
        },
        "model": {
            "path": config["model_name_or_path"],
            "trust_remote_code": config["trust_remote_code"],
            "enable_gradient_checkpointing": config["enable_gradient_checkpointing"],
            "use_remove_padding": runtime.use_remove_padding,
            "override_config": {
                **model_hf_overrides(config),
                "attn_implementation": config.get("attn_implementation"),
            },
        },
        "engine": {
            "dtype": verl_model_dtype(config),
            # Keep optimizer-owned parameters in FP32; engine.dtype controls mixed-precision compute.
            "model_dtype": "fp32",
            "seed": config["seed"],
            "ulysses_sequence_parallel_size": runtime.ulysses_sequence_parallel_size,
        },
        "optim": {
            "lr": config["learning_rate"],
            "lr_warmup_steps": config["lr_warmup_steps"],
            "lr_warmup_steps_ratio": config["lr_warmup_steps_ratio"],
            "lr_scheduler_type": scheduler,
            "weight_decay": config["weight_decay"],
        },
        "checkpoint": {
            "save_contents": ["hf_model"],
            "load_contents": ["model", "optimizer", "extra"],
        },
        "trainer": {
            "default_local_dir": config["output_dir"],
            "project_name": config["project_name"],
            "experiment_name": config.get("run_name") or Path(config["output_dir"]).name,
            "logger": logger_names,
            "total_epochs": runtime.total_epochs,
            "total_training_steps": config["total_training_steps"] if config["total_training_steps"] > 0 else None,
            "save_freq": checkpoint_trigger_freq,
            "test_freq": config["test_freq"] if val_files else -1,
            "max_ckpt_to_keep": config["max_actor_ckpt_to_keep"],
            "seed": config["seed"],
            "nnodes": runtime.nnodes,
            "n_gpus_per_node": runtime.n_gpus_per_node,
        },
        "vc": {
            "algorithm": config["trainer_name"],
            "sft_loss_mask": config["sft_loss_mask"],
            "sft_confidence_loss_weight": config.get("sft_confidence_loss_weight", 0.5),
            "sft_balance_confidence_targets": config.get("sft_balance_confidence_targets", False),
            "sft_balance_confidence_target_count": config.get("sft_balance_confidence_target_count"),
            "generation_eval": generation_eval_enabled,
            "generation_eval_freq": generation_eval_freq,
            "checkpoint_steps": config.get("checkpoint_steps", []),
            "generation_eval_dataset": config.get("generation_eval_dataset"),
            "generation_eval_datasets": generation_eval_datasets,
            "generation_eval_sample_size": config.get("generation_eval_sample_size"),
            "generation_eval_confidence_format": config["confidence_format"],
            "generation_eval_max_new_tokens": config["generation_eval_max_new_tokens"],
            "generation_eval_ece_bins": config.get("generation_eval_ece_bins", 10),
            "generation_eval_tensor_parallel_size": config.get("generation_eval_tensor_parallel_size", 1),
            "generation_eval_gpu_memory_utilization": config.get("generation_eval_gpu_memory_utilization", 0.8),
            "generation_eval_temperature": config.get("generation_eval_temperature", 0.0),
        },
    }
    with open_dict(verl_config):
        return OmegaConf.merge(verl_config, OmegaConf.create(overrides))


def compose_verl_config(config: dict, paths: dict[str, str], runtime) -> object:
    from omegaconf import OmegaConf, open_dict

    verl_config = load_verl_base_config(runtime.verl_config)
    hf_overrides = model_hf_overrides(config)
    algorithm = config["trainer_name"]
    train_batch_size = runtime.train_batch_size or config["train_batch_size"]
    gen_batch_size = getattr(runtime, "gen_batch_size", None)
    if gen_batch_size is not None and train_batch_size % gen_batch_size != 0:
        raise ValueError(
            f"V1 gen_batch_size ({gen_batch_size}) must divide train_batch_size ({train_batch_size})"
        )
    ppo_mini_batch_size = runtime.ppo_mini_batch_size or config["ppo_mini_batch_size"]
    ppo_micro_batch_size = (
        runtime.ppo_micro_batch_size_per_gpu
        or config["ppo_micro_batch_size_per_gpu"]
    )
    scheduler = verl_scheduler_type(config.get("lr_scheduler_type", "constant"))
    model_dtype = verl_model_dtype(config)
    ulysses_size = getattr(runtime, "ulysses_sequence_parallel_size", 1)

    train_split = config["dataset_train_split"]
    val_split = config["dataset_test_split"]
    if train_split not in paths:
        raise KeyError(f"Prepared data has no train split {train_split!r}")
    if val_split not in paths:
        raise KeyError(f"Prepared data has no validation split {val_split!r}")

    output_dir = config["output_dir"]
    batch_confidence_dir = str(Path(output_dir, "batch_confidence"))
    logger_names = ["console"]
    for logger_name in config.get("logger", []):
        if logger_name in {"wandb", "swanlab"} and logger_name not in logger_names:
            logger_names.append(logger_name)

    overrides = {
        "data": {
            "train_files": [paths[train_split]],
            "val_files": [paths[val_split]],
            "prompt_key": "prompt",
            "reward_fn_key": "data_source",
            "max_prompt_length": config["max_prompt_length"],
            "max_response_length": config["max_response_length"],
            "train_batch_size": train_batch_size,
            "gen_batch_size": gen_batch_size,
            "shuffle": config.get("shuffle_dataset", True),
            "filter_overlong_prompts": True,
            "truncation": "error",
            "trust_remote_code": config["trust_remote_code"],
            "train_max_samples": config.get("train_subset_size") or -1,
            "val_max_samples": config.get("eval_subset_size") or -1,
            "seed": config["seed"],
        },
        "actor_rollout_ref": {
            "model": {
                "path": config["model_name_or_path"],
                "load_processor": config.get("load_processor", True),
                "enable_gradient_checkpointing": config["enable_gradient_checkpointing"],
                "use_remove_padding": runtime.use_remove_padding,
                "trust_remote_code": config["trust_remote_code"],
                "override_config": {
                    **hf_overrides,
                    "attn_implementation": config.get("attn_implementation"),
                },
            },
            "actor": {
                "ppo_mini_batch_size": ppo_mini_batch_size,
                "ppo_micro_batch_size": None,
                "ppo_micro_batch_size_per_gpu": ppo_micro_batch_size,
                "clip_ratio_low": config["clip_ratio_low"],
                "clip_ratio_high": config.get("clip_ratio_high") or config["clip_ratio_low"],
                "entropy_coeff": 0.0,
                "ppo_epochs": config["ppo_epochs"],
                "use_kl_loss": config["kl_loss_coef"] != 0.0,
                "kl_loss_coef": config["kl_loss_coef"],
                "kl_loss_type": "low_var_kl",
                "loss_agg_mode": config.get("loss_agg_mode", "token-mean"),
                "fsdp_config": {
                    "param_offload": config.get("actor_param_offload", False),
                    "optimizer_offload": config.get("actor_optimizer_offload", False),
                    # verl's FSDP default keeps master parameters in FP32 and casts compute to dtype below.
                    "model_dtype": "fp32",
                    "dtype": model_dtype,
                    "seed": config["seed"],
                    "ulysses_sequence_parallel_size": ulysses_size,
                },
                "optim": {
                    "lr": config["learning_rate"],
                    "min_lr_ratio": config["min_lr_ratio"],
                    "lr_warmup_steps": config["lr_warmup_steps"],
                    "lr_warmup_steps_ratio": config["lr_warmup_steps_ratio"],
                    "lr_scheduler_type": scheduler,
                    "weight_decay": config["weight_decay"],
                },
                "checkpoint": {
                    "save_contents": ["hf_model"],
                    "load_contents": ["model", "optimizer", "extra"],
                },
            },
            "ref": {
                "log_prob_micro_batch_size": None,
                "log_prob_micro_batch_size_per_gpu": ppo_micro_batch_size,
                "fsdp_config": {
                    "param_offload": config.get("ref_param_offload", False),
                    "model_dtype": model_dtype,
                    "dtype": model_dtype,
                    "seed": config["seed"],
                    "ulysses_sequence_parallel_size": ulysses_size,
                },
            },
            "rollout": {
                "name": "vllm",
                "enable_sleep_mode": config.get("vllm_enable_sleep_mode", True),
                "temperature": config["temperature"],
                "n": config["rollout_n"],
                "agent": {"num_workers": runtime.rollout_workers},
                "gpu_memory_utilization": config["rollout_gpu_memory_utilization"],
                "tensor_model_parallel_size": runtime.tensor_model_parallel_size,
                "seed": config["seed"],
                "log_prob_micro_batch_size": None,
                "log_prob_micro_batch_size_per_gpu": ppo_micro_batch_size,
            },
        },
        "algorithm": {
            "adv_estimator": algorithm
            if algorithm
            in {"coca", "dcpo", "redoubt"}
            else "grpo",
            "norm_adv_by_std_in_grpo": config["norm_adv_by_std_in_grpo"],
            "use_kl_in_reward": False,
            "filter_groups": {
                "enable": config.get("enable_filter_groups", False),
                "metric": config.get("filter_groups_metric", "acc"),
                "max_num_gen_batches": config.get("max_num_gen_batches", 20),
            },
        },
        "reward": {
            "num_workers": runtime.reward_workers,
            "reward_manager": {
                "source": "importlib",
                "name": "VerbalizedRewardManager",
                "module": {
                    "path": str((TRAIN_DIR / "rewards" / "manager.py").resolve()),
                    "name": "src.train.rewards.manager",
                },
            },
            "reward_model": {"enable": False},
        },
        "trainer": {
            "project_name": config["project_name"],
            "experiment_name": config.get("run_name") or Path(output_dir).name,
            "logger": logger_names,
            "nnodes": runtime.nnodes,
            "n_gpus_per_node": runtime.n_gpus_per_node,
            "default_local_dir": output_dir,
            "rollout_data_dir": batch_confidence_dir if config.get("log_batch_confidences", True) else None,
            "validation_data_dir": batch_confidence_dir if config.get("log_batch_confidences", True) else None,
            "max_actor_ckpt_to_keep": config["max_actor_ckpt_to_keep"],
            "total_training_steps": config["total_training_steps"] if config["total_training_steps"] > 0 else None,
            "total_epochs": runtime.total_epochs,
            "save_freq": config["save_freq"],
            "test_freq": config["test_freq"],
            "val_before_train": config["val_before_train"],
            "use_v1": True,
            "v1": {
                "trainer_mode": "sync",
                "sampler": {
                    "custom_sampler": {
                        "path": str((TRAIN_DIR / "trainers" / "replay_buffer.py").resolve())
                        if config.get("enable_filter_groups", False)
                        else None,
                        "name": "DapoFilterReplayBuffer"
                        if config.get("enable_filter_groups", False)
                        else None,
                    },
                    "sampler_kwargs": {
                        "filter_metric": config.get("filter_groups_metric", "acc"),
                        "max_num_gen_batches": config.get("max_num_gen_batches", 20),
                    },
                },
            },
        },
        "vc": {
            "algorithm": algorithm,
            "log_batch_confidences": config.get("log_batch_confidences", True),
            "log_token_entropy": config.get("log_token_entropy", True),
            "optimization_rewards": config["optimization_rewards"],
            "confidence_reward": config.get("confidence_reward"),
            "confidence_reward_weight": config.get("confidence_reward_weight", 1.0),
            "monitoring_rewards": config["monitoring_rewards"],
            "format_pattern": config["format_pattern"],
            "log_score_clip": config.get("log_score_clip", 0.01),
            "group_correct_weight": config.get("group_correct_weight", 0.5),
            "ece_bins": config.get("generation_eval_ece_bins", 10),
            "scale_rewards": config["norm_adv_by_std_in_grpo"],
        },
    }
    engine_kwargs = dict(config.get("rollout_engine_kwargs", {}))
    if hf_overrides:
        engine_kwargs["hf_overrides"] = hf_overrides
    if engine_kwargs:
        # vLLM reloads the HF config independently of the training model.
        overrides["actor_rollout_ref"]["rollout"]["engine_kwargs"] = {
            "vllm": engine_kwargs
        }
    with open_dict(verl_config):
        return OmegaConf.merge(verl_config, OmegaConf.create(overrides))


def write_run_metadata(config: dict, selection: dict, cli_overrides: dict):
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(config)
    payload.update(selection)
    payload["backend"] = "verl"
    payload["cli_overrides"] = cli_overrides
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
