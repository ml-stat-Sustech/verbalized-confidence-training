import argparse
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

from src.common.runtime_env import configure_log_noise_environment

configure_log_noise_environment()

from src.train.datasets.preparation import prepare_rl_data, prepare_sft_data  # noqa: E402
from src.train.configs.runtime import (  # noqa: E402
    build_runtime_config_from_cli,
    resolve_output_dir,
    resolve_run_timestamp,
    resolve_tracking_run_name,
    write_run_metadata,
)


def build_parser():
    parser = argparse.ArgumentParser(description="Train VerbalizedConfidence methods with verl")
    parser.add_argument("--dataset", required=True, help="Dataset preset class, for example DeepScaleRRL")
    parser.add_argument("--method", required=True, help="Method preset class")
    parser.add_argument("--model", required=True, help="Model preset class")
    parser.add_argument("--rl-recipe", choices=("dapo", "grpo"), default=None)
    parser.add_argument("--train-file")
    parser.add_argument("--val-file")
    parser.add_argument("--data-dir", default="temp/verl_data")
    parser.add_argument("--verl-config", default=None, help="Optional path to verl's PPO or SFT base config")
    parser.add_argument("--n-gpus-per-node", type=int, default=8)
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--tensor-model-parallel-size", type=int, default=1)
    parser.add_argument("--train-batch-size", type=int, default=None)
    parser.add_argument("--gen-batch-size", type=int, default=None)
    parser.add_argument("--ppo-mini-batch-size", type=int, default=None)
    parser.add_argument("--ppo-micro-batch-size-per-gpu", type=int, default=None)
    parser.add_argument("--micro-batch-size-per-gpu", type=int, default=None)
    parser.add_argument("--ulysses-sequence-parallel-size", type=int, default=1)
    parser.add_argument("--total-epochs", type=int, default=1)
    parser.add_argument("--rollout-workers", type=int, default=8)
    parser.add_argument("--reward-workers", type=int, default=8)
    parser.add_argument("--use-remove-padding", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prepare-only", action="store_true")
    return parser


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    runtime, remaining = build_parser().parse_known_args(argv)
    runtime_overrides = {"rl_recipe": runtime.rl_recipe} if runtime.rl_recipe else {}
    config, cli_overrides = build_runtime_config_from_cli(
        runtime.dataset,
        runtime.method,
        runtime.model,
        remaining,
        runtime_overrides,
    )
    run_timestamp = resolve_run_timestamp()
    config["output_dir"] = resolve_output_dir(
        config,
        runtime.dataset,
        runtime.method,
        runtime.model,
        timestamp=run_timestamp,
    )
    config["run_name"] = resolve_tracking_run_name(
        config,
        runtime.dataset,
        runtime.method,
        runtime.model,
        timestamp=run_timestamp,
    )
    print(f"TRAIN_OUTPUT_DIR={config['output_dir']}", flush=True)
    is_sft = config["trainer_name"] == "sft_ce"

    if is_sft:
        from src.train.datasets.sft_generation import ensure_confidence_sft_data

        ensure_confidence_sft_data(SimpleNamespace(**config), SimpleNamespace(process_index=0))

    if bool(runtime.train_file) != bool(runtime.val_file):
        raise ValueError("--train-file and --val-file must be provided together")
    if runtime.train_file:
        paths = {
            config["dataset_train_split"]: str(Path(runtime.train_file).resolve()),
            config["dataset_test_split"]: str(Path(runtime.val_file).resolve()),
        }
    else:
        data_root = Path(
            runtime.data_dir,
            config.get("task_name") or runtime.dataset,
            runtime.method.lower(),
            config["confidence_format"],
        )
        data_root.mkdir(parents=True, exist_ok=True)
        # Each launch owns its parquet files, including simultaneous retries.
        data_dir = Path(tempfile.mkdtemp(prefix=f"{run_timestamp}-", dir=data_root))
        prepare = prepare_sft_data if is_sft else prepare_rl_data
        paths = prepare(config, data_dir)

    write_run_metadata(
        config,
        {
            "dataset_config_class": runtime.dataset,
            "method_config_class": runtime.method,
            "model_config_class": runtime.model,
            "trainer_name": config["trainer_name"],
        },
        cli_overrides,
    )
    if runtime.prepare_only:
        print("\n".join(f"[Data] {split}={path}" for split, path in paths.items()))
        return

    if is_sft:
        from src.train.trainers.sft import launch_sft

        launch_sft(config, paths, runtime)
        return

    from src.train.trainers.ppo import launch_ppo

    launch_ppo(config, paths, runtime)


if __name__ == "__main__":
    main()
