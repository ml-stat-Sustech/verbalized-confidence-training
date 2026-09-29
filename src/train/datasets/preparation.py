from pathlib import Path
from types import SimpleNamespace

from src.train.datasets.registry import build_dataset


def to_rl_record(example: dict, index: int, task_name: str) -> dict:
    source = example.get("source") or task_name
    extra_info = {
        "index": example.get("id", index),
        "source": source,
        "split": example.get("split"),
    }
    reward_model = {"style": "rule", "ground_truth": example["answer"]}
    return {
        "data_source": source,
        "prompt": example["prompt"],
        "ability": "math",
        "reward_model": reward_model,
        "extra_info": {key: value for key, value in extra_info.items() if value is not None},
    }


def prepare_rl_data(config: dict, output_dir: str | Path) -> dict[str, str]:
    dataset = build_dataset(SimpleNamespace(**config))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {}
    for split, split_dataset in dataset.items():
        include_path = config.get("train_include_ids_file")
        if split == config["dataset_train_split"] and include_path:
            selected_ids = set(Path(include_path).read_text().split())
            missing = selected_ids - {str(value) for value in split_dataset["id"]}
            if not selected_ids or missing:
                raise ValueError(f"Invalid train ID manifest {include_path}: empty or {len(missing)} IDs missing")
            split_dataset = split_dataset.filter(lambda row: str(row["id"]) in selected_ids)
            print(f"[Data] Selected {len(split_dataset)} train examples using {include_path}")
        exclude_path = config.get("train_exclude_ids_file")
        if split == config["dataset_train_split"] and exclude_path:
            if "id" not in split_dataset.column_names:
                raise ValueError(
                    f"Cannot exclude train IDs: split {split!r} has no id column"
                )
            exclude_ids_file = Path(exclude_path)
            excluded_ids = {
                line.strip()
                for line in exclude_ids_file.read_text().splitlines()
                if line.strip()
            }
            before = len(split_dataset)
            matched_ids = excluded_ids & {str(value) for value in split_dataset["id"]}
            split_dataset = split_dataset.filter(
                lambda example: str(example["id"]) not in excluded_ids
            )
            print(
                f"[Data] Excluded {before - len(split_dataset)}/{before} train examples using "
                f"{exclude_ids_file} ({len(excluded_ids) - len(matched_ids)} unmatched IDs)"
            )
        if split == config["dataset_train_split"] and config.get("train_subset_size") is not None:
            split_dataset = split_dataset.select(range(min(config["train_subset_size"], len(split_dataset))))
        if split == config["dataset_test_split"] and config.get("eval_subset_size") is not None:
            split_dataset = split_dataset.select(range(min(config["eval_subset_size"], len(split_dataset))))

        converted = split_dataset.map(
            lambda example, index: to_rl_record(example, index, config.get("task_name") or config["dataset_cls"]),
            with_indices=True,
            remove_columns=split_dataset.column_names,
        )
        path = output_dir / f"{split}.parquet"
        converted.to_parquet(path)
        paths[split] = str(path.resolve())
    return paths


def prepare_sft_data(config: dict, output_dir: str | Path) -> dict[str, str]:
    dataset = build_dataset(SimpleNamespace(**config))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {}
    for split, split_dataset in dataset.items():
        limit = config.get("train_subset_size") if split == config["dataset_train_split"] else config.get("eval_subset_size")
        if limit is not None:
            split_dataset = split_dataset.select(range(min(int(limit), len(split_dataset))))
        required = ["messages"]
        if config["sft_loss_mask"] == "completion_selective":
            required.append("rollout_correctness")
        if config.get("sft_balance_confidence_targets"):
            required.append("confidence_target")
        missing = [name for name in required if name not in split_dataset.column_names]
        if missing:
            raise ValueError(f"SFT split {split!r} is missing columns: {', '.join(missing)}")
        converted = split_dataset.select_columns(required)
        path = output_dir / f"{split}.parquet"
        converted.to_parquet(path)
        paths[split] = str(path.resolve())
    return paths
