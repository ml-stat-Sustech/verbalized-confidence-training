import logging
import json
import os
import random
from pathlib import Path
from types import SimpleNamespace

import torch


logger = logging.getLogger(__name__)


def _has_train_split(dataset_path, train_split):
    path = Path(dataset_path)
    if path.is_file():
        return True
    return (path / f"{train_split}.jsonl").exists() or (path / f"{train_split}.json").exists()


def _split_path(dataset_path: str | Path, split: str) -> Path:
    path = Path(dataset_path)
    for suffix in (".jsonl", ".json"):
        candidate = path / f"{split}{suffix}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Missing SFT split {split!r} under {path}")


def _build_static_balanced_split(dataset_path: str | Path, seed: int = 43) -> Path:
    """Materialize an equal-count confidence-target split from train.jsonl."""
    source = _split_path(dataset_path, "train")
    destination = Path(dataset_path) / "train_static_balanced.jsonl"
    rows_by_bucket: dict[float, list[str]] = {}
    seen_ids: set[str] = set()
    with source.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            target = row.get("confidence_target")
            if target is None:
                continue
            row_id = str(row.get("id", index))
            if row_id in seen_ids:
                raise ValueError(f"Duplicate SFT row id while balancing: {row_id}")
            seen_ids.add(row_id)
            rows_by_bucket.setdefault(round(float(target), 6), []).append(row_id)
    if not rows_by_bucket:
        raise ValueError(f"Cannot build balanced SFT split without confidence_target: {source}")

    count_per_bucket = min(len(ids) for ids in rows_by_bucket.values())
    selected_ids: set[str] = set()
    rng = random.Random(seed)
    for ids in rows_by_bucket.values():
        rng.shuffle(ids)
        selected_ids.update(ids[:count_per_bucket])

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".jsonl.tmp")
    with source.open(encoding="utf-8") as source_handle, temporary.open("w", encoding="utf-8") as output:
        for index, line in enumerate(source_handle):
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("id", index)) in selected_ids:
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(destination)
    logger.info(
        "Built %s with %d examples (%d per confidence target, %d targets)",
        destination,
        len(selected_ids),
        count_per_bucket,
        len(rows_by_bucket),
    )
    return destination


def _wait_for_everyone():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def ensure_confidence_sft_data(script_args, training_args):
    if not getattr(script_args, "sft_rollout_dataset", None):
        return
    requested_split = script_args.dataset_train_split
    if _has_train_split(script_args.dataset_name, requested_split):
        return

    if requested_split == "train_static_balanced" and _has_train_split(script_args.dataset_name, "train"):
        if getattr(training_args, "process_index", 0) == 0:
            _build_static_balanced_split(
                script_args.dataset_name,
                seed=int(getattr(script_args, "seed", 43)),
            )
        _wait_for_everyone()
        return

    if getattr(training_args, "process_index", 0) == 0:
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        from data_processing.sft_generation.generate_sft_data import main_from_namespace

        rollout_model = script_args.sft_rollout_model or script_args.source_checkpoint or script_args.dataset_cls
        model_name_or_path = getattr(script_args, "model_name_or_path", None) or script_args.source_checkpoint
        if rollout_model and (Path(str(rollout_model)).exists() or "/" in str(rollout_model)):
            model_name_or_path = rollout_model
        generation_args = SimpleNamespace(
            stage=script_args.sft_rollout_stage,
            dataset=script_args.sft_rollout_dataset,
            model=rollout_model,
            model_name_or_path=model_name_or_path,
            confidence_format=script_args.confidence_format,
            rollout_confidence_format=getattr(script_args, "sft_rollout_confidence_format", "same"),
            output_root=getattr(script_args, "sft_output_root", None) or "data/sft_data",
            output_dir=script_args.dataset_name,
            rollout_dir=None,
            splits=script_args.sft_rollout_splits,
            limit=None,
            train_size=None,
            test_size=None,
            validation_size=getattr(script_args, "sft_rollout_validation_size", 200),
            validation_source_split=getattr(script_args, "sft_rollout_validation_source_split", "test"),
            k=script_args.sft_k,
            temperature=script_args.sft_rollout_temperature,
            top_p=script_args.sft_rollout_top_p,
            max_completion_length=None,
            prompt_batch_size=script_args.sft_rollout_prompt_batch_size,
            fill_batch_size=script_args.sft_rollout_fill_batch_size,
            dtype="bfloat16",
            tensor_parallel_size=script_args.sft_rollout_tensor_parallel_size,
            gpu_memory_utilization=script_args.sft_rollout_gpu_memory_utilization,
            seed=script_args.sft_rollout_seed,
            shuffle=False,
            target_lambda=script_args.sft_target_lambda,
            binning="round",
            linguistic_low_medium_boundary=1.0 / 3.0,
            linguistic_medium_high_boundary=2.0 / 3.0,
            drop_invalid_format=True,
            save_audit_completions=False,
        )
        logger.info("Generating missing confidence SFT data at %s", script_args.dataset_name)
        main_from_namespace(generation_args)

        if requested_split == "train_static_balanced" and _has_train_split(script_args.dataset_name, "train"):
            _build_static_balanced_split(
                script_args.dataset_name,
                seed=int(getattr(script_args, "seed", 43)),
            )

    _wait_for_everyone()
    if requested_split == "train_static_balanced" and not _has_train_split(
        script_args.dataset_name, requested_split
    ) and _has_train_split(script_args.dataset_name, "train"):
        if getattr(training_args, "process_index", 0) == 0:
            _build_static_balanced_split(
                script_args.dataset_name,
                seed=int(getattr(script_args, "seed", 43)),
            )
        _wait_for_everyone()
    if not _has_train_split(script_args.dataset_name, script_args.dataset_train_split):
        raise FileNotFoundError(f"Confidence SFT data was not generated: {script_args.dataset_name}")
