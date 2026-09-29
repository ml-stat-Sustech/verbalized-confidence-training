#!/usr/bin/env python3
"""Generate confidence SFT data with a single stage-controlled entrypoint."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_processing.sft_generation.sft_data_utils import (  # noqa: E402
    USER_PROMPT_TEMPLATE,
    answers_equivalent,
    append_jsonl,
    build_assistant_message,
    build_messages,
    build_prompt_text,
    confidence_target,
    extract_reasoning_and_answer,
    format_confidence_target,
    has_answer,
    load_standard_split,
    normalize_confidence_format,
    parse_splits,
    read_json,
    read_jsonl,
    resolve_dataset_max_completion_length,
    resolve_model_hf_overrides,
    resolve_model_max_len,
    resolve_model_trust_remote_code,
    resolve_model_name_or_path,
    resolve_output_dir,
    restore_prefilled_think_tag,
    write_json,
    write_jsonl,
)
from src.common.system_prompts import get_sys_prompt, get_system_prompt_name  # noqa: E402
from src.common.calibration_metrics import compute_binary_auroc, compute_calibration_metrics  # noqa: E402


FILL_MISSING_ANSWER_INST = "Thinking time ended \n\n. My final answer is "
ROLLOUT_CONFIDENCE_FORMAT_SAME = "same"
VALIDATION_SPLIT = "validation"
STRUCTURE_TAG_RE = re.compile(r"</?(?:think|answer|confidence)>", re.IGNORECASE)
ANSWER_OPEN_TAG_RE = re.compile(r"<answer>", re.IGNORECASE)


@dataclass
class RolloutRecord:
    id: str
    split: str
    source_index: int
    question: str
    gold_answer: str
    raw_solution: str
    source: str
    prompt: list[dict[str, str]]
    prompt_text: str
    completions: list[str]
    completion_metadata: list[dict[str, Any]]
    answer_fill_count: int


def record_id_for_message(row: dict[str, Any], idx: int) -> str:
    return str(row.get("id", f"row-{idx}"))


def has_completion_quality_metadata(row: dict[str, Any]) -> bool:
    completions = row.get("completions", [])
    metadata = row.get("completion_metadata")
    return (
        isinstance(metadata, list)
        and len(metadata) == len(completions)
        and all(
            isinstance(item, dict)
            and "finish_reason" in item
            and "answer_was_filled" in item
            for item in metadata
        )
    )


def rollout_raw_path(args: argparse.Namespace, split: str) -> Path:
    output_dir = Path(args.output_dir)
    rollout_dir = Path(args.rollout_dir) if args.rollout_dir else output_dir / "rollouts"
    num_shards = int(getattr(args, "num_shards", 1))
    shard_index = getattr(args, "shard_index", None)
    if num_shards > 1 and shard_index is not None:
        return rollout_dir / f"{split}_raw_rollouts.shard-{shard_index:05d}-of-{num_shards:05d}.jsonl"
    return rollout_dir / f"{split}_raw_rollouts.jsonl"


def rollout_raw_paths_for_build(args: argparse.Namespace, split: str) -> list[Path]:
    num_shards = int(getattr(args, "num_shards", 1))
    if num_shards == 1:
        return [rollout_raw_path(args, split)]

    paths = []
    for shard_index in range(num_shards):
        shard_values = {**vars(args), "shard_index": shard_index}
        shard_args = argparse.Namespace(**shard_values)
        paths.append(rollout_raw_path(shard_args, split))
    missing = [path for path in paths if not path.is_file()]
    if missing:
        preview = ", ".join(str(path) for path in missing[:5])
        raise FileNotFoundError(f"Missing {len(missing)} rollout shard(s) for {split}: {preview}")
    return paths


def split_limit(args: argparse.Namespace, split: str) -> int | None:
    split_key = split.replace("-", "_")
    return getattr(args, f"{split_key}_size", None) or args.limit


def source_split(args: argparse.Namespace, output_split: str) -> str:
    if output_split == VALIDATION_SPLIT:
        return args.validation_source_split
    return output_split


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def log_progress(label: str, done: int, total: int, started_at: float, unit: str) -> None:
    elapsed = time.monotonic() - started_at
    rate = done / elapsed if elapsed > 0 else 0.0
    remaining = max(0, total - done)
    eta = remaining / rate if rate > 0 else 0.0
    pct = 100.0 * done / total if total else 100.0
    print(
        f"{label}: {done}/{total} {unit} ({pct:.1f}%) "
        f"elapsed={format_duration(elapsed)} eta={format_duration(eta)} rate={rate:.2f} {unit}/s",
        flush=True,
    )


def extract_filled_answer(text: str) -> str:
    _, tagged_answer = extract_reasoning_and_answer(text)
    if tagged_answer is not None:
        candidate = tagged_answer
    else:
        answer_starts = list(ANSWER_OPEN_TAG_RE.finditer(text))
        candidate = text[answer_starts[-1].end() :] if answer_starts else text
    return STRUCTURE_TAG_RE.split(candidate, maxsplit=1)[0].strip()


def has_clean_completion_parts(reasoning: str | None, answer: str | None) -> bool:
    return bool(
        reasoning
        and answer
        and not STRUCTURE_TAG_RE.search(reasoning)
        and not STRUCTURE_TAG_RE.search(answer)
    )


def fill_missing_answers(
    llm: Any,
    prompt_texts: list[str],
    completions_by_prompt: list[list[str]],
    batch_size: int,
) -> tuple[list[list[str]], set[tuple[int, int]]]:
    from vllm import SamplingParams

    missing_refs = []
    fill_prompts = []
    for prompt_idx, (prompt_text, completions) in enumerate(zip(prompt_texts, completions_by_prompt)):
        for sample_idx, completion in enumerate(completions):
            if not has_answer(completion):
                missing_refs.append((prompt_idx, sample_idx))
                fill_prompts.append(prompt_text + completion + FILL_MISSING_ANSWER_INST)

    if not fill_prompts:
        return completions_by_prompt, set()

    sampling_params = SamplingParams(n=1, temperature=0, max_tokens=50)
    fill_outputs = []
    started_at = time.monotonic()
    fill_total = len(fill_prompts)
    fill_ranges = range(0, fill_total, batch_size)
    for start in tqdm(fill_ranges, desc="fill missing <answer>"):
        fill_outputs.extend(llm.generate(fill_prompts[start : start + batch_size], sampling_params, use_tqdm=False))
        log_progress(
            "fill missing <answer>",
            min(start + batch_size, fill_total),
            fill_total,
            started_at,
            "prompts",
        )

    for (prompt_idx, sample_idx), output in zip(missing_refs, fill_outputs):
        answer_text = extract_filled_answer(output.outputs[0].text)
        completions_by_prompt[prompt_idx][sample_idx] = (
            completions_by_prompt[prompt_idx][sample_idx] + f"<answer>{answer_text}</answer>"
        )

    return completions_by_prompt, set(missing_refs)


def validate_existing_rollout_file(args: argparse.Namespace, split: str) -> None:
    raw_path = rollout_raw_path(args, split)
    existing = read_jsonl(raw_path)

    mismatched_existing = [
        record_id_for_message(row, idx)
        for idx, row in enumerate(existing)
        if len(row.get("completions", [])) != args.k
    ]
    if mismatched_existing:
        preview = ", ".join(mismatched_existing[:5])
        raise ValueError(
            f"{raw_path} contains records whose rollout count does not match --k={args.k}: {preview}. "
            "Use a fresh output directory or rebuild the raw rollout file."
        )

    if args.drop_invalid_format:
        missing_quality_metadata = [
            record_id_for_message(row, idx)
            for idx, row in enumerate(existing)
            if not has_completion_quality_metadata(row)
        ]
        if missing_quality_metadata:
            preview = ", ".join(missing_quality_metadata[:5])
            raise ValueError(
                f"{raw_path} lacks per-rollout quality metadata for: {preview}. "
                "Use a fresh output directory or rebuild the raw rollout file."
            )

    expected_system_prompt = build_messages(
        "",
        args.rollout_confidence_format,
    )[0]["content"]
    mismatched_prompt_existing = [
        record_id_for_message(row, idx)
        for idx, row in enumerate(existing)
        if not row.get("prompt") or row["prompt"][0].get("content") != expected_system_prompt
    ]
    if mismatched_prompt_existing:
        preview = ", ".join(mismatched_prompt_existing[:5])
        raise ValueError(
            f"{raw_path} contains records whose rollout prompt does not match "
            f"rollout_confidence_format={args.rollout_confidence_format}: {preview}. "
            "Use a fresh output directory or rebuild the raw rollout file."
        )


def load_base_rollouts(args: argparse.Namespace, split: str) -> tuple[dict[str, dict[str, Any]], int]:
    if not args.base_rollout_dir:
        return {}, 0

    base_num_shards = int(getattr(args, "base_num_shards", None) or args.num_shards)
    if args.num_shards % base_num_shards != 0:
        raise ValueError(
            f"--num-shards={args.num_shards} must be divisible by "
            f"--base-num-shards={base_num_shards}"
        )
    base_shard_index = None
    if base_num_shards > 1:
        if args.shard_index is None:
            raise ValueError("--shard-index is required when --base-num-shards is greater than 1")
        base_shard_index = args.shard_index * base_num_shards // args.num_shards
    base_args = argparse.Namespace(
        **{
            **vars(args),
            "rollout_dir": args.base_rollout_dir,
            "num_shards": base_num_shards,
            "shard_index": base_shard_index,
        }
    )
    base_path = rollout_raw_path(base_args, split)
    if not base_path.is_file():
        raise FileNotFoundError(f"Missing base rollout file for {split}: {base_path}")
    records = read_jsonl(base_path)
    rollout_counts = {len(row.get("completions", [])) for row in records}
    if len(rollout_counts) != 1:
        raise ValueError(f"Base rollout file has inconsistent rollout counts: {base_path}")
    base_k = rollout_counts.pop()
    if not 0 < base_k < args.k:
        raise ValueError(f"Base rollout count must be between 1 and --k={args.k - 1}, got {base_k}")

    base_args.k = base_k
    validate_existing_rollout_file(base_args, split)
    by_id = {row["id"]: row for row in records}
    if len(by_id) != len(records):
        raise ValueError(f"Base rollout file contains duplicate IDs: {base_path}")
    return by_id, base_k


def rollout_split(args: argparse.Namespace, split: str, llm: Any, tokenizer: Any) -> None:
    from vllm import SamplingParams

    raw_path = rollout_raw_path(args, split)
    validate_existing_rollout_file(args, split)
    existing = read_jsonl(raw_path)
    base_by_id, base_k = load_base_rollouts(args, split)

    existing_ids = {row["id"] for row in existing}
    dataset = load_standard_split(
        dataset_name=args.dataset,
        split=source_split(args, split),
        limit=split_limit(args, split),
        seed=args.seed,
        shuffle=args.shuffle or split == VALIDATION_SPLIT,
    )
    if args.num_shards > 1:
        base_num_shards = int(getattr(args, "base_num_shards", None) or args.num_shards)
        if args.base_rollout_dir and base_num_shards < args.num_shards:
            subdivisions = args.num_shards // base_num_shards
            dataset = dataset.shard(
                num_shards=base_num_shards,
                index=args.shard_index // subdivisions,
                contiguous=True,
            )
            dataset = dataset.shard(
                num_shards=subdivisions,
                index=args.shard_index % subdivisions,
                contiguous=True,
            )
        else:
            dataset = dataset.shard(
                num_shards=args.num_shards,
                index=args.shard_index,
                contiguous=True,
            )
    pending = [row for row in dataset if row["id"] not in existing_ids]
    if not pending:
        print(f"{split}: raw rollouts already complete at {raw_path}")
        return
    missing_base_ids = [row["id"] for row in pending if base_by_id and row["id"] not in base_by_id]
    if missing_base_ids:
        preview = ", ".join(missing_base_ids[:5])
        raise ValueError(f"Base rollout file is missing {len(missing_base_ids)} IDs for {split}: {preview}")

    prompt_texts = [
        build_prompt_text(
            tokenizer,
            row["question"],
            args.rollout_confidence_format,
        )
        for row in pending
    ]
    sampling_params = SamplingParams(
        n=args.k - base_k,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_completion_length,
        seed=args.sampling_seed if args.sampling_seed is not None else args.seed,
    )

    total_answer_fills = 0
    started_at = time.monotonic()
    total_batches = (len(pending) + args.prompt_batch_size - 1) // args.prompt_batch_size
    print(
        f"rollout {split}: start total_records={len(pending)} k={args.k} "
        f"reused_per_record={base_k} new_per_record={args.k - base_k} "
        f"total_generations={len(pending) * (args.k - base_k)} batches={total_batches}",
        flush=True,
    )
    for batch_idx, start in enumerate(
        tqdm(range(0, len(pending), args.prompt_batch_size), desc=f"rollout {split}"),
        start=1,
    ):
        batch_rows = pending[start : start + args.prompt_batch_size]
        batch_prompt_texts = prompt_texts[start : start + args.prompt_batch_size]
        outputs = llm.generate(batch_prompt_texts, sampling_params, use_tqdm=False)
        new_completions_by_prompt = [[candidate.text for candidate in output.outputs] for output in outputs]
        finish_reasons_by_prompt = [
            [candidate.finish_reason for candidate in output.outputs]
            for output in outputs
        ]
        new_completions_by_prompt, filled_refs = fill_missing_answers(
            llm,
            batch_prompt_texts,
            new_completions_by_prompt,
            batch_size=args.fill_batch_size,
        )
        new_completions_by_prompt = [
            [restore_prefilled_think_tag(prompt, completion) for completion in completions]
            for prompt, completions in zip(batch_prompt_texts, new_completions_by_prompt)
        ]
        total_answer_fills += len(filled_refs)

        records = []
        for row_idx, (row, prompt_text, new_completions, batch_output) in enumerate(
            zip(batch_rows, batch_prompt_texts, new_completions_by_prompt, outputs)
        ):
            base_record = base_by_id.get(row["id"], {})
            base_completions = list(base_record.get("completions", []))
            base_metadata = list(base_record.get("completion_metadata", []))
            records.append(
                asdict(
                    RolloutRecord(
                        id=row["id"],
                        split=split,
                        source_index=int(row["source_index"]),
                        question=row["question"],
                        gold_answer=row["gold_answer"],
                        raw_solution=row["raw_solution"],
                        source=row["source"],
                        prompt=build_messages(
                            row["question"],
                            args.rollout_confidence_format,
                        ),
                        prompt_text=prompt_text,
                        completions=base_completions + new_completions,
                        completion_metadata=base_metadata + [
                            {
                                "finish_reason": finish_reason,
                                "answer_was_filled": (row_idx, sample_idx)
                                in filled_refs,
                            }
                            for sample_idx, finish_reason in enumerate(
                                finish_reasons_by_prompt[row_idx]
                            )
                        ],
                        answer_fill_count=int(base_record.get("answer_fill_count", 0)) + sum(
                            1
                            for sample_idx in range(len(new_completions))
                            if (row_idx, sample_idx) in filled_refs
                        ),
                    )
                )
            )
        append_jsonl(raw_path, records)
        log_progress(
            f"rollout {split} batch {batch_idx}/{total_batches}",
            min(start + len(batch_rows), len(pending)),
            len(pending),
            started_at,
            "records",
        )

    elapsed = time.monotonic() - started_at
    print(
        json.dumps(
            {
                "split": split,
                "new_records": len(pending),
                "rollouts_per_record": args.k,
                "reused_rollouts_per_record": base_k,
                "new_rollouts_per_record": args.k - base_k,
                "elapsed_seconds": round(elapsed, 2),
                "elapsed": format_duration(elapsed),
                "filled_missing_answers": total_answer_fills,
                "raw_path": str(raw_path),
            },
            indent=2,
            sort_keys=True,
        )
    )


def run_rollout(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    rollout_dir = Path(args.rollout_dir) if args.rollout_dir else output_dir / "rollouts"
    rollout_dir.mkdir(parents=True, exist_ok=True)
    for split in parse_splits(args.splits):
        validate_existing_rollout_file(args, split)
    if args.shard_index in {None, 0}:
        write_json(
            output_dir / "rollout_config.json",
            {
                **vars(args),
                "shard_index": None,
                "rollout_dir": str(rollout_dir),
                "system_prompt_name": get_system_prompt_name(args.rollout_confidence_format),
                "system_prompt": build_messages(
                    "",
                    args.rollout_confidence_format,
                )[0]["content"],
                "target_system_prompt_name": get_system_prompt_name(args.confidence_format),
                "target_system_prompt": get_sys_prompt(args.confidence_format),
                "fill_missing_answer_inst": FILL_MISSING_ANSWER_INST,
            },
        )

    from transformers import AutoTokenizer
    from vllm import LLM

    trust_remote_code = resolve_model_trust_remote_code(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=trust_remote_code)
    llm = LLM(
        model=args.model_name_or_path,
        trust_remote_code=trust_remote_code,
        max_model_len=resolve_model_max_len(args.model),
        hf_overrides=args.model_hf_overrides,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    for split in parse_splits(args.splits):
        rollout_split(args, split=split, llm=llm, tokenizer=tokenizer)


def build_split(
    args: argparse.Namespace,
    split: str,
) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    raw_paths = rollout_raw_paths_for_build(args, split)
    sft_path = output_dir / f"{split}.jsonl"
    audit_path = output_dir / f"{split}_rollout_audit.jsonl"

    raw_records = [row for raw_path in raw_paths for row in read_jsonl(raw_path)]
    if not raw_records:
        raise FileNotFoundError(f"No raw rollouts found for split {split}: {raw_paths}")

    mismatched_rollout_counts = [
        row.get("id", f"row-{idx}")
        for idx, row in enumerate(raw_records)
        if len(row.get("completions", [])) != args.k
    ]
    if mismatched_rollout_counts:
        preview = ", ".join(mismatched_rollout_counts[:5])
        raise ValueError(
            f"Raw rollout data contains records whose rollout count does not match --k={args.k}: {preview}. "
            "Use the matching K or rebuild the raw rollout file."
        )

    sft_rows = []
    audit_rows = []
    skipped_missing_tags = 0
    answer_filled_rollouts = 0
    skipped_low_quality_rollouts = 0
    skipped_training_rollouts = 0
    total_rollouts = 0
    correct_rollouts = 0
    confidence_counts: Counter[str] = Counter()
    target_values: list[float] = []
    target_correctness: list[float] = []
    target_source = getattr(args, "confidence_target_source", "mixed")

    system_prompt = get_sys_prompt(args.confidence_format)
    for record in tqdm(raw_records, desc=f"build {split}"):
        completions = record["completions"]
        completion_metadata = record.get("completion_metadata")
        if args.drop_invalid_format and not has_completion_quality_metadata(record):
            raise ValueError(
                f"Raw rollout data lacks per-rollout quality metadata for record {record.get('id')!r}. "
                "Regenerate the raw rollouts before building a cleaned SFT split."
            )
        if completion_metadata is None:
            completion_metadata = [{} for _ in completions]
        rollout_infos = []
        source = record.get("source")
        for rollout_index, (completion, metadata) in enumerate(
            zip(completions, completion_metadata, strict=True)
        ):
            reasoning, answer = extract_reasoning_and_answer(completion)
            is_correct = bool(answer is not None and answers_equivalent(answer, record["gold_answer"], source=source))
            answer_was_filled = bool(metadata.get("answer_was_filled", False))
            finish_reason = metadata.get("finish_reason")
            has_native_quality_metadata = bool(metadata)
            is_low_quality = has_native_quality_metadata and finish_reason != "stop"
            has_valid_format = has_clean_completion_parts(reasoning, answer)
            rollout_infos.append(
                {
                    "rollout_index": rollout_index,
                    "completion": completion,
                    "reasoning": reasoning,
                    "answer": answer,
                    "is_correct": is_correct,
                    "finish_reason": finish_reason,
                    "answer_was_filled": answer_was_filled,
                    "is_low_quality": is_low_quality,
                    "has_valid_format": has_valid_format,
                    "kept_for_sft": not (
                        args.drop_invalid_format
                        and (is_low_quality or not has_valid_format)
                    ),
                }
            )
            correct_rollouts += int(is_correct)
        total_rollouts += len(rollout_infos)
        group_success_rate = (
            sum(int(info["is_correct"]) for info in rollout_infos) / len(rollout_infos) if rollout_infos else 0.0
        )
        audit_rows.append(
            {
                "id": record["id"],
                "split": split,
                "source_index": record["source_index"],
                "question": record["question"],
                "gold_answer": record["gold_answer"],
                "source": source,
                "group_success_rate": group_success_rate,
                "sft_method": "calibsft",
                "answer_fill_count": record.get("answer_fill_count", 0),
                "rollouts": rollout_infos
                if args.save_audit_completions
                else [
                    {
                        "rollout_index": info["rollout_index"],
                        "answer": info["answer"],
                        "is_correct": info["is_correct"],
                        "finish_reason": info["finish_reason"],
                        "answer_was_filled": info["answer_was_filled"],
                        "kept_for_sft": info["kept_for_sft"],
                    }
                    for info in rollout_infos
                ],
            }
        )

        for info in rollout_infos:
            if not info["has_valid_format"]:
                skipped_missing_tags += 1
            if info["answer_was_filled"]:
                answer_filled_rollouts += 1
            if info["is_low_quality"]:
                skipped_low_quality_rollouts += 1
            if not info["kept_for_sft"]:
                skipped_training_rollouts += 1
                continue
            reasoning = info["reasoning"] if info["reasoning"] is not None else info["completion"].strip()
            answer = info["answer"] if info["answer"] is not None else ""
            rollout_correctness = 1.0 if info["is_correct"] else 0.0
            target = confidence_target(
                group_success_rate=group_success_rate,
                rollout_correctness=rollout_correctness,
                target_lambda=args.target_lambda,
            )
            effective_target_source = target_source
            confidence_text, confidence_fields = format_confidence_target(
                target,
                args.confidence_format,
                args.binning,
                args.linguistic_low_medium_boundary,
                args.linguistic_medium_high_boundary,
            )
            if confidence_text is not None:
                confidence_counts[confidence_text] += 1
            target_values.append(float(target))
            target_correctness.append(rollout_correctness)
            assistant_content = build_assistant_message(
                reasoning=reasoning,
                answer=answer,
                confidence_format=args.confidence_format,
                confidence_text=confidence_text,
            )
            sft_rows.append(
                {
                    "id": f"{record['id']}-r{info['rollout_index']}",
                    "source_id": record["id"],
                    "split": split,
                    "source_index": record["source_index"],
                    "rollout_index": info["rollout_index"],
                    "question": record["question"],
                    "gold_answer": record["gold_answer"],
                    "source": source,
                    "rollout_answer": answer,
                    "rollout_correctness": rollout_correctness,
                    "group_success_rate": group_success_rate,
                    "confidence_target": target if args.confidence_format != "none" else None,
                    "confidence_target_source": effective_target_source,
                    "sft_method": "calibsft",
                    **confidence_fields,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": USER_PROMPT_TEMPLATE.format(question=record["question"]),
                        },
                        {"role": "assistant", "content": assistant_content},
                    ],
                }
            )

    rng = random.Random(args.seed)
    rng.shuffle(sft_rows)
    write_jsonl(sft_path, sft_rows)
    write_jsonl(audit_path, audit_rows)

    target_metrics = compute_calibration_metrics(target_correctness, target_values) if target_values else {}
    target_metrics["auroc"] = compute_binary_auroc(target_correctness, target_values)
    return {
        "raw_records": len(raw_records),
        "total_rollouts": total_rollouts,
        "correct_rollouts": correct_rollouts,
        "mean_rollout_accuracy": correct_rollouts / total_rollouts if total_rollouts else 0.0,
        "sft_examples": len(sft_rows),
        "skipped_missing_tags": skipped_missing_tags,
        "answer_filled_rollouts": answer_filled_rollouts,
        "skipped_low_quality_rollouts": skipped_low_quality_rollouts,
        "skipped_training_rollouts": skipped_training_rollouts,
        "sft_method": "calibsft",
        "target_formula": "lambda * group_success_rate + (1 - lambda) * rollout_correctness",
        "target_metrics": target_metrics,
        "confidence_counts": dict(sorted(confidence_counts.items())),
    }


def run_build(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    requested_splits = parse_splits(args.splits)
    config_path = output_dir / "dataset_config.json"
    existing_config = read_json(config_path)
    existing_splits = parse_splits(existing_config.get("splits", ""))
    generated_splits = list(dict.fromkeys([*existing_splits, *requested_splits]))
    write_json(
        config_path,
        {
            **vars(args),
            "splits": ",".join(generated_splits),
            "rollout_dir": str(Path(args.rollout_dir) if args.rollout_dir else output_dir / "rollouts"),
            "system_prompt_name": get_system_prompt_name(args.confidence_format),
            "system_prompt": get_sys_prompt(args.confidence_format),
            "target_formula": "lambda * group_success_rate + (1 - lambda) * rollout_correctness",
            "confidence_target_source": "mixed",
            "sft_method": getattr(args, "sft_method", "calibsft"),
        },
    )

    summary_path = output_dir / "summary.json"
    summaries = read_json(summary_path)
    for split in requested_splits:
        summaries[split] = build_split(args, split=split)
    write_json(summary_path, summaries)
    print(json.dumps(summaries, indent=2, sort_keys=True))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["all", "stage1", "stage2"], default="all")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True, help="Training model config class or a Hugging Face model path.")
    parser.add_argument("--model_name_or_path", default=None, help="Override resolved model path.")
    parser.add_argument("--confidence_format", required=True, choices=["none", "probability", "digit", "linguistic"])
    parser.add_argument(
        "--rollout_confidence_format",
        default=ROLLOUT_CONFIDENCE_FORMAT_SAME,
        choices=[ROLLOUT_CONFIDENCE_FORMAT_SAME, "none", "probability", "digit", "linguistic"],
        help=(
            "Confidence format used in the rollout prompt. The default 'same' matches --confidence_format. "
            "Use 'none' only when reusing "
            "answer-only rollouts."
        ),
    )
    parser.add_argument("--output_root", default="data/sft_data")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--rollout_dir", "--rollout-dir", default=None)
    parser.add_argument("--base_rollout_dir", "--base-rollout-dir", default=None)
    parser.add_argument("--base_num_shards", "--base-num-shards", type=int, default=None)
    parser.add_argument("--splits", default="train,validation")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--train_size", type=int, default=None)
    parser.add_argument("--test_size", type=int, default=None)
    parser.add_argument("--validation_size", "--validation-size", type=int, default=200)
    parser.add_argument(
        "--validation_source_split",
        "--validation-source-split",
        default="test",
        help="Raw dataset split used to build validation.jsonl.",
    )
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument(
        "--max_completion_length",
        type=int,
        default=None,
        help="Max generated tokens. Defaults to the dataset train config max_completion_length.",
    )
    parser.add_argument("--prompt_batch_size", type=int, default=256)
    parser.add_argument("--fill_batch_size", type=int, default=256)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--num_shards", "--num-shards", type=int, default=1)
    parser.add_argument("--shard_index", "--shard-index", type=int, default=None)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--sampling_seed", "--sampling-seed", type=int, default=None)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--target_lambda", type=float, default=0.5)
    parser.add_argument(
        "--confidence_target_source",
        "--confidence-target-source",
        choices=["mixed"],
        default="mixed",
        help="CalibSFT confidence target source.",
    )
    parser.add_argument(
        "--sft_method",
        "--sft-method",
        choices=["calibsft"],
        default="calibsft",
        help="Data-construction strategy.",
    )
    parser.add_argument("--binning", choices=["round", "floor", "ceil"], default="round")
    parser.add_argument("--linguistic_low_medium_boundary", type=float, default=1.0 / 3.0)
    parser.add_argument("--linguistic_medium_high_boundary", type=float, default=2.0 / 3.0)
    parser.add_argument(
        "--drop_invalid_format",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop non-stopping or malformed rollouts from SFT rows.",
    )
    parser.add_argument("--save_audit_completions", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    main_from_namespace(args)


def main_from_namespace(args: argparse.Namespace) -> None:
    for action in build_arg_parser()._actions:
        if not hasattr(args, action.dest) and action.default is not argparse.SUPPRESS:
            setattr(args, action.dest, action.default)
    args.num_shards = int(getattr(args, "num_shards", 1))
    args.shard_index = getattr(args, "shard_index", None)
    args.confidence_format = normalize_confidence_format(args.confidence_format)
    if args.rollout_confidence_format == ROLLOUT_CONFIDENCE_FORMAT_SAME:
        args.rollout_confidence_format = args.confidence_format
    else:
        args.rollout_confidence_format = normalize_confidence_format(args.rollout_confidence_format)
    args.model_name_or_path = args.model_name_or_path or resolve_model_name_or_path(args.model)
    args.model_hf_overrides = resolve_model_hf_overrides(args.model)
    args.output_dir = args.output_dir or str(
        resolve_output_dir(
            args.output_root,
            args.dataset,
            args.model,
            args.confidence_format,
            k=args.k,
            target_lambda=args.target_lambda,
            target_source=args.confidence_target_source,
        )
    )
    if args.k <= 0:
        raise ValueError(f"--k must be positive, got {args.k}")
    if args.num_shards <= 0:
        raise ValueError(f"--num_shards must be positive, got {args.num_shards}")
    if args.shard_index is not None and not 0 <= args.shard_index < args.num_shards:
        raise ValueError(
            f"--shard_index must be in [0, {args.num_shards}), got {args.shard_index}"
        )
    if args.stage in {"all", "stage1"} and args.num_shards > 1 and args.shard_index is None:
        raise ValueError("Distributed rollout requires --shard_index for stage1 generation")
    if args.stage == "stage2" and args.shard_index is not None:
        raise ValueError("Stage2 reads all stage1 shards; omit --shard_index")
    if not 0.0 <= args.target_lambda <= 1.0:
        raise ValueError(f"--target_lambda must be in [0, 1], got {args.target_lambda}")
    if args.stage in {"all", "stage1"}:
        if args.max_completion_length is None:
            args.max_completion_length = resolve_dataset_max_completion_length(args.dataset)
        if args.max_completion_length <= 0:
            raise ValueError(f"--max_completion_length must be positive, got {args.max_completion_length}")

    if args.stage in {"all", "stage1"}:
        run_rollout(args)
    if args.stage in {"all", "stage2"}:
        run_build(args)


if __name__ == "__main__":
    main()
