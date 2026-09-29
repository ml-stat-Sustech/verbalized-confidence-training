#!/usr/bin/env python3
"""Convert an existing confidence SFT JSONL to another output format."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_processing.sft_generation.sft_data_utils import format_confidence_target
from src.common.calibration_metrics import compute_binary_auroc, compute_calibration_metrics
from src.common.confidence_formats import parse_confidence_text
from src.common.system_prompts import get_sys_prompt


CONFIDENCE_TAG_AT_END = re.compile(r"<confidence>.*?</confidence>\s*\Z", re.DOTALL)


def convert_row(row: dict, confidence_format: str) -> tuple[dict, str]:
    target = float(row["confidence_target"])
    confidence_text, confidence_fields = format_confidence_target(
        target,
        confidence_format,
        "round",
        1.0 / 3.0,
        2.0 / 3.0,
    )
    messages = [dict(message) for message in row["messages"]]
    messages[0]["content"] = get_sys_prompt(confidence_format)
    assistant = messages[-1]["content"]
    if not CONFIDENCE_TAG_AT_END.search(assistant):
        raise ValueError(f"Row {row.get('id')} has no final <confidence> tag")
    messages[-1]["content"] = CONFIDENCE_TAG_AT_END.sub(
        f"<confidence>{confidence_text}</confidence>", assistant
    )

    converted = dict(row)
    converted.pop("confidence_probability", None)
    converted.pop("confidence_digit", None)
    converted.pop("confidence_linguistic", None)
    converted.update(confidence_fields)
    converted["messages"] = messages
    return converted, str(confidence_text)


def _metrics(correctness: list[float], confidence: list[float]) -> dict:
    metrics = compute_calibration_metrics(correctness, confidence)
    metrics["auroc"] = compute_binary_auroc(correctness, confidence)
    return metrics


def convert_file(source: Path, destination: Path, confidence_format: str) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    counts: Counter[str] = Counter()
    correctness: list[float] = []
    targets: list[float] = []
    represented_confidences: list[float] = []
    with source.open(encoding="utf-8") as input_handle, temporary.open(
        "w", encoding="utf-8"
    ) as output_handle:
        for line in input_handle:
            if not line.strip():
                continue
            converted, confidence_text = convert_row(json.loads(line), confidence_format)
            counts[confidence_text] += 1
            correctness.append(float(converted["rollout_correctness"]))
            targets.append(float(converted["confidence_target"]))
            represented_confidences.append(
                parse_confidence_text(confidence_text, confidence_format).value
            )
            output_handle.write(json.dumps(converted, ensure_ascii=False) + "\n")
    temporary.replace(destination)
    return {
        "source": str(source),
        "confidence_format": confidence_format,
        "sft_examples": len(correctness),
        "correct_rollouts": int(sum(correctness)),
        "mean_rollout_accuracy": sum(correctness) / len(correctness),
        "confidence_counts": dict(sorted(counts.items())),
        "target_metrics": _metrics(correctness, targets),
        "represented_confidence_metrics": _metrics(correctness, represented_confidences),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--confidence-format", choices=["digit", "linguistic"], required=True)
    args = parser.parse_args()

    split_summary = convert_file(args.source, args.destination, args.confidence_format)
    summary = {args.destination.stem: split_summary}
    summary_path = args.destination.parent / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
