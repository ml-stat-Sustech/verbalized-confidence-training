from __future__ import annotations

import importlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.confidence_formats import (  # noqa: E402
    CONFIDENCE_DIGIT,
    CONFIDENCE_LINGUISTIC,
    CONFIDENCE_NONE,
    CONFIDENCE_PROBABILITY,
    normalize_confidence_format,
)
from src.common.system_prompts import get_sys_prompt  # noqa: E402


USER_PROMPT_TEMPLATE = "\n\nPROBLEM: {question}\n\n"

ROLLOUT_DATASET_CONFIG_ALIASES = {
    "DeepScaleR": "DeepScaleRRL",
}


def normalize_answer(text: Any) -> str:
    if text is None:
        text = ""
    elif not isinstance(text, str):
        text = str(text)

    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def white_space_fix(value: str) -> str:
        return " ".join(value.split())

    def remove_punc(value: str) -> str:
        exclude = set("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")
        return "".join(ch for ch in value if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(text.lower())))


def exact_match_score(prediction: Any, ground_truth: Any) -> bool:
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def parse_splits(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("_").lower()


def format_float_for_path(value: float) -> str:
    return f"{float(value):g}"


def resolve_output_dir(
    output_root: str | Path,
    dataset: str,
    model: str,
    confidence_format: str,
    k: int | None = None,
    target_lambda: float | None = None,
    target_source: str = "mixed",
) -> Path:
    output_dir = Path(output_root) / slugify(dataset) / slugify(model) / normalize_confidence_format(confidence_format)
    if k is not None or target_lambda is not None:
        if k is None or target_lambda is None:
            raise ValueError("Both k and target_lambda are required when adding run parameters to output_dir.")
        suffix = f"k{int(k)}_lambda{format_float_for_path(target_lambda)}"
        if target_source != "mixed":
            suffix += f"_{target_source}"
        output_dir = output_dir / suffix
    return output_dir


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def load_config_class(module_suffix: str, class_name: str):
    module_aliases = {"methods": "algorithms"}
    module_suffix = module_aliases.get(module_suffix, module_suffix)
    module_name = module_suffix if module_suffix.startswith("src.") else f"src.train.configs.{module_suffix}"
    module = importlib.import_module(module_name)
    config_cls = getattr(module, class_name)
    return config_cls()


def resolve_model_name_or_path(model: str) -> str:
    try:
        model_config = load_config_class("models", model)
    except (ImportError, AttributeError):
        return model
    return model_config.model_name_or_path


def resolve_model_hf_overrides(model: str) -> dict[str, Any] | None:
    try:
        model_config = load_config_class("models", model)
    except (ImportError, AttributeError):
        return None
    overrides = dict(getattr(model_config, "hf_overrides", None) or {})
    if getattr(model_config, "force_long_rope", False):
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_config.model_name_or_path, trust_remote_code=False)
        rope = dict(config.rope_parameters)
        rope["short_factor"] = list(rope["long_factor"])
        overrides["rope_parameters"] = rope
    return overrides or None


def resolve_model_trust_remote_code(model: str) -> bool:
    try:
        return bool(load_config_class("models", model).trust_remote_code)
    except (ImportError, AttributeError):
        return True


def resolve_model_max_len(model: str) -> int | None:
    try:
        return getattr(load_config_class("models", model), "max_model_len", None)
    except (ImportError, AttributeError):
        return None


def resolve_rollout_dataset_config_name(dataset: str) -> str:
    return ROLLOUT_DATASET_CONFIG_ALIASES.get(dataset, dataset)


def resolve_dataset_max_completion_length(dataset: str) -> int:
    dataset_config_name = resolve_rollout_dataset_config_name(dataset)
    dataset_config = load_config_class("datasets", dataset_config_name)
    max_completion_length = getattr(dataset_config, "max_response_length", None)
    if max_completion_length is None:
        max_completion_length = getattr(dataset_config, "max_completion_length", None)
    if max_completion_length is None:
        raise ValueError(
            f"Dataset config {dataset_config_name} does not define max_response_length or max_completion_length."
        )
    return int(max_completion_length)


def _load_raw_dataset_from_train_dataset_class(dataset_name: str):
    from src.train.datasets.registry import DATASET_REGISTRY

    dataset_config_name = resolve_rollout_dataset_config_name(dataset_name)
    dataset_config = load_config_class("datasets", dataset_config_name)
    dataset_cls_name = getattr(dataset_config, "dataset_cls", None) or dataset_name
    dataset_cls = DATASET_REGISTRY.get(dataset_cls_name)
    if dataset_cls is None:
        raise KeyError(f"Unknown training dataset class: {dataset_cls_name}. Available: {sorted(DATASET_REGISTRY)}")
    if dataset_cls_name.endswith("SFT"):
        raise ValueError(f"{dataset_name} is already an SFT dataset and is not supported as a rollout source.")

    dataset_obj = object.__new__(dataset_cls)
    dataset_obj.args = dataset_config
    if hasattr(dataset_obj, "_load_raw_dataset"):
        raw_dataset = dataset_obj._load_raw_dataset()
    else:
        raw_dataset = dataset_obj.load_dataset()
    return dataset_config, dataset_obj.reformat(raw_dataset)


def load_standard_split(
    dataset_name: str,
    split: str,
    limit: int | None,
    seed: int,
    shuffle: bool,
):
    from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict

    dataset_config, dataset_dict = _load_raw_dataset_from_train_dataset_class(dataset_name)
    if isinstance(dataset_dict, (Dataset, IterableDataset)):
        dataset_dict = DatasetDict({dataset_config.dataset_train_split: dataset_dict})
    if not isinstance(dataset_dict, (DatasetDict, IterableDatasetDict)):
        raise TypeError(f"Unsupported dataset type for {dataset_name}: {type(dataset_dict)}")
    if split not in dataset_dict:
        raise KeyError(f"Split {split!r} not found in {dataset_name}. Available splits: {list(dataset_dict.keys())}")

    dataset = dataset_dict[split]

    def mapping(example: dict[str, Any], idx: int) -> dict[str, Any]:
        question = example.get("question") or example.get("problem")
        if question is None:
            raise KeyError(f"Could not find question/problem in example keys: {list(example.keys())}")
        answer = example.get("answer")
        if answer is None:
            raise KeyError(f"Could not find answer in example keys: {list(example.keys())}")
        source_id = example.get("id", f"{split}-{idx}")
        return {
            "id": str(source_id),
            "split": split,
            "source_index": idx,
            "question": str(question),
            "gold_answer": str(answer),
            "raw_solution": str(example.get("solution", example.get("raw_solution", answer))),
            "source": str(example.get("source", slugify(dataset_name))),
        }

    dataset = dataset.map(mapping, with_indices=True)
    if shuffle:
        dataset = dataset.shuffle(seed=seed)
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))
    return dataset


def build_messages(
    question: str,
    confidence_format: str,
) -> list[dict[str, str]]:
    system_prompt = get_sys_prompt(confidence_format)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(question=question)},
    ]


def build_prompt_text(
    tokenizer: Any,
    question: str,
    confidence_format: str,
) -> str:
    prompt = tokenizer.apply_chat_template(
        build_messages(question, confidence_format),
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt


def restore_prefilled_think_tag(prompt: str, completion: str) -> str:
    if prompt.rstrip().endswith("<think>") and not completion.lstrip().startswith("<think>"):
        return "<think>" + completion
    return completion


def last_tag_value(text: str, tag: str) -> str | None:
    matches = re.findall(rf"<{tag}>(.*?)</{tag}>", text, flags=re.DOTALL | re.MULTILINE)
    if not matches:
        return None
    return matches[-1].strip()


def extract_reasoning_and_answer(completion: str) -> tuple[str | None, str | None]:
    return last_tag_value(completion, "think"), last_tag_value(completion, "answer")


def has_answer(completion: str) -> bool:
    answer = last_tag_value(completion, "answer")
    return answer is not None and answer != ""


def answers_equivalent(prediction: str, gold_answer: str, source: str | None = None) -> bool:
    if not prediction or not gold_answer:
        return False
    try:
        from math_verify import parse, verify

        if verify(parse(gold_answer), parse(prediction)):
            return True
    except Exception:
        pass
    try:
        from math_verify import parse, verify

        if verify(parse(prediction), parse(gold_answer)):
            return True
    except Exception:
        pass
    return bool(exact_match_score(prediction, gold_answer))


def clamp_probability(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def confidence_target(
    group_success_rate: float,
    rollout_correctness: float,
    target_lambda: float,
) -> float:
    target = target_lambda * group_success_rate + (1.0 - target_lambda) * rollout_correctness
    return clamp_probability(target)


def discretize_confidence(target: float, method: str) -> int:
    target = clamp_probability(target)
    scaled = target * 9.0
    if method == "floor":
        digit = math.floor(scaled)
    elif method == "ceil":
        digit = math.ceil(scaled)
    elif method == "round":
        digit = math.floor(scaled + 0.5)
    else:
        raise ValueError(f"Unknown binning method: {method}")
    return int(min(9, max(0, digit)))


def linguistic_confidence_label(target: float, low_high_boundary: float, medium_high_boundary: float) -> str:
    target = clamp_probability(target)
    if not 0.0 <= low_high_boundary <= medium_high_boundary <= 1.0:
        raise ValueError(
            "Expected linguistic boundaries to satisfy 0 <= low_high_boundary <= medium_high_boundary <= 1."
        )
    if target < low_high_boundary:
        return "low"
    if target < medium_high_boundary:
        return "medium"
    return "high"


def format_confidence_target(
    target: float,
    confidence_format: str,
    binning: str,
    linguistic_low_medium_boundary: float,
    linguistic_medium_high_boundary: float,
) -> tuple[str | None, dict[str, Any]]:
    confidence_format = normalize_confidence_format(confidence_format)
    target = clamp_probability(target)
    if confidence_format == CONFIDENCE_NONE:
        return None, {}
    if confidence_format == CONFIDENCE_PROBABILITY:
        return f"{target:.2f}", {"confidence_probability": round(target, 6)}
    if confidence_format == CONFIDENCE_DIGIT:
        digit = discretize_confidence(target, binning)
        return str(digit), {"confidence_digit": digit}
    if confidence_format == CONFIDENCE_LINGUISTIC:
        label = linguistic_confidence_label(
            target,
            linguistic_low_medium_boundary,
            linguistic_medium_high_boundary,
        )
        return label, {"confidence_linguistic": label}
    raise ValueError(f"Invalid confidence_format: {confidence_format}")


def build_assistant_message(
    reasoning: str,
    answer: str,
    confidence_format: str,
    confidence_text: str | None,
) -> str:
    content = f"<think>{reasoning.strip()}</think> <answer>{answer.strip()}</answer>"
    if normalize_confidence_format(confidence_format) != CONFIDENCE_NONE:
        if confidence_text is None:
            raise ValueError("confidence_text is required when confidence_format is not none.")
        content += f" <confidence>{confidence_text}</confidence>"
    return content
