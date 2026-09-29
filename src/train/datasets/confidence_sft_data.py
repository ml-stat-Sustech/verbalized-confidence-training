from pathlib import Path
from typing import Any

from datasets import load_dataset


class ConfidenceSFTJSONLoader:
    """Confidence SFT data stored as JSON/JSONL records with a messages field."""

    def __init__(self, config):
        self.args = config
        self.dataset = self.load_dataset()

    def _candidate_split_paths(self, dataset_path: Path, split: str) -> list[Path]:
        return [
            dataset_path / f"{split}.jsonl",
            dataset_path / f"{split}.json",
        ]

    def _resolve_split_paths(self) -> dict[str, str]:
        dataset_path = Path(self.args.dataset_name)
        train_split = self.args.dataset_train_split
        test_split = self.args.dataset_test_split

        if dataset_path.is_file():
            return {train_split: str(dataset_path)}

        if not dataset_path.exists():
            raise FileNotFoundError(f"Confidence SFT dataset path does not exist: {dataset_path}")
        if not dataset_path.is_dir():
            raise ValueError(f"Confidence SFT dataset path must be a JSON/JSONL file or directory: {dataset_path}")

        split_paths = {}
        missing_splits = {}
        requested_splits = [train_split]
        if test_split != train_split:
            requested_splits.append(test_split)

        for split in requested_splits:
            for candidate in self._candidate_split_paths(dataset_path, split):
                if candidate.exists():
                    split_paths[split] = str(candidate)
                    break
            else:
                missing_splits[split] = ", ".join(str(path) for path in self._candidate_split_paths(dataset_path, split))

        if train_split not in split_paths:
            raise FileNotFoundError(
                f"Could not find confidence SFT train split {train_split!r}. Tried: {missing_splits[train_split]}"
            )
        return split_paths

    def _to_prompt_completion(self, example: dict[str, Any]) -> dict[str, Any]:
        messages = example.get("messages")
        if not isinstance(messages, list) or len(messages) < 2:
            raise ValueError(f"Expected messages with at least user/assistant turns, got: {messages!r}")
        if messages[-1].get("role") != "assistant":
            raise ValueError(f"Expected final message role to be assistant, got: {messages[-1]!r}")
        return {
            "messages": messages,
            "prompt": messages[:-1],
            "completion": [messages[-1]],
        }

    def _validate_confidence_targets(self, dataset):
        if not getattr(self.args, "sft_balance_confidence_targets", False):
            return dataset

        train_split = self.args.dataset_train_split
        train_dataset = dataset[train_split]
        if "confidence_target" not in train_dataset.column_names:
            raise ValueError("SFT confidence balancing requires a confidence_target column.")
        return dataset

    def load_dataset(self):
        split_paths = self._resolve_split_paths()
        dataset = load_dataset("json", data_files=split_paths)
        dataset = dataset.map(self._to_prompt_completion)

        dataset = self._validate_confidence_targets(dataset)
        return dataset


class DeepScaleRSFT(ConfidenceSFTJSONLoader):
    pass
