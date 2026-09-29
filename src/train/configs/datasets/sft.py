from dataclasses import dataclass

from src.train.configs.base import TrainConfig


@dataclass
class ConfidenceSFTBase(TrainConfig):
    dataset_name: str = "data/sft_data"
    dataset_config: str | None = None
    dataset_train_split: str = "train"
    dataset_test_split: str = "validation"
    max_prompt_length: int = 1024
    max_response_length: int = 1536
    max_sequence_length: int = 16384
    generation_eval_max_new_tokens: int = 16384


@dataclass
class DeepScaleRSFT(ConfidenceSFTBase):
    task_name: str = "deep_scale_r"
    dataset_cls: str = "DeepScaleRSFT"
    sft_rollout_dataset: str = "DeepScaleR"
    generation_eval_dataset: str = "DeepScaleR_Eval"
    generation_eval_sample_size: int = 200
    max_prompt_length: int = 2048
    max_response_length: int = 8192
    generation_eval_max_new_tokens: int = 8192
    sft_k: int = 50
    sft_target_lambda: float = 0.5
