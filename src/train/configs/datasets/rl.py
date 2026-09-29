from dataclasses import dataclass

from src.train.configs.base import TrainConfig


@dataclass
class DeepScaleRRL(TrainConfig):
    task_name: str = "deep_scale_r"
    dataset_cls: str = "DeepScaleR"
    dataset_name: str = "agentica-org/DeepScaleR-Preview-Dataset"
    max_prompt_length: int = 2048
    max_response_length: int = 8192
    train_batch_size: int = 64
    ppo_mini_batch_size: int = 32
    ppo_micro_batch_size_per_gpu: int = 4
    rollout_n: int = 8
    rollout_gpu_memory_utilization: float = 0.4
    total_training_steps: int = 160
    save_freq: int = 40
    test_freq: int = 40
