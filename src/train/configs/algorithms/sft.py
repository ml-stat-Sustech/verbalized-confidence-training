from dataclasses import dataclass


@dataclass
class SFTAlgorithmBase:
    train_batch_size: int = 256
    micro_batch_size_per_gpu: int = 4
    total_training_steps: int = 180
    test_freq: int = -1
    save_freq: int = 180
    learning_rate: float = 2e-6
    lr_scheduler_type: str = "constant_with_warmup"
    min_lr_ratio: float = 0.0
    lr_warmup_steps: int = 10
    lr_warmup_steps_ratio: float = 0.0
    generation_eval: bool = True
    generation_eval_freq: int = -1
    dataset_train_split: str = "train_static_balanced"
    sft_confidence_targets_prebalanced: bool = True
    sft_balance_confidence_targets: bool = False
    sft_balance_confidence_target_count: int | None = None
    sft_confidence_target_source: str = "mixed"


@dataclass
class CalibSFT(SFTAlgorithmBase):
    trainer_name: str = "sft_ce"
    sft_loss_mask: str = "completion_selective"
    sft_confidence_loss_weight: float = 0.5
