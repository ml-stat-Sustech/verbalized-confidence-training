from dataclasses import dataclass, field


@dataclass
class RLAlgorithmBase:
    clip_ratio_low: float = 0.2
    clip_ratio_high: float = 0.28
    norm_adv_by_std_in_grpo: bool = True
    enable_filter_groups: bool = True
    filter_groups_metric: str = "acc"
    max_num_gen_batches: int = 20
    monitoring_rewards: list[str] = field(default_factory=list)


@dataclass
class RLVR(RLAlgorithmBase):
    trainer_name: str = "rlvr"
    kl_loss_coef: float = 1e-3
    optimization_rewards: dict[str, float] = field(default_factory=lambda: {"format": 1.0, "accuracy": 1.0})


@dataclass
class ConfRLAlgorithmBase(RLAlgorithmBase):
    optimization_rewards: dict[str, float] = field(
        default_factory=lambda: {"format": 1.0, "accuracy": 1.0, "brier": 1.0}
    )
    monitoring_rewards: list[str] = field(default_factory=lambda: ["mean_confidence", "confidence_one_or_zero"])


@dataclass
class RLCR(ConfRLAlgorithmBase):
    trainer_name: str = "rlcr"


@dataclass
class ReDoubt(ConfRLAlgorithmBase):
    trainer_name: str = "redoubt"
    clip_ratio_high: float = 0.28
    norm_adv_by_std_in_grpo: bool = False
    enable_filter_groups: bool = True
    filter_groups_metric: str = "accuracy"
    optimization_rewards: dict[str, float] = field(default_factory=lambda: {"log_score": 1.0})
    confidence_reward: str = "log_score"
    log_score_clip: float = 0.001
    # Continues from an RLVR checkpoint.
    total_training_steps: int = 80
    save_freq: int = 80
    test_freq: int = 20


@dataclass
class CoCA(ConfRLAlgorithmBase):
    trainer_name: str = "coca"
    optimization_rewards: dict[str, float] = field(
        default_factory=lambda: {"format": 1.0, "accuracy": 1.0}
    )
    confidence_reward: str = "brier"
    confidence_reward_weight: float = 1.0


@dataclass
class DCPO(ConfRLAlgorithmBase):
    trainer_name: str = "dcpo"
    kl_loss_coef: float = 0.0
    group_correct_weight: float = 0.5
    optimization_rewards: dict[str, float] = field(
        default_factory=lambda: {"format": 1.0, "accuracy": 1.0}
    )
    confidence_reward: str = "brier"
    confidence_reward_weight: float = 1.0
