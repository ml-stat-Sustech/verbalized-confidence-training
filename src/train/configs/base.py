from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass
class TrainConfig:
    """Project presets expressed in verl training terms."""

    # Run paths
    logs_root: str = "logs/train"
    run_name: Optional[str] = None
    output_dir: Optional[str] = None

    # Model
    model_name_or_path: str = "Qwen/Qwen3-8B"
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "flash_attention_2"
    trust_remote_code: bool = True

    # Dataset
    dataset_name: str = ""
    dataset_cls: Optional[str] = None
    dataset_config: Optional[str] = None
    dataset_train_split: str = "train"
    dataset_test_split: str = "test"
    train_subset_size: Optional[int] = None
    train_include_ids_file: str = ""
    train_exclude_ids_file: str = ""
    eval_subset_size: Optional[int] = None
    task_name: Optional[str] = None
    stage: Optional[str] = None

    # SFT data lineage and generation
    source_stage: Optional[str] = None
    source_method: Optional[str] = None
    source_checkpoint: Optional[str] = None
    base_model: Optional[str] = None
    sft_data_dir: Optional[str] = None
    sft_k: Optional[int] = None
    sft_target_lambda: Optional[float] = None
    sft_balance_confidence_targets: bool = False
    sft_balance_confidence_target_count: Optional[int] = None
    sft_confidence_targets_prebalanced: bool = False
    sft_rollout_dataset: Optional[str] = None
    sft_rollout_model: Optional[str] = None
    sft_source_checkpoints: Optional[dict[str, str]] = None
    sft_output_root: Optional[str] = None
    sft_rollout_stage: str = "all"
    sft_rollout_splits: str = "train,validation"
    sft_rollout_validation_size: int = 200
    sft_rollout_validation_source_split: str = "test"
    sft_rollout_confidence_format: str = "same"
    sft_rollout_temperature: float = 1.0
    sft_rollout_top_p: float = 0.95
    sft_rollout_prompt_batch_size: int = 256
    sft_rollout_fill_batch_size: int = 256
    sft_rollout_tensor_parallel_size: int = 1
    sft_rollout_gpu_memory_utilization: float = 0.8
    sft_rollout_seed: int = 43

    log_token_entropy: bool = True

    # verl batch sizes
    train_batch_size: int = 64
    micro_batch_size_per_gpu: int = 4
    ppo_mini_batch_size: int = 64
    ppo_micro_batch_size_per_gpu: int = 8
    rollout_n: int = 16
    ppo_epochs: int = 1

    # PPO/GRPO optimization
    kl_loss_coef: float = 0.0
    clip_ratio_low: float = 0.2
    clip_ratio_high: Optional[float] = None
    norm_adv_by_std_in_grpo: bool = False
    learning_rate: float = 2e-6
    lr_scheduler_type: str = "cosine"
    min_lr_ratio: float = 0.5
    lr_warmup_steps: int = 10
    lr_warmup_steps_ratio: float = 0.0
    weight_decay: float = 0.0

    # verl trainer
    enable_gradient_checkpointing: bool = True
    max_prompt_length: int = 3072
    max_response_length: int = 1536
    total_training_steps: int = -1
    test_freq: int = 20
    val_before_train: bool = False
    save_freq: int = 20
    # Optional explicit save/evaluation steps. Empty keeps save_freq behavior.
    checkpoint_steps: list[int] = field(default_factory=list)
    max_actor_ckpt_to_keep: int = 1
    logger: list[str] = field(default_factory=lambda: ["console"])
    project_name: str = "VerbalizedConfidence"

    # Validation metrics
    log_batch_confidences: bool = True
    generation_eval: bool = False
    generation_eval_dataset: Optional[str] = None
    generation_eval_datasets: list[str] = field(default_factory=list)
    generation_eval_sample_size: Optional[int] = None
    generation_eval_freq: int = -1
    generation_eval_ece_bins: int = 10
    generation_eval_tensor_parallel_size: int = 1
    generation_eval_gpu_memory_utilization: float = 0.8
    generation_eval_temperature: float = 0.0

    # SFT objective
    sft_loss_mask: str = "completion_selective"

    # RL objective and DAPO sampling
    optimization_rewards: dict[str, float] = field(default_factory=dict)
    confidence_reward: Optional[str] = None
    confidence_reward_weight: float = 1.0
    monitoring_rewards: list[str] = field(default_factory=list)
    group_correct_weight: float = 0.5

    # Generation
    seed: int = 43
    confidence_format: str | None = "none"
    task_spec: str = "generation"
    temperature: float = 1.0
    rollout_gpu_memory_utilization: float = 0.7

    def __post_init__(self):
        if self.output_dir is None and self.run_name is not None:
            self.output_dir = f"{self.logs_root}/{self.run_name}"

    def to_config_dict(self) -> dict:
        return asdict(self)
