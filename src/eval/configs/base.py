from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class EvalBaseConfig:
    # dataset
    dataset_config_name: str | None = None
    dataset_cls: str | None = None
    dataset_name: str | None = None
    dataset_config: Optional[str] = None
    split: str = "test"
    include_multi_choice: bool = False
    sample_size: int | None = None

    # logging
    logs_root: str = "logs/eval"
    log_path: str | None = None
    save_predictions_jsonl: bool = True
    predictions_jsonl_name: str = "predictions.jsonl"
    max_question_save_tokens: int | None = 200
    name: str = "Baseline"
    model_config_name: str | None = None
    policy_name: str | None = None
    checkpoint_name: str | None = None
    train_task: str | None = None
    train_stage: str | None = None
    train_method: str | None = None
    trainer_name: str | None = None
    source_stage: str | None = None
    source_method: str | None = None
    source_checkpoint: str | None = None
    base_model: str | None = None
    train_confidence_format: str | None = None
    train_format_pattern: str | None = None
    sft_data_dir: str | None = None
    sft_k: int | None = None
    sft_target_lambda: float | None = None
    kl_loss_coef: float | None = None
    sft_loss_mask: str | None = None
    checkpoint_step: int | None = None
    train_run_dir: str | None = None
    lineage_path: str | None = None
    run_config_path: str | None = None

    # model
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int | None = None
    model_name_or_path: str | None = None
    trust_remote_code: bool = True
    hf_overrides: dict | None = None
    num_generations: int = 1
    temperature: float = 0
    top_p: float | None = None
    top_k: int | None = None
    repetition_penalty: float | None = None
    max_tokens: int = 16384
    max_model_len: int | None = None
    seed: Optional[int] = 42

    # evaluation
    answer_verifier_name: str | None = None
    answer_verifier_args: Dict = field(default_factory=dict)
    pass_k_vals: List = field(default_factory=list)
    ece_bins: int = 10
    save_reliability_diagram: bool = True

    # inference
    confidence_format: str | None = "probability"
    inferencer_name: str = "verbalized_confidence"
    confidence_mode: str = "vanilla"

    # verification
    judge_model_name_or_path: str = "Qwen/Qwen3-8B"
    judge_gpu_memory_utilization: float = 0.8
    judge_max_model_len: int = 8192
    judge_max_tokens: int = 20
