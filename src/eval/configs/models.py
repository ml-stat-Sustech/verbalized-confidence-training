from dataclasses import dataclass

from src.common.model_configs import MODEL_PATHS


@dataclass
class BaseModelConfig:
    model_name_or_path: str
    hf_overrides: dict | None = None
    num_generations: int = 1
    temperature: float = 0
    top_p: float | None = None
    top_k: int | None = None
    repetition_penalty: float | None = None
    max_tokens: int = 16384
    max_model_len: int | None = None
    seed: int = 42


@dataclass
class Qwen3_8B(BaseModelConfig):
    model_name_or_path: str = MODEL_PATHS["Qwen3_8B"]
    top_p: float | None = 0.95
    top_k: int | None = 20


@dataclass
class Gemma4_E2B_Instruct(BaseModelConfig):
    model_name_or_path: str = MODEL_PATHS["Gemma4_E2B_Instruct"]
    top_p: float | None = 0.95
    top_k: int | None = 64
