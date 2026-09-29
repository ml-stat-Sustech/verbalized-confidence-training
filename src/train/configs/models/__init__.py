from dataclasses import dataclass

from src.common.model_configs import MODEL_PATHS


@dataclass
class Qwen3_8B:
    model_name_or_path: str = MODEL_PATHS["Qwen3_8B"]


@dataclass
class Gemma4_E2B_Instruct:
    model_name_or_path: str = MODEL_PATHS["Gemma4_E2B_Instruct"]
    # Gemma 4 head dimensions exceed FlashAttention's 256 limit.
    attn_implementation: str = "sdpa"
