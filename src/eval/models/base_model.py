import gc
import importlib
from contextlib import contextmanager
from pathlib import Path

from huggingface_hub import constants as hf_hub_constants
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from src.common.runtime_env import suppress_common_noisy_loggers


class BaseModel:
    def __init__(self, config):
        self.config = config
        suppress_common_noisy_loggers()
        self.model_path_for_generation = self._resolve_generation_model_path(config.model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path_for_generation, trust_remote_code=getattr(config, "trust_remote_code", True),
        )
        self.llm = LLM(
            model=self.model_path_for_generation,
            max_model_len=getattr(config, "max_model_len", None),
            gpu_memory_utilization=config.gpu_memory_utilization,
            tensor_parallel_size=config.tensor_parallel_size,
            trust_remote_code=getattr(config, "trust_remote_code", True),
            hf_overrides=getattr(config, "hf_overrides", None),
        )

    def _resolve_generation_model_path(self, model_name_or_path: str) -> str:
        local_path = Path(model_name_or_path).expanduser()
        if local_path.exists():
            return str(local_path)
        if hf_hub_constants.HF_HUB_OFFLINE:
            config_path = hf_hub_download(
                repo_id=model_name_or_path,
                filename="config.json",
                local_files_only=True,
            )
            return str(Path(config_path).parent)
        return model_name_or_path

    def build_generation_inputs(self, prompts):
        texts = self.tokenizer.apply_chat_template(prompts, add_generation_prompt=True, tokenize=False)
        if isinstance(texts, str):
            texts = [texts]
        prompt_ids = self.tokenizer(texts, add_special_tokens=False).input_ids
        return texts, prompt_ids

    @contextmanager
    def override_vllm_progress_desc(self, progress_desc):
        if progress_desc is None:
            yield
            return

        llm_module = importlib.import_module("vllm.entrypoints.llm")
        original_tqdm = llm_module.tqdm

        def wrapped_tqdm(*args, **kwargs):
            kwargs["desc"] = progress_desc
            return original_tqdm(*args, **kwargs)

        llm_module.tqdm = wrapped_tqdm
        try:
            yield
        finally:
            llm_module.tqdm = original_tqdm

    def generate(
        self,
        texts,
        n=None,
        temperature=None,
        top_p=None,
        max_tokens=None,
        seed=None,
        logprobs=None,
        progress_desc=None,
    ):
        sampling_kwargs = {
            "n": self.config.num_generations if n is None else n,
            "temperature": self.config.temperature if temperature is None else temperature,
            "max_tokens": self.config.max_tokens if max_tokens is None else max_tokens,
            "seed": self.config.seed if seed is None else seed,
            "logprobs": logprobs,
        }
        effective_top_p = getattr(self.config, "top_p", None) if top_p is None else top_p
        if effective_top_p is not None:
            sampling_kwargs["top_p"] = effective_top_p
        effective_top_k = getattr(self.config, "top_k", None)
        if effective_top_k is not None:
            sampling_kwargs["top_k"] = effective_top_k
        effective_repetition_penalty = getattr(self.config, "repetition_penalty", None)
        if effective_repetition_penalty is not None:
            sampling_kwargs["repetition_penalty"] = effective_repetition_penalty
        sampling_params = SamplingParams(**sampling_kwargs)
        with self.override_vllm_progress_desc(progress_desc):
            return self.llm.generate(texts, sampling_params=sampling_params)

    def close(self):
        try:
            del self.llm
        except Exception:
            pass
        gc.collect()
