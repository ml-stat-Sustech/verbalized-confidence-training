import os
import logging


def configure_log_noise_environment():
    os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    os.environ["HF_HUB_VERBOSITY"] = "warning"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "1"


def suppress_common_noisy_loggers():
    for logger_name in (
        "httpx",
        "httpcore",
        "huggingface_hub",
        "huggingface_hub.utils._http",
        "urllib3",
    ):
        logging.getLogger(logger_name).setLevel(logging.WARNING)
