import logging
import os
import sys

from src.common.runtime_env import suppress_common_noisy_loggers


def setup_eval_logger(log_path: str | None):
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_path is not None:
        os.makedirs(log_path, exist_ok=True)
        handlers.append(logging.FileHandler(os.path.join(log_path, "log.txt"), mode="w"))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )
    suppress_common_noisy_loggers()


def print_eval_summary(config, evaluator, metrics):
    lines = [
        "[Eval] finished",
        f"[Eval] policy={config.name}",
    ]
    if config.log_path is not None:
        lines.append(f"[Eval] log_dir={os.path.abspath(config.log_path)}")
        lines.append(f"[Eval] log_file={os.path.abspath(os.path.join(config.log_path, 'log.txt'))}")
    if getattr(evaluator, "last_predictions_path", None):
        lines.append(f"[Eval] predictions={os.path.abspath(evaluator.last_predictions_path)}")
    if getattr(evaluator, "last_eval_csv_path", None):
        lines.append(f"[Eval] eval_csv={os.path.abspath(evaluator.last_eval_csv_path)}")

    metric_order = [
        "pass@k",
        "format_legal_ratio",
        "accuracy",
        "brier_score",
        "brier_skill_score",
        "ece",
        "macro_ce",
        "corp_mcb",
        "corp_dsc",
        "eaurc",
        "auroc",
        "confidence_avg",
        "confidence_legal_ratio",
        "generation_length",
    ]
    for key in metric_order:
        if key in metrics:
            lines.append(f"[Eval] {key}={evaluator.format_metric_value(key, metrics[key])}")
    for key, value in metrics.items():
        if key not in metric_order:
            lines.append(f"[Eval] {key}={evaluator.format_metric_value(key, value)}")
    print("\n".join(lines), flush=True)
