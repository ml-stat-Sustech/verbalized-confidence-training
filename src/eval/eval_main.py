import gc
import json
import os

from src.common.runtime_env import configure_log_noise_environment

configure_log_noise_environment()

from src.common.cli import parse_bool  # noqa: E402
from src.eval.configs.config_utils import build_eval_config_from_model_path  # noqa: E402
from src.eval.datasets.utils import build_dataset  # noqa: E402
from src.eval.evaluators.confidence_evaluator import ConfidenceEvaluator  # noqa: E402
from src.eval.inferencers.utils import build_inferencer  # noqa: E402
from src.eval.logger import print_eval_summary, setup_eval_logger  # noqa: E402
from src.eval.models.base_model import BaseModel  # noqa: E402


def main(config):
    setup_eval_logger(config.log_path)
    dataset = build_dataset(config)
    model = BaseModel(config)
    inferencer = build_inferencer(config, model)
    evaluator = ConfidenceEvaluator(config)
    dataset_eval = inferencer.run(dataset)
    model.close()
    del model
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    _, metrics = evaluator.run(dataset_eval)
    print_eval_summary(config, evaluator, metrics)
    return metrics


def cli_main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="Eval dataset config class name")
    parser.add_argument(
        "--model-name-or-path",
        "--model_name_or_path",
        dest="model_name_or_path",
        default=None,
        help="Model directory or Hugging Face repo id. Model configuration and training metadata are auto-detected.",
    )
    parser.add_argument("--inferencer", type=str, default=None, help="Eval inferencer name")
    parser.add_argument(
        "--confidence_format",
        type=str,
        default=None,
        choices=["none", "probability", "digit", "linguistic"],
        help="Verbalized confidence output format used by the response prompt.",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=None,
        help="Tensor parallel size for vLLM. Choose a size that divides the model's attention head count.",
    )
    parser.add_argument(
        "--confidence_mode",
        type=str,
        default=None,
        choices=["vanilla"],
        help="How to populate evaluation confidences: direct prompt/inferencer output.",
    )
    parser.add_argument(
        "--save_predictions_jsonl",
        type=parse_bool,
        default=None,
        help="Whether to save per-example predictions.jsonl (default: true).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Randomly sample this many evaluation examples using the configured seed.",
    )
    parser.add_argument(
        "--include-multi-choice",
        type=parse_bool,
        default=None,
        help="Include WebQuestions examples with multiple gold answers (default: false).",
    )
    parser.add_argument("--output-dir", type=str, default=None, help="Override the evaluation output directory.")
    parser.add_argument("--output-tag", type=str, default=None, help="Append a tag to the evaluation output directory.")
    parser.add_argument("--metrics-json", type=str, default=None, help="Write summary metrics to this JSON file.")
    parser.add_argument("--max-tokens", type=int, default=None, help="Maximum generated tokens per example.")
    parser.add_argument("--max-model-len", type=int, default=None, help="Total context limit for prompt and output.")
    parser.add_argument("--temperature", type=float, default=None, help="Evaluation sampling temperature.")
    parser.add_argument(
        "--num-generations",
        type=int,
        default=None,
        help="Number of independent rollouts per example.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=None,
        help="vLLM GPU memory utilization for evaluation.",
    )
    parser.add_argument("--ece-bins", type=int, default=None, help="Number of bins used for ECE.")
    args = parser.parse_args()

    if not args.dataset or not args.model_name_or_path:
        raise ValueError("Evaluation requires --dataset <Class> --model-name-or-path <path-or-repo>.")
    if args.num_generations is not None and args.num_generations < 1:
        raise ValueError("--num-generations must be >= 1.")
    config = build_eval_config_from_model_path(
        args.dataset,
        args.model_name_or_path,
        args.inferencer,
        args.confidence_format,
        args.confidence_mode,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    if args.save_predictions_jsonl is not None:
        config.save_predictions_jsonl = args.save_predictions_jsonl
    if args.include_multi_choice is not None:
        config.include_multi_choice = args.include_multi_choice
    if args.sample_size is not None:
        config.sample_size = args.sample_size
        sample_run = f"sample_{args.sample_size}_seed_{config.seed}"
        config.log_path = os.path.join(config.log_path, sample_run)
    if args.num_generations is not None:
        config.num_generations = args.num_generations
        generation_run = f"n{args.num_generations}"
        config.log_path = os.path.join(config.log_path, generation_run)
    if args.output_tag is not None:
        config.log_path = os.path.join(config.log_path, args.output_tag)
    if args.output_dir is not None:
        config.log_path = args.output_dir
    if args.max_tokens is not None:
        config.max_tokens = args.max_tokens
    if args.max_model_len is not None:
        config.max_model_len = args.max_model_len
    if args.temperature is not None:
        config.temperature = args.temperature
    if args.gpu_memory_utilization is not None:
        config.gpu_memory_utilization = args.gpu_memory_utilization
    if args.ece_bins is not None:
        config.ece_bins = args.ece_bins
    metrics = main(config)
    if args.metrics_json is not None:
        metrics_dir = os.path.dirname(os.path.abspath(args.metrics_json))
        os.makedirs(metrics_dir, exist_ok=True)
        with open(args.metrics_json, "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, ensure_ascii=False, indent=2, default=lambda value: value.item())
            handle.write("\n")


if __name__ == "__main__":
    cli_main()
