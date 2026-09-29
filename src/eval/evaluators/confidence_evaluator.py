import os

import numpy as np

from src.common.calibration_metrics import compute_binary_auroc, compute_calibration_metrics
from src.common.confidence_formats import CONFIDENCE_NONE, normalize_confidence_format
from src.eval.evaluators.base_evaluator import BaseEvaluator
from src.eval.evaluators.metrics import compute_pass_n, plot_reliability_diagram
from src.eval.verifiers import (
    alias_verifier,
    drop_verifier,
    livebench_reasoning_verifier,
    llm_verifier,
    popqa_verifier,
    rule_verifier,
    web_questions_verifier,
)


class ConfidenceEvaluator(BaseEvaluator):
    def run_answer_verifier(self, dataset_eval):
        if self.config.answer_verifier_name is None:
            return None
        if self.config.answer_verifier_name == "rule_verifier":
            return rule_verifier(dataset_eval, self.config, **self.config.answer_verifier_args)
        if self.config.answer_verifier_name == "alias_verifier":
            return alias_verifier(dataset_eval, self.config, **self.config.answer_verifier_args)
        if self.config.answer_verifier_name == "popqa_verifier":
            return popqa_verifier(dataset_eval, self.config, **self.config.answer_verifier_args)
        if self.config.answer_verifier_name == "web_questions_verifier":
            return web_questions_verifier(dataset_eval, self.config, **self.config.answer_verifier_args)
        if self.config.answer_verifier_name == "llm_verifier":
            return llm_verifier(dataset_eval, self.config, **self.config.answer_verifier_args)
        if self.config.answer_verifier_name == "drop_verifier":
            return drop_verifier(dataset_eval, self.config, **self.config.answer_verifier_args)
        if self.config.answer_verifier_name == "livebench_reasoning_verifier":
            return livebench_reasoning_verifier(dataset_eval, self.config, **self.config.answer_verifier_args)
        raise ValueError(f"Unknown answer_verifier_name: {self.config.answer_verifier_name}")

    def verify_results(self, dataset_eval):
        label_dict = self.run_answer_verifier(dataset_eval)
        if label_dict is not None:
            dataset_eval = self.merge_output_columns(dataset_eval, label_dict)
        if "generation_len" not in dataset_eval.column_names:
            generation_len = [[len(generation_text) for generation_text in generations] for generations in dataset_eval["generations"]]
            dataset_eval = self.merge_output_columns(dataset_eval, {"generation_len": generation_len})
        return dataset_eval

    def summarize_results(self, dataset_eval):
        if "is_correct" not in dataset_eval.column_names:
            return {}

        is_correct = dataset_eval["is_correct"]
        generation_len = dataset_eval["generation_len"]
        confidences = dataset_eval["confidences"]
        is_conf_legal = dataset_eval["is_conf_legal"]

        metrics = {}
        n = self.config.num_generations
        if n > 1:
            if n not in self.config.pass_k_vals:
                self.config.pass_k_vals.append(n)
            for k in self.config.pass_k_vals:
                if 1 < k <= n:
                    metrics[f"pass@{k}"] = compute_pass_n(is_correct, k)

        correctness_array = np.array(is_correct).flatten()
        metrics["accuracy"] = np.mean(correctness_array)
        if "answer_f1" in dataset_eval.column_names:
            metrics["answer_f1"] = np.mean(np.array(dataset_eval["answer_f1"]))
        metrics["generation_length"] = np.mean(np.array(generation_len))
        if "is_format_legal" in dataset_eval.column_names:
            metrics["format_legal_ratio"] = np.mean(np.array(dataset_eval["is_format_legal"]))

        confidence_format = normalize_confidence_format(getattr(self.config, "confidence_format", None))
        if confidence_format != CONFIDENCE_NONE:
            confidence_array = np.array(confidences).flatten()
            calibration = compute_calibration_metrics(
                correctness_array,
                confidence_array,
                n_bins=self.config.ece_bins,
            )
            metrics["brier_score"] = calibration["brier_score"]
            metrics["ece"] = calibration["ece"]
            metrics["abs_confidence_accuracy_gap"] = calibration["abs_confidence_accuracy_gap"]
            for key in (
                "brier_skill_score",
                "macro_ce",
                "corp_mcb",
                "corp_dsc",
                "eaurc",
                "confidence_std",
                "confidence_range",
                "confidence_unique_count",
                "confidence_top1_share",
                "confidence_top3_share",
                "confidence_extreme_share",
                "confidence_one_or_zero_ratio",
                "confidence_entropy",
                "confidence_effective_values",
            ):
                if calibration[key] is not None:
                    metrics[key] = calibration[key]
            for key, value in calibration.items():
                if key.startswith("confidence_correct_") or key.startswith("confidence_incorrect_"):
                    if value is not None:
                        metrics[key] = value
            if calibration.get("confidence_correct_incorrect_mean_gap") is not None:
                metrics["confidence_correct_incorrect_mean_gap"] = calibration[
                    "confidence_correct_incorrect_mean_gap"
                ]
            auroc = compute_binary_auroc(correctness_array, confidence_array)
            if auroc is not None:
                metrics["auroc"] = auroc
            metrics["confidence_avg"] = calibration["mean_confidence"]
            metrics["confidence_legal_ratio"] = np.mean(np.array(is_conf_legal))
        return metrics

    def save_results(self, metrics, dataset_eval=None):
        super().save_results(metrics, dataset_eval=dataset_eval)
        confidence_format = normalize_confidence_format(getattr(self.config, "confidence_format", None))
        if (
            confidence_format == CONFIDENCE_NONE
            or
            not self.config.save_reliability_diagram
            or self.config.log_path is None
            or dataset_eval is None
            or "is_correct" not in dataset_eval.column_names
        ):
            return

        correctness_array = np.array(dataset_eval["is_correct"]).flatten()
        confidence_array = np.array(dataset_eval["confidences"]).flatten()
        save_path = os.path.join(self.config.log_path, "reliability_diagram.png")
        plot_reliability_diagram(
            correctness_array,
            confidence_array,
            n_bins=self.config.ece_bins,
            title=self.config.name,
            save_path=save_path,
        ).close()
