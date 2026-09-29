import csv
import json
import os
import re


class BaseEvaluator:
    CONFIDENCE_SUBGROUP_COLUMNS = [
        f"confidence_{group}_{metric}"
        for group in ("correct", "incorrect")
        for metric in (
            "mean",
            "std",
            "range",
            "unique_count",
            "top1_share",
            "top3_share",
            "extreme_share",
            "one_or_zero_ratio",
            "entropy",
            "effective_values",
        )
    ] + ["confidence_correct_incorrect_mean_gap"]

    EVAL_CSV_COLUMNS = [
        "dataset_config_name",
        "model_config_name",
        "model_name_or_path",
        "train_task",
        "train_stage",
        "train_method",
        "source_stage",
        "source_method",
        "stage",
        "policy_name",
        "inferencer_name",
        "confidence_format",
        "num_generations",
        "pass@k",
        "format_legal_ratio",
        "accuracy",
        "answer_f1",
        "auroc",
        "brier_score",
        "brier_skill_score",
        "ece",
        "abs_confidence_accuracy_gap",
        "macro_ce",
        "corp_mcb",
        "corp_dsc",
        "eaurc",
        "confidence_avg",
        "confidence_std",
        "confidence_range",
        "confidence_unique_count",
        "confidence_top1_share",
        "confidence_top3_share",
        "confidence_extreme_share",
        "confidence_one_or_zero_ratio",
        "confidence_entropy",
        "confidence_effective_values",
        *CONFIDENCE_SUBGROUP_COLUMNS,
        "confidence_legal_ratio",
        "generation_length",
        "log_path",
        "sft_k",
        "sft_target_lambda",
        "kl_loss_coef",
    ]

    def __init__(self, config):
        self.config = config
        self.last_predictions_path = None
        self.last_eval_csv_path = None

    @staticmethod
    def _csv_value(value):
        return "" if value is None else value

    @staticmethod
    def _resolve_run_dir(checkpoint_name):
        if not checkpoint_name:
            return None
        normalized = str(checkpoint_name).rstrip("/")
        if os.path.basename(normalized).startswith("checkpoint-"):
            return os.path.dirname(normalized)
        return normalized

    @classmethod
    def _read_run_metadata(cls, checkpoint_name):
        run_dir = cls._resolve_run_dir(checkpoint_name)
        if run_dir is None:
            return {}

        metadata = {}
        for filename in ("lineage.json", "run_config.json"):
            path = os.path.join(run_dir, filename)
            if not os.path.isfile(path):
                continue
            try:
                with open(path, encoding="utf-8") as handle:
                    payload = json.load(handle)
            except Exception:
                continue
            if isinstance(payload, dict):
                metadata.update(payload)
        return metadata

    @staticmethod
    def _looks_like_acc_rl(stage, method):
        stage = str(stage or "").lower()
        method = str(method or "").lower()
        return stage in {"task_rl", "acc_rl", "accuracy_rl"} or "rlvr" in method or "accuracy" in method

    @classmethod
    def _stage_label_from_values(cls, policy_name, train_stage, source_stage, train_method="", source_method="", source_checkpoint=""):
        if not train_stage:
            return "zero-shot" if policy_name == "Baseline" else ""

        train_stage_normalized = str(train_stage).lower()
        if train_stage_normalized in {"task_rl", "acc_rl", "accuracy_rl"}:
            return "acc RL"
        if train_stage_normalized in {"direct_rl", "confidence_rl"}:
            return "acc RL + conf RL" if cls._looks_like_acc_rl(source_stage, source_method) else "direct RL"
        if train_stage_normalized == "confidence_sft":
            return "acc RL + conf SFT" if cls._looks_like_acc_rl(source_stage, source_method) else "conf SFT"
        if train_stage_normalized == "post_sft_confidence_rl":
            source_metadata = cls._read_run_metadata(source_checkpoint)
            source_sft_source_stage = source_metadata.get("source_stage")
            source_sft_source_method = source_metadata.get("source_method")
            if cls._looks_like_acc_rl(source_sft_source_stage, source_sft_source_method):
                return "acc RL + conf SFT + conf RL"
            return "conf SFT + conf RL"
        return train_stage

    def _stage_label(self):
        return self._stage_label_from_values(
            self.config.policy_name,
            getattr(self.config, "train_stage", None),
            getattr(self.config, "source_stage", None),
            train_method=getattr(self.config, "train_method", None),
            source_method=getattr(self.config, "source_method", None),
            source_checkpoint=getattr(self.config, "source_checkpoint", None),
        )

    def run(self, dataset_eval):
        dataset_eval = self.verify_results(dataset_eval)
        metrics = self.summarize_results(dataset_eval)
        self.save_results(metrics, dataset_eval=dataset_eval)
        return dataset_eval, metrics

    def merge_output_columns(self, dataset, output_columns):
        for key, value in output_columns.items():
            if key in dataset.column_names:
                dataset = dataset.remove_columns([key])
            dataset = dataset.add_column(key, value)
        return dataset

    def verify_results(self, dataset_eval):
        return dataset_eval

    def summarize_results(self, dataset_eval):
        return {}

    @staticmethod
    def format_metric_value(key, value):
        percent_metric_keys = {
            "brier_score",
            "brier_skill_score",
            "ece",
            "macro_ce",
            "corp_mcb",
            "corp_dsc",
            "eaurc",
            "auroc",
            "accuracy",
            "answer_f1",
            "abs_confidence_accuracy_gap",
            "confidence_avg",
            "confidence_std",
            "confidence_range",
            "confidence_top1_share",
            "confidence_top3_share",
            "confidence_extreme_share",
            "confidence_one_or_zero_ratio",
            "confidence_legal_ratio",
            "format_legal_ratio",
        }
        if key.startswith("confidence_") and (
            key.endswith("_mean")
            or key.endswith("_std")
            or key.endswith("_range")
            or key.endswith("_share")
            or key.endswith("_ratio")
            or key.endswith("_gap")
        ):
            return f"{value * 100:.2f}"
        if key.startswith("pass@"):
            return f"{value * 100:.2f}"
        if key in percent_metric_keys:
            return f"{value * 100:.2f}"
        if key == "generation_length":
            return str(int(round(value)))
        return str(value)

    def save_results(self, metrics, dataset_eval=None):
        if self.config.log_path is not None:
            os.makedirs(self.config.log_path, exist_ok=True)
            log_file = os.path.join(self.config.log_path, "log.txt")
            with open(log_file, "a") as f:
                f.write("\n")
                f.write(f"[Evaluation] {self.config.name}\n")
                for key, value in metrics.items():
                    f.write(f"{key}: {self.format_metric_value(key, value)}\n")
            self.save_eval_metadata(metrics)

        self.last_predictions_path = self.save_predictions(dataset_eval)
        self.last_eval_csv_path = self.record_eval_csv(metrics)

    @staticmethod
    def _json_default(value):
        if hasattr(value, "item"):
            return value.item()
        if hasattr(value, "tolist"):
            return value.tolist()
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

    @staticmethod
    def _truncate_text_for_storage(text, max_tokens):
        if not isinstance(text, str) or max_tokens is None:
            return text

        token_matches = list(re.finditer(r"\S+", text))
        if len(token_matches) <= max_tokens:
            return text

        truncated_end = token_matches[max_tokens - 1].end()
        return text[:truncated_end].rstrip() + "\n\n[Truncated for storage]"

    def save_predictions(self, dataset_eval):
        if (
            not getattr(self.config, "save_predictions_jsonl", True)
            or self.config.log_path is None
            or dataset_eval is None
        ):
            return None

        os.makedirs(self.config.log_path, exist_ok=True)
        predictions_path = os.path.join(
            self.config.log_path,
            getattr(self.config, "predictions_jsonl_name", "predictions.jsonl"),
        )
        with open(predictions_path, "w", encoding="utf-8") as f:
            for row in dataset_eval:
                row_dict = dict(row)
                row_dict["question"] = self._truncate_text_for_storage(
                    row_dict.get("question"),
                    getattr(self.config, "max_question_save_tokens", 200),
                )
                f.write(json.dumps(row_dict, ensure_ascii=False, default=self._json_default))
                f.write("\n")
        return predictions_path

    def save_eval_metadata(self, metrics):
        if self.config.log_path is None:
            return None
        os.makedirs(self.config.log_path, exist_ok=True)
        path = os.path.join(self.config.log_path, "eval_metadata.json")
        payload = {
            "dataset_config_name": self.config.dataset_config_name,
            "dataset_name": self.config.dataset_name,
            "model_config_name": self.config.model_config_name,
            "model_name_or_path": self.config.model_name_or_path,
            "inferencer_name": self.config.inferencer_name,
            "policy_name": self.config.policy_name,
            "checkpoint_name": self.config.checkpoint_name,
            "train_task": getattr(self.config, "train_task", None),
            "train_stage": getattr(self.config, "train_stage", None),
            "train_method": getattr(self.config, "train_method", None),
            "trainer_name": getattr(self.config, "trainer_name", None),
            "source_stage": getattr(self.config, "source_stage", None),
            "source_method": getattr(self.config, "source_method", None),
            "source_checkpoint": getattr(self.config, "source_checkpoint", None),
            "base_model": getattr(self.config, "base_model", None),
            "train_confidence_format": getattr(self.config, "train_confidence_format", None),
            "sft_data_dir": getattr(self.config, "sft_data_dir", None),
            "sft_k": getattr(self.config, "sft_k", None),
            "sft_target_lambda": getattr(self.config, "sft_target_lambda", None),
            "kl_loss_coef": getattr(self.config, "kl_loss_coef", None),
            "sft_loss_mask": getattr(self.config, "sft_loss_mask", None),
            "checkpoint_step": getattr(self.config, "checkpoint_step", None),
            "train_run_dir": getattr(self.config, "train_run_dir", None),
            "lineage_path": getattr(self.config, "lineage_path", None),
            "run_config_path": getattr(self.config, "run_config_path", None),
            "metrics": metrics,
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, default=self._json_default)
            handle.write("\n")
        return path

    def _append_eval_csv_row(self, csv_path, row):
        file_exists = os.path.exists(csv_path)
        if not file_exists:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.EVAL_CSV_COLUMNS)
                writer.writeheader()
                writer.writerow({key: row.get(key, "") for key in self.EVAL_CSV_COLUMNS})
            return

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_fieldnames = reader.fieldnames or []
            existing_rows = list(reader)

        if existing_fieldnames == self.EVAL_CSV_COLUMNS:
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.EVAL_CSV_COLUMNS)
                writer.writerow({key: row.get(key, "") for key in self.EVAL_CSV_COLUMNS})
            return

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.EVAL_CSV_COLUMNS)
            writer.writeheader()
            for existing_row in existing_rows:
                migrated_row = {key: existing_row.get(key, "") for key in self.EVAL_CSV_COLUMNS}
                if not migrated_row.get("kl_loss_coef"):
                    migrated_row["kl_loss_coef"] = existing_row.get("beta", "")
                if not migrated_row.get("stage"):
                    migrated_row["stage"] = self._stage_label_from_values(
                        existing_row.get("policy_name"),
                        existing_row.get("train_stage"),
                        existing_row.get("source_stage"),
                        train_method=existing_row.get("train_method"),
                        source_method=existing_row.get("source_method"),
                        source_checkpoint=existing_row.get("source_checkpoint"),
                    )
                writer.writerow(migrated_row)
            writer.writerow({key: row.get(key, "") for key in self.EVAL_CSV_COLUMNS})

    def record_eval_csv(self, metrics):
        os.makedirs(self.config.logs_root, exist_ok=True)
        csv_path = os.path.join(self.config.logs_root, "eval.csv")
        pass_at_k_key = f"pass@{getattr(self.config, 'num_generations', 1)}"
        row = {
            "dataset_config_name": self.config.dataset_config_name,
            "model_config_name": self.config.model_config_name,
            "model_name_or_path": self.config.model_name_or_path,
            "train_task": self._csv_value(getattr(self.config, "train_task", None)),
            "train_stage": self._csv_value(getattr(self.config, "train_stage", None)),
            "train_method": self._csv_value(getattr(self.config, "train_method", None)),
            "source_stage": self._csv_value(getattr(self.config, "source_stage", None)),
            "source_method": self._csv_value(getattr(self.config, "source_method", None)),
            "stage": self._stage_label(),
            "policy_name": self.config.policy_name,
            "inferencer_name": self.config.inferencer_name,
            "confidence_format": self._csv_value(getattr(self.config, "confidence_format", None)),
            "num_generations": self._csv_value(getattr(self.config, "num_generations", 1)),
            "pass@k": self.format_metric_value("pass@k", metrics.get("pass@k", metrics.get(pass_at_k_key)))
            if "pass@k" in metrics or pass_at_k_key in metrics
            else "",
            "accuracy": self.format_metric_value("accuracy", metrics.get("accuracy")) if "accuracy" in metrics else "",
            "answer_f1": self.format_metric_value("answer_f1", metrics.get("answer_f1"))
            if "answer_f1" in metrics
            else "",
            "auroc": self.format_metric_value("auroc", metrics.get("auroc")) if "auroc" in metrics else "",
            "brier_score": self.format_metric_value("brier_score", metrics.get("brier_score"))
            if "brier_score" in metrics
            else "",
            "brier_skill_score": self.format_metric_value(
                "brier_skill_score", metrics.get("brier_skill_score")
            )
            if "brier_skill_score" in metrics
            else "",
            "ece": self.format_metric_value("ece", metrics.get("ece")) if "ece" in metrics else "",
            "abs_confidence_accuracy_gap": self.format_metric_value(
                "abs_confidence_accuracy_gap", metrics.get("abs_confidence_accuracy_gap")
            )
            if "abs_confidence_accuracy_gap" in metrics
            else "",
            "macro_ce": self.format_metric_value("macro_ce", metrics.get("macro_ce"))
            if "macro_ce" in metrics
            else "",
            "corp_mcb": self.format_metric_value("corp_mcb", metrics.get("corp_mcb"))
            if "corp_mcb" in metrics
            else "",
            "corp_dsc": self.format_metric_value("corp_dsc", metrics.get("corp_dsc"))
            if "corp_dsc" in metrics
            else "",
            "eaurc": self.format_metric_value("eaurc", metrics.get("eaurc")) if "eaurc" in metrics else "",
            "confidence_avg": self.format_metric_value("confidence_avg", metrics.get("confidence_avg"))
            if "confidence_avg" in metrics
            else "",
            "confidence_std": self.format_metric_value("confidence_std", metrics.get("confidence_std"))
            if "confidence_std" in metrics
            else "",
            "confidence_range": self.format_metric_value("confidence_range", metrics.get("confidence_range"))
            if "confidence_range" in metrics
            else "",
            "confidence_unique_count": self.format_metric_value(
                "confidence_unique_count", metrics.get("confidence_unique_count")
            )
            if "confidence_unique_count" in metrics
            else "",
            "confidence_top1_share": self.format_metric_value(
                "confidence_top1_share", metrics.get("confidence_top1_share")
            )
            if "confidence_top1_share" in metrics
            else "",
            "confidence_top3_share": self.format_metric_value(
                "confidence_top3_share", metrics.get("confidence_top3_share")
            )
            if "confidence_top3_share" in metrics
            else "",
            "confidence_extreme_share": self.format_metric_value(
                "confidence_extreme_share", metrics.get("confidence_extreme_share")
            )
            if "confidence_extreme_share" in metrics
            else "",
            "confidence_one_or_zero_ratio": self.format_metric_value(
                "confidence_one_or_zero_ratio", metrics.get("confidence_one_or_zero_ratio")
            )
            if "confidence_one_or_zero_ratio" in metrics
            else "",
            "confidence_entropy": self.format_metric_value(
                "confidence_entropy", metrics.get("confidence_entropy")
            )
            if "confidence_entropy" in metrics
            else "",
            "confidence_effective_values": self.format_metric_value(
                "confidence_effective_values", metrics.get("confidence_effective_values")
            )
            if "confidence_effective_values" in metrics
            else "",
            "confidence_legal_ratio": self.format_metric_value(
                "confidence_legal_ratio", metrics.get("confidence_legal_ratio")
            )
            if "confidence_legal_ratio" in metrics
            else "",
            "generation_length": self.format_metric_value("generation_length", metrics.get("generation_length"))
            if "generation_length" in metrics
            else "",
            "log_path": self.config.log_path,
            "sft_k": self._csv_value(getattr(self.config, "sft_k", None)),
            "sft_target_lambda": self._csv_value(getattr(self.config, "sft_target_lambda", None)),
            "kl_loss_coef": self._csv_value(getattr(self.config, "kl_loss_coef", None)),
        }
        for key in self.CONFIDENCE_SUBGROUP_COLUMNS:
            if key in metrics:
                row[key] = self.format_metric_value(key, metrics[key])
        self._append_eval_csv_row(csv_path, row)
        return csv_path
