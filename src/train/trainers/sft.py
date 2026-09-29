import json
import logging
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DistributedSampler

from src.train.trainers import finish_swanlab


logger = logging.getLogger(__name__)


class UniformTargetDistributedSampler(DistributedSampler):
    def __init__(
        self,
        dataset,
        targets,
        *,
        num_replicas: int,
        rank: int,
        seed: int,
        samples_per_target: int | None = None,
        drop_last: bool = True,
    ):
        super().__init__(
            dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=False,
            seed=seed,
            drop_last=drop_last,
        )
        buckets = {}
        for index, target in enumerate(targets):
            buckets.setdefault(f"{float(target):.6f}", []).append(index)
        if not buckets:
            raise ValueError("Uniform confidence target sampling requires a non-empty dataset")
        if samples_per_target is not None and samples_per_target <= 0:
            raise ValueError("sft_balance_confidence_target_count must be positive")

        self.buckets = [buckets[key] for key in sorted(buckets)]
        self.samples_per_target = samples_per_target or min(map(len, self.buckets))
        epoch_size = len(self.buckets) * self.samples_per_target
        self.num_samples = epoch_size // self.num_replicas if drop_last else math.ceil(epoch_size / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas
        if self.num_samples == 0:
            raise ValueError("Balanced SFT epoch is smaller than the data-parallel world size")

    @staticmethod
    def _sample_bucket(indices, count: int, generator: torch.Generator) -> list[int]:
        sampled = []
        while len(sampled) < count:
            order = torch.randperm(len(indices), generator=generator).tolist()
            remaining = count - len(sampled)
            sampled.extend(indices[position] for position in order[:remaining])
        return sampled

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        indices = []
        for bucket in self.buckets:
            indices.extend(self._sample_bucket(bucket, self.samples_per_target, generator))
        order = torch.randperm(len(indices), generator=generator).tolist()
        indices = [indices[position] for position in order]

        if self.drop_last:
            indices = indices[: self.total_size]
        else:
            padding = self.total_size - len(indices)
            indices.extend((indices * math.ceil(padding / len(indices)))[:padding])

        rank_indices = indices[self.rank : self.total_size : self.num_replicas]
        return iter(rank_indices)


def total_epochs_for_steps(
    total_steps: int | None, steps_per_epoch: int, configured_epochs: int
) -> int:
    if total_steps is None:
        return configured_epochs
    return max(configured_epochs, math.ceil(total_steps / steps_per_epoch))


def _find_subsequence(values: list[int], pattern: list[int], start: int = 0) -> int | None:
    for index in range(max(0, start), len(values) - len(pattern) + 1):
        if values[index : index + len(pattern)] == pattern:
            return index
    return None


def confidence_marker_token_ids(tokenizer) -> list[list[int]]:
    candidates = []
    seen = set()
    for marker in (" <confidence>", "<confidence>", "\t<confidence>", "\n<confidence>", "\n\n<confidence>"):
        token_ids = tokenizer.encode(marker, add_special_tokens=False)
        key = tuple(token_ids)
        if token_ids and key not in seen:
            candidates.append(token_ids)
            seen.add(key)
    return sorted(candidates, key=len, reverse=True)


def confidence_suffix_start(input_ids: torch.Tensor, loss_mask: torch.Tensor, markers: list[list[int]]) -> int | None:
    active = torch.nonzero(loss_mask, as_tuple=False).flatten()
    if active.numel() == 0:
        return None
    active_start = int(active[0].item())
    values = input_ids.tolist()
    matches = []
    for marker in markers:
        search_start = max(0, active_start - len(marker) + 1)
        while (marker_start := _find_subsequence(values, marker, search_start)) is not None:
            marker_end = marker_start + len(marker)
            if marker_end > active_start:
                matches.append((marker_end, marker_start))
            search_start = marker_start + 1
    if not matches:
        return None
    last_marker_end = max(end for end, _ in matches)
    return max(min(start for end, start in matches if end == last_marker_end), active_start)


def mask_confidence_suffix(input_ids: torch.Tensor, loss_mask: torch.Tensor, markers: list[list[int]]) -> torch.Tensor:
    start = confidence_suffix_start(input_ids, loss_mask, markers)
    masked = loss_mask.clone()
    if start is None:
        masked.zero_()
    else:
        masked[:start] = 0
    return masked


def completion_without_confidence_mask(
    loss_mask: torch.Tensor,
    confidence_loss_mask: torch.Tensor,
) -> torch.Tensor:
    if loss_mask.is_nested:
        values = loss_mask.values() * (~confidence_loss_mask.values().to(dtype=torch.bool)).to(
            dtype=loss_mask.dtype
        )
        return torch.nested.nested_tensor_from_jagged(values, offsets=loss_mask.offsets())
    return loss_mask * (~confidence_loss_mask.to(dtype=torch.bool)).to(dtype=loss_mask.dtype)


def _flatten_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.values() if tensor.is_nested else tensor.reshape(-1)


def full_sequence_token_budget(input_ids: torch.Tensor, configured_budget: int | None) -> int:
    if input_ids.is_nested:
        sequence_length = int(input_ids.offsets().diff().max().item())
    else:
        sequence_length = int(input_ids.shape[-1])
    return max(0 if configured_budget is None else int(configured_budget), sequence_length)


def _preserve_inline_reasoning(message: dict, chat_template: str | None) -> dict:
    """Preserve Qwen3 reasoning across verl's per-message SFT rendering.

    MultiTurnSFTDataset renders each turn in isolation and warns when those tokens differ
    from a full-conversation render. Qwen3's isolated assistant path extracts inline
    ``<think>`` text but does not re-emit it. Supplying both fields on a copy keeps the
    isolated render identical to the full-conversation assistant suffix.
    """
    content = message.get("content")
    template = chat_template or ""
    if (
        message.get("role") != "assistant"
        or not isinstance(content, str)
        or "<think>" not in content
        or "</think>" not in content
        or "message.reasoning_content is string" not in template
        or "content.split('</think>')" not in template
    ):
        return message

    reasoning = content.split("</think>")[0].rstrip("\n").split("<think>")[-1].lstrip("\n")
    final_content = content.split("</think>")[-1].lstrip("\n")
    prepared = dict(message)
    prepared["reasoning_content"] = reasoning
    prepared["content"] = f"<think>\n{reasoning}\n</think>\n\n{final_content}"
    return prepared


class VerbalizedSFTDatasetMixin:
    def _process_single_message(self, index, message, full_message, tools=None, enable_thinking=None):
        message = _preserve_inline_reasoning(message, getattr(self.tokenizer, "chat_template", None))
        return super()._process_single_message(index, message, full_message, tools, enable_thinking)

    def _configure_vc_fields(self, config):
        vc = config.get("vc", {})
        self.sft_loss_mask = str(vc.get("sft_loss_mask", "completion_selective"))
        self._confidence_markers = confidence_marker_token_ids(self.tokenizer)

    def _add_vc_fields(self, item: dict, row: dict) -> dict:
        base_loss_mask = item["loss_mask"]
        if self.sft_loss_mask != "completion_selective":
            raise ValueError(f"Unsupported sft_loss_mask: {self.sft_loss_mask!r}")
        item["confidence_loss_mask"] = mask_confidence_suffix(
            item["input_ids"], base_loss_mask, self._confidence_markers
        )
        if item["confidence_loss_mask"].sum().item() == 0:
            raise ValueError("SFT sample has no confidence suffix")
        if "rollout_correctness" not in row:
            raise KeyError("Missing SFT rollout_correctness field")
        if float(row["rollout_correctness"]) < 0.5:
            # Incorrect rollouts only supervise the confidence suffix.
            item["loss_mask"] = item["confidence_loss_mask"].clone()
        return item


def build_verbalized_sft_dataset_class():
    from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset

    class VerbalizedSFTDataset(VerbalizedSFTDatasetMixin, MultiTurnSFTDataset):
        def __init__(self, parquet_files, tokenizer, config, processor=None, max_samples=-1):
            super().__init__(parquet_files, tokenizer, config, processor, max_samples)
            self._configure_vc_fields(config)

        def __getitem__(self, item):
            output = super().__getitem__(item)
            row = self.dataframe.iloc[item].to_dict()
            return self._add_vc_fields(output, row)

    return VerbalizedSFTDataset


VerbalizedSFTDataset = build_verbalized_sft_dataset_class()


def _per_sample_segment_sft_loss(
    log_prob: torch.Tensor,
    loss_mask: torch.Tensor,
    dp_group,
    dp_size: int,
    global_sample_count: int | float | None = None,
) -> torch.Tensor:
    log_prob_rows = log_prob.unbind() if log_prob.ndim > 1 else (log_prob,)
    loss_mask_rows = loss_mask.unbind() if loss_mask.ndim > 1 else (loss_mask,)
    if len(log_prob_rows) != len(loss_mask_rows):
        raise ValueError("SFT log probabilities and masks must have the same batch size")

    sample_losses = []
    active_samples = []
    for log_prob_row, loss_mask_row in zip(log_prob_rows, loss_mask_rows, strict=True):
        mask = torch.roll(loss_mask_row, shifts=-1, dims=0).to(
            device=log_prob_row.device, dtype=log_prob_row.dtype
        )
        mask[-1] = 0
        active_count = mask.sum()
        is_active = (active_count > 0).to(dtype=log_prob_row.dtype)
        sample_losses.append(-(log_prob_row * mask).sum() / active_count.clamp(min=1.0) * is_active)
        active_samples.append(is_active)

    loss_sum = torch.stack(sample_losses).sum()
    sample_count = torch.stack(active_samples).sum()
    if global_sample_count is not None:
        sample_count = torch.as_tensor(
            global_sample_count, device=loss_sum.device, dtype=loss_sum.dtype
        )
    elif dp_group is not None:
        torch.distributed.all_reduce(sample_count, op=torch.distributed.ReduceOp.SUM, group=dp_group)
    return loss_sum / sample_count.clamp(min=1.0) * dp_size


def _active_sample_count(loss_mask: torch.Tensor) -> torch.Tensor:
    rows = loss_mask.unbind() if loss_mask.ndim > 1 else (loss_mask,)
    return torch.stack([(row.sum() > 0).to(dtype=torch.float32) for row in rows]).sum()


class SelectiveCompletionSFTLoss:
    def __init__(self, confidence_weight: float = 0.5):
        if not 0.0 <= confidence_weight <= 1.0:
            raise ValueError("sft_confidence_loss_weight must be between 0 and 1")
        self.confidence_weight = confidence_weight

    def __call__(self, model_output=None, data=None, dp_group=None, **kwargs):
        from verl.utils import tensordict_utils as tu

        dp_size = int(data["dp_size"])
        confidence_mask = data["confidence_loss_mask"]
        main_mask = completion_without_confidence_mask(data["loss_mask"], confidence_mask)
        main_loss = _per_sample_segment_sft_loss(
            model_output["log_probs"],
            main_mask,
            dp_group,
            dp_size,
            tu.get_non_tensor_data(data, "selective_main_sample_count", None),
        )
        confidence_loss = _per_sample_segment_sft_loss(
            model_output["log_probs"],
            confidence_mask,
            dp_group,
            dp_size,
            tu.get_non_tensor_data(data, "selective_confidence_sample_count", None),
        )
        loss = (1.0 - self.confidence_weight) * main_loss + self.confidence_weight * confidence_loss
        return loss, {}


def build_verbalized_sft_trainer_class():
    from verl.trainer.sft_trainer import SFTTrainer
    from verl.utils import tensordict_utils as tu

    class VerbalizedSFTTrainer(SFTTrainer):
        def _build_dataloader(self):
            super()._build_dataloader()
            if not bool(self.config.vc.get("sft_balance_confidence_targets", False)):
                return
            if "confidence_target" not in self.train_dataset.dataframe.columns:
                raise ValueError("Balanced SFT parquet is missing confidence_target")

            from torchdata.stateful_dataloader import StatefulDataLoader
            from verl.utils.device import get_device_name

            self.train_sampler = UniformTargetDistributedSampler(
                self.train_dataset,
                self.train_dataset.dataframe["confidence_target"].tolist(),
                num_replicas=self.engine.get_data_parallel_size(),
                rank=self.engine.get_data_parallel_rank(),
                seed=int(self.config.trainer.seed),
                samples_per_target=self.config.vc.get("sft_balance_confidence_target_count"),
                drop_last=True,
            )
            self.train_dataloader = StatefulDataLoader(
                dataset=self.train_dataset,
                batch_size=self.train_batch_size_per_dp,
                sampler=self.train_sampler,
                collate_fn=self.collate_fn,
                num_workers=self.config.data.num_workers,
                pin_memory=False,
                drop_last=True,
                pin_memory_device=get_device_name(),
            )

        def _init_engine(self):
            super()._init_engine()
            configured_epochs = int(self.config.trainer.total_epochs)
            effective_epochs = total_epochs_for_steps(
                self.config.trainer.total_training_steps,
                self.steps_per_epoch,
                configured_epochs,
            )
            if effective_epochs > configured_epochs:
                logger.info(
                    "Extending SFT training from %d to %d epochs to reach %d total steps",
                    configured_epochs,
                    effective_epochs,
                    self.total_training_steps,
                )
                self.config.trainer.total_epochs = effective_epochs

        def _build_ckpt_handler(self):
            super()._build_ckpt_handler()
            save_checkpoint = self.ckpt_handler.save_checkpoint

            def save_checkpoint_and_evaluate(step):
                checkpoint_steps = {
                    int(value) for value in self.config.vc.get("checkpoint_steps", [])
                }
                is_final_step = step >= self.total_training_steps
                if checkpoint_steps and step not in checkpoint_steps and not is_final_step:
                    return
                save_checkpoint(step)
                self._run_generation_eval(step)
                if self.rank == 0 and is_final_step:
                    self._finalize_generation_checkpoints(step)

            self.ckpt_handler.save_checkpoint = save_checkpoint_and_evaluate

        @staticmethod
        def _summarize_generation_metrics(metrics_by_dataset):
            metrics = [
                metrics
                for metrics in metrics_by_dataset.values()
                if "accuracy" in metrics and "auroc" in metrics and "brier_score" in metrics
            ]
            if not metrics:
                return None
            return {
                "accuracy": sum(item["accuracy"] for item in metrics) / len(metrics),
                "auroc": sum(item["auroc"] for item in metrics) / len(metrics),
                "brier_score": sum(item["brier_score"] for item in metrics) / len(metrics),
                "datasets": len(metrics),
            }

        def _finalize_generation_checkpoints(self, final_step: int) -> None:
            output_dir = Path(self.config.trainer.default_local_dir)
            generation_root = output_dir / "generation_eval"
            candidates = []
            for step_dir in sorted(generation_root.glob("step_*")):
                try:
                    step = int(step_dir.name.removeprefix("step_"))
                except ValueError:
                    continue
                if not (output_dir / f"global_step_{step}" / "huggingface").is_dir():
                    continue
                metrics_by_dataset = {}
                for metrics_path in step_dir.rglob("metrics.json"):
                    with metrics_path.open(encoding="utf-8") as handle:
                        metrics_by_dataset[metrics_path.parent.name] = json.load(handle)
                summary = self._summarize_generation_metrics(metrics_by_dataset)
                if summary is not None:
                    candidates.append({"step": step, **summary})
            if not candidates:
                logger.warning("No complete generation-eval metrics found; skipping best checkpoint selection")
                return

            max_accuracy = max(item["accuracy"] for item in candidates)
            eligible = [item for item in candidates if item["accuracy"] >= max_accuracy - 0.01]
            best = max(
                eligible,
                key=lambda item: (item["auroc"], -item["brier_score"], item["step"]),
            )
            best_source = output_dir / f"global_step_{best['step']}" / "huggingface"
            if not best_source.is_dir():
                logger.warning("Best checkpoint source is missing: %s", best_source)
                return

            best_dir = output_dir / "best"
            shutil.rmtree(best_dir, ignore_errors=True)
            try:
                shutil.copytree(best_source, best_dir / "huggingface", copy_function=os.link)
            except OSError:
                shutil.rmtree(best_dir, ignore_errors=True)
                shutil.copytree(best_source, best_dir / "huggingface")
            selection = {
                "best_step": best["step"],
                "final_step": final_step,
                "max_accuracy": max_accuracy,
                "accuracy_tolerance": 0.01,
                "selected": best,
                "candidates": candidates,
            }
            (best_dir / "selection.json").write_text(
                json.dumps(selection, indent=2), encoding="utf-8"
            )

            keep_steps = {final_step, best["step"]}
            for step_dir in output_dir.glob("global_step_*"):
                try:
                    step = int(step_dir.name.removeprefix("global_step_"))
                except ValueError:
                    continue
                if step not in keep_steps:
                    shutil.rmtree(step_dir)
            logger.info(
                "Selected SFT best checkpoint: step=%s accuracy=%.4f auroc=%.4f brier=%.4f; kept final step=%s",
                best["step"],
                best["accuracy"],
                best["auroc"],
                best["brier_score"],
                final_step,
            )

        def _build_engine(self):
            super()._build_engine()
            confidence_weight = float(self.config.vc.get("sft_confidence_loss_weight", 0.5))
            self.loss_fn = SelectiveCompletionSFTLoss(confidence_weight)
            self.training_client.set_loss_fn(loss_fn=self.loss_fn)

            original_train_batch = self.training_client.train_batch
            original_infer_batch = self.training_client.infer_batch

            def prepare_batch(data):
                if bool(tu.get_non_tensor_data(data, "use_dynamic_bsz", True)):
                    configured_budget = tu.get_non_tensor_data(data, "max_token_len_per_gpu", None)
                    budget = full_sequence_token_budget(data["input_ids"], configured_budget)
                    tu.assign_non_tensor(data, max_token_len_per_gpu=budget)
                confidence_mask = data["confidence_loss_mask"]
                main_mask = completion_without_confidence_mask(data["loss_mask"], confidence_mask)
                sample_counts = torch.stack(
                    [_active_sample_count(main_mask), _active_sample_count(confidence_mask)]
                ).to(self.device_name)
                if self.engine.get_data_parallel_size() > 1:
                    torch.distributed.all_reduce(
                        sample_counts,
                        op=torch.distributed.ReduceOp.SUM,
                        group=self.engine.get_data_parallel_group(),
                    )
                tu.assign_non_tensor(
                    data,
                    selective_main_sample_count=float(sample_counts[0].item()),
                    selective_confidence_sample_count=float(sample_counts[1].item()),
                )

            def train_batch(data):
                prepare_batch(data)
                return original_train_batch(data)

            def infer_batch(data):
                prepare_batch(data)
                return original_infer_batch(data)

            self.training_client.train_batch = train_batch
            self.training_client.infer_batch = infer_batch

        def fit(self):
            import verl.trainer.sft_trainer as verl_sft_trainer

            original_tracking = verl_sft_trainer.Tracking

            def capture_tracking(*args, **kwargs):
                tracking = original_tracking(*args, **kwargs)
                self._tracking = tracking
                return tracking

            verl_sft_trainer.Tracking = capture_tracking
            try:
                result = super().fit()
                finish_swanlab(getattr(self, "_tracking", None))
                return result
            finally:
                verl_sft_trainer.Tracking = original_tracking

        def _should_run_generation_eval(self, step: int) -> bool:
            vc = self.config.get("vc", {})
            if not bool(vc.get("generation_eval", False)):
                return False
            checkpoint_steps = {
                int(value) for value in vc.get("checkpoint_steps", [])
            }
            if checkpoint_steps:
                return step in checkpoint_steps or step >= self.total_training_steps
            frequency = int(vc.get("generation_eval_freq", -1))
            if step >= self.total_training_steps:
                return True
            return frequency > 0 and step % frequency == 0

        @staticmethod
        def _eval_cuda_visible_devices() -> str:
            visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
            if visible_devices:
                return visible_devices.split(",", 1)[0].strip()
            return "0"

        def _generation_eval_command(
            self, step: int, dataset: str, metrics_path: Path
        ) -> list[str]:
            vc = self.config.vc
            checkpoint = Path(self.config.trainer.default_local_dir) / f"global_step_{step}" / "huggingface"
            eval_dir = metrics_path.parent
            is_competition_math = str(dataset).lower() in {
                "aime2024",
                "aime2025",
                "aime2026",
            }
            eval_temperature = 0.6 if is_competition_math else vc.generation_eval_temperature
            eval_num_generations = 32 if is_competition_math else 1
            command = [
                sys.executable,
                "-m",
                "src.eval.eval_main",
                "--dataset",
                dataset,
                "--model-name-or-path",
                str(checkpoint.resolve()),
                "--inferencer",
                "verbalized_confidence",
                "--confidence_format",
                str(vc.generation_eval_confidence_format),
                "--tensor_parallel_size",
                str(vc.generation_eval_tensor_parallel_size),
                "--max-tokens",
                str(vc.generation_eval_max_new_tokens),
                "--temperature",
                str(eval_temperature),
                "--num-generations",
                str(eval_num_generations),
                "--gpu-memory-utilization",
                str(vc.generation_eval_gpu_memory_utilization),
                "--ece-bins",
                str(vc.generation_eval_ece_bins),
                "--output-dir",
                str(eval_dir.resolve()),
                "--metrics-json",
                str(metrics_path.resolve()),
            ]
            if vc.get("generation_eval_sample_size") is not None:
                command.extend(["--sample-size", str(vc.generation_eval_sample_size)])
            return command

        @staticmethod
        def _swanlab_generation_metrics(metrics: dict, dataset: str | None = None) -> dict[str, float]:
            metric_names = [
                ("format_legal_ratio", "format"),
                ("accuracy", "accuracy"),
                ("confidence_avg", "confidence_mean"),
                ("ece", "ece"),
                ("brier_score", "brier_score"),
                ("brier_skill_score", "brier_skill_score"),
                ("macro_ce", "macro_ce"),
                ("corp_mcb", "corp_mcb"),
                ("corp_dsc", "corp_dsc"),
                ("eaurc", "eaurc"),
                ("auroc", "auroc"),
                ("confidence_legal_ratio", "confidence_legal_ratio"),
                ("generation_length", "generation_length"),
            ]
            prefix = f"val/metrics/{dataset}" if dataset else "val/metrics"
            return {
                f"{prefix}/{target_name}": float(metrics[source_name])
                for source_name, target_name in metric_names
                if source_name in metrics
            }

        def _run_generation_eval(self, step: int) -> dict:
            if not self._should_run_generation_eval(step):
                return {}

            output_dir = Path(self.config.trainer.default_local_dir)
            eval_root = output_dir / "generation_eval" / f"step_{step}"
            datasets = list(self.config.vc.get("generation_eval_datasets", []))
            if not datasets:
                datasets = [str(self.config.vc.generation_eval_dataset)]
            status_path = eval_root / "status.json"
            status = 0
            metrics_by_dataset = {}

            self.training_client.to("cpu", model=True, optimizer=True, grad=True)
            torch.cuda.empty_cache()
            if self.rank == 0:
                eval_root.mkdir(parents=True, exist_ok=True)
                status_path.unlink(missing_ok=True)
            torch.distributed.barrier()
            try:
                if self.rank == 0:
                    env = os.environ.copy()
                    env["CUDA_VISIBLE_DEVICES"] = self._eval_cuda_visible_devices()
                    for name in (
                        "RANK",
                        "WORLD_SIZE",
                        "LOCAL_RANK",
                        "LOCAL_WORLD_SIZE",
                        "GROUP_RANK",
                        "GROUP_WORLD_SIZE",
                        "ROLE_RANK",
                        "ROLE_WORLD_SIZE",
                        "ROLE_NAME",
                        "MASTER_ADDR",
                        "MASTER_PORT",
                    ):
                        env.pop(name, None)
                    for name in [name for name in env if name.startswith("TORCHELASTIC_")]:
                        env.pop(name, None)
                    for dataset in datasets:
                        eval_dir = eval_root / dataset if len(datasets) > 1 else eval_root
                        eval_dir.mkdir(parents=True, exist_ok=True)
                        metrics_path = eval_dir / "metrics.json"
                        command = self._generation_eval_command(step, dataset, metrics_path)
                        logger.info(
                            "Running SFT generation eval for %s at step %d: %s",
                            dataset,
                            step,
                            " ".join(command),
                        )
                        completed = subprocess.run(command, env=env, check=False)
                        status = completed.returncode
                        if status != 0:
                            break
                        with metrics_path.open(encoding="utf-8") as handle:
                            metrics_by_dataset[dataset] = json.load(handle)
            except Exception:
                logger.exception("SFT generation eval failed at step %d", step)
                status = 1
            finally:
                if self.rank == 0:
                    temporary_status_path = status_path.with_suffix(".tmp")
                    temporary_status_path.write_text(json.dumps({"returncode": status}), encoding="utf-8")
                    temporary_status_path.replace(status_path)

            while not status_path.exists():
                time.sleep(1)
            with status_path.open(encoding="utf-8") as handle:
                status = int(json.load(handle)["returncode"])
            self.training_client.to("device", model=True, optimizer=True, grad=True)
            torch.distributed.barrier()

            if status != 0:
                raise RuntimeError(f"SFT generation eval failed at step {step}; see {eval_root}")
            if self.rank == 0 and hasattr(self, "_tracking"):
                for dataset, metrics in metrics_by_dataset.items():
                    self._tracking.log(
                        data=self._swanlab_generation_metrics(metrics, dataset),
                        step=step,
                    )
            return metrics_by_dataset

    return VerbalizedSFTTrainer


VerbalizedSFTTrainer = build_verbalized_sft_trainer_class()


def run_sft(config):
    from verl.utils.distributed import destroy_global_process_group, initialize_global_process_group

    initialize_global_process_group()
    try:
        trainer = VerbalizedSFTTrainer(config=config)
        trainer.fit()
    except BaseException:
        logger.exception("SFT worker failed before distributed cleanup")
        raise
    finally:
        destroy_global_process_group()


def launch_sft(config, paths, runtime):
    import subprocess
    import sys
    from pathlib import Path

    from omegaconf import OmegaConf

    from src.train.configs.runtime import compose_verl_sft_config

    if runtime.nnodes != 1:
        raise ValueError("The unified SFT launcher currently supports one node")

    verl_config = compose_verl_sft_config(config, paths, runtime)
    config_path = Path(config["output_dir"], "verl_sft_config.yaml")
    OmegaConf.save(verl_config, config_path)
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc-per-node={runtime.n_gpus_per_node}",
        "-m",
        "src.train.trainers.sft",
        "--config",
        str(config_path.resolve()),
    ]
    subprocess.run(command, check=True)


def main(argv=None):
    import argparse

    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser(description="Torchrun worker for verl SFT")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    run_sft(OmegaConf.load(args.config))


if __name__ == "__main__":
    main()
