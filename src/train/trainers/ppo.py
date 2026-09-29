import json
import re
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from src.common.calibration_metrics import (
    compute_binary_auroc,
    compute_calibration_metrics,
    compute_confidence_distribution_metrics,
)
from src.train.algorithms.segmented import SEGMENTED_ADV_ESTIMATORS, compute_segmented_advantage
from src.train.rewards.reward_functions import extract_answer_value
from src.train.trainers import finish_swanlab, shutdown_train_dataloader
from src.train.trainers.replay_buffer import reward_extra_info_arrays


CONFIDENCE_ALGORITHMS = {
    "rlcr",
    "redoubt",
    "coca",
    "dcpo",
}


def build_batch_confidence_records(
    mode,
    global_step,
    question_ids,
    confidences,
    correctness,
    answers=None,
):
    answers = answers if answers is not None else [""] * len(question_ids)
    lengths = {len(question_ids), len(confidences), len(correctness), len(answers)}
    if len(lengths) != 1:
        raise ValueError("Batch confidence fields must have the same length")

    grouped = OrderedDict()
    for question_id, confidence, is_correct, answer in zip(
        question_ids,
        confidences,
        correctness,
        answers,
        strict=True,
    ):
        group = grouped.setdefault(
            str(question_id),
            {"confidences": [], "is_correct": [], "answers": []},
        )
        group["confidences"].append(round(float(confidence), 2))
        group["is_correct"].append(float(is_correct))
        group["answers"].append(str(answer))

    records = []
    for question_id, group in grouped.items():
        records.append(
            {
                "mode": mode,
                "batch_index": int(global_step),
                "global_step": int(global_step),
                "question_id": question_id,
                "num_rollouts": len(group["confidences"]),
                "group_success_rate": float(np.mean(group["is_correct"])),
                "is_correct": group["is_correct"],
                "answers": group["answers"],
                "confidences": group["confidences"],
            }
        )
    return records


def write_batch_confidence_records(output_dir, mode, records):
    if mode not in {"train", "eval"}:
        raise ValueError(f"Unsupported batch confidence mode: {mode}")
    record_dir = Path(output_dir, "batch_confidence")
    record_dir.mkdir(parents=True, exist_ok=True)
    record_path = record_dir / f"{mode}.jsonl"
    file_mode = "a" if mode == "train" else "w"
    with record_path.open(file_mode, encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    return record_path


def _finite_values(values):
    return np.asarray([value for value in values if value is not None], dtype=float)


def aggregate_validation_metrics(
    reward_extra_infos_dict,
    n_bins=10,
):
    accuracies = reward_extra_infos_dict.get("accuracy", [])
    confidences = reward_extra_infos_dict.get("confidence", [])
    paired = [
        (accuracy, confidence)
        for accuracy, confidence in zip(accuracies, confidences, strict=False)
        if accuracy is not None and confidence is not None
    ]
    if not paired:
        return {}

    correctness = np.asarray([value[0] for value in paired], dtype=float)
    confidence = np.asarray([value[1] for value in paired], dtype=float)
    calibration = compute_calibration_metrics(correctness, confidence, n_bins=n_bins)
    metrics = OrderedDict()

    rewards = _finite_values(reward_extra_infos_dict.get("reward", []))
    formats = _finite_values(reward_extra_infos_dict.get("format", []))
    if rewards.size:
        metrics["reward"] = float(rewards.mean())
    if formats.size:
        metrics["format"] = float(formats.mean())
    metrics["accuracy"] = calibration["accuracy"]
    metrics["confidence_mean"] = calibration["mean_confidence"]
    metrics["ece"] = calibration["ece"]
    metrics["brier_score"] = calibration["brier_score"]
    for key in ("brier_skill_score", "macro_ce", "corp_mcb", "corp_dsc", "eaurc"):
        if calibration[key] is not None:
            metrics[key] = calibration[key]
    auroc = compute_binary_auroc(correctness, confidence)
    if auroc is not None:
        metrics["auroc"] = auroc
    metrics["abs_confidence_accuracy_gap"] = calibration["abs_confidence_accuracy_gap"]
    if rewards.size:
        metrics["reward_std"] = float(rewards.std())
    metrics.update(compute_confidence_distribution_metrics(confidence))

    legal = _finite_values(reward_extra_infos_dict.get("confidence_legal", []))
    if legal.size:
        metrics["confidence_legal_ratio"] = float(legal.mean())
    return metrics


def build_tag_value_token_mask(tokenizer, token_ids, tag_name):
    token_ids = [int(token_id) for token_id in token_ids]
    completion = tokenizer.decode(
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    matches = list(re.finditer(rf"<{re.escape(tag_name)}>(.*?)</{re.escape(tag_name)}>", completion, re.DOTALL))
    if not matches:
        return np.zeros(len(token_ids), dtype=bool)

    match = matches[-1]
    value = match.group(1)
    span_start = match.start(1) + len(value) - len(value.lstrip())
    span_end = match.start(1) + len(value.rstrip())
    mask = np.zeros(len(token_ids), dtype=bool)
    special_ids = set(getattr(tokenizer, "all_special_ids", []))
    try:
        encoded = tokenizer(
            completion,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        actual_positions = [index for index, token_id in enumerate(token_ids) if token_id not in special_ids]
        actual_ids = [token_ids[index] for index in actual_positions]
        if encoded["input_ids"] == actual_ids:
            for token_position, (token_start, token_end) in zip(
                actual_positions,
                encoded["offset_mapping"],
                strict=True,
            ):
                if token_end > span_start and token_start < span_end:
                    mask[token_position] = True
            return mask
    except (KeyError, NotImplementedError, TypeError, ValueError):
        pass

    token_start = 0
    for index in range(len(token_ids)):
        token_end = len(
            tokenizer.decode(
                token_ids[: index + 1],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        )
        if token_end > span_start and token_start < span_end:
            mask[index] = True
        token_start = token_end
    return mask


def aggregate_token_entropy_metrics(entropy, response_mask, answer_mask, confidence_mask):
    entropy = np.asarray(entropy, dtype=float)
    response_mask = np.asarray(response_mask, dtype=bool)
    answer_mask = np.asarray(answer_mask, dtype=bool) & response_mask
    confidence_mask = np.asarray(confidence_mask, dtype=bool) & response_mask
    if not (entropy.shape == response_mask.shape == answer_mask.shape == confidence_mask.shape):
        raise ValueError("entropy and token masks must have the same shape")

    finite = np.isfinite(entropy)
    metrics = OrderedDict()
    presence = {}
    for name, mask in (("answer", answer_mask), ("confidence", confidence_mask)):
        valid = mask & finite
        if valid.any():
            metrics[f"{name}_mean"] = float(entropy[valid].mean())
        presence[f"{name}_present_ratio"] = float(mask.any(axis=1).mean())
    metrics.update(presence)
    return metrics


def aggregate_segment_signal_metrics(
    advantages,
    response_mask,
    confidence_starts,
    loss_agg_mode="token-mean",
    confidence_ends=None,
):
    """Measure each segment's configured PPO aggregation signal at ratio one."""
    advantages = np.asarray(advantages, dtype=float)
    response_mask = np.asarray(response_mask, dtype=bool)
    confidence_starts = np.asarray(confidence_starts, dtype=int)
    if advantages.shape != response_mask.shape or confidence_starts.shape != (advantages.shape[0],):
        raise ValueError("advantages, response_mask, and confidence_starts must describe the same batch")

    positions = np.arange(advantages.shape[1])[None, :]
    confidence_mask = positions >= confidence_starts[:, None]
    if confidence_ends is not None:
        confidence_ends = np.asarray(confidence_ends, dtype=int)
        if confidence_ends.shape != (advantages.shape[0],):
            raise ValueError("confidence_ends must describe the same batch")
        confidence_mask &= positions < confidence_ends[:, None]
    confidence_mask &= response_mask
    answer_mask = response_mask & ~confidence_mask
    token_count = int(response_mask.sum())
    if token_count == 0:
        return OrderedDict()

    metrics = OrderedDict()
    for name, mask in (("answer", answer_mask), ("confidence", confidence_mask)):
        signal = np.where(mask, advantages, 0.0)
        if loss_agg_mode == "seq-mean-token-mean":
            response_lengths = response_mask.sum(axis=1, keepdims=True).clip(min=1)
            normalized = signal / response_lengths / advantages.shape[0]
            metrics[f"{name}_loss_abs"] = float(np.abs(normalized).sum())
            metrics[f"{name}_logprob_grad_norm"] = float(np.sqrt(np.square(normalized).sum()))
        else:
            metrics[f"{name}_loss_abs"] = float(np.abs(signal).sum() / token_count)
            metrics[f"{name}_logprob_grad_norm"] = float(np.sqrt(np.square(signal).sum()) / token_count)
    return metrics


def _segmented_reward_keys(vc_config):
    keys = {
        "accuracy",
        "confidence",
        "confidence_legal",
        "answer_reward",
        "confidence_start",
    }
    if str(vc_config.algorithm).lower() == "redoubt":
        keys.add(str(getattr(vc_config, "confidence_reward", None) or "brier"))
    if str(vc_config.algorithm).lower() == "coca":
        keys.add("confidence_end")
    return keys


def _register_v1_trainer():
    import transfer_queue as tq
    from tensordict import TensorDict
    from verl.protocol import DataProto
    from verl.trainer.ppo.ray_trainer import apply_kl_penalty
    from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch
    from verl.trainer.ppo.v1 import PPOTrainerSync, get_trainer_cls, register_trainer
    from verl.workers.utils.padding import response_to_nested

    try:
        return get_trainer_cls("verbalized_sync")
    except ValueError:
        pass

    @register_trainer("verbalized_sync")
    class VerbalizedPPOTrainerSync(PPOTrainerSync):
        def _batch_confidence_enabled(self):
            return bool(self.config.vc.get("log_batch_confidences", True))

        def _batch_confidence_dir(self):
            return Path(self.config.trainer.default_local_dir, "batch_confidence")

        @staticmethod
        def _reward_infos(extra_fields):
            infos = []
            for extra_field in extra_fields:
                if hasattr(extra_field, "data"):
                    extra_field = extra_field.data
                infos.append(extra_field.get("reward_extra_info", {}))
            return infos

        def _log_rollout_data(self, batch, timing_raw, rollout_data_dir):
            if not self._batch_confidence_enabled() or Path(rollout_data_dir) != self._batch_confidence_dir():
                return super()._log_rollout_data(batch, timing_raw, rollout_data_dir)

            data = tq.kv_batch_get(
                keys=batch.keys,
                partition_id=batch.partition_id,
                select_fields=["responses", "extra_fields"],
            )
            responses = data["responses"].to_padded_tensor(padding=self.tokenizer.pad_token_id)
            outputs = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in responses]
            reward_infos = self._reward_infos(list(data["extra_fields"]))
            question_ids = [key.rsplit("_", 2)[0] for key in batch.keys]
            records = build_batch_confidence_records(
                "train",
                self.global_steps,
                question_ids,
                [info["confidence"] for info in reward_infos],
                [info["accuracy"] for info in reward_infos],
                [extract_answer_value(output) for output in outputs],
            )
            write_batch_confidence_records(self.config.trainer.default_local_dir, "train", records)

        def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
            if not self._batch_confidence_enabled() or Path(dump_path) != self._batch_confidence_dir():
                return super()._dump_generations(
                    inputs,
                    outputs,
                    gts,
                    scores,
                    reward_extra_infos_dict,
                    dump_path,
                )

            records = build_batch_confidence_records(
                "eval",
                self.global_steps,
                [key.rsplit("_", 2)[0] for key in reward_extra_infos_dict["uid"]],
                reward_extra_infos_dict["confidence"],
                reward_extra_infos_dict["accuracy"],
                [extract_answer_value(output) for output in outputs],
            )
            write_batch_confidence_records(self.config.trainer.default_local_dir, "eval", records)

        def _val_metrics_update(self, data_sources, sample_uids, reward_extra_infos_dict, sample_turns):
            if str(self.config.vc.algorithm).lower() not in CONFIDENCE_ALGORITHMS:
                return super()._val_metrics_update(
                    data_sources,
                    sample_uids,
                    reward_extra_infos_dict,
                    sample_turns,
                )

            validation_metrics = aggregate_validation_metrics(
                reward_extra_infos_dict,
                n_bins=int(self.config.vc.get("ece_bins", 10)),
            )
            return {f"val/metrics/{name}": value for name, value in validation_metrics.items()}

        def _compute_metrics(self, batch, metrics, timing_raw, global_steps, epoch):
            super()._compute_metrics(batch, metrics, timing_raw, global_steps, epoch)
            if str(self.config.vc.algorithm).lower() not in CONFIDENCE_ALGORITHMS:
                return

            keep_indices = [index for index, tag in enumerate(batch.tags) if not tag.get("is_padding", False)]
            if not keep_indices:
                return

            batch_data = tq.kv_batch_get(
                keys=batch.keys,
                partition_id=batch.partition_id,
                select_fields=["responses", "response_mask", "advantages", "rm_scores", "extra_fields"],
            )
            reward_infos = self._reward_infos(list(batch_data.pop("extra_fields")))
            padded = batch_data.to_padded_tensor()
            index_tensor = torch.as_tensor(keep_indices, dtype=torch.long)
            responses = padded["responses"][index_tensor]
            response_mask = padded["response_mask"][index_tensor].bool()
            rewards = padded["rm_scores"][index_tensor].sum(dim=-1).detach().cpu().numpy()
            kept_reward_infos = [reward_infos[index] for index in keep_indices]
            reward_extra_infos_dict = {
                name: [info.get(name) for info in kept_reward_infos]
                for name in {
                    "accuracy",
                    "confidence",
                    "confidence_legal",
                    "format",
                }
            }
            reward_extra_infos_dict["reward"] = rewards.tolist()
            train_metrics = aggregate_validation_metrics(
                reward_extra_infos_dict,
                n_bins=int(self.config.vc.get("ece_bins", 10)),
            )
            metrics.update({f"train/metrics/{name}": value for name, value in train_metrics.items()})

            algorithm = str(self.config.vc.algorithm).lower()
            if algorithm in SEGMENTED_ADV_ESTIMATORS:
                segment_metrics = aggregate_segment_signal_metrics(
                    padded["advantages"][index_tensor].detach().cpu().numpy(),
                    response_mask.detach().cpu().numpy(),
                    [info["confidence_start"] for info in kept_reward_infos],
                    confidence_ends=[info["confidence_end"] for info in kept_reward_infos]
                    if algorithm == "coca"
                    else None,
                    loss_agg_mode=self.config.actor_rollout_ref.actor.loss_agg_mode,
                )
                metrics.update({f"train/segment_signal/{name}": value for name, value in segment_metrics.items()})

            if not self.config.vc.get("log_token_entropy", True) or "actor/entropy" not in metrics:
                return
            entropy_data = tq.kv_batch_get(
                keys=batch.keys,
                partition_id=batch.partition_id,
                select_fields=["entropy"],
            ).to_padded_tensor()
            entropy = entropy_data["entropy"][index_tensor].detach().cpu().numpy()
            response_mask_array = response_mask.detach().cpu().numpy()
            response_array = responses.detach().cpu().numpy()
            answer_mask = np.zeros_like(response_mask_array, dtype=bool)
            confidence_mask = np.zeros_like(response_mask_array, dtype=bool)
            for row, (token_ids, token_mask) in enumerate(zip(response_array, response_mask_array, strict=True)):
                token_positions = np.flatnonzero(token_mask)
                valid_token_ids = token_ids[token_positions].tolist()
                answer_mask[row, token_positions] = build_tag_value_token_mask(
                    self.tokenizer,
                    valid_token_ids,
                    "answer",
                )
                confidence_mask[row, token_positions] = build_tag_value_token_mask(
                    self.tokenizer,
                    valid_token_ids,
                    "confidence",
                )
            entropy_metrics = aggregate_token_entropy_metrics(
                entropy,
                response_mask_array,
                answer_mask,
                confidence_mask,
            )
            metrics.update({f"train/token_entropy/{name}": value for name, value in entropy_metrics.items()})

        def _compute_advantage(self, batch, metrics):
            algorithm = str(self.config.algorithm.adv_estimator).lower()
            if algorithm not in SEGMENTED_ADV_ESTIMATORS:
                return super()._compute_advantage(batch, metrics)

            fields = [
                "uid",
                "response_mask",
                "rm_scores",
                "rollout_log_probs",
                "old_log_probs",
                "ref_log_prob",
                "values",
                "extra_fields",
            ]
            nested_data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=fields)
            extra_fields = list(nested_data.pop("extra_fields"))
            response_mask = nested_data["response_mask"]
            data = DataProto(batch=nested_data.to_padded_tensor())
            data.batch["token_level_scores"] = data.batch["rm_scores"]
            data.non_tensor_batch["uid"] = np.asarray(data.batch.pop("uid").tolist(), dtype=object)
            data.non_tensor_batch.update(
                reward_extra_info_arrays(extra_fields, _segmented_reward_keys(self.config.vc))
            )

            if self.config.algorithm.use_kl_in_reward:
                data, kl_metrics = apply_kl_penalty(
                    data,
                    kl_ctrl=self.kl_ctrl_in_reward,
                    kl_penalty=self.config.algorithm.kl_penalty,
                )
                metrics.update(kl_metrics)
            else:
                data.batch["token_level_rewards"] = data.batch["token_level_scores"]

            rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
            bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
            rollout_correction = (
                rollout_corr_config is not None
                and "rollout_log_probs" in data.batch
                and not bypass_recomputing_logprobs
            )
            if rollout_correction:
                data, is_metrics = compute_rollout_correction_and_add_to_batch(data, rollout_corr_config)
                metrics.update(is_metrics)

            data = compute_segmented_advantage(data, self.config.vc)
            output_fields = ["advantages", "returns"]
            if self.config.algorithm.use_kl_in_reward:
                output_fields.append("token_level_rewards")
            if rollout_correction:
                output_fields.append("response_mask")
                if "rollout_is_weights" in data.batch:
                    output_fields.append("rollout_is_weights")

            output = {
                field: response_to_nested(data.batch[field], response_mask)
                for field in output_fields
            }
            output = TensorDict(output, batch_size=len(batch))
            return tq.kv_batch_put(keys=batch.keys, partition_id=batch.partition_id, fields=output)

    return VerbalizedPPOTrainerSync


def build_task_runner_class():
    import ray

    @ray.remote
    class VerbalizedTaskRunnerV1:
        def __init__(self):
            self.config = None
            self.trainer = None
            self.agent_loop_manager = None

        def init_agent_loop_manager(self):
            from verl.trainer.ppo.v1 import AgentLoopManagerTQ
            from verl.utils.import_utils import load_class_from_fqn

            manager_class_fqn = self.config.actor_rollout_ref.rollout.get("agent", {}).get(
                "agent_loop_manager_class"
            )
            manager_cls = (
                load_class_from_fqn(manager_class_fqn, "AgentLoopManager")
                if manager_class_fqn
                else AgentLoopManagerTQ
            )
            self.agent_loop_manager = manager_cls.create(
                config=self.config,
                llm_client=self.trainer.get_llm_client(),
                teacher_client=self.trainer.get_teacher_client(),
                reward_loop_worker_handles=self.trainer.get_reward_handles(),
            )

        def run(self, config):
            import transfer_queue as tq
            from omegaconf import OmegaConf
            trainer_cls = _register_v1_trainer()
            config.transfer_queue.enable = True
            OmegaConf.resolve(config)
            self.config = config
            tq.init(config.transfer_queue)
            try:
                self.trainer = trainer_cls(config=config)
                self.trainer.init()
                self.init_agent_loop_manager()
                self.trainer.fit(self.agent_loop_manager)
            finally:
                try:
                    shutdown_train_dataloader(self.trainer)
                finally:
                    tq.close()
            finish_swanlab(getattr(self.trainer, "logger", None))

    return VerbalizedTaskRunnerV1


def run_ppo(config):
    from verl.trainer.main_ppo import run_ppo as verl_run_ppo
    from verl.trainer.ppo.utils import need_critic, need_reference_policy
    from verl.utils.config import validate_config

    validate_config(
        config=config,
        use_reference_policy=need_reference_policy(config),
        use_critic=need_critic(config),
    )
    verl_run_ppo(config, task_runner_class=build_task_runner_class())


def launch_ppo(config, paths, runtime):
    from src.train.configs.runtime import compose_verl_config

    run_ppo(compose_verl_config(config, paths, runtime))
