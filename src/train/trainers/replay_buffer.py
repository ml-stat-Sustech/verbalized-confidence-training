from collections import defaultdict
from collections.abc import Callable

import numpy as np
import transfer_queue as tq
from transfer_queue import KVBatchMeta
from verl.trainer.ppo.v1.replay_buffer import ReplayBuffer


def reward_extra_info_arrays(extra_fields, required_keys):
    reward_infos = []
    for extra_field in extra_fields:
        if hasattr(extra_field, "data"):
            extra_field = extra_field.data
        reward_infos.append(extra_field.get("reward_extra_info", {}))

    missing = [key for key in required_keys if any(key not in info for info in reward_infos)]
    if missing:
        raise KeyError(f"V1 rollout is missing reward metadata: {sorted(set(missing))}")
    return {key: np.asarray([info[key] for info in reward_infos]) for key in required_keys}


def prompt_uid_from_trajectory_key(key: str) -> str:
    parts = key.rsplit("_", 2)
    if len(parts) != 3:
        raise ValueError(f"Invalid V1 trajectory key: {key!r}")
    return parts[0]


def classify_acc_group(values) -> tuple[bool, str]:
    values = np.asarray(values, dtype=float)
    if len(values) == 1 or np.std(values) > 0:
        return True, "mixed"
    if values[0] == 0.0:
        return False, "all_wrong"
    if values[0] == 1.0:
        return False, "all_correct"
    return False, "constant"


class DapoFilterReplayBuffer(ReplayBuffer):
    """Replay buffer implementing DAPO prompt-group dynamic sampling."""

    def __init__(
        self,
        trainer_mode,
        trainer_config,
        max_off_policy_threshold,
        max_off_policy_strategy,
        sampler_kwargs,
        poll_interval=2.0,
        refill_fn: Callable[[int], int] | None = None,
    ):
        super().__init__(
            trainer_mode=trainer_mode,
            trainer_config=trainer_config,
            max_off_policy_threshold=max_off_policy_threshold,
            max_off_policy_strategy=max_off_policy_strategy,
            sampler_kwargs=sampler_kwargs,
            poll_interval=poll_interval,
            refill_fn=refill_fn,
        )
        self.filter_metric = str(sampler_kwargs.get("filter_metric", "acc"))
        self.max_num_gen_batches = int(sampler_kwargs.get("max_num_gen_batches", 20))

    def _sample_candidate(self, global_steps: int, partition_id: str, batch_size: int):
        return super().sample(global_steps=global_steps, partition_id=partition_id, batch_size=batch_size)

    def _candidate_metric_values(self, batch: KVBatchMeta) -> np.ndarray:
        metadata_key = "accuracy" if self.filter_metric == "acc" else self.filter_metric
        data = tq.kv_batch_get(
            keys=batch.keys,
            partition_id=batch.partition_id,
            select_fields=["extra_fields"],
        )
        return reward_extra_info_arrays(list(data["extra_fields"]), {metadata_key})[metadata_key]

    def sample(self, global_steps: int, partition_id: str, batch_size: int):
        if partition_id != "train":
            return self._sample_candidate(global_steps, partition_id, batch_size)

        accepted: dict[str, list[tuple[str, dict]]] = {}
        off_policy_metrics = {}
        candidate_groups = all_wrong_groups = all_correct_groups = 0
        generation_batches = 0

        while len(accepted) < batch_size:
            generation_batches += 1
            candidate, candidate_metrics = self._sample_candidate(global_steps, partition_id, batch_size)
            off_policy_metrics.update(candidate_metrics)
            metric_values = self._candidate_metric_values(candidate)

            grouped = defaultdict(list)
            for key, tag, value in zip(candidate.keys, candidate.tags, metric_values, strict=True):
                grouped[prompt_uid_from_trajectory_key(key)].append((key, tag, value))

            rejected_keys = []
            for uid, trajectories in grouped.items():
                candidate_groups += 1
                keep, reason = classify_acc_group([trajectory[2] for trajectory in trajectories])
                if keep:
                    accepted[uid] = [(key, tag) for key, tag, _ in trajectories]
                else:
                    rejected_keys.extend(key for key, _, _ in trajectories)
                    all_wrong_groups += reason == "all_wrong"
                    all_correct_groups += reason == "all_correct"
            if rejected_keys:
                tq.kv_clear(keys=rejected_keys, partition_id=partition_id)

            if len(accepted) >= batch_size:
                break
            if self.max_num_gen_batches > 0 and generation_batches >= self.max_num_gen_batches:
                accepted_keys = [key for trajectories in accepted.values() for key, _ in trajectories]
                if accepted_keys:
                    tq.kv_clear(keys=accepted_keys, partition_id=partition_id)
                raise ValueError(
                    f"DAPO filtering accepted {len(accepted)}/{batch_size} prompt groups after "
                    f"{generation_batches} generation batches; max_num_gen_batches={self.max_num_gen_batches}"
                )
            if self.refill_fn is None:
                raise RuntimeError("DAPO filtering requires the V1 replay-buffer refill callback")
            self.refill_fn(batch_size)

        selected_uids = list(accepted)[:batch_size]
        unused_uids = list(accepted)[batch_size:]
        unused_keys = [key for uid in unused_uids for key, _ in accepted[uid]]
        if unused_keys:
            tq.kv_clear(keys=unused_keys, partition_id=partition_id)

        selected = [item for uid in selected_uids for item in accepted[uid]]
        valid_ratio = len(accepted) / candidate_groups if candidate_groups else 0.0
        prefix = "training/dynamic_sampling"
        off_policy_metrics.update(
            {
                f"{prefix}/valid_group_ratio": valid_ratio,
                f"{prefix}/all_wrong_group_ratio": all_wrong_groups / candidate_groups if candidate_groups else 0.0,
                f"{prefix}/all_correct_group_ratio": all_correct_groups / candidate_groups if candidate_groups else 0.0,
                f"{prefix}/num_candidate_groups": candidate_groups,
                f"{prefix}/num_accepted_groups": len(accepted),
                f"{prefix}/num_used_groups": len(selected_uids),
                f"{prefix}/num_generation_batches": generation_batches,
            }
        )
        return (
            KVBatchMeta(
                partition_id=partition_id,
                keys=[key for key, _ in selected],
                tags=[tag for _, tag in selected],
            ),
            off_policy_metrics,
        )
