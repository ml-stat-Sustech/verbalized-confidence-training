import torch

from src.train.algorithms.advantages import compute_coca_token_advantages, group_success_rates
from src.train.rewards.scoring import confidence_reward


SEGMENTED_ADV_ESTIMATORS = {"coca", "dcpo", "redoubt"}


def compute_segmented_confidence_rewards(
    algorithm,
    accuracy,
    group_targets,
    confidences,
    *,
    reward_name,
    reward_weight,
    group_correct_weight,
    precomputed_rewards=None,
    confidence_legal=None,
):
    if algorithm == "redoubt":
        if precomputed_rewards is None:
            raise ValueError("ReDoubt requires per-rollout clipped log scores")
        return accuracy, precomputed_rewards * reward_weight
    if algorithm == "dcpo":
        if not 0.0 <= group_correct_weight <= 1.0:
            raise ValueError(
                "group_correct_weight must be between 0 and 1, "
                f"got {group_correct_weight}"
            )
        targets = (
            group_correct_weight * group_targets
            + (1.0 - group_correct_weight) * accuracy
        )
        rewards = 1.0 - torch.square(confidences - targets)
        if confidence_legal is not None:
            rewards = rewards * confidence_legal
        return targets, rewards * reward_weight

    targets = group_targets
    rewards = torch.tensor(
        [
            confidence_reward(
                reward_name,
                target=float(target.item()),
                confidence=float(confidence.item()),
                shifted_brier=True,
            )
            for target, confidence in zip(targets, confidences)
        ],
        dtype=torch.float32,
        device=accuracy.device,
    )
    if confidence_legal is not None:
        rewards = rewards * confidence_legal
    return targets, rewards * reward_weight


def compute_segmented_advantage(data, vc_config):
    algorithm = str(vc_config.get("algorithm", "coca")).lower()
    if algorithm not in SEGMENTED_ADV_ESTIMATORS:
        raise ValueError(f"Unsupported segmented advantage estimator: {algorithm}")
    reward_name = str(vc_config.get("confidence_reward") or "brier")
    reward_weight = float(vc_config.get("confidence_reward_weight", 1.0))
    scale_rewards = bool(vc_config.scale_rewards)
    group_correct_weight = float(vc_config.get("group_correct_weight", 0.5))
    device = data.batch["response_mask"].device
    group_ids = list(data.non_tensor_batch["uid"])
    accuracy = torch.as_tensor(
        data.non_tensor_batch["accuracy"].astype(float), dtype=torch.float32, device=device
    )
    group_targets = group_success_rates(accuracy, group_ids)
    confidences = torch.as_tensor(
        data.non_tensor_batch["confidence"].astype(float), dtype=torch.float32, device=device
    )
    precomputed_rewards = None
    confidence_legal = torch.as_tensor(
        data.non_tensor_batch["confidence_legal"].astype(float),
        dtype=torch.float32,
        device=device,
    )
    if algorithm == "redoubt":
        precomputed_rewards = torch.as_tensor(
            data.non_tensor_batch[reward_name].astype(float), dtype=torch.float32, device=device
        )
    confidence_targets, confidence_rewards = compute_segmented_confidence_rewards(
        algorithm,
        accuracy,
        group_targets,
        confidences,
        reward_name=reward_name,
        reward_weight=reward_weight,
        group_correct_weight=group_correct_weight,
        precomputed_rewards=precomputed_rewards,
        confidence_legal=confidence_legal,
    )
    answer_rewards = torch.as_tensor(
        data.non_tensor_batch["answer_reward"].astype(float), dtype=torch.float32, device=device
    )
    starts = torch.as_tensor(
        data.non_tensor_batch["confidence_start"].astype(int), dtype=torch.long, device=device
    )
    positions = torch.arange(data.batch["response_mask"].shape[1], device=device).unsqueeze(0)
    if algorithm == "coca":
        ends = torch.as_tensor(
            data.non_tensor_batch["confidence_end"].astype(int), dtype=torch.long, device=device
        )
        confidence_mask = (
            (positions >= starts.unsqueeze(1)) & (positions < ends.unsqueeze(1))
        ).to(data.batch["response_mask"].dtype)
    else:
        confidence_mask = (positions >= starts.unsqueeze(1)).to(data.batch["response_mask"].dtype)
    advantages, returns = compute_coca_token_advantages(
        answer_rewards,
        confidence_rewards,
        data.batch["response_mask"],
        confidence_mask,
        group_ids,
        scale_rewards=scale_rewards,
    )
    data.batch["advantages"] = advantages
    data.batch["returns"] = returns
    data.non_tensor_batch["group_success_rate"] = group_targets.detach().cpu().numpy()
    data.non_tensor_batch["confidence_target"] = confidence_targets.detach().cpu().numpy()
    data.non_tensor_batch["segmented_confidence_reward"] = confidence_rewards.detach().cpu().numpy()
    return data
