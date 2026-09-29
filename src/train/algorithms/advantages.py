from collections import defaultdict
from collections.abc import Sequence

import torch


def group_success_rates(correctness: torch.Tensor, group_ids: Sequence[object]) -> torch.Tensor:
    if correctness.ndim != 1 or len(correctness) != len(group_ids):
        raise ValueError("correctness and group_ids must describe the same flat batch")

    grouped = defaultdict(list)
    for index, group_id in enumerate(group_ids):
        grouped[group_id].append(correctness[index])

    means = {group_id: torch.stack(values).mean() for group_id, values in grouped.items()}
    return torch.stack([means[group_id] for group_id in group_ids])


def normalize_group_rewards(
    rewards: torch.Tensor,
    group_ids: Sequence[object],
    *,
    scale_rewards: bool,
    epsilon: float = 1e-4,
) -> torch.Tensor:
    if rewards.ndim != 1 or len(rewards) != len(group_ids):
        raise ValueError("rewards and group_ids must describe the same flat batch")

    grouped = defaultdict(list)
    for index, group_id in enumerate(group_ids):
        grouped[group_id].append(rewards[index])

    stats = {}
    for group_id, values in grouped.items():
        values_tensor = torch.stack(values)
        mean = values_tensor.mean()
        std = values_tensor.std() if len(values) > 1 else torch.ones_like(mean)
        stats[group_id] = (mean, std)

    advantages = []
    for reward, group_id in zip(rewards, group_ids):
        mean, std = stats[group_id]
        advantage = reward - mean
        if scale_rewards:
            advantage = advantage / (std + epsilon)
        advantages.append(advantage)
    return torch.stack(advantages)


def compute_coca_token_advantages(
    answer_rewards: torch.Tensor,
    confidence_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    confidence_mask: torch.Tensor,
    group_ids: Sequence[object],
    *,
    scale_rewards: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if response_mask.shape != confidence_mask.shape:
        raise ValueError("response_mask and confidence_mask must have the same shape")
    if response_mask.shape[0] != answer_rewards.shape[0] or answer_rewards.shape != confidence_rewards.shape:
        raise ValueError("reward vectors must match the response batch size")

    response_mask = response_mask.to(dtype=answer_rewards.dtype)
    confidence_mask = confidence_mask.to(device=response_mask.device, dtype=response_mask.dtype) * response_mask
    answer_mask = (response_mask - confidence_mask).clamp(min=0)

    answer_advantages = normalize_group_rewards(
        answer_rewards,
        group_ids,
        scale_rewards=scale_rewards,
    )
    confidence_advantages = normalize_group_rewards(
        confidence_rewards,
        group_ids,
        scale_rewards=scale_rewards,
    )
    advantages = (
        answer_advantages.unsqueeze(1) * answer_mask
        + confidence_advantages.unsqueeze(1) * confidence_mask
    )
    return advantages, advantages.clone()
