import re

from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase

from src.train.algorithms.segmented import SEGMENTED_ADV_ESTIMATORS
from src.train.rewards.scoring import (
    clipped_log_score,
    confidence_reward,
    score_completion,
)


def _token_span_mask(tokenizer, token_ids, text: str, span) -> list[int]:
    token_ids = [int(token_id) for token_id in token_ids]
    mask = [0] * len(token_ids)
    if span is None:
        return mask

    span_start, span_end = span
    special_ids = set(getattr(tokenizer, "all_special_ids", []))
    try:
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        actual_positions = [
            index for index, token_id in enumerate(token_ids) if token_id not in special_ids
        ]
        actual_ids = [token_ids[index] for index in actual_positions]
        if encoded["input_ids"] == actual_ids:
            for token_position, (token_start, token_end) in zip(
                actual_positions,
                encoded["offset_mapping"],
                strict=True,
            ):
                if token_end > span_start and token_start < span_end:
                    mask[token_position] = 1
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
            mask[index] = 1
        token_start = token_end
    return mask


def build_confidence_mask(tokenizer, token_ids, completion: str) -> list[int]:
    answer_matches = list(re.finditer(r"<answer>.*?</answer>", completion, re.DOTALL))
    if answer_matches:
        answer_end = answer_matches[-1].end()
        return _token_span_mask(tokenizer, token_ids, completion, (answer_end, len(completion)))

    confidence_matches = list(re.finditer(r"<confidence>.*?</confidence>", completion, re.DOTALL))
    span = None if not confidence_matches else confidence_matches[-1].span()
    return _token_span_mask(tokenizer, token_ids, completion, span)


def confidence_start_index(tokenizer, token_ids, completion: str) -> int:
    mask = build_confidence_mask(tokenizer, token_ids, completion)
    return next((index for index, value in enumerate(mask) if value), len(mask))


def confidence_tag_span_indices(tokenizer, token_ids, completion: str) -> tuple[int, int]:
    """Return token bounds for the final confidence tag only."""
    matches = list(re.finditer(r"<confidence>.*?</confidence>", completion, re.DOTALL))
    if not matches:
        return len(token_ids), len(token_ids)
    mask = _token_span_mask(tokenizer, token_ids, completion, matches[-1].span())
    marked = [index for index, value in enumerate(mask) if value]
    if not marked:
        return len(token_ids), len(token_ids)
    return marked[0], marked[-1] + 1


class VerbalizedRewardManager(RewardManagerBase):
    """Compute scalar rewards and metadata for the verl RL trainers."""

    def __init__(self, config, tokenizer, compute_score=None, **kwargs):
        super().__init__(config, tokenizer, compute_score)
        vc_config = config.vc
        self.algorithm = str(vc_config.algorithm).lower()
        self.optimization_rewards = dict(vc_config.optimization_rewards)
        configured_confidence_reward = vc_config.get("confidence_reward", None)
        self.confidence_reward_name = str(
            configured_confidence_reward
            or ("log_score" if self.algorithm == "redoubt" else "brier")
        )
        self.monitoring_rewards = list(vc_config.monitoring_rewards)
        self.format_pattern = str(vc_config.format_pattern)
        self.log_score_clip = float(vc_config.get("log_score_clip", 0.01))
        if self.algorithm not in {
            "rlvr",
            "rlcr",
            "redoubt",
            "coca",
            "dcpo",
        }:
            raise ValueError(f"Unsupported verl algorithm: {self.algorithm}")

    def _component_values(self, scores, calibration_reward):
        confidence = 0.0 if scores.confidence is None else scores.confidence
        return {
            "format": scores.format,
            "accuracy": scores.accuracy,
            self.confidence_reward_name: calibration_reward,
            "mean_confidence": confidence,
            "confidence_one_or_zero": float(abs(confidence) < 0.01 or abs(confidence - 1.0) < 0.01),
        }

    async def run_single(self, data):
        if len(data) != 1:
            raise ValueError("VerbalizedRewardManager.run_single expects one rollout")
        item = data[0]
        response_ids = item.batch["responses"]
        valid_length = int(item.batch["attention_mask"][-response_ids.shape[-1] :].sum().item())
        valid_ids = response_ids[:valid_length]
        completion = self.tokenizer.decode(valid_ids, skip_special_tokens=True)
        ground_truth = item.non_tensor_batch["reward_model"]["ground_truth"]
        source = str(item.non_tensor_batch.get("data_source", ""))

        scores = score_completion(
            completion,
            ground_truth,
            source,
            self.format_pattern,
            require_format_for_accuracy=self.optimization_rewards.get("format", 0.0) > 0.0,
        )
        reward_confidence = scores.confidence
        if scores.format == 0.0:
            reward_confidence = None
        if self.algorithm == "redoubt":
            calibration_reward = (
                clipped_log_score(
                    scores.accuracy,
                    reward_confidence,
                    clip=self.log_score_clip,
                )
                if reward_confidence is not None
                else clipped_log_score(1.0, 0.0, clip=self.log_score_clip)
            )
        else:
            calibration_reward = confidence_reward(
                self.confidence_reward_name,
                target=scores.accuracy,
                confidence=reward_confidence,
                shifted_brier=self.algorithm != "rlvr",
            )
        components = self._component_values(scores, calibration_reward)

        if self.algorithm in {"coca", "dcpo"}:
            reward_score = (
                self.optimization_rewards.get("format", 0.0) * scores.format
                + self.optimization_rewards.get("accuracy", 0.0) * scores.accuracy
            )
        elif self.algorithm == "redoubt":
            reward_score = calibration_reward
        else:
            reward_score = sum(weight * components[name] for name, weight in self.optimization_rewards.items())

        extra_info = {
            "format": scores.format,
            "accuracy": scores.accuracy,
            "confidence": 0.0 if scores.confidence is None else scores.confidence,
            "confidence_legal": float(scores.format > 0.0 and scores.confidence is not None),
            "brier_score": (scores.accuracy - (0.0 if scores.confidence is None else scores.confidence)) ** 2,
            self.confidence_reward_name: calibration_reward,
            "answer_reward": 0.0 if self.algorithm == "redoubt" else reward_score,
        }
        if self.algorithm in SEGMENTED_ADV_ESTIMATORS:
            if self.algorithm == "coca":
                extra_info["confidence_start"], extra_info["confidence_end"] = confidence_tag_span_indices(
                    self.tokenizer,
                    valid_ids,
                    completion,
                )
            else:
                extra_info["confidence_start"] = confidence_start_index(self.tokenizer, valid_ids, completion)
        for name in self.monitoring_rewards:
            extra_info[name] = components[name]
        return {"reward_score": reward_score, "reward_extra_info": extra_info}
