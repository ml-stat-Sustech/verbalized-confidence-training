import math
import re
from dataclasses import dataclass

from src.common.answer_verification import exact_match_score, math_answers_equivalent
from src.common.confidence_formats import extract_normalized_confidence_value
from src.train.rewards.reward_functions import (
    confidence_format_for_pattern,
    format_reward,
)


@dataclass(frozen=True)
class CompletionScores:
    format: float
    accuracy: float
    confidence: float | None


def _completion_payload(completion: str):
    return [[{"role": "assistant", "content": completion}]]


def score_format(completion: str, format_pattern: str) -> float:
    return float(format_reward(format_pattern, _completion_payload(completion))[0])


def score_accuracy(completion: str, ground_truth, source: str | None) -> float:
    matches = re.findall(r"<answer>(.*?)</answer>", completion, re.DOTALL | re.MULTILINE)
    answer = matches[-1].strip() if matches else ""
    if not answer:
        return 0.0
    return float(exact_match_score(answer, ground_truth) or math_answers_equivalent(answer, ground_truth))


def score_completion(
    completion: str,
    ground_truth,
    source: str | None,
    format_pattern: str,
    *,
    require_format_for_accuracy: bool,
) -> CompletionScores:
    format_score = score_format(completion, format_pattern)
    accuracy = score_accuracy(completion, ground_truth, source)
    if require_format_for_accuracy and format_score == 0.0:
        accuracy = 0.0
    confidence_format = confidence_format_for_pattern(format_pattern)
    confidence = extract_normalized_confidence_value(completion, confidence_format=confidence_format)
    return CompletionScores(format=format_score, accuracy=accuracy, confidence=confidence)


def confidence_reward(
    reward_name: str,
    *,
    target: float,
    confidence: float | None,
    shifted_brier: bool,
) -> float:
    if confidence is None:
        return 0.0
    if reward_name == "brier":
        value = -((float(target) - confidence) ** 2)
        return value + 1.0 if shifted_brier else value
    raise ValueError(f"Unsupported confidence reward: {reward_name}")


def clipped_log_score(target: float, confidence: float, clip: float = 0.01) -> float:
    if not 0.0 < clip < 1.0:
        raise ValueError(f"log-score clip must be between 0 and 1, got {clip}")
    confidence = min(max(float(confidence), 0.0), 1.0)
    return float(target) * math.log(max(confidence, clip)) + (
        1.0 - float(target)
    ) * math.log(max(1.0 - confidence, clip))
