import re
from typing import Optional

from src.common.confidence_formats import (
    CONFIDENCE_DIGIT,
    CONFIDENCE_LINGUISTIC,
    CONFIDENCE_NONE,
    CONFIDENCE_PROBABILITY,
    confidence_value_regex,
    extract_last_tag_value,
    extract_normalized_confidence_value,
)

def _extract_last_tag_value(content, tag_name):
    return extract_last_tag_value(content, tag_name)


def extract_answer_value(content) -> str:
    return _extract_last_tag_value(content, "answer").strip()


def extract_confidence_value(content) -> Optional[float]:
    return extract_normalized_confidence_value(content, confidence_format=CONFIDENCE_PROBABILITY)


def extract_digit_confidence_value(content) -> Optional[float]:
    return extract_normalized_confidence_value(content, confidence_format=CONFIDENCE_DIGIT)


def extract_linguistic_confidence_value(content) -> Optional[float]:
    return extract_normalized_confidence_value(content, confidence_format=CONFIDENCE_LINGUISTIC)


def confidence_format_for_pattern(format_pattern):
    if format_pattern == "answer_confidence_digit":
        return CONFIDENCE_DIGIT
    if format_pattern == "answer_confidence_linguistic":
        return CONFIDENCE_LINGUISTIC
    if "confidence" in format_pattern:
        return CONFIDENCE_PROBABILITY
    return CONFIDENCE_NONE


def _confidence_extractor_for_format(format_pattern):
    confidence_format = confidence_format_for_pattern(format_pattern)
    return lambda content: extract_normalized_confidence_value(content, confidence_format=confidence_format)


def format_reward(format_pattern, completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    if format_pattern == "answer":
        pattern = r".*?</think>\s*<answer>.*?</answer>\s*\Z"
    elif format_pattern == "answer_confidence_probability":
        pattern = rf".*?</think>\s*<answer>.*?</answer>\s*<confidence>{confidence_value_regex(CONFIDENCE_PROBABILITY)}</confidence>\s*\Z"
    elif format_pattern == "answer_confidence_digit":
        pattern = rf".*?</think>\s*<answer>.*?</answer>\s*<confidence>{confidence_value_regex(CONFIDENCE_DIGIT)}</confidence>\s*\Z"
    elif format_pattern == "answer_confidence_linguistic":
        pattern = rf".*?</think>\s*<answer>.*?</answer>\s*<confidence>{confidence_value_regex(CONFIDENCE_LINGUISTIC)}</confidence>\s*\Z"
    else:
        raise ValueError(f"Invalid format pattern: {format_pattern}")

    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
    matches = [1.0 if match else 0.0 for match in matches]

    if "confidence" in format_pattern:
        extractor = _confidence_extractor_for_format(format_pattern)
        for i, match in enumerate(matches):
            if match and extractor(completion_contents[i]) is None:
                matches[i] = 0.0
    return matches
