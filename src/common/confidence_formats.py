from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


CONFIDENCE_NONE = "none"
CONFIDENCE_PROBABILITY = "probability"
CONFIDENCE_DIGIT = "digit"
CONFIDENCE_LINGUISTIC = "linguistic"

VALID_CONFIDENCE_FORMATS = {
    CONFIDENCE_NONE,
    CONFIDENCE_PROBABILITY,
    CONFIDENCE_DIGIT,
    CONFIDENCE_LINGUISTIC,
}

LINGUISTIC_CONFIDENCE_VALUES = {
    "low": 0.25,
    "medium": 0.5,
    "high": 0.75,
}


@dataclass(frozen=True)
class ParsedConfidence:
    is_legal: int
    value: float
    raw_text: str


def normalize_confidence_format(confidence_format: str | None) -> str:
    if confidence_format is None:
        return CONFIDENCE_NONE
    key = str(confidence_format).strip()
    if key not in VALID_CONFIDENCE_FORMATS:
        valid = sorted(VALID_CONFIDENCE_FORMATS)
        raise ValueError(f"Invalid confidence_format: {confidence_format}. Valid formats: {valid}")
    return key


def has_confidence(confidence_format: str | None) -> bool:
    return normalize_confidence_format(confidence_format) != CONFIDENCE_NONE


def extract_last_tag_value(content: str, tag_name: str) -> str:
    pattern = rf"<{tag_name}>(.*?)</{tag_name}>"
    matches = re.findall(pattern, content, re.DOTALL | re.MULTILINE)
    return matches[-1] if matches else ""


def extract_last_confidence_text(content: str) -> str:
    return extract_last_tag_value(content, "confidence").strip()


def _parse_probability_confidence(confidence_text: str, allow_percent: bool) -> Optional[float]:
    if confidence_text == "":
        return None

    def normalize_number(text: str) -> Optional[float]:
        try:
            confidence = round(float(text), 6)
        except Exception:
            return None
        if 0.0 <= confidence <= 1.0:
            return confidence
        if allow_percent and 1.0 < confidence <= 100.0:
            return confidence / 100.0
        return None

    normalized = normalize_number(confidence_text)
    if normalized is not None:
        return normalized

    first_number = re.search(r"-?\d+(?:\.\d+)?", confidence_text)
    if first_number:
        return normalize_number(first_number.group())
    return None


def parse_confidence_text(
    confidence_text: str,
    confidence_format: str | None = CONFIDENCE_PROBABILITY,
    allow_percent: bool = False,
) -> ParsedConfidence:
    confidence_format = normalize_confidence_format(confidence_format)
    confidence_text = "" if confidence_text is None else str(confidence_text).strip()
    if confidence_format == CONFIDENCE_NONE or confidence_text == "":
        return ParsedConfidence(0, 0.0, confidence_text)

    if confidence_format == CONFIDENCE_PROBABILITY:
        confidence = _parse_probability_confidence(confidence_text, allow_percent=allow_percent)
        if confidence is None:
            return ParsedConfidence(0, 0.0, confidence_text)
        return ParsedConfidence(1, confidence, confidence_text)

    if confidence_format == CONFIDENCE_DIGIT:
        digit_match = re.fullmatch(r"[0-9]", confidence_text)
        if digit_match is None:
            return ParsedConfidence(0, 0.0, confidence_text)
        return ParsedConfidence(1, int(confidence_text) / 9.0, confidence_text)

    if confidence_format == CONFIDENCE_LINGUISTIC:
        label = confidence_text.lower()
        if label not in LINGUISTIC_CONFIDENCE_VALUES:
            return ParsedConfidence(0, 0.0, confidence_text)
        return ParsedConfidence(1, LINGUISTIC_CONFIDENCE_VALUES[label], confidence_text)

    raise ValueError(f"Invalid confidence_format: {confidence_format}")


def parse_confidence_from_text(
    content: str,
    confidence_format: str | None = CONFIDENCE_PROBABILITY,
    allow_percent: bool = False,
) -> ParsedConfidence:
    return parse_confidence_text(
        extract_last_confidence_text(content),
        confidence_format=confidence_format,
        allow_percent=allow_percent,
    )


def extract_normalized_confidence_value(
    content: str,
    confidence_format: str | None = CONFIDENCE_PROBABILITY,
    allow_percent: bool = False,
) -> Optional[float]:
    parsed = parse_confidence_from_text(
        content,
        confidence_format=confidence_format,
        allow_percent=allow_percent,
    )
    return parsed.value if parsed.is_legal else None


def confidence_value_regex(confidence_format: str | None) -> str:
    confidence_format = normalize_confidence_format(confidence_format)
    if confidence_format == CONFIDENCE_DIGIT:
        return r"\s*[0-9]\s*"
    if confidence_format == CONFIDENCE_LINGUISTIC:
        return r"\s*(?:low|medium|high)\s*"
    if confidence_format == CONFIDENCE_PROBABILITY:
        return r"\s*(?:0\.\d{2}|1\.00)\s*"
    return ""


def confidence_fill_instruction(confidence_format: str | None) -> tuple[str, int]:
    confidence_format = normalize_confidence_format(confidence_format)
    if confidence_format == CONFIDENCE_DIGIT:
        return (
            "Thinking time ended \n\n. My verbalized confidence in my answer as a single digit from 0 to 9 is equal to ",
            3,
        )
    if confidence_format == CONFIDENCE_LINGUISTIC:
        return (
            "Thinking time ended \n\n. My verbalized confidence in my answer as one of low, medium, or high is equal to ",
            5,
        )
    if confidence_format == CONFIDENCE_PROBABILITY:
        return (
            "Thinking time ended \n\n. My verbalized confidence in my answer as a decimal between 0 and 1 with exactly two decimal places is equal to ",
            20,
        )
    raise ValueError("No confidence fill instruction for answer-only format.")
