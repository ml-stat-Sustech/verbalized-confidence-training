from src.common.confidence_formats import (
    CONFIDENCE_DIGIT,
    CONFIDENCE_LINGUISTIC,
    CONFIDENCE_NONE,
    CONFIDENCE_PROBABILITY,
    normalize_confidence_format,
)


SYSTEM_PROMPTS = {
    "think_answer": (
        "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
        "first thinks about the reasoning process in the mind and then provides the user with the answer. The response "
        "must strictly follow this format: <think> reasoning process here </think> <answer> final short answer only "
        "</answer>. The <answer> tag must contain only the final answer string needed for exact-match evaluation, not a "
        "full sentence, explanation, or reasoning."
    ),
    "think_answer_confidence_probability": (
        "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
        "first thinks about the reasoning process in the mind and analyzes its confidence about the solution and then "
        "provides the user with the final answer as well as its confidence level. The confidence level indicates how "
        "certain the Assistant is about its answer, expressed as a decimal between 0 and 1 with exactly two decimal "
        "places, enclosed within <confidence> </confidence> tags. The response must strictly follow this format: "
        "<think> reasoning process here </think> <answer> final short answer only </answer> "
        "<confidence> 0.xx </confidence>. The <answer> tag must contain only the final answer string needed for "
        "exact-match evaluation, not a full sentence, explanation, or reasoning."
    ),
    "think_answer_confidence_digit": (
        "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
        "first thinks about the reasoning process in the mind and analyzes its confidence about the solution and then "
        "provides the user with the final answer as well as its confidence score. The confidence score indicates how "
        "certain the Assistant is about its answer, expressed as a single integer from 0 to 9, enclosed within "
        "<confidence> </confidence> tags. A score of 0 means very low confidence and the answer is likely incorrect. "
        "A score of 9 means very high confidence and the answer is very likely correct. The response must strictly "
        "follow this format: <think> reasoning process here </think> <answer> final short answer only </answer> "
        "<confidence>X</confidence>, where X is one of 0, 1, 2, 3, 4, 5, 6, 7, 8, 9. The <answer> tag must contain "
        "only the final answer string needed for exact-match evaluation, not a full sentence, explanation, or reasoning."
    ),
    "think_answer_confidence_linguistic": (
        "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
        "first thinks about the reasoning process in the mind and analyzes its confidence about the solution and then "
        "provides the user with the final answer as well as its confidence level. The confidence level indicates how "
        "certain the Assistant is about its answer, expressed as exactly one of low, medium, or high, enclosed within "
        "<confidence> </confidence> tags. Low means the answer is likely incorrect, medium means the answer is uncertain, "
        "and high means the answer is likely correct. The response must strictly follow this format: "
        "<think> reasoning process here </think> <answer> final short answer only </answer> "
        "<confidence>label</confidence>. The <answer> tag must contain only the final answer string needed for "
        "exact-match evaluation, not a full sentence, explanation, or reasoning."
    ),
}


def get_system_prompt_name(confidence_format=None):
    confidence_format = normalize_confidence_format(confidence_format)
    if confidence_format == CONFIDENCE_NONE:
        return "think_answer"
    return f"think_answer_confidence_{confidence_format}"


def get_sys_prompt_by_name(prompt_name):
    if prompt_name not in SYSTEM_PROMPTS:
        valid_names = sorted(SYSTEM_PROMPTS)
        raise ValueError(f"Invalid system prompt name: {prompt_name}. Valid names: {valid_names}")
    return SYSTEM_PROMPTS[prompt_name]


def get_sys_prompt(confidence_format=None):
    prompt_name = get_system_prompt_name(confidence_format)
    return get_sys_prompt_by_name(prompt_name)

OUTPUT_FORMAT_PATTERNS_BY_CONFIDENCE_FORMAT = {
    CONFIDENCE_NONE: "answer",
    CONFIDENCE_PROBABILITY: "answer_confidence_probability",
    CONFIDENCE_DIGIT: "answer_confidence_digit",
    CONFIDENCE_LINGUISTIC: "answer_confidence_linguistic",
}


def get_format_pattern(confidence_format=None):
    return OUTPUT_FORMAT_PATTERNS_BY_CONFIDENCE_FORMAT[normalize_confidence_format(confidence_format)]
