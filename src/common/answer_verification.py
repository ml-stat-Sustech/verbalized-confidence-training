import re
import string

from math_verify import parse, verify

UNIT_MACRO = re.compile(r"\\(?:si|text|mathrm|textrm|unit|operatorname)\s*\{[^{}]*\}")
SCIENTIFIC = re.compile(r"\\times\s*10\s*\^\s*\{?\s*([+-]?\d+)\s*\}?")
POWER_OF_TEN = re.compile(r"(?<![\d.])10\s*\^\s*\{?\s*([+-]?\d+)\s*\}?")
THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
ASSIGNMENT = re.compile(r"^[A-Za-z\\][^=<>]{0,30}?=\s*(?=[^=])(.+)$", re.DOTALL)
NUMBER = re.compile(r"^[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?$")
# An answer short enough to be a final answer rather than a whole solution.
ANSWER_LENGTH = 80
# Measured quantities are reported to a few digits, so they are compared within a relative
# tolerance. Integers are compared exactly.
NUMERIC_TOLERANCE = 1e-2


def _strip_boxed_answer(text):
    text = "" if text is None else str(text)
    return re.sub(r"^\s*\\(?:boxed|fbox)\{(.*)\}\s*$", r"\1", text, flags=re.DOTALL)


def normalize_answer(text):
    text = _strip_boxed_answer(text)
    text = text.lower()
    text = "".join(character for character in text if character not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def _is_final_answer(text):
    return len(text) <= ANSWER_LENGTH and "\n" not in text


def _clean_math_answer(text):
    """The answer without the wrappers that carry no mathematical content."""
    text = _strip_boxed_answer(text).strip()
    text = THOUSANDS.sub("", text)
    if _is_final_answer(text):
        text = text.strip("$").strip()
        assignment = ASSIGNMENT.match(text)
        if assignment:  # "R = expr" states the same answer as "expr"
            text = assignment.group(1)
        text = UNIT_MACRO.sub("", text)
    return text.strip().rstrip(".,;").strip()


def _as_number(text):
    """The value of an answer that is a single number, in decimal or scientific notation."""
    text = _clean_math_answer(text)
    if not _is_final_answer(text):
        return None
    text = POWER_OF_TEN.sub(r"1e\1", SCIENTIFIC.sub(r"e\1", text))
    text = re.sub(r"\\left|\\right|\\cdot|\\,|[{}$\s,]", "", text)
    return float(text) if NUMBER.match(text) else None


def _listed_items(text):
    """The comma-separated parts of an answer that lists several required values."""
    text = _strip_boxed_answer(text).strip()
    if not _is_final_answer(text):
        return [text]
    text = THOUSANDS.sub("", text)
    parts, depth, current = [], 0, ""
    for character in text:
        depth += {"{": 1, "(": 1, "[": 1, "}": -1, ")": -1, "]": -1}.get(character, 0)
        if character == "," and depth == 0:
            parts.append(current)
            current = ""
        else:
            current += character
    parts.append(current)
    parts = [part.strip().strip("$").strip() for part in parts]
    return [part for part in parts if part]


def _values_equivalent(prediction, ground_truth) -> bool:
    predicted_value, gold_value = _as_number(prediction), _as_number(ground_truth)
    if predicted_value is not None and gold_value is not None:
        # math_verify reads "2.75 \times 10^{11}" as 10, so numbers never reach its parser.
        if float(gold_value).is_integer() and float(predicted_value).is_integer():
            return predicted_value == gold_value
        return abs(predicted_value - gold_value) <= NUMERIC_TOLERANCE * max(abs(gold_value), 1e-12)

    prediction, ground_truth = _clean_math_answer(prediction), _clean_math_answer(ground_truth)
    try:
        if verify(parse(ground_truth), parse(prediction)):
            return True
    except Exception:
        pass
    return exact_match_score(prediction, ground_truth)


def math_answers_equivalent(prediction, ground_truth) -> bool:
    if prediction is None or ground_truth is None:
        return False

    gold_items = _listed_items(ground_truth)
    # A list of numbers asks for every one of them. A comma inside prose ("September 8, 2017")
    # does not, so the rule is limited to lists whose every entry is a number.
    if len(gold_items) > 1 and all(_as_number(item) is not None for item in gold_items):
        predicted_items = _listed_items(prediction)
        if len(predicted_items) != len(gold_items):
            return False
        return all(any(_values_equivalent(predicted, gold) for predicted in predicted_items)
                   for gold in gold_items)
    return _values_equivalent(prediction, ground_truth)
