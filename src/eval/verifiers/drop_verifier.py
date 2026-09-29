import re
import string

def _normalize(text):
    tokens = re.split(r"\s+|-", str(text).lower())
    normalized = []
    for token in tokens:
        if not token:
            continue
        try:
            token = str(float(token))
        except ValueError:
            token = "".join(character for character in token if character not in string.punctuation)
        if token not in {"", "a", "an", "the"}:
            normalized.append(token)
    return " ".join(normalized)


def drop_metrics(prediction, references):
    predicted = set(_normalize(prediction).split())
    exact = 0.0
    best_f1 = 0.0
    for reference in references:
        normalized_reference = _normalize(reference)
        expected = set(normalized_reference.split())
        exact = max(exact, float(_normalize(prediction) == normalized_reference))
        expected_numbers = {token for token in expected if _is_number(token)}
        predicted_numbers = {token for token in predicted if _is_number(token)}
        overlap = len(predicted & expected) if not expected_numbers or expected_numbers & predicted_numbers else 0
        if predicted and expected and overlap:
            best_f1 = max(best_f1, 2 * overlap / (len(predicted) + len(expected)))
    return exact, round(best_f1, 2)


def _is_number(text):
    try:
        float(text)
        return True
    except ValueError:
        return False


def drop_verifier(local_dataset, config, **kwargs):
    is_correct = []
    answer_f1 = []
    for row in local_dataset:
        references = row["answer"]
        row_correct = []
        row_f1 = []
        for prediction in row["predictions"]:
            exact, f1 = drop_metrics(prediction, references)
            row_correct.append(int(exact))
            row_f1.append(f1)
        is_correct.append(row_correct)
        answer_f1.append(row_f1)
    return {"is_correct": is_correct, "answer_f1": answer_f1}
