"""Deterministic graders adapted from LiveBench's reasoning evaluators."""

import itertools
import re


def _last_boxed_only_string(text: str) -> str | None:
    index = text.rfind(r"\boxed")
    if r"\boxed " in text:
        return r"\boxed " + text.split(r"\boxed ")[-1].split("$")[0]
    if index < 0:
        index = text.rfind(r"\fbox")
        if index < 0:
            return None

    open_braces = 0
    for position in range(index, len(text)):
        if text[position] == "{":
            open_braces += 1
        elif text[position] == "}":
            open_braces -= 1
            if open_braces == 0:
                return text[index : position + 1].replace("$", "").replace("fbox", "boxed")
    return None


def _remove_boxed(text: str) -> str:
    if text.startswith(r"\boxed "):
        return text[len(r"\boxed ") :]
    if not text.startswith(r"\boxed{") or not text.endswith("}"):
        raise ValueError(f"Invalid boxed answer: {text}")
    return text[len(r"\boxed{") : -1]


def spatial_score(ground_truth: str, prediction: str) -> float:
    if prediction == ground_truth:
        return 1.0

    word_to_number = {
        "zero": "0",
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "10",
        "eleven": "11",
        "twelve": "12",
        "thirteen": "13",
        "fourteen": "14",
        "fifteen": "15",
        "sixteen": "16",
        "seventeen": "17",
        "eighteen": "18",
        "nineteen": "19",
        "twenty": "20",
    }
    expected = ground_truth.strip().lower()
    bold_words = re.findall(r"\*\*([^*]+)\*\*", prediction)
    for word in (value.strip().lower() for value in bold_words[-3:]):
        if word == expected or word_to_number.get(word) == expected:
            return 1.0
        for shape in ("tetrahedra", "tetrahedron", "triangle", "square"):
            if expected == shape and shape in word and len(word) < 2 * len(shape) + 5:
                return 1.0

    boxed = _last_boxed_only_string(prediction.replace(r"\fbox{", r"\boxed{"))
    if boxed is not None:
        parsed = _remove_boxed(boxed)
        for wrapper in (r"\textbf{", r"\mathbf{", r"\text{"):
            parsed = parsed.replace(wrapper, "")
        if parsed.replace("}", "") == ground_truth:
            return 1.0
    return 0.0


def web_of_lies_score(ground_truth: str, prediction: str) -> float:
    solution_matches = re.findall(r"<solution>(.*?)</solution>", prediction)
    if not solution_matches:
        solution_matches = re.findall(r"</solution>(.*?)</solution>", prediction)
    parsed_answer = solution_matches[-1] if solution_matches else None

    if parsed_answer is None:
        bold_words = re.findall(r"\*\*(.*?)\*\*", prediction)
        normalized_words = [
            word.lower().strip().replace(",", "").replace(".", "")[: max(len(word), 3)]
            for match in bold_words
            for word in match.split()
        ]
        answers = []
        for word in reversed(normalized_words):
            if word in {"yes", "no", "unknown"}:
                answers.insert(0, word)
                if len(answers) == 3:
                    break
        if answers:
            parsed_answer = ", ".join(answers)

    if parsed_answer is None or parsed_answer.strip() == "":
        normalized = prediction.replace(r"\boxed{\textbf{", r"\boxed{")
        normalized = normalized.replace(r"\fbox{", r"\boxed{").replace(r"\textbf{", r"\boxed{")
        boxed = _last_boxed_only_string(normalized)
        if boxed is not None:
            parsed_answer = _remove_boxed(boxed)

    if parsed_answer is None:
        final_combination = None
        final_index = -1
        for combination in itertools.product(("yes", "no", "unknown"), repeat=3):
            candidate = ", ".join(combination)
            index = prediction.lower().find(candidate)
            if index > final_index:
                final_combination = combination
                final_index = index
        if final_combination is not None:
            parsed_answer = ", ".join(final_combination)

    if not parsed_answer:
        return 0.0
    expected = ground_truth.lower()
    if parsed_answer == expected:
        return 1.0
    label_count = sum(parsed_answer.count(label) for label in ("yes", "no", "unknown"))
    return float(label_count == 3 and expected in parsed_answer)


def _old_zebra_score(ground_truth: str, prediction: str) -> float:
    if prediction.strip().lower() == ground_truth.strip().lower():
        return 1.0

    number_to_word = {
        "1": "one",
        "2": "two",
        "3": "three",
        "4": "four",
        "5": "five",
        "6": "six",
        "7": "seven",
        "8": "eight",
        "9": "nine",
    }
    bold_words = re.findall(r"\*\*\*(\w+)\*\*\*", prediction)
    words = re.findall(r"\b\w+\b", prediction)
    answer = bold_words[-1] if bold_words else (words[-1] if words else "")
    normalized = answer.lower()
    expected = ground_truth.lower()
    return float(
        normalized == expected
        or number_to_word.get(answer, "").lower() == expected
        or normalized + " movies" == expected
    )


def zebra_puzzle_score(ground_truth: str, prediction: str, release_date: str) -> float:
    if release_date < "2024-11-25":
        return _old_zebra_score(ground_truth, prediction)

    expected_answers = ground_truth.split(",")
    solution_matches = re.findall(r"<solution>(.*?)</solution>", prediction)
    if not solution_matches:
        solution_matches = re.findall(r"</solution>(.*?)</solution>", prediction)
    if not solution_matches:
        normalized = prediction.replace(r"\fbox{", r"\boxed{")
        boxed = _last_boxed_only_string(normalized)
        if boxed is not None:
            answer = _remove_boxed(boxed).replace(r"\text{", "").replace("}", "").replace("\\", "")
            solution_matches.append(answer)
    if not solution_matches:
        last_line = prediction.strip().split("\n")[-1]
        if last_line.count(",") == len(expected_answers) - 1:
            solution_matches.append(last_line)
    if not solution_matches:
        return 0.0

    if len(solution_matches) > 1:
        answer_parts = []
        for match in solution_matches:
            answer_parts.extend(match.split(","))
        predicted_answers = answer_parts[-len(expected_answers) :]
    else:
        predicted_answers = solution_matches[-1].split(",")

    num_correct = 0
    for expected, predicted in zip(expected_answers, predicted_answers):
        expected = expected.strip().lower().replace("-", " ")
        predicted = predicted.strip().lower().replace("-", " ").replace("position", "")
        if expected == predicted or expected in predicted:
            num_correct += 1
    fraction_correct = num_correct / len(expected_answers)
    return (float(num_correct == len(expected_answers)) + fraction_correct) / 2


def evaluate_livebench_reasoning(
    task: str,
    ground_truth: str,
    prediction: str,
    release_date: str,
) -> float:
    if task == "spatial":
        return spatial_score(ground_truth, prediction)
    if task == "web_of_lies_v2":
        return web_of_lies_score(ground_truth, prediction)
    if task == "zebra_puzzle":
        return zebra_puzzle_score(ground_truth, prediction, release_date)
    raise ValueError(f"Unsupported LiveBench reasoning task: {task}")


def livebench_reasoning_verifier(local_dataset, config, **kwargs):
    is_correct = []
    task_scores = []
    for row in local_dataset:
        scores = [
            evaluate_livebench_reasoning(
                row["task"],
                str(row["answer"]),
                str(prediction),
                row["livebench_release_date"],
            )
            for prediction in row["predictions"]
        ]
        task_scores.append(scores)
        is_correct.append([int(score == 1.0) for score in scores])
    return {"is_correct": is_correct, "task_score": task_scores}
