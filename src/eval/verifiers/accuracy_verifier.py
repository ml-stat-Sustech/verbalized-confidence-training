import re

from src.common.answer_verification import exact_match_score, math_answers_equivalent, normalize_answer


def answers_equivalent(answer_a: str, answer_b: str) -> bool:
    if not answer_a or not answer_b:
        return False

    return math_answers_equivalent(answer_a, answer_b) or exact_match_score(answer_a, answer_b)


def gen_correctness_reward(completions, answer, **kwargs):
    matches = []
    for completion, expected_answer in zip(completions, answer):
        answer_text = completion[0]["content"]
        matches.append(float(answers_equivalent(answer_text, expected_answer)))
    return matches


def rule_verifier(
    local_dataset,
    config,
    format_fn="confidence_format",
    format_pattern="answer_analysis_confidence",
    **kwargs,
):
    label_dict = {"is_correct": []}
    is_correct = []
    correctness_fn = gen_correctness_reward

    for i in range(len(local_dataset)):
        correctness_list = []
        predictions = local_dataset[i]["predictions"]
        gold_answer = local_dataset[i]["answer"]

        for answer_text in predictions:
            args = {"completions": [[{"role": "assistant", "content": answer_text}]], "answer": [gold_answer]}
            actual_correctness = correctness_fn(**args)[0]
            correctness_list.append(1 if actual_correctness == 1 else 0)

        is_correct.append(correctness_list)

    label_dict["is_correct"] = is_correct
    return label_dict


def alias_verifier(local_dataset, config, **kwargs):
    is_correct = []
    for row in local_dataset:
        aliases = row["answer"] if isinstance(row["answer"], list) else [row["answer"]]
        is_correct.append(
            [
                int(any(exact_match_score(prediction, alias) for alias in aliases))
                for prediction in row["predictions"]
            ]
        )
    return {"is_correct": is_correct}


def popqa_verifier(local_dataset, config, **kwargs):
    return {
        "is_correct": [
            [int(any(str(alias).lower() in prediction.lower() for alias in row["answer"])) for prediction in row["predictions"]]
            for row in local_dataset
        ]
    }


def web_questions_verifier(local_dataset, config, **kwargs):
    is_correct = []
    answer_f1 = []
    for row in local_dataset:
        gold = {normalize_answer(answer) for answer in row["answer"]}
        row_correct = []
        row_f1 = []
        for prediction in row["predictions"]:
            parts = re.split(r"[,;\n]+|\s+(?:and|or)\s+", prediction, flags=re.IGNORECASE)
            predicted = {normalize_answer(part) for part in parts if normalize_answer(part)}
            overlap = len(gold & predicted)
            f1 = 0.0 if overlap == 0 else 2 * overlap / (len(gold) + len(predicted))
            row_f1.append(f1)
            row_correct.append(int(f1 == 1.0))
        is_correct.append(row_correct)
        answer_f1.append(row_f1)
    return {"is_correct": is_correct, "answer_f1": answer_f1}
