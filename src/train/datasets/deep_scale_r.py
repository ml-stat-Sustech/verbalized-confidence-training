from datasets import Dataset, DatasetDict

from src.train.datasets.base_dataset import BaseDataset


DEEP_SCALE_R_TEST_SIZE = 2000
DEEP_SCALE_R_SEED = 43


def normalize_text(text):
    return " ".join(str(text or "").split()).casefold()


def prepare_deep_scale_r_dataset(
    dataset,
    test_size=DEEP_SCALE_R_TEST_SIZE,
    seed=DEEP_SCALE_R_SEED,
):
    if isinstance(dataset, Dataset):
        dataset = DatasetDict({"train": dataset})
    if not isinstance(dataset, DatasetDict) or "train" not in dataset:
        raise ValueError("DeepScaleR dataset must contain a train split")

    def mapping(example, index):
        problem = str(example.get("problem") or "").strip()
        return {
            "id": example.get("id", index + 1),
            "problem": problem,
            "question": problem,
            "answer": str(example.get("answer") or "").strip(),
            "source": "deep_scale_r",
        }

    formatted = dataset["train"].map(mapping, with_indices=True)
    first_indices = {}
    answers_by_question = {}
    for index, example in enumerate(formatted):
        question = normalize_text(example["question"])
        answer = normalize_text(example["answer"])
        if not question or not answer:
            continue
        first_indices.setdefault(question, index)
        answers_by_question.setdefault(question, set()).add(answer)

    conflicting_questions = {
        question for question, answers in answers_by_question.items() if len(answers) > 1
    }
    keep_indices = [
        index
        for question, index in first_indices.items()
        if question not in conflicting_questions
    ]
    cleaned = formatted.select(keep_indices).shuffle(seed=seed)
    if len(cleaned) <= test_size:
        raise ValueError(
            f"DeepScaleR needs more than {test_size} clean examples, got {len(cleaned)}"
        )

    train_size = len(cleaned) - int(test_size)
    return DatasetDict(
        {
            "train": cleaned.select(range(train_size)),
            "test": cleaned.select(range(train_size, len(cleaned))),
        }
    )


class DeepScaleR(BaseDataset):
    def reformat(self, dataset):
        return prepare_deep_scale_r_dataset(dataset)
