import json

from src.eval.datasets.base_dataset import BaseDataset


class NQOpen(BaseDataset):
    def reformat(self, dataset):
        return self.finalize_dataset(dataset, question_key="question", answer_key="answer")


class PopQA(BaseDataset):
    def reformat(self, dataset):
        def mapping(example):
            answers = example["possible_answers"]
            if isinstance(answers, str):
                answers = json.loads(answers)
            return {
                "id": example["id"],
                "question": example["question"],
                "answer": answers,
            }

        return dataset.map(mapping, remove_columns=dataset.column_names)


class WebQuestions(BaseDataset):
    def reformat(self, dataset):
        if not self.args.include_multi_choice:
            dataset = dataset.filter(lambda example: len(example["answers"]) == 1)
        return self.finalize_dataset(dataset, question_key="question", answer_key="answers")
