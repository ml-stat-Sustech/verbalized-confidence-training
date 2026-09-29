from src.eval.datasets.base_dataset import BaseDataset


class MathArenaAIME(BaseDataset):
    def reformat(self, dataset):
        dataset = dataset.map(lambda example: {"answer_text": str(example["answer"])})
        dataset = dataset.remove_columns("answer")
        return self.finalize_dataset(
            dataset,
            question_key="problem",
            answer_key="answer_text",
            id_key="problem_idx",
        )
