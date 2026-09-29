from src.eval.datasets.base_dataset import BaseDataset


class MinervaMath(BaseDataset):
    def reformat(self, dataset):
        return self.finalize_dataset(dataset, question_key="question", answer_key="answer")
