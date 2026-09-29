from src.eval.datasets.base_dataset import BaseDataset


class GSM8K(BaseDataset):
    def reformat(self, dataset):
        return self.finalize_dataset(dataset, question_key="problem", answer_key="answer")
