from src.eval.datasets.base_dataset import BaseDataset


class Trivia(BaseDataset):
    def reformat(self, dataset):
        return self.finalize_dataset(dataset, question_key="question", answer_key="answer", id_key="question_id")
