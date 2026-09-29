from src.eval.datasets.base_dataset import BaseDataset


class AIME2024(BaseDataset):
    def reformat(self, dataset):
        return self.finalize_dataset(dataset, question_key="Problem", answer_key="Answer", id_key="ID")
