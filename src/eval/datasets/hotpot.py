from src.eval.datasets.base_dataset import BaseDataset


class Hotpot(BaseDataset):
    def reformat(self, dataset):
        return self.finalize_dataset(dataset, question_key="problem", answer_key="answer", id_key="id")
