from datasets import load_dataset

from src.eval.datasets.base_dataset import BaseDataset
from src.train.datasets.deep_scale_r import prepare_deep_scale_r_dataset


class DeepScaleR(BaseDataset):
    def load_dataset(self):
        dataset = load_dataset(self.args.dataset_name)
        return prepare_deep_scale_r_dataset(dataset)[self.args.split]

    def reformat(self, dataset):
        return self.finalize_dataset(dataset, question_key="question", answer_key="answer", id_key="id")
