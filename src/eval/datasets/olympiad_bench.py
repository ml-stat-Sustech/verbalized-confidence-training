from src.eval.datasets.base_dataset import BaseDataset


class OlympiadBench(BaseDataset):
    def reformat(self, dataset):
        dataset = dataset.map(lambda example: {"answer": example["final_answer"][0]})
        return self.finalize_dataset(dataset, question_key="question", answer_key="answer", id_key="id")
