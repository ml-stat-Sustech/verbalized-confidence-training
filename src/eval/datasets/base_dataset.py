from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict, load_dataset


class BaseDataset:
    def __init__(self, config):
        self.args = config
        self.dataset = self.reformat(self.load_dataset())
        if self.args.sample_size is not None:
            if self.args.sample_size <= 0:
                raise ValueError(f"sample_size must be positive, got {self.args.sample_size}.")
            sample_size = min(self.args.sample_size, len(self.dataset))
            self.dataset = self.dataset.shuffle(seed=self.args.seed).select(range(sample_size))

    def load_dataset(self):
        load_args = [self.args.dataset_name]
        if self.args.dataset_config is not None:
            load_args.append(self.args.dataset_config)
        dataset = load_dataset(*load_args)

        if isinstance(dataset, (DatasetDict, IterableDatasetDict)):
            return dataset[self.args.split]
        if isinstance(dataset, (Dataset, IterableDataset)):
            return dataset
        raise TypeError(f"Unsupported dataset type: {type(dataset)}")

    def reformat(self, dataset):
        return dataset

    def finalize_dataset(self, dataset, question_key: str, answer_key: str, id_key: str | None = None):
        def mapping(example, idx):
            question = example[question_key]
            answer = example[answer_key]
            sample_id = example[id_key] if id_key is not None and id_key in example else idx + 1
            return {"id": sample_id, "question": question, "answer": answer}

        dataset = dataset.map(mapping, with_indices=True)
        keep_columns = [column for column in ["id", "question", "answer"] if column in dataset.column_names]
        return dataset.remove_columns([column for column in dataset.column_names if column not in keep_columns])
