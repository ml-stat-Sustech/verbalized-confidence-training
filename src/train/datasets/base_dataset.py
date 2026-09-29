import os

from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict, load_dataset, load_from_disk

from src.common.dataset_processing import process_dataset


class BaseDataset:
    def __init__(self, config):
        self.args = config
        dataset = self.load_dataset()
        dataset = self.reformat(dataset)
        self.dataset = process_dataset(dataset, self.args)

    def load_dataset(self):
        dataset_path = self.args.dataset_name
        if os.path.isdir(dataset_path) and os.path.exists(os.path.join(dataset_path, "dataset_dict.json")):
            return load_from_disk(dataset_path)

        load_args = [dataset_path]
        if self.args.dataset_config is not None:
            load_args.append(self.args.dataset_config)
        dataset = load_dataset(*load_args)

        if isinstance(dataset, (DatasetDict, IterableDatasetDict)):
            return dataset
        if isinstance(dataset, (Dataset, IterableDataset)):
            return DatasetDict({self.args.dataset_train_split: dataset})
        raise TypeError(f"Unsupported dataset type: {type(dataset)}")

    def reformat(self, dataset):
        return dataset
