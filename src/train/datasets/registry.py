from src.train.datasets.confidence_sft_data import DeepScaleRSFT
from src.train.datasets.deep_scale_r import DeepScaleR


DATASET_REGISTRY = {
    "DeepScaleRSFT": DeepScaleRSFT,
    "DeepScaleR": DeepScaleR,
}


def build_dataset(config):
    dataset_cls = DATASET_REGISTRY.get(config.dataset_cls)
    if dataset_cls is None:
        raise KeyError(f"Unknown training dataset class: {config.dataset_cls}")
    return dataset_cls(config).dataset
