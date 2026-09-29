from src.eval.inferencers.base_inferencer import BaseInferencer
from src.eval.inferencers.verbalized_confidence_inferencer import VerbalizedConfidenceInferencer


def build_inferencer(config, model):
    inferencer_name = getattr(config, "inferencer_name", None)
    if inferencer_name == "verbalized_confidence":
        return VerbalizedConfidenceInferencer(config, model)
    if inferencer_name == "base":
        return BaseInferencer(config, model)
    raise ValueError(f"Unknown inferencer_name: {inferencer_name}")
