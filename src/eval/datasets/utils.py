from src.eval.datasets.aime2024 import AIME2024
from src.eval.datasets.aime_matharena import MathArenaAIME
from src.eval.datasets.deep_scale_r import DeepScaleR
from src.eval.datasets.gsm8k import GSM8K
from src.eval.datasets.hotpot import Hotpot
from src.eval.datasets.livebench_reasoning import LiveBenchReasoning
from src.eval.datasets.math500 import Math500
from src.eval.datasets.minerva_math import MinervaMath
from src.eval.datasets.olympiad_bench import OlympiadBench
from src.eval.datasets.open_qa import NQOpen, PopQA, WebQuestions
from src.eval.datasets.reasoning_qa import DROP, MuSiQue
from src.eval.datasets.trivia import Trivia


DATASET_REGISTRY = {
    # In-distribution mathematical reasoning.
    "DeepScaleR_Eval": DeepScaleR,
    "Math500": Math500,
    "MinervaMath": MinervaMath,
    "OlympiadBench": OlympiadBench,
    "GSM8K": GSM8K,
    "AIME2024": AIME2024,
    "AIME2025": MathArenaAIME,
    "AIME2026": MathArenaAIME,
    # Out-of-distribution general reasoning.
    "HotpotVanilla": Hotpot,
    "MuSiQue": MuSiQue,
    "TriviaQA": Trivia,
    "NQOpen": NQOpen,
    "PopQA": PopQA,
    "WebQuestions": WebQuestions,
    "DROP": DROP,
    "LiveBenchReasoning": LiveBenchReasoning,
}


def build_dataset(config):
    dataset_cls = DATASET_REGISTRY[config.dataset_cls]
    return dataset_cls(config)
