from dataclasses import dataclass


@dataclass
class EvalDatasetConfig:
    dataset_name: str
    dataset_config: str | None = None
    split: str = "test"
    answer_verifier_name: str | None = "rule_verifier"
    answer_verifier_args: dict = None
    pass_k_vals: list = None

    def __post_init__(self):
        if self.answer_verifier_args is None:
            self.answer_verifier_args = {}
        if self.pass_k_vals is None:
            self.pass_k_vals = []


@dataclass
class TriviaQA(EvalDatasetConfig):
    dataset_name: str = "claytonwang/trivia_eval"
    answer_verifier_name: str = "llm_verifier"


@dataclass
class NQOpen(EvalDatasetConfig):
    dataset_name: str = "google-research-datasets/nq_open"
    split: str = "validation"
    answer_verifier_name: str = "alias_verifier"


@dataclass
class PopQA(EvalDatasetConfig):
    dataset_name: str = "akariasai/PopQA"
    answer_verifier_name: str = "popqa_verifier"


@dataclass
class WebQuestions(EvalDatasetConfig):
    dataset_name: str = "stanfordnlp/web_questions"
    answer_verifier_name: str = "web_questions_verifier"
    include_multi_choice: bool = False


@dataclass
class GSM8K(EvalDatasetConfig):
    dataset_name: str = "claytonwang/gsm8k_eval"
    answer_verifier_name: str = "rule_verifier"


@dataclass
class HotpotVanilla(EvalDatasetConfig):
    dataset_name: str = "claytonwang/hotpotvanilla_RLCR"
    answer_verifier_name: str = "rule_verifier"


@dataclass
class DeepScaleR_Eval(EvalDatasetConfig):
    dataset_name: str = "agentica-org/DeepScaleR-Preview-Dataset"
    answer_verifier_name: str = "rule_verifier"


@dataclass
class Math500(EvalDatasetConfig):
    dataset_name: str = "HuggingFaceH4/MATH-500"
    answer_verifier_name: str = "rule_verifier"


@dataclass
class MinervaMath(EvalDatasetConfig):
    dataset_name: str = "math-ai/minervamath"
    answer_verifier_name: str = "rule_verifier"


@dataclass
class OlympiadBench(EvalDatasetConfig):
    dataset_name: str = "math-ai/olympiadbench"
    answer_verifier_name: str = "rule_verifier"


@dataclass
class AIME2024(EvalDatasetConfig):
    dataset_name: str = "Maxwell-Jia/AIME_2024"
    split: str = "train"
    answer_verifier_name: str = "rule_verifier"


@dataclass
class AIME2025(EvalDatasetConfig):
    dataset_name: str = "MathArena/aime_2025"
    split: str = "train"
    answer_verifier_name: str = "rule_verifier"


@dataclass
class AIME2026(EvalDatasetConfig):
    dataset_name: str = "MathArena/aime_2026"
    split: str = "train"
    answer_verifier_name: str = "rule_verifier"


@dataclass
class LiveBenchReasoning(EvalDatasetConfig):
    dataset_name: str = "livebench/reasoning"
    answer_verifier_name: str = "livebench_reasoning_verifier"


@dataclass
class MuSiQue(EvalDatasetConfig):
    dataset_name: str = "dgslibisey/MuSiQue"
    split: str = "validation"
    answer_verifier_name: str = "alias_verifier"


@dataclass
class DROP(EvalDatasetConfig):
    dataset_name: str = "ucinlp/drop"
    split: str = "validation"
    answer_verifier_name: str = "drop_verifier"
