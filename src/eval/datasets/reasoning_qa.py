from datasets import load_dataset
from huggingface_hub import hf_hub_download

from src.eval.datasets.base_dataset import BaseDataset


def _context_prompt(paragraphs, question):
    context = "\n\n".join(f"[{title}]\n{text}" for title, text in paragraphs)
    return f"Context:\n{context}\n\nQuestion:\n{question}"


class MuSiQue(BaseDataset):
    def load_dataset(self):
        data_file = hf_hub_download(
            "dgslibisey/MuSiQue",
            "musique_ans_v1.0_dev.jsonl",
            repo_type="dataset",
        )
        dataset = load_dataset(
            "json",
            data_files={"validation": data_file},
        )
        return dataset[self.args.split]

    def reformat(self, dataset):
        def mapping(example):
            paragraphs = [(item["title"], item["paragraph_text"]) for item in example["paragraphs"]]
            aliases = [example["answer"], *example.get("answer_aliases", [])]
            return {
                "id": example["id"],
                "question": _context_prompt(paragraphs, example["question"]),
                "answer": list(dict.fromkeys(aliases)),
            }

        return dataset.map(mapping, remove_columns=dataset.column_names)


class DROP(BaseDataset):
    def reformat(self, dataset):
        def mapping(example):
            return {
                "id": example["query_id"],
                "question": _context_prompt([("Passage", example["passage"])], example["question"]),
                "answer": example["answers_spans"]["spans"],
            }

        return dataset.map(mapping, remove_columns=dataset.column_names)
