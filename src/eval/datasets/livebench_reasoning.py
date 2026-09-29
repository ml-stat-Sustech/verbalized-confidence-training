from src.eval.datasets.base_dataset import BaseDataset


class LiveBenchReasoning(BaseDataset):
    def reformat(self, dataset):
        def mapping(example, idx):
            turns = example["turns"]
            if len(turns) != 1:
                raise ValueError(f"LiveBench reasoning example {example['question_id']} has {len(turns)} turns")
            release_date = example["livebench_release_date"]
            return {
                "id": example.get("question_id", idx + 1),
                "question": turns[0].replace("<solution>", "").replace("</solution>", ""),
                "answer": example["ground_truth"],
                "task": example["task"],
                "level": example["level"],
                "_livebench_release_date": release_date.isoformat() if release_date else "",
                "livebench_removal_date": example["livebench_removal_date"],
            }

        dataset = dataset.map(mapping, with_indices=True)
        keep_columns = [
            "id",
            "question",
            "answer",
            "task",
            "level",
            "_livebench_release_date",
            "livebench_removal_date",
        ]
        dataset = dataset.remove_columns([column for column in dataset.column_names if column not in keep_columns])
        return dataset.rename_column("_livebench_release_date", "livebench_release_date")
