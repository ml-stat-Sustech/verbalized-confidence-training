import gc
import logging
import re

from src.common.confidence_formats import (
    CONFIDENCE_NONE,
    CONFIDENCE_PROBABILITY,
    confidence_value_regex,
    normalize_confidence_format,
    parse_confidence_text,
)
from src.common.system_prompts import get_sys_prompt


class BaseInferencer:

    def __init__(self, config, model):
        self.config = config
        self.model = model

    def run(self, dataset):
        dataset_eval = dataset.dataset
        sys_messages = self.build_sys_messages(dataset_eval)
        model_inputs, _ = self.model.build_generation_inputs(sys_messages)
        logging.info("Running eval for %s on %d samples", self.config.name, len(dataset_eval))
        generations = self.generate_outputs(model_inputs)
        format_legal = self.extract_format_legal(generations)
        generations = self.fill_missing_answers(model_inputs, generations)
        if not self.requires_model_for_confidence_estimation():
            self.model.close()
            gc.collect()
        generations = self.estimate_confidence(model_inputs, generations)

        output_columns = self.extract_output_columns(generations)
        output_columns["is_format_legal"] = format_legal
        dataset_eval = self.merge_output_columns(dataset_eval, output_columns)
        return dataset_eval

    def generate_outputs(self, texts):
        return self.model.generate(texts, progress_desc="Generate responses")

    @staticmethod
    def confidence_extractor(confidence_text, confidence_format=CONFIDENCE_PROBABILITY):
        confidence_format = normalize_confidence_format(confidence_format)
        parsed = parse_confidence_text(confidence_text, confidence_format)
        return parsed.is_legal, parsed.value

    def resolve_confidence_format(self):
        if getattr(self.config, "inferencer_name", None) == "base":
            return CONFIDENCE_NONE
        return normalize_confidence_format(getattr(self.config, "confidence_format", None))

    def fill_missing_answers(self, texts, outputs):
        inst = "Thinking time ended \n\n. My final answer is "
        missing_answer_indices = []
        prompts = []
        for output_idx, (text, output) in enumerate(zip(texts, outputs)):
            for sample_idx in range(self.config.num_generations):
                last_answer = self.extract_last_tagged_value(output.outputs[sample_idx].text, "answer")
                if last_answer == "":
                    missing_answer_indices.append((output_idx, sample_idx))
                    prompts.append(self.continuation_prompt(text, output.outputs[sample_idx], inst))

        ans_calls_needed = 0
        if prompts:
            ans_outputs = self.model.generate(
                prompts,
                n=1,
                temperature=0,
                max_tokens=50,
                logprobs=None,
                progress_desc="Filling missing <answer>",
            )
            for (output_idx, sample_idx), ans_output in zip(missing_answer_indices, ans_outputs):
                answer_text = ans_output.outputs[0].text
                tagged_answer = self.extract_last_tagged_value(answer_text, "answer")
                if tagged_answer:
                    answer_text = tagged_answer
                generation = outputs[output_idx].outputs[sample_idx]
                suffix = "<answer> " + answer_text.strip() + " </answer>"
                generation.text += suffix
                if isinstance(texts[output_idx], dict):
                    generation.token_ids = list(generation.token_ids) + self.model.tokenizer.encode(
                        suffix, add_special_tokens=False,
                    )
                ans_calls_needed += 1

        total_outputs = self.config.num_generations * len(outputs)
        logging.info(
            "Config %s: filled missing <answer> tags for %d/%d outputs",
            self.config.name,
            ans_calls_needed,
            total_outputs,
        )
        return outputs

    def continuation_prompt(self, prompt, generation, instruction):
        if isinstance(prompt, dict):
            return {"prompt_token_ids": (
                prompt["prompt_token_ids"] + list(generation.token_ids)
                + self.model.tokenizer.encode(instruction, add_special_tokens=False)
            )}
        return prompt + generation.text + instruction

    @staticmethod
    def extract_last_tagged_value(text, tag):
        opening_tag = f"<{tag}>"
        closing_tag = f"</{tag}>"
        start = text.rfind(opening_tag)
        if start == -1:
            return ""
        start += len(opening_tag)
        end = text.find(closing_tag, start)
        if end == -1:
            return ""
        return text[start:end].strip()

    def estimate_confidence(self, texts, outputs):
        return outputs

    def requires_model_for_confidence_estimation(self) -> bool:
        return True

    def extract_format_legal(self, outputs):
        confidence_format = self.resolve_confidence_format()
        if confidence_format == CONFIDENCE_NONE:
            pattern = r".*?</think>\s*<answer>.*?</answer>\s*\Z"
        else:
            pattern = (
                rf".*?</think>\s*<answer>.*?</answer>\s*"
                rf"<confidence>{confidence_value_regex(confidence_format)}</confidence>\s*\Z"
            )
        return [
            [bool(re.match(pattern, generation.text, re.DOTALL | re.MULTILINE)) for generation in output.outputs]
            for output in outputs
        ]

    def extract_output_columns(self, outputs):
        output_columns = {
            "generations": [],
            "predictions": [],
            "confidences": [],
            "is_conf_legal": [],
        }
        confidence_pattern = r"<confidence>(.*?)</confidence>"

        for output in outputs:
            row_generations = []
            row_predictions = []
            row_confidence_values = []
            row_conf_legal = []
            for generation in output.outputs:
                generation_text = generation.text
                row_generations.append(generation_text)

                confidence_matches = re.findall(confidence_pattern, generation_text, re.DOTALL | re.MULTILINE)
                confidence_text = confidence_matches[-1].strip() if confidence_matches else ""
                parsed_confidence = parse_confidence_text(
                    confidence_text,
                    confidence_format=self.resolve_confidence_format(),
                )
                row_predictions.append(self.extract_last_tagged_value(generation_text, "answer"))
                row_confidence_values.append(parsed_confidence.value)
                row_conf_legal.append(parsed_confidence.is_legal)

            output_columns["generations"].append(row_generations)
            output_columns["predictions"].append(row_predictions)
            output_columns["confidences"].append(row_confidence_values)
            output_columns["is_conf_legal"].append(row_conf_legal)

        return output_columns

    def resolve_sys_prompt(self):
        confidence_format = getattr(self.config, "confidence_format", None)
        if getattr(self.config, "inferencer_name", None) == "base":
            confidence_format = CONFIDENCE_NONE
        return get_sys_prompt(confidence_format)

    def build_sys_messages(self, dataset_eval):
        sys_prompt = self.resolve_sys_prompt()
        sys_messages = []
        for example in dataset_eval:
            sys_messages.append(
                [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": f"\n\nPROBLEM: {example['question']}\n\n"},
                ]
            )
        return sys_messages

    def merge_output_columns(self, dataset, output_columns):
        for key, value in output_columns.items():
            if key in dataset.column_names:
                dataset = dataset.remove_columns([key])
            dataset = dataset.add_column(key, value)
        return dataset
