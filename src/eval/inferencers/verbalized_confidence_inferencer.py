import logging
import re

from src.common.confidence_formats import CONFIDENCE_NONE, confidence_fill_instruction
from src.eval.inferencers.base_inferencer import BaseInferencer


class VerbalizedConfidenceInferencer(BaseInferencer):
    def estimate_confidence(self, texts, outputs):
        confidence_format = self.resolve_confidence_format()
        if confidence_format == CONFIDENCE_NONE:
            logging.info("Config %s: confidence_format=none; skipped filling <confidence> tags", self.config.name)
            return outputs
        inst, max_tokens = confidence_fill_instruction(confidence_format)
        missing_confidence_indices = []
        prompts = []
        for output_idx, (text, output) in enumerate(zip(texts, outputs)):
            for sample_idx in range(self.config.num_generations):
                conf_pattern = r"<confidence>(.*?)</confidence>"
                conf_matches = re.findall(conf_pattern, output.outputs[sample_idx].text, re.DOTALL | re.MULTILINE)
                last_confidence = conf_matches[-1] if conf_matches else ""
                if last_confidence == "":
                    missing_confidence_indices.append((output_idx, sample_idx))
                    prompts.append(self.continuation_prompt(text, output.outputs[sample_idx], inst))

        conf_calls_needed = 0
        if prompts:
            verb_outputs = self.model.generate(
                prompts,
                n=1,
                temperature=0,
                max_tokens=max_tokens,
                logprobs=None,
                progress_desc="Filling missing <confidence>",
            )
            for (output_idx, sample_idx), verb_output in zip(missing_confidence_indices, verb_outputs):
                confidence_text = verb_output.outputs[0].text
                outputs[output_idx].outputs[sample_idx].text = (
                    outputs[output_idx].outputs[sample_idx].text + "<confidence>" + confidence_text + "</confidence>"
                )
                conf_calls_needed += 1

        total_outputs = self.config.num_generations * len(outputs)
        logging.info(
            "Config %s: filled missing <confidence> tags for %d/%d outputs",
            self.config.name,
            conf_calls_needed,
            total_outputs,
        )
        return outputs
