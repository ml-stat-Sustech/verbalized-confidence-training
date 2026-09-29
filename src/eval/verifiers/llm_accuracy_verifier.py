import gc
import json

from transformers import AutoTokenizer


QWEN_JUDGE_PROMPT = """
Please as a grading expert, judge whether the final answers given by the candidates below are consistent with the standard answers, that is, whether the candidates answered correctly.
Here are some evaluation criteria:
1. Please refer to the given standard answer. You don't need to re-generate the answer to the question because the standard answer has been given. You only need to judge whether the candidate's answer is consistent with the standard answer according to the form of the question. THE STANDARD ANSWER IS ALWAYS CORRECT AND THE QUESTION IS PERFECTLY VALID. NEVER QUESTION THEM.
2. ONLY compare the FINAL ANSWER - COMPLETELY IGNORE any potential errors in the REASONING PROCESSES.
3. Some answers may be expressed in different ways, such as some answers may be a mathematical expression, some answers may be a textual description, as long as the meaning expressed is the same. Before making a judgment, please understand the question and the standard answer first, and then judge whether the candidate's answer is correct.
4. Some answers may consist of multiple items, such as multiple-choice questions, multiple-select questions, fill-in-the-blank questions, etc. Regardless of the question type, the final answer will be considered correct as long as it matches the standard answer, regardless of whether the reasoning process is correct. For multiple-select questions and multi-blank fill-in-the-blank questions, all corresponding options or blanks must be answered correctly and match the standard answer exactly to be deemed correct.
5. If the prediction is given with \\boxed{{}}, please ignore the \\boxed{{}} and only judge whether the candidate's answer is consistent with the standard answer.
6. If the candidate's answer is invalid (e.g., incomplete (cut off mid-response), lots of unnormal repetitive content, or irrelevant to the question, saying it can't answer the question because some irresistible factors, like ethical issues, no enough information, etc.), select option C (INVALID).
Please judge whether the following answers are consistent with the standard answer based on the above criteria. Grade the predicted answer of this new question as one of:
A: CORRECT
B: INCORRECT
C: INVALID
Just return the letters "A", "B", or "C", with no text around it.
Here is your task. Simply reply with either CORRECT, INCORRECT, or INVALID. Don't apologize or correct yourself if there was a mistake; we are just trying to grade the answer.
<Original Question Begin>:
{question}
<Original Question End>
<Standard Answer Begin>:
{reference_answer}
<Standard Answer End>
<Candidate's Answer Begin>:
{candidate_answer}
<Candidate's Answer End>
Judging the correctness of the candidate's answer:
""".strip()

REFERENCE_LIST_GUIDANCE = {
    "aliases": "The standard-answer list contains alternative names or forms. Matching any one item is sufficient; do not require every item.",
    "infer": "The standard-answer list may contain aliases for one answer or multiple jointly required answers. Infer which from the original question, and require a complete answer when the question asks for multiple items.",
}


def _display_answer(value):
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _build_qwen_judge_prompt(
    tokenizer,
    question,
    reference_answer,
    candidate_answer,
    reference_list_mode=None,
):
    user_prompt = QWEN_JUDGE_PROMPT.format(
        question=question,
        reference_answer=_display_answer(reference_answer),
        candidate_answer=_display_answer(candidate_answer),
    )
    if isinstance(reference_answer, list) and reference_list_mode is not None:
        user_prompt += "\n" + REFERENCE_LIST_GUIDANCE[reference_list_mode]
    user_prompt += "\n/no_think"
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def _parse_qwen_judge_response(text):
    label = text.strip().lstrip("*_`# ").upper()
    first_label = label[:1] if len(label) == 1 or label[1:2].isspace() or label[1:2] == ":" else ""
    if first_label == "A":
        return 1
    if first_label in {"B", "C"}:
        return 0
    raise ValueError(f"Qwen judge returned an invalid label: {text!r}")


def build_llm_judge(config):
    from vllm import LLM

    tokenizer = AutoTokenizer.from_pretrained(config.judge_model_name_or_path, trust_remote_code=True)
    llm = LLM(
        model=config.judge_model_name_or_path,
        gpu_memory_utilization=config.judge_gpu_memory_utilization,
        tensor_parallel_size=config.tensor_parallel_size,
        max_model_len=config.judge_max_model_len,
    )
    return tokenizer, llm


def close_llm_judge(llm):
    del llm
    gc.collect()


def llm_verifier(
    local_dataset,
    config,
    format_fn="confidence_format",
    reference_list_mode=None,
    **kwargs,
):
    from vllm import SamplingParams

    label_dict = {"is_correct": []}
    is_correct = []
    n = config.num_generations

    extracted_answers = []
    for i in range(len(local_dataset)):
        q_spec_ans = []
        for answer_text in local_dataset[i]["predictions"]:
            q_spec_ans.append(answer_text if answer_text else "I don't know")
        extracted_answers.append(q_spec_ans)

    prompts = []
    tokenizer, llm = build_llm_judge(config)
    for i in range(len(local_dataset)):
        for j in range(n):
            prompts.append(
                _build_qwen_judge_prompt(
                    tokenizer,
                    local_dataset[i]["question"],
                    local_dataset[i]["answer"],
                    extracted_answers[i][j],
                    reference_list_mode,
                )
            )

    sampling_params = SamplingParams(n=1, temperature=0, max_tokens=config.judge_max_tokens)
    outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=False)

    responses = [_parse_qwen_judge_response(output.outputs[0].text) for output in outputs]

    agg_responses = []
    for i in range(0, len(responses), n):
        agg_responses.append(responses[i : i + n])

    for i in range(len(local_dataset)):
        correctness_list = []
        for j in range(n):
            actual_correctness = agg_responses[i][j]
            correctness_list.append(1 if actual_correctness == 1 else 0)

        is_correct.append(correctness_list)

    label_dict["is_correct"] = is_correct

    close_llm_judge(llm)
    return label_dict
