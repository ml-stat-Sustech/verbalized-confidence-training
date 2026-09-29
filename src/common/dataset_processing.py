from src.common.system_prompts import get_sys_prompt, get_sys_prompt_by_name


def resolve_dataset_sys_prompt(script_args):
    if prompt_name := getattr(script_args, "system_prompt_name", None):
        return get_sys_prompt_by_name(prompt_name)
    return get_sys_prompt(getattr(script_args, "confidence_format", None))


def process_dataset(dataset, script_args):
    sys_prompt = resolve_dataset_sys_prompt(script_args)

    if script_args.task_spec == "generation":
        dataset = make_generation_dataset(dataset, sys_prompt)

    return dataset


def make_generation_dataset(dataset,sys_prompt):
    def make_generation_conversation(example):
        if 'question' in example.keys():
            user_format = (
                f"\n\nPROBLEM: {example['question']}\n\n"
                )
        else:
            user_format = (
                    f"\n\nPROBLEM: {example['problem']}\n\n"
                    )
        return {
            "prompt": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_format},
            ],
        }

    dataset = dataset.map(make_generation_conversation)
    return dataset
