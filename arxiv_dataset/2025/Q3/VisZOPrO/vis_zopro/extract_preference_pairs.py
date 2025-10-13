import re


def common_start(str1: str, str2: str) -> str:
    # Zip the two strings and iterate over them together
    common_chars = []
    for c1, c2 in zip(str1, str2):
        if c1 == c2:
            common_chars.append(c1)
        else:
            break
    # Join the common characters and return as a string
    return "".join(common_chars)


def split_dialogue_turns(text: str):
    pattern = r"(Human|Assistant):\s*(.*?)(?=\n\n(?:Human|Assistant):|$)"
    matches = re.findall(pattern, text, re.DOTALL)
    return [{"role": role.lower(), "content": content.strip()} for role, content in matches]


def extract_hh(example: str) -> list[dict[str, str]]:
    # Extract the prompt, which corresponds to the common start of the chosen and rejected dialogues
    prompt_text = common_start(example["chosen"], example["rejected"])

    # The chosen and rejected may share a common start, so we need to remove the common part
    if not prompt_text.endswith("\n\nAssistant: "):
        prompt_text = prompt_text[: prompt_text.rfind("\n\nAssistant: ")] + "\n\nAssistant: "

    # Extract the chosen and rejected lines
    chosen_line = example["chosen"][len(prompt_text):]
    rejected_line = example["rejected"][len(prompt_text):]

    # Remove the generation prompt ("\n\nAssistant: ") from the prompt
    prompt_text = prompt_text[: -len("\n\nAssistant: ")]

    # Split the string at every occurrence of "Human: " or "Assistant: "
    prompt_lines = re.split(r"(\n\nAssistant: |\n\nHuman: )", prompt_text)

    # Remove the first element as it's empty
    prompt_lines = prompt_lines[1:]

    prompt = []
    for idx in range(0, len(prompt_lines), 2):
        role = "user" if prompt_lines[idx] == "\n\nHuman: " else "assistant"
        content = prompt_lines[idx + 1]
        prompt.append({"role": role, "content": content})

    # Remove the prompt from the chosen and rejected dialogues
    chosen = [{"role": "assistant", "content": chosen_line}]
    rejected = [{"role": "assistant", "content": rejected_line}]

    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}


def extract_rlhf_trojan(example):
    human_pattern = r"\n\nHuman: (.*?)(?=\n\nAssistant: |\n\nHuman: |$)"
    first_human_match = re.search(human_pattern, example["chosen"], re.DOTALL)
    if not first_human_match:
        return {"prompt": [], "chosen": [], "rejected": []}

    prompt_text = first_human_match.group(1).strip()
    prompt = [{"role": "user", "content": prompt_text}]

    chosen_remaining = example["chosen"][first_human_match.end():].strip()
    rejected_remaining = example["rejected"][first_human_match.end():].strip()

    chosen_turns = split_dialogue_turns(chosen_remaining)
    rejected_turns = split_dialogue_turns(rejected_remaining)

    return {
        "prompt": prompt,
        "chosen": chosen_turns,
        "rejected": rejected_turns
    }


def extract_saferlhf(example):

    better_id = example["better_response_id"]
    chosen_idx = better_id
    rejected_idx = 1 - better_id

    prompt = [{
        "role": "user",
        "content": example["prompt"]
    }]

    chosen = [{
        "role": "assistant", "content": example[f"response_{chosen_idx}"]
    }]

    rejected = [{
        "role": "assistant", "content": example[f"response_{rejected_idx}"]
    }]

    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}


def extract_SHP(example):
    prompt_text = example["history"]

    chosen_line = example["human_ref_A"] if example["labels"] == 1 else example["human_ref_B"]
    rejected_line = example["human_ref_B"] if example["labels"] == 1 else example["human_ref_A"]

    prompt = [{
        "role": "user",
        "content": prompt_text
    }]

    chosen = [{
        "role": "assistant", "content": chosen_line
    }]

    rejected = [{
        "role": "assistant", "content": rejected_line
    }]

    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}


def extract_cvalues_rlhf(example):
    prompt = [{
        "role": "user",
        "content": example["prompt"]
    }]

    chosen = [{
        "role": "assistant", "content": example["pos_resp"]
    }]
    rejected = [{
        "role": "assistant", "content": example["neg_resp"]
    }]

    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}


def extract_openai_summary(example):
    # the example has info: a dict where we need to take 'post', summaries: a list of 2 dicts where we need to take 'text', choice: either 0 or 1
    prompt = [{
        "role": "Full text",  # use role as proxy for building the chat template
        "content": example["info"]["post"]
    }]
    chosen = [{
        "role": "Summary",
        "content": example["summaries"][example["choice"]]["text"]
    }]
    rejected = [{
        "role": "Summary",
        "content": example["summaries"][1 - example["choice"]]["text"]
    }]
    if not prompt[0]["content"]:
        prompt[0]["content"] = ""
    if not chosen[0]["content"]:
        chosen[0]["content"] = ""
    if not rejected[0]["content"]:
        rejected[0]["content"] = ""
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}


def extract_wmt(example):
    # in this case return original, translation, score (z_mean)
    prompt = [{
        "role": "Original",
        "content": example["translation"]["en"]
    }]
    chosen = [{
        "role": "Translation",
        "content": example["translation"]["de"]
    }]
    rejected = [{
        "role": "Translation",
        "content": example["translation"]["en"]
    }]

    return {"prompt": prompt, "chosen": chosen, "rejected": rejected, "completion": chosen, "score": example["z_mean"]}


def extract_dialogue(example, example_name, task="chat"):
    chat_datasets2fn = {
        'Anthropic/hh-rlhf': extract_hh,
        'ethz-spylab/rlhf_trojan_dataset': extract_rlhf_trojan,
        'PKU-Alignment/PKU-SafeRLHF-single-dimension': extract_saferlhf,
        'stanfordnlp/SHP': extract_SHP,
        'Skepsun/cvalues_rlhf': extract_cvalues_rlhf
    }
    summary_datasets2fn = {
        'openai/summarize_from_feedback': extract_openai_summary
    }
    mt_datasets2fn = {
        'wmt/wmt20_mlqe_task1': extract_wmt
    }

    if task == "chat":
        if example_name in chat_datasets2fn:
            return chat_datasets2fn[example_name](example)
    elif task == 'MT':
        if example_name in mt_datasets2fn:
            return mt_datasets2fn[example_name](example)
    elif task == 'summarization':
        if example_name in summary_datasets2fn:
            return summary_datasets2fn[example_name](example)
    raise ValueError(
        "Unsupported task, dataset or task-dataset combination."
    )
