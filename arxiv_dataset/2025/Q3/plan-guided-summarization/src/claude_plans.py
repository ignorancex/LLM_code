import argparse
import enum
import json
import re
import time
from dataclasses import dataclass
from typing import List

import boto3
import nltk
from claude_plan_prompts_with_links import *
from datasets import Dataset, concatenate_datasets, load_dataset

# from tqdm.notebook import tqdm
from tqdm import tqdm

# TODO: Update the profile_name to your profile's name
# TODO: Update the region if you want to use a different region
session = boto3.Session(profile_name="PLACEHOLDER")
bedrock_client = session.client(service_name="bedrock-runtime", region_name="us-west-2")


class ModelId(enum.Enum):
    CLAUDE_V2 = "anthropic.claude-v2"
    CLAUDE_V3 = "anthropic.claude-3-sonnet-20240229-v1:0"
    CLAUDE_V3_5 = "anthropic.claude-3-5-sonnet-20240620-v1:0"


@dataclass
class APIResponse:
    text: str
    local_latency: float
    invocation_latency: int
    input_token_count: int
    output_token_count: int


def get_claude_input_dict(
    prompt: str,
    max_tokens: int = 50,
    temperature: float = 1,
    top_p: float = 0.9,
    top_k: int = 50,
) -> dict:
    return {
        "prompt": prompt,
        "max_tokens_to_sample": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
    }


def get_messages_api_input_dict(
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    anthropic_version: str,
) -> dict:
    parts = [x.strip() for x in re.split("(Assistant:|Human:)", prompt) if x]

    string_to_role = {"Human:": "user", "Assistant:": "assistant"}
    role_to_string = {v: k for k, v in string_to_role.items()}

    messages = []
    current_role = "user"
    for part in parts:
        if part in string_to_role.keys():
            current_role = string_to_role[part]
        else:
            if messages and messages[-1]["role"] == current_role:
                messages[-1]["content"] += f" {role_to_string[current_role]} {part}"
            else:
                messages.append(
                    {
                        "role": current_role,
                        "content": part,
                    }
                )

    return {
        "anthropic_version": anthropic_version,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "messages": messages,
    }


def _get_input_dict(
    model_id: ModelId,
    prompt: str,
    max_tokens: int = 400,
    temperature: float = 1,
    top_p: float = 0.999,
    top_k: int = 250,
) -> dict:
    if model_id == ModelId.CLAUDE_V2:
        data = get_claude_input_dict(prompt, max_tokens, temperature, top_p, top_k)
    # Claude 3 is only supported through messages API as of 3/5/2024
    elif model_id == ModelId.CLAUDE_V3 or model_id == ModelId.CLAUDE_V3_5:
        data = get_messages_api_input_dict(
            prompt,
            max_tokens,
            temperature,
            top_p,
            top_k,
            anthropic_version="bedrock-2023-05-31",
        )
    else:
        raise ValueError(f"Unsupported model id: {model_id}.")
    return data


def parse_response(
    response: dict, local_latency: float, model_id: ModelId
) -> APIResponse:
    try:
        response_body = json.load(response.get("body"))  # type: ignore
    except TypeError:
        response_body = json.loads(response.get("body").read())

    try:
        if model_id == ModelId.CLAUDE_V2:
            response_text = response_body.get("completion")
        elif model_id == ModelId.CLAUDE_V3 or model_id == ModelId.CLAUDE_V3_5:
            response_text = response_body.get("content")[0].get("text")
        else:
            response_text = response_body.get("results")[0].get("outputText")
    except IndexError:
        response_text = ""

    metadata = response.get("ResponseMetadata", {}).get("HTTPHeaders", {})
    return APIResponse(
        response_text,
        local_latency,
        metadata.get("x-amzn-bedrock-invocation-latency"),
        metadata.get("x-amzn-bedrock-input-token-count"),
        metadata.get("x-amzn-bedrock-output-token-count"),
    )


def get_claude_response(
    prompt: str,
    max_tokens: int = 400,
    temperature: float = 1,
    top_p: float = 0.999,
    top_k: float = 250,
    model_id: ModelId = ModelId.CLAUDE_V3_5,
    add_human_and_assistant_tags: bool = False,
) -> APIResponse:
    if add_human_and_assistant_tags:
        prompt = f"Human: {prompt} \n\nAssistant:"

    content_type = "application/json"
    accept = "application/json"

    data = _get_input_dict(model_id, prompt, max_tokens, temperature, top_p, top_k)

    start = time.time()
    response = bedrock_client.invoke_model(
        body=json.dumps(data),
        modelId=model_id.value,
        accept=accept,
        contentType=content_type,
    )
    end = time.time()

    return parse_response(response, end - start, model_id)


def flatten_squality_ds(squality_ds: Dataset):
    # there are 4 summaries for each document so we flatten the dataset so that we have four times the training data.
    documents = squality_ds["document"]
    metadata = squality_ds["metadata"]
    questions = squality_ds["questions"]
    summary1 = [q[0]["responses"][0]["response_text"] for q in questions]
    summary2 = [q[0]["responses"][1]["response_text"] for q in questions]
    summary3 = [q[0]["responses"][2]["response_text"] for q in questions]
    summary4 = [q[0]["responses"][3]["response_text"] for q in questions]
    dataset1 = Dataset.from_dict(
        {"document": documents, "metadata": metadata, "summary": summary1}
    )
    dataset2 = Dataset.from_dict(
        {"document": documents, "metadata": metadata, "summary": summary2}
    )
    dataset3 = Dataset.from_dict(
        {"document": documents, "metadata": metadata, "summary": summary3}
    )
    dataset4 = Dataset.from_dict(
        {"document": documents, "metadata": metadata, "summary": summary4}
    )
    flattened_squality_ds = concatenate_datasets(
        [dataset1, dataset2, dataset3, dataset4]
    )
    return flattened_squality_ds


def get_claude_plans(summaries: List[str], claude_plans_file: str, offset=0):
    with open(claude_plans_file, "w") as f:
        for i, summary in enumerate(tqdm(summaries)):
            sentences = nltk.sent_tokenize(summary)
            summary_with_sent_nums = "".join(
                f"{t} [{i+1}] " for i, t in enumerate(sentences)
            ).strip()
            response = get_claude_response(
                prompt_icl_with_links_1a(summary_with_sent_nums),
                add_human_and_assistant_tags=False,
            )
            f.write(
                json.dumps(
                    {
                        "idx": i + offset,
                        "plan_with_citation": response.text,
                        "summary": summary_with_sent_nums,
                    }
                )
                + "\n"
            )
            time.sleep(10)


def get_squality_plans(
    claude_plans_file: str, dataset_cache: str, split: str = "train"
):
    squality_ds = load_dataset(
        "pszemraj/SQuALITY-v1.3", split=split, cache_dir=dataset_cache
    )
    flattened_squality_ds = flatten_squality_ds(squality_ds)
    summaries = flattened_squality_ds["summary"]
    get_claude_plans(summaries, claude_plans_file)


def get_summscreen_plans(
    claude_plans_file: str, dataset_cache: str, split: str = "train"
):
    summscreen_ds = load_dataset(
        "YuanPJ/summ_screen", "fd", split=split, cache_dir=dataset_cache
    )
    summaries = [recap[0] for recap in summscreen_ds["Recap"]]
    get_claude_plans(summaries, claude_plans_file, offset=0)


def parse_cmd_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--claude_plans_directory",
        type=str,
        required=True,
        help="",
    )
    parser.add_argument(
        "--dataset_cache",
        type=str,
        required=True,
        help="",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_cmd_args()
    # squality
    for split in ["train", "validation", "test"]:
        print(f"Processing split: {split}")
        claude_plans_file = os.path.join(
            args.claude_plans_directory,
            f"squality-plans-from-summary-claude-{split}.jsonl",
        )
        get_squality_plans(claude_plans_file, args.dataset_cache, split)

    # summscreen
    for split in ["train", "validation", "test"]:
        print(f"Processing split: {split}")
        claude_plans_file = os.path.join(
            args.claude_plans_directory,
            f"summscreen-fd-plans-from-summary-claude-{split}.jsonl",
        )
        get_summscreen_plans(claude_plans_file, args.dataset_cache, split)


if __name__ == "__main__":
    main()
