"""
Medical Subject Headings
https://www.nlm.nih.gov/mesh/meshhome.html

MeSH Qualifiers List
https://www.nlm.nih.gov/mesh/subhierarchy.html

MeSH Qualifiers with Scope Notes
https://www.nlm.nih.gov/mesh/qualifiers_scopenotes.html


Follow s1: https://github.com/simplescaling/s1/blob/main/data/featurization.py
"""

import dotenv

dotenv.load_dotenv()


import json
import multiprocessing as mp
from collections import defaultdict
from functools import partial
from multiprocessing import Pool
from typing import List, Tuple

import click
import datasets
from bespokelabs import curator
from datasets import Dataset, concatenate_datasets, load_dataset
from rapidfuzz import fuzz, process
from tqdm import tqdm

SYSTEM_PROMPT = (
    "You are a helpful assistant that classifies questions into different subjects based on the provided classification rubrics. "
    "You will be given a question and a list of subjects. "
    "You need to classify the question into one of the subjects. "
    "If the question has multiple subjects, you should classify the question into the most relevant subject. "
    "Explain your reasoning, and end your response on a new line with two-digit code of the subject that the question belongs to."
)


@click.command()
@click.option(
    "--repo_id",
    type=str,
    help="The repo id to load the dataset from.",
    default="mmqm/m196k-dedup-decon-filter_easy-r1-filter_wrong-decon_eval-tokenized-120325",
)
@click.option(
    "--domain_prompt_json_path", type=str, default="src/select_data/mesh_qualifier.json"
)
@click.option(
    "--dry_run", is_flag=True, default=False, help="Whether to run in dry run mode."
)
def main(
    repo_id,
    domain_prompt_json_path,
    dry_run,
):
    split = "train"

    dataset = load_dataset(repo_id, split=split)
    kept_columns = list(dataset.column_names)
    kept_columns.remove("prompt")
    dataset = dataset.remove_columns(kept_columns)

    if dry_run is True:
        dataset = dataset.take(3)

    domain_dataset = classify_sample_with_llm(domain_prompt_json_path, dataset)

    upload_repo_id = f"{repo_id}-domain-classification"
    domain_dataset.push_to_hub(upload_repo_id)
    print(f"Uploaded to {upload_repo_id}")


def classify_sample_with_llm(domain_prompt_json_path, dataset):
    with open(domain_prompt_json_path, "r") as f:
        domain_prompt = json.load(f)
    domain_prompt = "\n\n".join([node["prompt"] for node in domain_prompt])

    # Use openai compatible:
    # https://github.com/BerriAI/litellm/issues/8263
    # https://cloud.siliconflow.cn/models
    reasoner = PlainLLM(
        model_name="azure/gpt-4o-1120-nofilter-global",
        backend="litellm",
        # generation_params={"temp": 0.0, "max_tokens": 8_000},
        backend_params={
            "max_requests_per_minute": 900,
            "max_tokens_per_minute": 270_000,
            # https://docs.bespokelabs.ai/bespoke-curator/api-reference/llm-api-documentation#common-parameters
            "request_timeout": 30,
            "max_retries": 20,
        },
        custom_system_prompt=SYSTEM_PROMPT,
        custom_domain_prompt=domain_prompt,
    )
    domain_dataset = reasoner(dataset)
    return domain_dataset


class PlainLLM(curator.LLM):
    def __init__(
        self, custom_system_prompt=None, custom_domain_prompt=None, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        if custom_system_prompt is None:
            raise ValueError("Please provide a system prompt.")
        if custom_domain_prompt is None:
            raise ValueError("Please provide a domain prompt.")

        self._custom_system_prompt = custom_system_prompt
        self._custom_domain_prompt = custom_domain_prompt
        print("Custom system prompt:", self._custom_system_prompt)
        print("Custom domain prompt:", self._custom_domain_prompt)

    return_completions_object = True

    def prompt(self, input):
        """Create a prompt for the LLM to reason about the problem."""
        return [
            {
                "role": "system",
                "content": self._custom_system_prompt,
            },
            {
                "role": "user",
                "content": f'## Question\n{input["prompt"]}\n\n## Classification rubrics\n{self._custom_domain_prompt}',
            },
        ]

    def parse(self, input_sample, response):
        """Parse the LLM response to extract reasoning and solution."""
        return {
            **input_sample,
            # "reasoning": response["choices"][0]["message"]["reasoning_content"],
            "domain": response["choices"][0]["message"]["content"],
        }


if __name__ == "__main__":
    main()
