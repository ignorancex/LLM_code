import dotenv

dotenv.load_dotenv()


import logging
import os

import click
import prompt as distill_data_prompt
from bespokelabs import curator
from datasets import load_dataset


def mocked_reasoner(ds, answer_column: str = None):
    reasoning = "Deep Thought is thinking for 7.5 million years."
    if answer_column:
        ds = ds.map(
            lambda x: {
                "deepseek_solution": x[answer_column],
                "ground_truth_solution": x[answer_column],
            }
        )
    else:
        solution = "The answer is 42."
        ds = ds.add_column("deepseek_solution", [solution] * len(ds))
    ds = ds.add_column("reasoning", [reasoning] * len(ds))
    return ds


class PlainLLM(curator.LLM):
    return_completions_object = True

    def prompt(self, input):
        """Create a prompt for the LLM to reason about the problem."""
        return [
            # {"role": "system", "content": distill_data_prompt.DEEPSEEK_R1_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Return your final response within \\boxed{{}}. {input['prompt']}",
            },
        ]

    def parse(self, input_sample, response):
        """Parse the LLM response to extract reasoning and solution."""
        return {
            **input_sample,
            # "reasoning": response["choices"][0]["message"]["reasoning_content"],
            "distilled_answer_string": response["choices"][0]["message"]["content"],
        }


class R1Reasoner(PlainLLM):
    def parse(self, input_sample, response):
        """Parse the LLM response to extract reasoning and solution."""
        return {
            **input_sample,
            "reasoning": response["choices"][0]["message"]["reasoning_content"],
            "distilled_answer_string": response["choices"][0]["message"]["content"],
        }


def reason(ds, model_name="r1"):
    if os.environ.get("MOCK_REASON"):
        return mocked_reasoner(ds, answer_column="solution")

    if model_name == "r1":
        # Use openai compatible:
        # https://github.com/BerriAI/litellm/issues/8263
        # https://cloud.siliconflow.cn/models
        reasoner = R1Reasoner(
            model_name="openai/Pro/deepseek-ai/DeepSeek-R1",
            generation_params={"temp": 0.0, "max_tokens": 8_000},
            backend_params={
                "max_requests_per_minute": 1_000,
                "max_tokens_per_minute": 1_000_000,
                "base_url": "https://api.siliconflow.cn/v1",
                # https://docs.bespokelabs.ai/bespoke-curator/api-reference/llm-api-documentation#common-parameters
                # "request_timeout": 30,
                # "max_retries": 0,
                # https://docs.bespokelabs.ai/bespoke-curator/api-reference/llm-api-documentation#backend-parameters-configuration
                "require_all_responses": False,
            },
        )
    elif model_name.startswith("qwen-7b"):
        reasoner = PlainLLM(
            model_name="openai/Pro/Qwen/Qwen2.5-7B-Instruct",
            generation_params={"temp": 0.0, "max_tokens": 4_000},
            backend_params={
                "max_requests_per_minute": 2_000,
                "max_tokens_per_minute": 160_000,
                "base_url": "https://api.siliconflow.cn/v1",
            },
        )
    elif model_name.startswith("qwen-32b"):
        reasoner = PlainLLM(
            model_name="openai/Qwen/Qwen2.5-32B-Instruct",
            generation_params={"temp": 0.0, "max_tokens": 4_000},
            backend_params={
                "max_requests_per_minute": 2_000,
                "max_tokens_per_minute": 80_000,
                "base_url": "https://api.siliconflow.cn/v1",
            },
        )
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
    return reasoner(ds)


@click.command()
@click.option("--dry_run", is_flag=True, default=False)
@click.option("--repo_id", type=str, default="mmqm/m196k-dedup-decon")
@click.option("--model_name", type=str, default="qwen-7b")
@click.option("--debug", is_flag=True, default=False)
def main(
    dry_run=False,
    repo_id="mmqm/m196k-dedup-decon",
    model_name="qwen-7b",
    debug=False,
):
    dataset = load_dataset(repo_id, split="train")
    if dry_run:
        dataset = dataset.take(3)

    dataset = reason(dataset, model_name)

    if dry_run:
        print("======== MATH DATASET =======")
        print(dataset)
        print(dataset[0])
        print("================")

    if debug:
        # fmt: off
        import IPython; IPython.embed()
        # fmt: on

    output_repo_id = f"{repo_id}-{model_name}"
    dataset.push_to_hub(output_repo_id)


if __name__ == "__main__":
    main()
