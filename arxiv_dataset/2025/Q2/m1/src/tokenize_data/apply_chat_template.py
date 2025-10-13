import dotenv

dotenv.load_dotenv()


from datetime import datetime
from functools import partial

import click
import datasets
from transformers import AutoTokenizer

# NOTE There is no instruction on output format when SFT
# https://huggingface.co/datasets/simplescaling/s1K-1.1_tokenized/
# OUTPUT_FORMAT_PROMPT = "Return your final response within \\boxed{}."


def tokenize_sample(sample, tokenizer):
    prompt = sample["prompt"]
    thinking = sample["reasoning"]
    attempt = sample["distilled_answer_string"]

    # NOTE: we need "Answer: " to indicate the start of the answer.
    # Used in the evaluator to extract the answer in s1.
    # See OpanAI's evals / simple-evals.
    # But in eval, we would prefer \\box{} to indicate the answer.
    attempt = "Answer: " + attempt if "Answer:" not in attempt else attempt

    text = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": prompt},
            {
                "role": "assistant",
                "content": "<|im_start|>think\n"
                + thinking.strip()
                + "\n<|im_start|>answer\n"
                + attempt.strip(),
            },
        ],
        tokenize=False,
    )
    return {"text": text}


def tokenize_dataset(
    thinking_dataset,
    tokenizer,
    num_proc,
    push_repo_id,
):
    _tokenizer_sample = partial(tokenize_sample, tokenizer=tokenizer)
    _tokenizer_sample(thinking_dataset[0])

    thinking_dataset = thinking_dataset.map(
        _tokenizer_sample,
        num_proc=num_proc,
        desc="Tokenizing SFT data",
    )

    thinking_dataset.push_to_hub(push_repo_id)
    print(f"Pushed to {push_repo_id}")


@click.command()
@click.option(
    "--repo_id",
    type=str,
    help="The repo id to push the tokenized dataset to.",
    default="mmqm/m196k-dedup-decon-filter_easy-r1-filter_wrong",
)
@click.option(
    "--dry_run", is_flag=True, help="Whether to run the script in dry run mode."
)
def main(repo_id="mmqm/m196k-dedup-decon-filter_easy-r1-filter_wrong", dry_run=False):
    split = "train"

    current_date = datetime.now()
    formatted_date = current_date.strftime("%d%m%y")
    push_repo_id = f"{repo_id}-tokenized-{formatted_date}"

    thinking_dataset = datasets.load_dataset(repo_id, split=split)
    if dry_run:
        thinking_dataset = thinking_dataset.take(3)

    tokenizer_name = "Qwen/Qwen2.5-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    num_proc = 20
    tokenize_dataset(thinking_dataset, tokenizer, num_proc, push_repo_id)


if __name__ == "__main__":
    main()
