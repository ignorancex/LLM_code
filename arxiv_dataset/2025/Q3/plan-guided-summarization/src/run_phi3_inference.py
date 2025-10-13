"""
https://huggingface.co/docs/trl/v0.11.4/en/dataset_formats
for training, we need data in lm format
example: <|user|>\nSummarize the following text<|end|>\n<|assistant|>\nBlah blah blah<|end|>\n<|endoftext|>
for prediction, we need data in Prompt-only format
example: <|user|>\nSummarize the following text<|end|>\n<|assistant|>
"""

import argparse
import os

import jsonlines
import torch
from datasets import load_from_disk
from peft import AutoPeftModelForCausalLM, PeftModel
from other_utils import str2bool
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


def generate_and_write_summaries(
    dataset, model, tokenizer, generation_config, output_file
):
    # save summaries every n generations
    summaries = []
    save_every = len(dataset) // 4
    for datum in tqdm(dataset):
        datum = {k: v.unsqueeze(0).to("cuda") for k, v in datum.items()}
        outputs = model.generate(**datum, **generation_config)
        summary = tokenizer.batch_decode(outputs)[0]
        # find index of <|assistant|> in the summary and remove everything before it
        idx = summary.find("<|assistant|>")
        if idx == -1:
            summary = ""
        else:
            summary = summary[idx:]
            # replace <|assistant|> and <|endoftext|> with empty string
            summary = summary.replace("<|assistant|>", "")
            summary = summary.replace("<|end|>", "")
            summary = summary.replace("<|endoftext|>", "")
        summaries.append(summary.strip())
        if len(summaries) % save_every == 0:
            with jsonlines.open(output_file, "w") as fp:
                for i, summ in enumerate(summaries):
                    fp.write({"idx": i, "output": summ})

    with jsonlines.open(output_file, "w") as fp:
        for i, summ in tqdm(enumerate(summaries)):
            fp.write({"idx": i, "output": summ})


def parse_cmd_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--max_source_length", type=int, default=8192)
    parser.add_argument(
        "--checkpoint_path", type=str, default="microsoft/Phi-3-mini-128k-instruct"
    )
    parser.add_argument(
        "--model_name", type=str, default="microsoft/Phi-3-mini-128k-instruct"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="seed to use during training with the HF trainer",
    )
    parser.add_argument(
        "--num_reference_summaries",
        type=int,
        default=1,
        help="number of reference summaries per data point",
    )
    parser.add_argument(
        "--do_sample",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Is sampling enabled during generation",
    )
    parser.add_argument("--task", type=str, default="baseline")
    # parser.add_argument(
    #     "--force_words_ids", type=str, default=None, help="Pass space separated list of token IDs"
    # )
    # parser.add_argument(
    #     "--forced_bos_token_id",
    #     type=int,
    #     default=32012,
    #     help="forced_bos_token_id for generation",
    # )
    args = parser.parse_args()
    return args


def main():

    args = parse_cmd_args()
    set_seed(args.seed)

    # create args.output_dir if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    model_kwargs = dict(
        use_cache=False,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",  # loading the model with flash-attenstion support
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_path)
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3-mini-128k-instruct", **model_kwargs
    )

    model = PeftModel.from_pretrained(
        base_model,
        args.checkpoint_path,
    )

    generation_config = {
        "do_sample": args.do_sample,
        # 32011 for <|plan|> and 32012 for <|summary|>
        "max_new_tokens": args.max_new_tokens,
        "num_beams": 5,
        "length_penalty": 0.8,
        "no_repeat_ngram_size": 5,
        "pad_token_id": tokenizer.pad_token_id,
    }

    dataset = load_from_disk(args.dataset)

    for split in ["validation", "test"]:
        print(f"Running inference for {split} set . . .")
        dataset_split = dataset[split]
        if args.task == "baseline" or args.task == "e2e":
            summary_dataset = dataset_split
            plan_dataset = None
        elif args.task == "multitask":
            # split dataset_split into two equal sized datasets: plans and summaries
            plan_dataset = dataset_split.select(range(len(dataset_split) // 2))
            summary_dataset = dataset_split.select(
                range(len(dataset_split) // 2, len(dataset_split))
            )

        if args.num_reference_summaries > 1:
            # multiple ref summaries and no sampling
            if not args.do_sample:
                dataset_size = len(summary_dataset) // args.num_reference_summaries
                # select the first dataset_size data points from summary_dataset and plan_dataset
                summary_dataset = summary_dataset.select(range(dataset_size))
                if plan_dataset is not None:
                    plan_dataset = plan_dataset.select(range(dataset_size))

        # ["document", "metadata", "summary", "input_ids", "attention_mask", "text"]
        # drop columns except input_ids and attention_mask
        summary_dataset = summary_dataset.remove_columns(
            ["document", "metadata", "summary", "text"]
        )
        if "idx" in summary_dataset.column_names:
            summary_dataset = summary_dataset.remove_columns(["idx"])
        summary_dataset.set_format("pt", columns=["input_ids", "attention_mask"])
        # validation_dataset = validation_dataset.select(range(25))

        if plan_dataset is not None:
            # drop columns except input_ids and attention_mask
            plan_dataset = plan_dataset.remove_columns(
                ["document", "metadata", "summary", "text"]
            )
            if "idx" in plan_dataset.column_names:
                plan_dataset = plan_dataset.remove_columns(["idx"])
            plan_dataset.set_format("pt", columns=["input_ids", "attention_mask"])

        if args.task == "baseline" or args.task == "e2e":
            # gen summaries for the dataset
            output_file = os.path.join(args.output_dir, f"summaries_{split}.jsonl")
            print(f"Generating summaries for {split} split")
            generate_and_write_summaries(
                summary_dataset, model, tokenizer, generation_config, output_file
            )
        elif args.task == "multitask":
            # gen plans for the dataset
            output_file = os.path.join(args.output_dir, f"plans_{split}.jsonl")
            print(f"Generating plans for {split} split")
            generate_and_write_summaries(
                plan_dataset, model, tokenizer, generation_config, output_file
            )
            # gen summaries for the dataset
            output_file = os.path.join(args.output_dir, f"summaries_{split}.jsonl")
            print(f"Generating summaries for {split} split")
            generate_and_write_summaries(
                summary_dataset, model, tokenizer, generation_config, output_file
            )


if __name__ == "__main__":
    main()
