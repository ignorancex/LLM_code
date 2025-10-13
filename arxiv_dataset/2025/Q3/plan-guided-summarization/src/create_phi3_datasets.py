import argparse
import os
import re
from typing import Dict, List

import jsonlines
import spacy
from datasets import (
    Dataset,
    DatasetDict,
    concatenate_datasets,
    interleave_datasets,
    load_dataset,
)
from other_utils import (
    filter_by_num_sentences,
    flatten_squality,
    process_summscreen,
)
from transformers import AutoTokenizer


def preprocess_add_indices(example, index):
    return {"idx": index}


def preprocess(
    example: Dict,
    # document_key,
    task_prompt: str,
    tokenizer: AutoTokenizer,
    max_source_length: int = 8192,
):  # Batch of inputs
    prompt = f"{task_prompt}\n{example['document'].strip()}"
    prompt_token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    prompt_token_ids = prompt_token_ids[:max_source_length]
    truncated_prompt = tokenizer.decode(prompt_token_ids)

    conversational_template = {"prompt": [{"role": "user", "content": ""}]}
    conversational_template["prompt"][0]["content"] = truncated_prompt
    standard_output = tokenizer.apply_chat_template(
        conversational_template["prompt"], tokenize=False, add_generation_prompt=True
    )
    # for some reason the tokenizer is adding extra white space after special tokens such as "<|plan|>" and "<|summary|>"
    # so we need to remove them
    standard_output = standard_output.replace("  ", " ")
    model_inputs = tokenizer(
        standard_output, max_length=tokenizer.model_max_length, truncation=True
    )
    model_inputs["text"] = standard_output
    # model_inputs = tokenizer(standard_output["prompt"])
    # labels = tokenizer(text_target=example["summary"])
    # model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def preprocess_train(
    example: Dict,
    # document_key,
    # summary_key,
    # idx: int,
    task: str,
    task_prompt: str,
    tokenizer: AutoTokenizer,
    plans: List[str] = None,
    max_source_length: int = 8192,
):
    idx = example["idx"]
    prompt = f"{task_prompt}\n{example['document'].strip()}"
    # we tokenize the prompt and keep as many tokens as we can fit in 8k context
    # <|user|>, <|end|>, <|assistant|>, <|end|>, <|endoftext|>
    prompt_token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    prompt_token_ids = prompt_token_ids[:max_source_length]

    truncated_prompt = tokenizer.decode(prompt_token_ids)
    conversational_template = {
        "messages": [
            {"role": "user", "content": ""},
            {"role": "assistant", "content": ""},
        ]
    }
    conversational_template["messages"][0]["content"] = truncated_prompt
    if plans is not None:
        if task == "e2e":
            # conversational_template["messages"][1]["content"] = (
            #     "<|plan|> " + plans[idx] + " <|summary|> " + example["summary"]
            # )
            conversational_template["messages"][1][
                "content"
            ] = f'<plan>\n{plans[idx]}\n</plan>\n<summary>\n{example["summary"]}\n</summary>'
        elif task == "multitask":
            # conversational_template["messages"][1]["content"] = "<|plan|> " + plans[idx]
            conversational_template["messages"][1][
                "content"
            ] = f"<plan>\n{plans[idx]}\n</plan>"
    else:
        if task == "e2e":
            # raise ValueError
            raise ValueError("Plans should not be None for e2e task.")
        elif task == "multitask":
            # conversational_template["messages"][1]["content"] = "<|summary|> " + example["summary"]
            conversational_template["messages"][1][
                "content"
            ] = f'<summary>\n{example["summary"]}\n</summary>'
        else:
            conversational_template["messages"][1][
                "content"
            ] = f'<summary>\n{example["summary"]}\n</summary>'
    standard_output = tokenizer.apply_chat_template(
        conversational_template["messages"], tokenize=False, add_generation_prompt=False
    )
    # for some reason the tokenizer is adding extra white space after special tokens such as "<|plan|>" and "<|summary|>"
    # so we need to remove them
    standard_output = standard_output.replace("  ", " ")
    model_inputs = tokenizer(
        standard_output, max_length=tokenizer.model_max_length, truncation=True
    )
    model_inputs["text"] = standard_output
    # model_inputs = tokenizer(standard_output["text"])
    # labels = tokenizer(text_target=example["summary"])
    # model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def get_all_dataset_splits(
    tokenizer: AutoTokenizer,
    dataset_name: str,
    dataset_cache_path: str,
    # document_key: str,
    # summary_key: str,
    training_plan_file_path: str = None,
    val_plan_file_path: str = None,
    plan_key: str = "plan_without_citation",
    max_source_length: int = 8192,
    task: str = "baseline",
    min_num_sentences: int = 1,
    **kwargs,
) -> List[Dataset]:
    if training_plan_file_path:
        training_plans = get_plans_from_file(training_plan_file_path, plan_key)
        validation_plans = get_plans_from_file(val_plan_file_path, plan_key)
        if task == "e2e":
            # training_task_prompt = "Generate a plan starting with <|plan|> special token followed by a summary starting with <|summary|> special token for the following text:"
            training_task_prompt = "Generate a plan followed by a summary for the following text. Enclose the plan within <plan> and </plan> tags and enclose the summary within <summary> and </summary> tags.\n\nText:\n"
        elif task == "multitask":
            # plan prompts
            # training_task_plan_prompt = (
            # 'Generate a plan starting with "<|plan|>" special token for the following text:'
            # )
            # training_task_plan_prompt = "Generate a plan for the following text. Start the plan with <|plan|> token.\n\nText:\n"
            training_task_plan_prompt = "Generate a plan for the following text. Enclose the plan within <plan> and </plan> tags.\n\nText:\n"
            # summary prompts
            # training_task_summary_prompt = 'Generate a summary starting with "<|summary|>" special token for the following text:'
            # training_task_summary_prompt = "Generate a summary for the following text. Start the summary with <|summary|> token.\n\nText:\n"
            training_task_summary_prompt = "Generate a summary for the following text. Enclose the summary within <summary> and </summary> tags.\n\nText:\n"
    else:
        training_plans = None
        validation_plans = None
        # "Summarize the following text:"
        training_task_prompt = "Generate a summary for the following text. Enclose the summary within <summary> and </summary> tags.\n\nText:\n"
    dataset_splits = []
    for split in ["train", "validation", "test"]:
        print("Processing split: ", split)
        if dataset_name == "pszemraj/SQuALITY-v1.3":
            dataset_split = load_dataset(
                dataset_name, split=split, cache_dir=dataset_cache_path
            )
            dataset_split = flatten_squality(dataset_split)
            # add index to dataset split
            dataset_split = dataset_split.map(preprocess_add_indices, with_indices=True)
        elif dataset_name == "YuanPJ/summ_screen":
            dataset_split = load_dataset(
                dataset_name, "fd", split=split, cache_dir=dataset_cache_path
            )
            dataset_split = process_summscreen(dataset_split)
            # add index to dataset split
            dataset_split = dataset_split.map(preprocess_add_indices, with_indices=True)
        # filter by number of sentences
        if min_num_sentences > 1:
            nlp = spacy.load("en_core_web_md")
            dataset_split = dataset_split.filter(
                filter_by_num_sentences,
                fn_kwargs={"nlp": nlp, "min_num_sentences": min_num_sentences},
            )
        # columns = list(dataset_split.features)
        if split == "train":
            # DONE
            tokenizer.padding_side = "right"
            if task == "e2e" or task == "baseline":
                dataset_split = dataset_split.map(
                    preprocess_train,
                    fn_kwargs={
                        # "document_key": document_key,
                        # "summary_key": summary_key,
                        "task": task,
                        "task_prompt": training_task_prompt,
                        "tokenizer": tokenizer,
                        "plans": training_plans,
                        "max_source_length": max_source_length,
                    },
                    # with_indices=True,
                    num_proc=8,
                )
            elif task == "multitask":
                plan_task_ds = dataset_split.map(
                    preprocess_train,
                    fn_kwargs={
                        # "document_key": document_key,
                        # "summary_key": summary_key,
                        "task": task,
                        "task_prompt": training_task_plan_prompt,
                        "tokenizer": tokenizer,
                        "plans": training_plans,
                        "max_source_length": max_source_length,
                    },
                    # with_indices=True,
                    num_proc=8,
                )
                summary_task_ds = dataset_split.map(
                    preprocess_train,
                    fn_kwargs={
                        # "document_key": document_key,
                        # "summary_key": summary_key,
                        "task": task,
                        "task_prompt": training_task_summary_prompt,
                        "tokenizer": tokenizer,
                        "plans": None,
                        "max_source_length": max_source_length,
                    },
                    # with_indices=True,
                    num_proc=8,
                )
                dataset_split = interleave_datasets([plan_task_ds, summary_task_ds])
            dataset_splits.append(dataset_split)
            # dataset_split = dataset_split.remove_columns(columns)
        elif split == "validation":
            # DONE
            if task == "e2e" or task == "baseline":
                # -----------------------------------------------
                # validation set to compute loss during training|
                # -----------------------------------------------
                tokenizer.padding_side = "right"
                tmp_split = dataset_split.map(
                    preprocess_train,
                    fn_kwargs={
                        # "document_key": document_key,
                        # "summary_key": summary_key,
                        "task": task,
                        "task_prompt": training_task_prompt,
                        "tokenizer": tokenizer,
                        "plans": validation_plans,
                        "max_source_length": max_source_length,
                    },
                    # with_indices=True,
                    num_proc=8,
                )
                dataset_splits.append(tmp_split)
                # -----------------------------------------------------------------
                # validation set to generate plans and summaries during evaluation|
                # -----------------------------------------------------------------
                tokenizer.padding_side = "left"
                tmp_split = dataset_split.map(
                    preprocess,
                    fn_kwargs={
                        # "document_key": document_key,
                        "task_prompt": training_task_prompt,
                        "tokenizer": tokenizer,
                        "max_source_length": max_source_length,
                    },
                    num_proc=8,
                )
                dataset_splits.append(tmp_split)
            elif task == "multitask":
                # -----------------------------------------------
                # validation set to compute loss during training|
                # -----------------------------------------------
                tokenizer.padding_side = "right"
                plan_task_ds = dataset_split.map(
                    preprocess_train,
                    fn_kwargs={
                        # "document_key": document_key,
                        # "summary_key": summary_key,
                        "task": task,
                        "task_prompt": training_task_plan_prompt,
                        "tokenizer": tokenizer,
                        "plans": validation_plans,
                        "max_source_length": max_source_length,
                    },
                    # with_indices=True,
                    num_proc=8,
                )
                # not sure if we want to also compute validation loss over the plans
                # maybe we should
                summary_task_ds = dataset_split.map(
                    preprocess_train,
                    fn_kwargs={
                        # "document_key": document_key,
                        # "summary_key": summary_key,
                        "task": task,
                        "task_prompt": training_task_summary_prompt,
                        "tokenizer": tokenizer,
                        "plans": None,
                        "max_source_length": max_source_length,
                    },
                    # with_indices=True,
                    num_proc=8,
                )
                tmp_split = concatenate_datasets([plan_task_ds, summary_task_ds])
                dataset_splits.append(tmp_split)
                # -----------------------------------------------------------------
                # validation set to generate plans and summaries during evaluation|
                # -----------------------------------------------------------------
                tokenizer.padding_side = "left"
                plan_task_ds = dataset_split.map(
                    preprocess,
                    fn_kwargs={
                        # "document_key": document_key,
                        "task_prompt": training_task_plan_prompt,
                        "tokenizer": tokenizer,
                        "max_source_length": max_source_length,
                    },
                    num_proc=8,
                )
                summary_task_ds = dataset_split.map(
                    preprocess,
                    fn_kwargs={
                        # "document_key": document_key,
                        "task_prompt": training_task_summary_prompt,
                        "tokenizer": tokenizer,
                        "max_source_length": max_source_length,
                    },
                    num_proc=8,
                )
                # concatenate plan_task_ds and summary_task_ds
                tmp_split = concatenate_datasets([plan_task_ds, summary_task_ds])
                dataset_splits.append(tmp_split)
        elif split == "test":
            # DONE
            tokenizer.padding_side = "left"
            if task == "e2e" or task == "baseline":
                dataset_split = dataset_split.map(
                    preprocess,
                    fn_kwargs={
                        # "document_key": document_key,
                        "task_prompt": training_task_prompt,
                        "tokenizer": tokenizer,
                        "max_source_length": max_source_length,
                    },
                    num_proc=8,
                )
            elif task == "multitask":
                plan_task_ds = dataset_split.map(
                    preprocess,
                    fn_kwargs={
                        # "document_key": document_key,
                        "task_prompt": training_task_plan_prompt,
                        "tokenizer": tokenizer,
                        "max_source_length": max_source_length,
                    },
                    num_proc=8,
                )
                summary_task_ds = dataset_split.map(
                    preprocess,
                    fn_kwargs={
                        # "document_key": document_key,
                        "task_prompt": training_task_summary_prompt,
                        "tokenizer": tokenizer,
                        "max_source_length": max_source_length,
                    },
                    num_proc=8,
                )
                # concatenate plan_task_ds and summary_task_ds
                dataset_split = concatenate_datasets([plan_task_ds, summary_task_ds])
            dataset_splits.append(dataset_split)
    return dataset_splits


def numbered_list_to_paragraph(text):

    # Split the text into lines
    lines = text.strip().split("\n")

    # Remove numbering and extra whitespace from each line
    sentences = [
        re.sub(r"^\s*\d+\.\s*", "", line.strip()) for line in lines if line.strip()
    ]

    # Join the sentences into a single paragraph
    paragraph = " ".join(sentences)

    return paragraph


def get_plans_from_file(plan_file_path: str, plan_key: str):
    training_plans = {}
    with jsonlines.open(plan_file_path, "r") as f:
        for line in f:
            training_plans[line["idx"]] = line[plan_key]
    print(f"Plans found: {len(training_plans)}")
    return training_plans


def parse_cmd_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, default="microsoft/Phi-3-mini-128k-instruct"
    )
    parser.add_argument(
        "--dataset_name",
        choices=["squality", "summscreen"],
        default="squality",
        help="Dataset name",
    )
    parser.add_argument(
        "--dataset_cache_path", type=str, default="/home/ubuntu/hf_data_cache"
    )
    parser.add_argument("--tr_plan_file_path", type=str, default=None)
    parser.add_argument("--val_plan_file_path", type=str, default=None)
    parser.add_argument(
        "--plan_key",
        type=str,
        default="plan_without_citation",
        help="plan key in the JSON object",
    )
    parser.add_argument("--task", type=str, default="baseline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_source_length", type=int, default=8192)
    parser.add_argument("--max_summary_length", type=int, default=512)
    parser.add_argument("--output_dir", type=str, default=None, required=True)
    parser.add_argument(
        "--min_num_sentences",
        type=int,
        default=1,
        help="minimum number of sentences in the reference summary to retain that data point",
    )

    args = parser.parse_args()
    return args


def main():

    args = parse_cmd_args()

    if args.dataset_name == "squality":
        args.dataset_name = "pszemraj/SQuALITY-v1.3"
    elif args.dataset_name == "summscreen":
        args.dataset_name = "YuanPJ/summ_screen"
    else:
        raise ValueError("Invalid dataset name")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.model_max_length = args.max_source_length + args.max_summary_length
    tokenizer.pad_token = (
        tokenizer.unk_token
    )  # use unk rather than eos token to prevent endless generation
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    # if args.task == "e2e" or args.task == "multitask":
    #     # add special tokens
    #     special_tokens_dict = {"additional_special_tokens": ["<|plan|>", "<|summary|>"]}
    #     tokenizer.add_special_tokens(special_tokens_dict)

    train_dataset, val_dataset_for_tr, val_dataset, test_dataset = (
        get_all_dataset_splits(
            tokenizer,
            args.dataset_name,
            args.dataset_cache_path,
            args.tr_plan_file_path,
            args.val_plan_file_path,
            args.plan_key,
            args.max_source_length,
            args.task,
            args.min_num_sentences,
        )
    )
    # train_dataset.set_format("pt", columns=list(train_dataset.features))
    # validation_dataset.set_format("pt", columns=list(validation_dataset.features))
    # test_dataset.set_format("pt", columns=list(test_dataset.features))
    # test_dataset = test_dataset.select(range(52))

    dd = DatasetDict(
        {
            "train": train_dataset,
            "test": test_dataset,
            "validation": val_dataset,
            "validation_for_training": val_dataset_for_tr,
        }
    )

    # create output_dir if it doesn't exist
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
    dataset_file_prefix = f"{args.dataset_name.split('/')[-1]}_msrcl_{args.max_source_length}_msumml_{args.max_summary_length}_msent_{args.min_num_sentences}"
    dd.save_to_disk(os.path.join(args.output_dir, dataset_file_prefix))


if __name__ == "__main__":
    main()
