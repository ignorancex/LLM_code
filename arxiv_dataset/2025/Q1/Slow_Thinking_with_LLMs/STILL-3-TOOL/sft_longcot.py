import os
import json
import copy
import torch
import random
import logging
import transformers
from tqdm import tqdm
from torch.utils.data import Dataset
from dataclasses import dataclass, field
from torch.utils.data import random_split
from typing import Optional, Dict, Sequence
from torch.nn.utils.rnn import pad_sequence
from typing import List, Optional, Tuple, Union
from datasets import load_from_disk, load_dataset
from transformers import Trainer, AutoModelForCausalLM


IGNORE_INDEX = -100


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None)


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    prompt_version: str = field(
        default=None, metadata={"help": "Path to the prompt version."}
    )
    prompt_template: str = field(
        default=None, metadata={"help": "Path to the prompt template."}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        data_path: str = None,
        prompt_version: str = None,
        prompt_template: str = None,
    ):
        super(SupervisedDataset, self).__init__()
        logging.warning(f"Loading data from {data_path} and use {prompt_version}...")
        self.tokenizer = tokenizer
        self.input_ids, self.labels = [], []
        self.prompt_version = prompt_version
        self.prompt = self.load_prompt(prompt_version, prompt_template)
        if os.path.isdir(data_path):
            try:
                all_data = load_from_disk(data_path)
            except Exception:
                all_data = load_dataset(data_path)
            all_data = all_data["train"] if "train" in all_data else all_data
            for d in all_data:
                input_ids, labels = self.encode_src_tgt_v2(
                    d["source"], d["full"], tokenizer
                )
                self.input_ids.append(input_ids)
                self.labels.append(labels)
        else:
            all_paths = data_path.split(",")
            print(f"all_paths: {all_paths}")
            for path in all_paths:
                with open(path, "r") as f:
                    for i, line in tqdm(enumerate(f.readlines())):
                        c = json.loads(line)
                        # revise based on the specific data format
                        s = c["input"]
                        t = c["output"]
                        input_ids, labels = self.encode_src_tgt(s, t, tokenizer, c)
                        self.input_ids.append(input_ids)
                        self.labels.append(labels)

        input_tokens = self.tokenizer.convert_ids_to_tokens(input_ids.tolist())
        label_tokens = self.tokenizer.convert_ids_to_tokens(
            [l if l != -100 else self.tokenizer.pad_token_id for l in labels.tolist()]
        )
        print(f"Input: [{input_tokens}]")
        print(f"Output: [{label_tokens}]")

    def encode_src_tgt(self, s, t, tokenizer, c):
        # revise based on the specific prompt template
        chat = [
            {"role": "user", "content": f"{self.prompt}{s}"},
            {"role": "assistant", "content": f"{t}"},
        ]
        if self.prompt_version == "empty":
            # applied chat template
            source = s
            full = s + t
        else:
            source = tokenizer.apply_chat_template(
                chat[:-1], tokenize=False, add_generation_prompt=True
            )
            full = tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=False
            )
        source_id = tokenizer.encode(
            source,
            max_length=tokenizer.model_max_length,
            truncation=True,
            add_special_tokens=False,
            return_tensors="pt",
        )[0]
        full_id = tokenizer.encode(
            full,
            max_length=tokenizer.model_max_length,
            truncation=True,
            add_special_tokens=False,
            return_tensors="pt",
        )[0]
        label = full_id.clone()
        label[: len(source_id)] = IGNORE_INDEX
        return full_id, label

    def load_prompt(self, prompt_version, prompt_template):
        if prompt_version == "empty" or prompt_template is None:
            return ""
        with open(prompt_template) as f:
            prompt_dict = json.load(f)
            print(f"We utilize prompt of:\n{prompt_dict[prompt_version]}")
            return prompt_dict[prompt_version]

    def encode_src_tgt_v2(self, source, target, tokenizer):
        source_id = tokenizer.encode(
            source,
            max_length=tokenizer.model_max_length,
            truncation=True,
            add_special_tokens=False,
            return_tensors="pt",
        )[0]
        full_id = tokenizer.encode(
            target,
            max_length=tokenizer.model_max_length,
            truncation=True,
            add_special_tokens=False,
            return_tensors="pt",
        )[0]
        label = full_id.clone()
        label[: len(source_id)] = IGNORE_INDEX
        return full_id, label

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )

        return dict(
            input_ids=input_ids,
            labels=labels,
        )


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset = SupervisedDataset(
        tokenizer=tokenizer,
        data_path=data_args.data_path,
        prompt_version=data_args.prompt_version,
        prompt_template=data_args.prompt_template,
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=dataset, eval_dataset=None, data_collator=data_collator)


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    print("==========Model Args=========")
    print(model_args)
    print("==========Data Args=========")
    print(data_args)
    print("==========Training Args=========")
    print(training_args)

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        _attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, model_max_length=training_args.model_max_length
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # STILL-3-TOOL-32B: deprecated
    # special_tokens_dict = {
    #     "begin_of_step": "<|begin_of_step|>",
    #     "end_of_step": "<|end_of_step|>",
    #     "begin_of_thought": "<|begin_of_thought|>",
    #     "end_of_thought": "<|end_of_thought|>",
    #     "begin_of_solution": "<|begin_of_solution|>",
    #     "end_of_solution": "<|end_of_solution|>",
    # }
    # special_tokens_dict = {
    #     "additional_special_tokens": [
    #         "<|begin_of_step|>",
    #         "<|end_of_step|>",
    #         "<|begin_of_thought|>",
    #         "<|end_of_thought|>",
    #         "<|begin_of_solution|>",
    #         "<|end_of_solution|>",
    #     ]
    #     + tokenizer.special_tokens_map["additional_special_tokens"]
    # }

    # STILL-3-TOOL-32B: deprecated
    # num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    # print("We have added", num_added_toks, "tokens")
    # # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e., the length of the tokenizer.
    # model.resize_token_embeddings(len(tokenizer))

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model(training_args.output_dir)
    trainer.save_state()


if __name__ == "__main__":
    torch.manual_seed(42)
    train()
