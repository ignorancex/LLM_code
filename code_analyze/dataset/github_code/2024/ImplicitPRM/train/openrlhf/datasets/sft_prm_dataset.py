from typing import Callable
from itertools import chain

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .utils import zero_pad_sequences


def preprocess_data(data, input_template=None, input_key="input", output_key=None, apply_chat_template=None):
    if apply_chat_template:
        if output_key:
            prompt = apply_chat_template(data[input_key], tokenize=False, add_generation_prompt=True)
            response = apply_chat_template(data[input_key] + data[output_key], tokenize=False)[len(prompt) :]
        else:
            prompt = apply_chat_template(data[input_key][:-1], tokenize=False, add_generation_prompt=True)
            response = apply_chat_template(data[input_key], tokenize=False)[len(prompt) :]
    else:
        prompt = data[input_key]
        if input_template:
            prompt = input_template.format(prompt)
        # output_key is None for continue pretrain
        response = data[output_key] if output_key else ""
    return prompt, response, data["process_label"], data['correctness']

class SFTPRMDataset(Dataset):
    """
    Dataset for SFT model

    Args:
        dataset: dataset for SFT model
        tokenizer: tokenizer for SFT model
        max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer: Callable,
        max_length: int,
        strategy,
        input_template=None,
        pretrain_mode=False,
        num_processors=8,  # Specify the number of processors you want to use
        apply_template=False
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.pretrain_mode = pretrain_mode
        self.max_length = max_length
        self.apply_template = apply_template
        # chat template
        self.input_template = input_template
        self.input_key = getattr(self.strategy.args, "input_key", None)
        self.output_key = getattr(self.strategy.args, "output_key", None)
        self.apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)

        if self.apply_chat_template:
            self.apply_chat_template = self.tokenizer.apply_chat_template
            tokenizer_chat_template = getattr(self.strategy.args, "tokenizer_chat_template", None)
            if tokenizer_chat_template:
                self.tokenizer.chat_template = tokenizer_chat_template

        # Parallel loading datasets
        processed_dataset = dataset.map(
            self.process_data, remove_columns=dataset.column_names, num_proc=num_processors
        )
        processed_dataset = processed_dataset.filter(lambda x: x["prompt"] is not None)

        # Store the processed data in class attributes
        self.prompts = processed_dataset["prompt"]
        self.responses = processed_dataset["response"]
        self.process_labels = processed_dataset["process_label"]
        self.correctness = processed_dataset["correctness"]
        self.prompt_ids_lens = processed_dataset["prompt_ids_len"]

        for prompt, response in zip(self.prompts[:3], self.responses[:3]):
            print("prompt:", prompt)
            print("response:", response)
            print("------")

        self.eos_tid = self.tokenizer.eos_token_id
        # Usage: https://github.com/meta-llama/llama3/issues/77
        self.sep_tid, self.neg_tid, self.pos_tid = self.tokenizer.convert_tokens_to_ids([
            "<|reserved_special_token_0|>", 
            # "<|reserved_special_token_1|>",
            # "<|reserved_special_token_2|>"
            "-",
            "+"
        ])
        # neg_tid, pos_tid = self.tokenizer.convert_tokens_to_ids(["<NEG>", "<POS>"])
        self.plabel2tid = {
            0: [self.neg_tid],
            1: [self.pos_tid]
        }

    def process_data(self, data):
        prompt, response, process_label,correctness = preprocess_data(
            data,
            None if self.pretrain_mode else self.input_template,
            self.input_key,
            self.output_key,
            apply_chat_template=None if self.pretrain_mode else self.apply_chat_template,
        )
        if not self.pretrain_mode:
            prompt_token = self.tokenizer(
                prompt,
                max_length=self.max_length,
                padding=False,
                truncation=True,
                return_tensors="pt",
                add_special_tokens=False,
            )
            prompt_ids_len = prompt_token["attention_mask"].int().sum().item()

            # filter the sample whose length is greater than max_length (2 for answer length)
            if not prompt or not response or prompt_ids_len >= self.max_length - 2:
                prompt = None
        else:
            prompt_ids_len = 0

        if self.apply_template:
            prompt = self.tokenizer.apply_chat_template([
            {"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)

        return {"prompt": prompt, "response": response, "prompt_ids_len": prompt_ids_len, "process_label": process_label,'correctness':correctness}

    def __len__(self):
        length = len(self.prompts)
        return length

    def __getitem__(self, idx):
        prompt_ids_len = self.prompt_ids_lens[idx]
        prompt = self.prompts[idx]
        response = self.responses[idx]
        process_label = self.process_labels[idx]
        correctness = self.correctness[idx]
        # if not self.pretrain_mode:
        #     text = (prompt + response).rstrip("\n")
        #     if not text.endswith(self.tokenizer.eos_token):
        #         text += " " + self.tokenizer.eos_token
        # else:
        #     text = prompt

        # input_token = self.tokenizer(
        #     text,
        #     max_length=self.max_length,
        #     padding=False,
        #     truncation=True,
        #     return_tensors="pt",
        #     add_special_tokens=False,
        # )

        # if not self.pretrain_mode:
        #     # to avoid EOS_token truncation
        #     input_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
        #     input_token["attention_mask"][0][-1] = True
        if isinstance(response, list):
            response_lst = response
        else:
            response_lst = response.split("\n\n")
    
        prompt_ids = self.tokenizer(prompt, padding=False, add_special_tokens=False if self.apply_template else True).input_ids
        processes_ids = self.tokenizer(response_lst, padding=False, add_special_tokens=False).input_ids
        assert len(processes_ids) == len(process_label), (len(processes_ids), len(process_label))

        # prompt <sep> process1 <sep> process2 <sep> process3 <sep> <eos>
        # -100   -100   -100    <pos>  -100    <neg>  -100    <neg> -100
        if self.apply_template:
            sep_lst1 = []
            sep_lst2 = []
        else:
            sep_lst1 = [self.sep_tid]
            sep_lst2 = [-100]

        # input_ids = prompt_ids + sep_lst1 + list(chain(*[lst + [self.sep_tid] for lst in processes_ids])) + [self.eos_tid]
        # labels = [-100] * len(prompt_ids) + sep_lst2 + list(chain(*[[-100]*len(lst) + self.plabel2tid[process_label[lid]] for lid, lst in enumerate(processes_ids)])) + [self.eos_tid]

        input_ids = prompt_ids + sep_lst1 + list(chain(*[lst for lst in processes_ids])) + [self.sep_tid] + [self.eos_tid]
        labels = [-100] * len(prompt_ids) + sep_lst2 + list(chain(*[[-100] * len(lst) for lid, lst in enumerate(processes_ids)]))+ self.plabel2tid[correctness]+ [self.eos_tid]

        assert len(input_ids) == len(labels)

        length_gap = self.max_length - len(input_ids)
        if length_gap >= 0:
            attention_mask = [1] * len(input_ids) + [0] * length_gap
            input_ids += [self.eos_tid] * length_gap
            labels += [-100] * length_gap
        else:
            attention_mask = [1] * self.max_length
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]
            input_ids[-1] = self.eos_tid

        for idx, inp_id in enumerate(input_ids):
            if inp_id in [self.sep_tid, self.pos_tid, self.neg_tid]:
                attention_mask[idx] = 0

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.bool)
        labels = torch.tensor(labels, dtype=torch.long)

        info = {"input": prompt, "output": response, "input_length": attention_mask.int().sum().item(), "labels": labels}

        return prompt_ids_len, input_ids, attention_mask, info

    def collate_fn(self, item_list):
        prompt_ids_lens = []
        input_ids = []
        attention_masks = []
        infos = {"input": [], "output": [], "labels": []}

        for prompt_ids_len, input_id, attention_mask, info in item_list:
            prompt_ids_lens.append(prompt_ids_len)
            input_ids.append(input_id)
            attention_masks.append(attention_mask)
            infos["input"].append(info["input"])
            infos["output"].append(info["output"])
            infos["labels"].append(info["labels"])
            
        input_ids = torch.stack(input_ids, dim=0)
        attention_masks = torch.stack(attention_masks, dim=0)
        infos["labels"] = torch.stack(infos["labels"], dim=0)
        # input_ids = zero_pad_sequences(input_ids, "right", self.tokenizer.pad_token_id)
        # attention_masks = zero_pad_sequences(attention_masks, "right")
        return prompt_ids_lens, input_ids, attention_masks, infos

    def packing_collate_fn(self, item_list):
        packed_input_ids = []
        packed_attention_masks = []
        prompt_ids_lens = []
        infos = {"input_length": []}

        index = 1
        for prompt_ids_len, input_id, attention_mask, info in item_list:
            packed_input_ids.append(input_id.flatten())
            packed_attention_masks.append(torch.ones_like(input_id.flatten()) * index)
            prompt_ids_lens.append(prompt_ids_len)
            infos["input_length"].append(info["input_length"])
            index += 1

        packed_input_ids = torch.cat(packed_input_ids, dim=0).unsqueeze(0)
        packed_attention_masks = torch.cat(packed_attention_masks, dim=0).unsqueeze(0)

        return prompt_ids_lens, packed_input_ids, packed_attention_masks, infos
