from typing import Callable

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm

from .utils import exist_and_not_none, zero_pad_sequences


def preprocess_data(
    data,
    input_template=None,
    prompt_key=None,
    chosen_key="chosen",
    rejected_key="rejected",
    apply_chat_template=None,
    is_dpo=False,
) -> str:
    if apply_chat_template:
        if prompt_key:
            prompt = apply_chat_template(data[prompt_key], tokenize=False, add_generation_prompt=True)
            chosen = apply_chat_template(data[prompt_key] + data[chosen_key], tokenize=False)[len(prompt) :]
            rejected = apply_chat_template(data[prompt_key] + data[rejected_key], tokenize=False)[len(prompt) :]
        else:
            prompt = ""
            chosen = apply_chat_template(data[chosen_key], tokenize=False,add_generation_prompt=False)
            rejected = apply_chat_template(data[rejected_key], tokenize=False,add_generation_prompt=False)
            # if chosen.endswith("<|eot_id|><|start_header_id|>assistant<|end_header_id|>"):
            #     chosen = chosen.strip()
            #     rejected = rejected.strip()
            #     chosen = chosen[:-len("<|eot_id|><|start_header_id|>assistant<|end_header_id|>")]
            #     rejected = rejected[:-len("<|eot_id|><|start_header_id|>assistant<|end_header_id|>")]

            if is_dpo:
                prompt = apply_chat_template(data[chosen_key][:-1], tokenize=False, add_generation_prompt=True)
                chosen = chosen[len(prompt) :]
                rejected = rejected[len(prompt) :]
    else:
        if prompt_key:
            prompt = data[prompt_key]
            if input_template:
                prompt = input_template.format(prompt)
        else:
            prompt = ""
        chosen = data[chosen_key]
        rejected = data[rejected_key]

    # margin loss
    margin = data["margin"] if exist_and_not_none(data, "margin") else 0

    return prompt, chosen, rejected, margin


class RewardDataset(Dataset):
    """
    Dataset for reward model

    Args:
        dataset: dataset for reward model
        self.tokenizer: self.tokenizer for reward model
        self.max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer: Callable,
        max_length: int,
        strategy,
        input_template=None,
        is_dpo=False,
        num_processors=8,
        multiple_of=1,
    ) -> None:
        super().__init__()
        self.is_dpo = is_dpo
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.max_length = max_length
        self.multiple_of = multiple_of

        # chat_template
        self.input_template = input_template
        self.prompt_key = getattr(self.strategy.args, "prompt_key", None)
        self.chosen_key = getattr(self.strategy.args, "chosen_key", None)
        self.rejected_key = getattr(self.strategy.args, "rejected_key", None)
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
        print(processed_dataset[0])
        # Filter out None values if necessary
        processed_dataset = processed_dataset.filter(lambda x: x["prompt"] is not None)

        # Store the processed data in class attributes
        self.prompts = processed_dataset["prompt"]
        self.chosens = processed_dataset["chosen"]
        self.rejects = processed_dataset["reject"]
        self.extras = processed_dataset["extra"]

    def process_data(self, data):
        prompt, chosen, reject, margin = preprocess_data(
            data,
            self.input_template,
            self.prompt_key,
            self.chosen_key,
            self.rejected_key,
            self.apply_chat_template,
            self.is_dpo,
        )

        if self.is_dpo:
            prompt_token = self.tokenizer(
                prompt,
                max_length=self.max_length,
                padding=False,
                truncation=True,
                return_tensors="pt",
                add_special_tokens=False,
            )
            prompt_ids_len = prompt_token["attention_mask"].int().sum().item()

            # Filter the sample whose length is greater than max_length (2 for answer length)
            if prompt_ids_len >= self.max_length - 2:
                prompt = None

        return {
            "prompt": prompt,
            "chosen": chosen,
            "reject": reject,
            "extra": prompt_ids_len if self.is_dpo else margin,
        }

    def __len__(self):
        length = len(self.chosens)
        return length

    def __getitem__(self, idx):
        prompt, chosen, reject, extra = self.prompts[idx], self.chosens[idx], self.rejects[idx], self.extras[idx]

        chosen = (prompt + chosen).rstrip("\n")
        if not chosen.endswith(self.tokenizer.eos_token):
            chosen += " " + self.tokenizer.eos_token
        chosen_token = self.tokenizer(
            chosen,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )

        reject = (prompt + reject).rstrip("\n")
        if not reject.endswith(self.tokenizer.eos_token):
            reject += " " + self.tokenizer.eos_token
        reject_token = self.tokenizer(
            reject,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )

        # to avoid EOS_token truncation
        chosen_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
        reject_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
        chosen_token["attention_mask"][0][-1] = True
        reject_token["attention_mask"][0][-1] = True

        return (
            chosen_token["input_ids"],
            chosen_token["attention_mask"],
            reject_token["input_ids"],
            reject_token["attention_mask"],
            extra,
        )

    def collate_fn(self, item_list):
        chosen_ids = []
        chosen_masks = []
        reject_ids = []
        rejects_masks = []
        extras = []
        for chosen_id, chosen_mask, reject_id, rejects_mask, extra in item_list:
            chosen_ids.append(chosen_id)
            chosen_masks.append(chosen_mask)
            reject_ids.append(reject_id)
            rejects_masks.append(rejects_mask)
            extras.append(extra)

        if self.is_dpo:
            padding_side = "right"
        else:
            padding_side = "left"
        chosen_ids = zero_pad_sequences(chosen_ids, side=padding_side, value=self.tokenizer.pad_token_id)
        chosen_masks = zero_pad_sequences(chosen_masks, side=padding_side)
        reject_ids = zero_pad_sequences(reject_ids, side=padding_side, value=self.tokenizer.pad_token_id)
        rejects_masks = zero_pad_sequences(rejects_masks, side=padding_side)
        return chosen_ids, chosen_masks, reject_ids, rejects_masks, extras

    def packing_collate_fn(self, item_list):
        extras = []

        chosen_ids = []
        chosen_att_masks = []
        chosen_seq_lens = []
        rejected_ids = []
        rejected_att_masks = []
        rejected_seq_lens = []
        index = 1
        for chosen_id, chosen_mask, reject_id, rejects_mask, extra in item_list:
            chosen_ids.append(chosen_id.flatten())
            chosen_att_masks.append(torch.full_like(chosen_id.flatten(), index))
            chosen_seq_lens.append(len(chosen_id.flatten()))
            extras.append(extra)

            rejected_ids.append(reject_id.flatten())
            rejected_att_masks.append(torch.full_like(reject_id.flatten(), index + len(item_list)))
            rejected_seq_lens.append(len(reject_id.flatten()))
            index += 1

        packed_input_ids = torch.cat(chosen_ids + rejected_ids, dim=0).unsqueeze(0)
        packed_attention_masks = torch.cat(chosen_att_masks + rejected_att_masks, dim=0).unsqueeze(0)
        packed_seq_lens = chosen_seq_lens + rejected_seq_lens

        if self.multiple_of > 1 and packed_input_ids.numel() % self.multiple_of != 0:
            padding_len = self.multiple_of - (packed_input_ids.numel() % self.multiple_of)
            packed_input_ids = F.pad(packed_input_ids, (0, padding_len), value=self.tokenizer.pad_token_id)
            packed_attention_masks = F.pad(packed_attention_masks, (0, padding_len), value=0)

        return packed_input_ids, packed_attention_masks, packed_seq_lens, extras


class RewardDataset_ICB(Dataset):
    """
    Dataset for reward model

    Args:
        dataset: dataset for reward model
        self.tokenizer: self.tokenizer for reward model
        self.max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer: Callable,
        max_length: int,
        strategy,
        input_template=None,
        is_dpo=False,
        num_processors=8,
        multiple_of=1,
        batch_size =4,
    ) -> None:
        super().__init__()
        self.is_dpo = is_dpo
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.max_length = max_length
        self.multiple_of = multiple_of
        self.batch_size = batch_size

        # chat_template
        self.input_template = input_template
        self.prompt_key = getattr(self.strategy.args, "prompt_key", None)
        self.chosen_key = getattr(self.strategy.args, "chosen_key", None)
        self.rejected_key = getattr(self.strategy.args, "rejected_key", None)
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

        # Filter out None values if necessary
        processed_dataset = processed_dataset.filter(lambda x: x["prompt"] is not None)
        processed_dataset = self.batching_data(processed_dataset)

        # Store the processed data in class attributes
        self.prompts = processed_dataset["prompt"]
        self.chosens = processed_dataset["chosen"]
        self.rejects = processed_dataset["reject"]
        self.extras = processed_dataset["extra"]

    def batching_data(self,dataset):
        i=0
        pre_prompt = None
        final_dataset = {'prompt':[],'chosen':[],'reject':[],'extra':[]}
        while i < len(dataset):
            if dataset[i]['prompt'] != pre_prompt or len(prompts)>=self.batch_size:
                pre_prompt = dataset[i]['prompt']
                if i!=0:
                    final_dataset['prompt'].append(prompts)
                    final_dataset['chosen'].append(chosen)
                    final_dataset['reject'].append(rejected)
                    final_dataset['extra'].append(extra)
                prompts, chosen, rejected, extra = [], [], [], []
            prompts.append(dataset[i]['prompt'])
            chosen.append(dataset[i]['chosen'])
            rejected.append(dataset[i]['reject'])
            extra.append(dataset[i]['extra'])
            i+=1
        return final_dataset

    def batching_data_by_length(self,dataset):
        i=0
        pre_prompt = None
        final_dataset = {'prompt':[],'chosen':[],'reject':[],'extra':[]}
        while i < len(dataset):
            if dataset[i]['prompt'] != pre_prompt:
                pre_prompt = dataset[i]['prompt']
                if i!=0:
                    final_dataset['prompt'].append(prompts)
                    final_dataset['chosen'].append(chosen)
                    final_dataset['reject'].append(rejected)
                    final_dataset['extra'].append(extra)
                prompts, chosen, rejected, extra = [], [], [], []
                token_length = 0

            chosen_token = self.tokenizer(
                dataset[i]['prompt']+dataset[i]['chosen'],
                return_tensors="pt",
                add_special_tokens=False,
            )
            reject_token = self.tokenizer(
                dataset[i]['prompt']+dataset[i]['reject'],
                return_tensors="pt",
                add_special_tokens=False,
            )
            if token_length + chosen_token["attention_mask"].int().sum().item() + reject_token["attention_mask"].int().sum().item() > self.max_length:
                final_dataset['prompt'].append(prompts)
                final_dataset['chosen'].append(chosen)
                final_dataset['reject'].append(rejected)
                final_dataset['extra'].append(extra)
                prompts, chosen, rejected, extra = [dataset[i]['prompt']], [dataset[i]['chosen']], [dataset[i]['reject']], [dataset[i]['extra']]
                token_length = chosen_token["attention_mask"].int().sum().item()+reject_token["attention_mask"].int().sum().item()
            else:
                prompts.append(dataset[i]['prompt'])
                chosen.append(dataset[i]['chosen'])
                rejected.append(dataset[i]['reject'])
                extra.append(dataset[i]['extra'])
                token_length += chosen_token["attention_mask"].int().sum().item() + reject_token["attention_mask"].int().sum().item()
            i+=1
        return final_dataset


    def process_data(self, data):
        prompt, chosen, reject, margin = preprocess_data(
            data,
            self.input_template,
            self.prompt_key,
            self.chosen_key,
            self.rejected_key,
            self.apply_chat_template,
            self.is_dpo,
        )

        if self.is_dpo:
            prompt_token = self.tokenizer(
                prompt,
                max_length=self.max_length,
                padding=False,
                truncation=True,
                return_tensors="pt",
                add_special_tokens=False,
            )
            prompt_ids_len = prompt_token["attention_mask"].int().sum().item()

            # Filter the sample whose length is greater than max_length (2 for answer length)
            if prompt_ids_len >= self.max_length - 2:
                prompt = None

        return {
            "prompt": prompt,
            "chosen": chosen,
            "reject": reject,
            "extra": prompt_ids_len if self.is_dpo else margin,
        }

    def __len__(self):
        length = len(self.chosens)
        return length

    def __getitem__(self, idx):
        prompt_list, chosen_list, reject_list, extra_list = self.prompts[idx], self.chosens[idx], self.rejects[idx], self.extras[idx]
        item_list = []
        for prompt,chosen,reject,extra in zip(prompt_list, chosen_list, reject_list, extra_list):
            chosen = (prompt + chosen).rstrip("\n")
            if not chosen.endswith(self.tokenizer.eos_token):
                chosen += " " + self.tokenizer.eos_token
            chosen_token = self.tokenizer(
                chosen,
                max_length=self.max_length,
                padding=False,
                truncation=True,
                return_tensors="pt",
                add_special_tokens=False,
            )

            reject = (prompt + reject).rstrip("\n")
            if not reject.endswith(self.tokenizer.eos_token):
                reject += " " + self.tokenizer.eos_token
            reject_token = self.tokenizer(
                reject,
                max_length=self.max_length,
                padding=False,
                truncation=True,
                return_tensors="pt",
                add_special_tokens=False,
            )

            # to avoid EOS_token truncation
            chosen_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
            reject_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
            chosen_token["attention_mask"][0][-1] = True
            reject_token["attention_mask"][0][-1] = True

            item_list.append(
                (chosen_token["input_ids"],
                chosen_token["attention_mask"],
                reject_token["input_ids"],
                reject_token["attention_mask"],
                extra)
            )
        if not self.strategy.args.packing_samples:
            chosen_ids, chosen_masks, reject_ids, rejects_masks, extras = self.batch_collate_fn(item_list)
            return chosen_ids, chosen_masks, reject_ids, rejects_masks, extras
        else:
            packed_input_ids, packed_attention_masks, packed_seq_lens, extras = self.batch_packing_collate_fn(item_list)
            return packed_input_ids, packed_attention_masks, packed_seq_lens, extras


    def batch_collate_fn(self, item_list):
        chosen_ids = []
        chosen_masks = []
        reject_ids = []
        rejects_masks = []
        extras = []
        for chosen_id, chosen_mask, reject_id, rejects_mask, extra in item_list:
            chosen_ids.append(chosen_id)
            chosen_masks.append(chosen_mask)
            reject_ids.append(reject_id)
            rejects_masks.append(rejects_mask)
            extras.append(extra)

        if self.is_dpo:
            padding_side = "right"
        else:
            padding_side = "left"
        chosen_ids = zero_pad_sequences(chosen_ids, side=padding_side, value=self.tokenizer.pad_token_id)
        chosen_masks = zero_pad_sequences(chosen_masks, side=padding_side)
        reject_ids = zero_pad_sequences(reject_ids, side=padding_side, value=self.tokenizer.pad_token_id)
        rejects_masks = zero_pad_sequences(rejects_masks, side=padding_side)
        return chosen_ids, chosen_masks, reject_ids, rejects_masks, extras

    def batch_packing_collate_fn(self, item_list):
        extras = []

        chosen_ids = []
        chosen_att_masks = []
        chosen_seq_lens = []
        rejected_ids = []
        rejected_att_masks = []
        rejected_seq_lens = []
        index = 1
        for chosen_id, chosen_mask, reject_id, rejects_mask, extra in item_list:
            chosen_ids.append(chosen_id.flatten())
            chosen_att_masks.append(torch.full_like(chosen_id.flatten(), index))
            chosen_seq_lens.append(len(chosen_id.flatten()))
            extras.append(extra)

            rejected_ids.append(reject_id.flatten())
            rejected_att_masks.append(torch.full_like(reject_id.flatten(), index + len(item_list)))
            rejected_seq_lens.append(len(reject_id.flatten()))
            index += 1

        packed_input_ids = torch.cat(chosen_ids + rejected_ids, dim=0).unsqueeze(0)
        packed_attention_masks = torch.cat(chosen_att_masks + rejected_att_masks, dim=0).unsqueeze(0)
        packed_seq_lens = chosen_seq_lens + rejected_seq_lens

        if self.multiple_of > 1 and packed_input_ids.numel() % self.multiple_of != 0:
            padding_len = self.multiple_of - (packed_input_ids.numel() % self.multiple_of)
            packed_input_ids = F.pad(packed_input_ids, (0, padding_len), value=self.tokenizer.pad_token_id)
            packed_attention_masks = F.pad(packed_attention_masks, (0, padding_len), value=0)

        return packed_input_ids, packed_attention_masks, packed_seq_lens, extras


    def collate_fn(self, item_list):
        assert len(item_list)==1,item_list
        chosen_ids, chosen_masks, reject_ids, rejects_masks, extras = item_list[0]
        return chosen_ids, chosen_masks, reject_ids, rejects_masks, extras

    def packing_collate_fn(self, item_list):
        assert len(item_list)==1,item_list
        packed_input_ids, packed_attention_masks, packed_seq_lens, extras = item_list[0]
        return packed_input_ids, packed_attention_masks, packed_seq_lens, extras



def preprocess_data_twostage(
    data,
    input_template=None,
    prompt_key=None,
    chosen_key="chosen",
    rejected_key="rejected",
    apply_chat_template=None,
    is_dpo=False,
    tokenizer=None,
) -> str:

    if prompt_key:
        prompt = apply_chat_template(data[prompt_key], tokenize=False, add_generation_prompt=True)
        chosen = apply_chat_template(data[prompt_key] + data[chosen_key], tokenize=False)[len(prompt) :]
        # rejected = apply_chat_template(data[prompt_key] + data[rejected_key], tokenize=False)[len(prompt) :]
    else:
        prompt = ""
        chosen = apply_chat_template(data[chosen_key], tokenize=False)
        # rejected = apply_chat_template(data[rejected_key], tokenize=False)

        if is_dpo:
            prompt = apply_chat_template(data[chosen_key][:-1], tokenize=False, add_generation_prompt=True)
            chosen = chosen[len(prompt) :]
            # rejected = rejected[len(prompt) :]

    # margin loss
    margin = data["margin"] if exist_and_not_none(data, "margin") else 0
    return prompt,data['prompt'], data['steps'], margin, data['process_label']

class RewardDataset_twostage(Dataset):
    """
    Dataset for reward model

    Args:
        dataset: dataset for reward model
        self.tokenizer: self.tokenizer for reward model
        self.max_length: max length of input
    """
    def __init__(
        self,
        dataset,
        tokenizer: Callable,
        max_length: int,
        strategy,
        input_template=None,
        is_dpo=False,
        num_processors=8,
        multiple_of=1,
    ) -> None:
        super().__init__()
        self.is_dpo = is_dpo
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.max_length = max_length
        self.multiple_of = multiple_of

        # chat_template
        self.input_template = input_template
        self.prompt_key = getattr(self.strategy.args, "prompt_key", None)
        self.chosen_key = getattr(self.strategy.args, "chosen_key", None)
        self.rejected_key = getattr(self.strategy.args, "rejected_key", None)
        self.apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)

        if self.apply_chat_template:
            self.apply_chat_template = self.tokenizer.apply_chat_template
            tokenizer_chat_template = getattr(self.strategy.args, "tokenizer_chat_template", None)
            if tokenizer_chat_template:
                self.tokenizer.chat_template = tokenizer_chat_template
        self.special_ids = [tokenizer.encode('\n\nStep')[-1],
                   tokenizer.encode(' Step')[-1],
                   tokenizer.encode('Step 2:')[-1]]

        # Parallel loading datasets
        processed_dataset = dataset.map(
            self.process_data, remove_columns=dataset.column_names, num_proc=num_processors
        )

        print('processed dataset',len(processed_dataset))
        # Filter out None values if necessary
        processed_dataset = processed_dataset.filter(lambda x: x["prompt"] is not None)
        print('processed dataset1', len(processed_dataset))
        # Store the processed data in class attributes
        self.prompts = processed_dataset["prompt"]
        self.chosens = processed_dataset["chosen"]
        self.extras = processed_dataset["extra"]
        self.step_labels = processed_dataset["step_labels"]


    def process_data(self, data):
        prompt, original_prompt, chosen, margin, step_labels = preprocess_data_twostage(
            data,
            self.input_template,
            self.prompt_key,
            self.chosen_key,
            self.rejected_key,
            self.apply_chat_template,
            self.is_dpo,
            self.tokenizer,
        )

        if self.is_dpo:
            prompt_token = self.tokenizer(
                prompt,
                max_length=self.max_length,
                padding=False,
                truncation=True,
                return_tensors="pt",
                add_special_tokens=False,
            )
            prompt_ids_len = prompt_token["attention_mask"].int().sum().item()

            # Filter the sample whose length is greater than max_length (2 for answer length)
            if prompt_ids_len >= self.max_length - 2:
                prompt = None


        return {
            "prompt": original_prompt,
            "chosen": chosen,
            "extra": prompt_ids_len if self.is_dpo else margin,
            'step_labels':step_labels,
        }

    def __len__(self):
        length = len(self.chosens)
        return length

    def __getitem__(self, idx):
        prompt, step_list, extra, step_labels = self.prompts[idx], self.chosens[idx], self.extras[idx], self.step_labels[idx]

        cur_special_ids = []
        chosen_token = self.tokenizer.apply_chat_template([
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "\n\n".join(step_list)},
        ], tokenize=True, add_generation_prompt=False)
        chosen_token = torch.tensor(chosen_token)
        cur_special_ids = []
        intermediate_token_ids = []
        for step_num in range(0, len(step_list) + 1):
            conv = self.tokenizer.apply_chat_template([
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": "\n\n".join(step_list[:step_num])},
            ], tokenize=False, add_generation_prompt=False)

            conv = conv.strip()

            if step_num != len(step_list) and conv.endswith("<|start_header_id|>assistant<|end_header_id|>"):
                conv = conv[:-len("<|start_header_id|>assistant<|end_header_id|>")]
            if step_num != len(step_list) and conv.endswith("<|eot_id|>"):
                conv = conv[:-len("<|eot_id|>")]

            if step_num != 0 and step_num != len(step_list):
                conv += '\n\n'

            currect_ids = self.tokenizer.encode(conv, add_special_tokens=False)

            if step_num > 1:
                last_ids = intermediate_token_ids[-1]
                assert 'ĊĊ' in self.tokenizer.convert_ids_to_tokens([currect_ids[len(last_ids) - 1]])[-1], (
                self.tokenizer.convert_ids_to_tokens(last_ids), self.tokenizer.convert_ids_to_tokens(currect_ids[:len(last_ids)]),
                step_list[:step_num])

            intermediate_token_ids.append(currect_ids)
            cur_special_ids.append(len(currect_ids) - 2)
        cur_special_ids = cur_special_ids[1:]
        assert len(cur_special_ids) == len(step_labels),(cur_special_ids,len(step_labels),self.tokenizer.tokenize(self.tokenizer.decode(input_ids)),input_ids,self.special_ids)

        if chosen_token.shape[-1] > self.max_length:
            chosen_token = chosen_token[:,:self.max_length]
            # to avoid EOS_token truncation
            chosen_token[0][-1] = self.tokenizer.eos_token_id
            # chosen_token["attention_mask"][0][-1] = True
            cur_special_ids_ = [id for id,label in zip(cur_special_ids,step_labels) if id<self.max_length-1]
            step_labels_ = [id for id, label in zip(cur_special_ids, step_labels) if id< self.max_length - 1]
        else:
            cur_special_ids_ = cur_special_ids
            step_labels_ = step_labels

        return (
            chosen_token,
            chosen_token!=self.tokenizer.pad_token_id,
            torch.tensor(cur_special_ids_),
            torch.tensor(step_labels_),
            extra,
        )

    def collate_fn(self, item_list):
        chosen_ids = []
        chosen_masks = []
        step_labels = []
        special_tokens = []
        extras = []
        for chosen_id, chosen_mask, cur_special_ids_,step_labels_,extra in item_list:
            chosen_ids.append(chosen_id)
            chosen_masks.append(chosen_mask)
            special_tokens.append(torch.tensor(cur_special_ids_).unsqueeze(0))
            step_labels.append(torch.tensor(step_labels_).unsqueeze(0))
            extras.append(extra)

        if self.is_dpo:
            padding_side = "right"
        else:
            padding_side = "left"
        chosen_ids = zero_pad_sequences(chosen_ids, side=padding_side, value=self.tokenizer.pad_token_id)
        chosen_masks = zero_pad_sequences(chosen_masks, side=padding_side)
        special_tokens = zero_pad_sequences(special_tokens, side=padding_side, value=-100)
        step_labels = zero_pad_sequences(step_labels, side=padding_side,value=-100)
        return chosen_ids, chosen_masks, special_tokens, step_labels, extras

