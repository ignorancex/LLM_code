import os
import torch
from torch.utils.data import IterableDataset
from datasets import load_dataset, load_from_disk, Dataset
from functools import partial
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from torch.nn.utils.rnn import pad_sequence


class SFTDataset(IterableDataset):
    """
    Iterable dataset that returns constant length chunks of tokens.
    Support the datasets is a dictionary with the following structure:
    {
        'src': str,
        'sample': List[OpenAI messages].
    }
    where OpenAI messages are of the form:
    {
        'role': str,
        'content': str,
    }
    The data is preprocessed, so the `sample` are with length around max_seq_length

        Args:
            tokenizer (`transformers.PreTrainedTokenizer`):
                The processor used for processing the data.
            dataset (`dataset.Dataset`):
                Dataset with text files.
            infinite (`bool`, *optional*, defaults to `False`):
                If True the iterator is reset after dataset reaches end else stops.
            max_seq_length (`int`, *optional*, defaults to `1024`):
                Length of token sequences to return.
            eos_token_id (`int`, *optional*, defaults to `0`):
                Id of the end of sequence token if the passed tokenizer does not have an EOS token.
            shuffle ('bool', *optional*, defaults to True)
                Shuffle the examples before they are returned
    """

    def __init__(
        self,
        tokenizer,
        dataset,
        infinite=False,
        max_seq_length=1024,
        shuffle=True,
        ignore_index=-100,
        remove_dummy_prefix=True
    ):
        self.tokenizer = tokenizer
        self.eos_token_id = tokenizer.eos_token_id
        self.dataset = dataset
        self.max_seq_length = max_seq_length
        self.infinite = infinite
        self.current_size = 0
        self.shuffle = shuffle
        self.ignore_index = ignore_index
        self.extra_id_0 = 260
        self.extra_id_1 = 261
        self.extra_id_2 = 262
        self.remove_dummy_prefix = remove_dummy_prefix

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        dataset_index = 0

        while True:
            if self.shuffle and dataset_index == 0:
                self.dataset = self.dataset.shuffle()

            for example in self.dataset:
                yield self._process_messages(example['sample'])

                dataset_index += 1
                if dataset_index >= len(self.dataset):
                    if not self.infinite:
                        break
                    dataset_index = 0

            if not self.infinite:
                break
    
    def _remove_dummy_prefix(self, list_of_tokens):
        if len(list_of_tokens) == 1 or list_of_tokens[0] != 484:
            return list_of_tokens
        else:
            return list_of_tokens[1:]
    
    def _process_messages(self, list_of_messages):
        input_ids = []
        labels = []
        for messages in list_of_messages:
            for message in messages['messages']:
                tokenized_content = self.tokenizer.encode(
                    message['content'],
                    add_special_tokens=False,
                    max_length=self.max_seq_length,
                    truncation=True
                )
                if message['role'] in ['user', 'system']:
                    current_input_ids = [self.extra_id_1] + tokenized_content + [self.extra_id_0]
                    current_labels = [self.ignore_index] * len(current_input_ids)
                elif message['role'] == 'assistant':
                    if self.remove_dummy_prefix:
                        tokenized_content = self._remove_dummy_prefix(tokenized_content)
                    current_input_ids = [self.extra_id_2] + tokenized_content + [self.extra_id_0]
                    current_labels = [self.ignore_index] + tokenized_content + [self.extra_id_0]
                else:
                    raise ValueError('Invalid role')
                input_ids += current_input_ids
                labels += current_labels
        
        if len(input_ids) > self.max_seq_length:
            input_ids = input_ids[:self.max_seq_length]
            labels = labels[:self.max_seq_length]
            if all([label == self.ignore_index for label in labels]):
                raise ValueError('All labels are ignore index')
                
        return {
            'input_ids': input_ids,
            'labels': labels,
        }

class SFTDataCollator:
    """simple data collator for SFT dataset
    """
    def __init__(self, tokenizer, ignore_index=-100):
        self.tokenizer = tokenizer
        self.ignore_index = ignore_index

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        input_ids = [torch.LongTensor(f['input_ids']) for f in features]
        labels = [torch.LongTensor(f['labels']) for f in features]

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=self.ignore_index)

        return {
            'input_ids': input_ids,
            'labels': labels,
        }


def get_sft_data(data_path: str, split: str, tokenizer, max_seq_length: int, infinite: bool=True, shuffle: bool=True):
    dataset = load_dataset('json', data_files=os.path.join(data_path, 'train.json'), split='train', field=split)
    
    def filter_fn(example):
        list_of_messages = example['sample']
        for messages in list_of_messages:
            for message in messages['messages']:
                if message['role'] == 'assistant':
                    return True
        return False
    
    dataset = dataset.filter(filter_fn, num_proc=32)        
    
    dataset = SFTDataset(
        tokenizer=tokenizer,
        dataset=dataset,
        infinite=infinite,
        max_seq_length=max_seq_length,
        shuffle=shuffle,
    )
    return dataset


def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

# for fast development and save storage space
def save_weights_decorator(Class):
    class WrappedClass(Class):
        def _save_checkpoint(self, model, trial, metrics=None):
            checkpoint_folder = f"checkpoint-{self.state.global_step}"
            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # only save model weights
            self.save_model(output_dir)

    return WrappedClass

############ RLHF Dataset #####################

def extract_anthropic_prompt(prompt_and_response):
    """Extract the anthropic prompt from a prompt and response pair."""
    search_term = "\n\nAssistant:"
    search_term_idx = prompt_and_response.rfind(search_term)
    assert search_term_idx != -1, f"Prompt and response does not contain '{search_term}'"
    return prompt_and_response[: search_term_idx + len(search_term)]


def get_hh(split: str, sanity_check: bool = False, silent: bool = False, cache_dir: str = None) -> Dataset:
    """Load the Anthropic Helpful-Harmless dataset from Hugging Face and convert it to the necessary format.

    The dataset is converted to a dictionary with the following structure:
    {
        'prompt': List[str],
        'chosen': List[str],
        'rejected': List[str],
    }

    Prompts should be structured as follows:
      \n\nHuman: <prompt>\n\nAssistant:
    Multiple turns are allowed, but the prompt should always start with \n\nHuman: and end with \n\nAssistant:.
    """
    # dataset = load_dataset("Anthropic/hh-rlhf", split=split, cache_dir=cache_dir)
    dataset = load_from_disk('data/rl/hh_rlhf')
    dataset = dataset[split]
    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 1000)))

    def split_prompt_and_responses(sample) -> Dict[str, str]:
        prompt = extract_anthropic_prompt(sample["chosen"])
        return {
            "prompt": prompt,
            "chosen": sample["chosen"][len(prompt) :],
            "rejected": sample["rejected"][len(prompt) :],
        }

    return dataset.map(split_prompt_and_responses)

def get_dpo_data(data_path: str, split: str, tokenizer, max_length: int, max_prompt_length: int) -> Dataset:
    """get dpo dataset

    Args:
        data_path (str): data path
        split (str): train / valid / test splits
        tokenizer (Tokenizer): tokenizer
        max_length (int): max length of the input
        max_prompt_length (int): max length of the prompt

    Returns:
        Dataset: huggingface dataset
        
    The data record is a dictionary with the following structure:
    {
        "messages": OpenAI messages,
        "chosen": str,
        "rejected": str,
    }
    """
    dataset = load_from_disk(data_path)
    dataset = dataset[split]

    def split_prompt_and_responses(sample) -> Dict[str, str]:
        assert "messages" in sample
        assert sample["messages"][-1]['role'] == 'user'

        prompt = ""
        for message in sample['messages']:
            if message['role'] in ['system', 'user']:
                prompt += f"[extra_id_1]{message['content']}[extra_id_0]"
            elif message['role'] == 'assistant':
                prompt += f"[extra_id_2]{message['content']}[extra_id_0]"

        prompt += '[extra_id_2]'

        return {
            "prompt": prompt,
            "chosen": sample["chosen"]+'[extra_id_0]',
            "rejected": sample["rejected"]+'[extra_id_0]',
        }

    def length_filter(example):
        plen = len(tokenizer.encode(example['prompt']))
        clen = len(tokenizer.encode(example['chosen']))
        rlen = len(tokenizer.encode(example['rejected']))
        return plen < max_prompt_length and plen + clen < max_length and plen + rlen < max_length

    dataset = dataset.map(split_prompt_and_responses, num_proc=32)
    dataset = dataset.filter(length_filter)

    return dataset

# This data collator is just for model max length
@dataclass
class DPODebugDataCollator:
    max_length: Optional[int] = None

    def tokenize_batch_element(
        self,
        prompt: str,
        chosen: str,
        rejected: str,
    ) -> Dict:
        """Tokenize a single batch element.

        At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
            in case the prompt + chosen or prompt + rejected responses is/are too long. First
            we truncate the prompt; if we're still too long, we truncate the chosen/rejected.

        We also create the labels for the chosen/rejected responses, which are of length equal to
            the sum of the length of the prompt and the chosen/rejected response, with
            label_pad_token_id  for the prompt tokens.
        """
        batch = {}
        batch['chosen_input_ids'] = [5] * self.max_length
        batch['chosen_attention_mask'] = [1] * self.max_length
        batch['chosen_labels'] = [5] * (self.max_length - 1)
        batch['rejected_input_ids'] = [5] * self.max_length
        batch['rejected_attention_mask'] = [1] * self.max_length
        batch['rejected_labels'] = [5] * self.max_length

        return batch

    def collate(self, batch):
        # first, pad everything to the same length
        padded_batch = {}
        for k in batch[0].keys():
            padded_batch[k] = torch.LongTensor([ex[k] for ex in batch])

        return padded_batch

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        tokenized_batch = []

        for feature in features:
            prompt = feature["prompt"]
            chosen = feature["chosen"]
            rejected = feature["rejected"]

            batch_element = self.tokenize_batch_element(prompt, chosen, rejected)
            tokenized_batch.append(batch_element)

        # return collated batch
        return self.collate(tokenized_batch)
    
    
    
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import IterableDataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase
    
    
def get_joint_data(data_path: str, split: str, tokenizer, max_length: int, max_prompt_length: int) -> Dataset:
    """get dpo dataset

    Args:
        data_path (str): data path
        split (str): train / valid / test splits
        tokenizer (Tokenizer): tokenizer
        max_length (int): max length of the input
        max_prompt_length (int): max length of the prompt

    Returns:
        Dataset: huggingface dataset
        
    The data record is a dictionary with the following structure:
    {
        "messages": OpenAI messages,
        "chosen": str,
        "rejected": str,
    }
    """
    dataset = load_dataset("json", data_dir=data_path,split=split)
    
    # FIXME: if cannot convert token in this way
    extra_id_0 = tokenizer.convert_tokens_to_ids('[extra_id_0]')
    extra_id_1 = tokenizer.convert_tokens_to_ids('[extra_id_1]')
    extra_id_2 = tokenizer.convert_tokens_to_ids('[extra_id_2]')

    print(extra_id_0, extra_id_1, extra_id_2)    

    def split_prompt_and_responses(sample) -> Dict[str, str]:
        ignore_index = -100
        if sample["isdpo"]:
            assert "messages" in sample
            assert sample["messages"][-1]['role'] == 'user' ,sample["messages"][-1]['role']
        
            prompt = ""
            for message in sample['messages']:
                if message['role'] in ['system', 'user']:
                    prompt += f"[extra_id_1]{message['content']}[extra_id_0]"
                elif message['role'] == 'assistant':
                    prompt += f"[extra_id_2]{message['content']}[extra_id_0]"

            prompt += '[extra_id_2]'

            return {
                "prompt": prompt,
                "chosen": sample["chosen"]+'[extra_id_0]',
                "rejected": sample["rejected"]+'[extra_id_0]',
                "isdpo": True
            }
        else :
            input_ids = []
            labels = []
            for message in sample['messages']:
                
                # FIXME: Iss that the correct way to tokenize?
                
                if message['role'] in ['user', 'system']:
                    input_str = message['content']
                    tokenized_content = tokenizer.encode(
                        input_str,
                        add_special_tokens=False,
                        max_length=max_length,
                        truncation=True
                    ) 
                    current_input_ids = [extra_id_1] + tokenized_content + [extra_id_0]
                    current_labels = [ignore_index] * len(current_input_ids)
                elif message['role'] == 'assistant':
                    input_str = message['content']
                    tokenized_content = tokenizer.encode(
                        input_str,
                        add_special_tokens=False,
                        max_length=max_length,
                        truncation=True
                    ) 
                    current_input_ids = [extra_id_2] + tokenized_content + [extra_id_0]
                    
                    current_labels = [ignore_index] + tokenized_content + [ignore_index]
                else:
                    raise ValueError('Invalid role')
                
                assert len(current_input_ids) == len(current_labels)
                
                input_ids += current_input_ids
                labels += current_labels
            
            if len(input_ids) > max_length:
                input_ids = input_ids[:max_length]
                labels = labels[:max_length]
            return {
                'input_ids': input_ids,
                'labels': labels,
                "isdpo": False
            }
            

    def length_filter(example):
        if example["isdpo"]:
            plen = len(tokenizer.encode(example['prompt']))
            clen = len(tokenizer.encode(example['chosen']))
            rlen = len(tokenizer.encode(example['rejected']))
        else:
            plen = len(example['input_ids'])
            clen = 0
            rlen = 0
        return plen < max_prompt_length and plen + clen < max_length and plen + rlen < max_length

    dataset = dataset.map(split_prompt_and_responses, num_proc=32)
    dataset = dataset.filter(length_filter)

    return dataset
    
@dataclass
class JointTrainingCollator:
    r"""
    DPO DataCollator class that pads the inputs to the maximum length of the batch.
    Args:
        tokenizer (`PreTrainedTokenizerBase`):
            The tokenizer used for encoding the data.
        model (Optional[`PreTrainedModel`]):
            The model that is being trained. If set and has the *prepare_decoder_input_ids_from_labels*, use it to
            prepare the *decoder_input_ids*.
        padding (`Union[bool, str, `PaddingStrategy`]`, `optional`, defaults to `True`):
            padding_strategy to pass to the tokenizer.
        max_length (`Optional[int]`, `optional`, defaults to `None`):
            The maximum length of the sequence to be processed.
        max_prompt_length (`Optional[int]`, `optional`, defaults to `None`):
            The maximum length of the prompt to be processed.
        label_pad_token_id (`int`, defaults to -100):
            The label used for masking.
        padding_value (`int`, defaults to 0):
            The value used for padding.
        is_encoder_decoder (`Optional[bool]`, `optional`, defaults to `None`):
            Whether or not you model has an encoder_decoder architecture.
        max_target_length (`Optional[int]`, `optional`, defaults to `None`):
            The maximum length of the target to be processed. Only useful for encoder-decoder architectures.
        truncation_mode: (`str`, defaults to "keep_end"):
            The truncation mode to use when truncating the prompt.
    """
    tokenizer: PreTrainedTokenizerBase
    model: Optional[PreTrainedModel] = None
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_prompt_length: Optional[int] = None
    label_pad_token_id: int = -100
    padding_value: int = 0
    truncation_mode: str = "keep_end"
    is_encoder_decoder: Optional[bool] = False
    max_target_length: Optional[int] = None
    ignore_index: int = -100

    def tokenize_batch_element_dpo(
        self,
        prompt: str,
        chosen: str,
        rejected: str,
    ) -> Dict:
        """Tokenize a single batch element.

        At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
            in case the prompt + chosen or prompt + rejected responses is/are too long. First
            we truncate the prompt; if we're still too long, we truncate the chosen/rejected.

        We also create the labels for the chosen/rejected responses, which are of length equal to
            the sum of the length of the prompt and the chosen/rejected response, with
            label_pad_token_id  for the prompt tokens.
        """
        batch = {}

        if not self.is_encoder_decoder:
            chosen_tokens = self.tokenizer(chosen, add_special_tokens=False)
            rejected_tokens = self.tokenizer(rejected, add_special_tokens=False)
            prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)

            eos_token_id = self.tokenizer.eos_token_id
            # Get indices in list prompt_tokens["input_ids"] that equals the EOS token (often 0)
            eos_indices_prompt = [i for i, x in enumerate(prompt_tokens["input_ids"]) if x == eos_token_id]
            # attention mask these indices to eos_token_id
            new_attention_mask = [
                0 if i in eos_indices_prompt else p for i, p in enumerate(prompt_tokens["attention_mask"])
            ]
            prompt_tokens["attention_mask"] = new_attention_mask

            # do the same for chosen and rejected
            eos_indices_chosen = [i for i, x in enumerate(chosen_tokens["input_ids"]) if x == eos_token_id]
            new_attention_mask_c = [
                0 if i in eos_indices_chosen else p for i, p in enumerate(chosen_tokens["attention_mask"])
            ]
            chosen_tokens["attention_mask"] = new_attention_mask_c

            eos_indices_rejected = [i for i, x in enumerate(rejected_tokens["input_ids"]) if x == eos_token_id]
            new_attention_mask_r = [
                0 if i in eos_indices_rejected else p for i, p in enumerate(rejected_tokens["attention_mask"])
            ]
            rejected_tokens["attention_mask"] = new_attention_mask_r

            # add EOS token to end of prompt
            chosen_tokens["input_ids"].append(self.tokenizer.eos_token_id)
            chosen_tokens["attention_mask"].append(1)

            rejected_tokens["input_ids"].append(self.tokenizer.eos_token_id)
            rejected_tokens["attention_mask"].append(1)

            longer_response_length = max(len(chosen_tokens["input_ids"]), len(rejected_tokens["input_ids"]))

            # if combined sequence is too long, truncate the prompt
            if len(prompt_tokens["input_ids"]) + longer_response_length > self.max_length:
                if self.truncation_mode == "keep_start":
                    prompt_tokens = {k: v[: self.max_prompt_length] for k, v in prompt_tokens.items()}
                elif self.truncation_mode == "keep_end":
                    prompt_tokens = {k: v[-self.max_prompt_length :] for k, v in prompt_tokens.items()}
                else:
                    raise ValueError(f"Unknown truncation mode: {self.truncation_mode}")

            # if that's still too long, truncate the response
            if len(prompt_tokens["input_ids"]) + longer_response_length > self.max_length:
                chosen_tokens = {k: v[: self.max_length - self.max_prompt_length] for k, v in chosen_tokens.items()}
                rejected_tokens = {
                    k: v[: self.max_length - self.max_prompt_length] for k, v in rejected_tokens.items()
                }

            # Create labels
            chosen_sequence_tokens = {k: prompt_tokens[k] + chosen_tokens[k] for k in chosen_tokens}
            rejected_sequence_tokens = {k: prompt_tokens[k] + rejected_tokens[k] for k in rejected_tokens}
            chosen_sequence_tokens["labels"] = chosen_sequence_tokens["input_ids"][:]
            chosen_sequence_tokens["labels"][: len(prompt_tokens["input_ids"])] = [self.label_pad_token_id] * len(
                prompt_tokens["input_ids"]
            )
            rejected_sequence_tokens["labels"] = rejected_sequence_tokens["input_ids"][:]
            rejected_sequence_tokens["labels"][: len(prompt_tokens["input_ids"])] = [self.label_pad_token_id] * len(
                prompt_tokens["input_ids"]
            )

            for k, toks in {
                "chosen": chosen_sequence_tokens,
                "rejected": rejected_sequence_tokens,
                "prompt": prompt_tokens,
            }.items():
                for type_key, tokens in toks.items():
                    if type_key == "token_type_ids":
                        continue
                    batch[f"{k}_{type_key}"] = tokens

        else:
            chosen_tokens = self.tokenizer(
                chosen, truncation=True, max_length=self.max_target_length, add_special_tokens=True
            )
            rejected_tokens = self.tokenizer(
                rejected, truncation=True, max_length=self.max_target_length, add_special_tokens=True
            )
            prompt_tokens = self.tokenizer(
                prompt, truncation=True, max_length=self.max_prompt_length, add_special_tokens=True
            )

            batch["chosen_labels"] = chosen_tokens["input_ids"]
            batch["rejected_labels"] = rejected_tokens["input_ids"]
            batch["prompt_input_ids"] = prompt_tokens["input_ids"]
            batch["prompt_attention_mask"] = prompt_tokens["attention_mask"]

            if self.model is not None and hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
                batch["rejected_decoder_input_ids"] = self.model.prepare_decoder_input_ids_from_labels(
                    labels=batch["rejected_labels"]
                )
                batch["chosen_decoder_input_ids"] = self.model.prepare_decoder_input_ids_from_labels(
                    labels=batch["chosen_labels"]
                )

        batch["prompt"] = prompt
        batch["chosen"] = prompt + chosen
        batch["rejected"] = prompt + rejected
        batch["chosen_response_only"] = chosen
        batch["rejected_response_only"] = rejected

        return batch

    def collate_dpo(self, batch):
        # first, pad everything to the same length
        padded_batch = {}
        for k in batch[0].keys():
            if k.endswith("_input_ids") or k.endswith("_attention_mask") or k.endswith("_labels"):
                if self.is_encoder_decoder:
                    to_pad = [torch.LongTensor(ex[k]) for ex in batch]

                    if (k.startswith("prompt")) and (k.endswith("input_ids")):
                        padding_value = self.tokenizer.pad_token_id
                    elif k.endswith("_attention_mask"):
                        padding_value = 0
                    elif (k.startswith("chosen")) or (k.startswith("rejected")) or ("decoder" in k):
                        padding_value = self.label_pad_token_id
                    else:
                        raise ValueError(f"Unexpected key in batch '{k}'")
                    padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
                else:
                    # adapted from https://stackoverflow.com/questions/73256206
                    if "prompt" in k:
                        to_pad = [torch.LongTensor(ex[k][::-1]) for ex in batch]
                    else:
                        to_pad = [torch.LongTensor(ex[k]) for ex in batch]
                    if k.endswith("_input_ids"):
                        padding_value = self.tokenizer.pad_token_id
                    elif k.endswith("_labels"):
                        padding_value = self.label_pad_token_id
                    elif k.endswith("_attention_mask"):
                        padding_value = self.padding_value
                    else:
                        raise ValueError(f"Unexpected key in batch '{k}'")

                    padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
                    # for the prompt, flip back so padding is on left side
                    if "prompt" in k:
                        padded_batch[k] = padded_batch[k].flip(dims=[1])
            else:
                padded_batch[k] = [ex[k] for ex in batch]
        padded_batch['isdpo'] = True
        return padded_batch

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        tokenized_batch = []
        if features[0]["isdpo"] ==False:
            input_ids = [torch.LongTensor(f['input_ids']) for f in features]
            labels = [torch.LongTensor(f['labels']) for f in features]

            input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            labels = pad_sequence(labels, batch_first=True, padding_value=self.ignore_index)

            return {
                'input_ids': input_ids,
                'labels': labels,
                'isdpo': False
            }

        for feature in features:
            prompt = feature["prompt"]
            chosen = feature["chosen"]
            rejected = feature["rejected"]

            batch_element = self.tokenize_batch_element_dpo(prompt, chosen, rejected)
            tokenized_batch.append(batch_element)

            

        # return collated batch
        return self.collate_dpo(tokenized_batch)
    
        
    # def __init__(self, tokenizer, ignore_index=-100):
    #     self.tokenizer = tokenizer
    #     self.ignore_index = ignore_index

    # def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
    #     input_ids = [torch.LongTensor(f['input_ids']) for f in features]
    #     labels = [torch.LongTensor(f['labels']) for f in features]

    #     input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
    #     labels = pad_sequence(labels, batch_first=True, padding_value=self.ignore_index)

    #     return {
    #         'input_ids': input_ids,
    #         'labels': labels,
    #     }

# 对many-shot context 找出连续大于0的位置片段
def find_consecutive_greater_than_zero(tensor):
    start = None
    end = None
    consecutive_segments = []
    for i in range(len(tensor)):
        if tensor[i] > 0:
            if start is None:
                start = i
            end = i
        else:
            if start is not None:
                consecutive_segments.append((start, end))
                start = None
    if start is not None:
        consecutive_segments.append((start, end))
    return consecutive_segments


