#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author             : 陈蔚 (weichen.cw@zju.edu.cn)
Date               : 2024-10-09 10:19
Last Modified By   : 陈蔚 (weichen.cw@zju.edu.cn)
Last Modified Date : 2024-10-14 00:17
Description        : JSON dataset.
-------- 
Copyright (c) 2024 Wei Chen. 
'''

import json
import logging
import os
import concurrent.futures
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor
from typing import List, Literal, Tuple

import torch
import torch.distributed
from addict import Dict
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from utils import IGNORE_INDEX, logging_rank

logger = logging.getLogger("DeepSpeed")


def build_dataset_json(args: Namespace,
                       tokenizer: PreTrainedTokenizer) -> Dataset:
    """Build dataset from json file.

    Parameters
    ----------
    args : Namespace
        Arguments from argparse.
    tokenizer : PreTrainedTokenizer
        Tokenizer used to tokenize data.

    Returns
    -------
    Dataset
        Tokenized dataset.
    """
    train_dataset = JsonDataset(args, tokenizer)
    return train_dataset


class JsonDataset(Dataset):
    """
    JSON dataset.
    """

    def __init__(self, args: Namespace,
                 tokenizer: PreTrainedTokenizer) -> None:
        """Build dataset from json file.

        Parameters
        ----------
        args : Namespace
            Arguments from argparse.
        tokenizer : PreTrainedTokenizer
            Tokenizer used to tokenize data.
        """
        
        super().__init__()

        self.args = args
        self.tokenizer = tokenizer
        self.rank, self.world_size = self.get_rank_and_word_size()
        self.load_data()

    def get_rank_and_word_size(self) -> Tuple[int, int]:
        """Get rank and world size in distributed training.

        Returns
        -------
        Tuple[int, int]
            Rank and world size.
        """
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            rank = 0
            world_size = 1

        return rank, world_size

    def logging_message(self, msg) -> None:
        """Logging message on rank 0.

        Parameters
        ----------
        msg : _type_
            Message to log.
        """
        if self.rank == 0:
            logger.info(msg)
        return

    def load_data(self) -> None:
        """Load data from json file. If not exists, tokenize data and save to cache.
        """
        cache_path = self.get_cache_path()

        if not os.path.exists(cache_path) and self.rank == 0:
            with open(self.args.data_path, 'r', encoding='utf-8') as file:
                data = [json.loads(line) for line in file]

            self.logging_message(f"Tokenizing data and saving to {cache_path}")
            self.tokenized_data = self._tokenize_data(
                data,
                self.tokenizer,
                self.args.max_seq_length,
                self.args.padding,
                self.args.data_type,
                train_on_prompt=self.args.train_on_prompt,
            )

            torch.save(self.tokenized_data, cache_path)

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        logger.info(f"Loading tokenized data from {cache_path}.")
        self.tokenized_data = torch.load(cache_path, weights_only=True)

    def get_cache_path(self) -> str:
        """Get the name of the cache file.

        Returns
        -------
        str
            The path of the cache file.
        """
        assert self.args.cache_dir is not None, "Cache directory must be specified."
        
        dataset_name = os.path.basename(self.args.data_path)
        tokenizer_name = type(self.tokenizer).__name__
        model_type = self.args.model_type
        max_seq_length = self.args.max_seq_length
        padding = self.args.padding
        train_on_prompt = self.args.train_on_prompt
        
        cache_path = os.path.join(
            self.args.cache_dir,
            f"{dataset_name}.cache_{model_type}_{tokenizer_name}_{max_seq_length}_{padding}_{train_on_prompt}.pt"
        )
        return cache_path

    def __len__(self):
        return len(self.tokenized_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return self.tokenized_data[i]

    @staticmethod
    def _tokenize_data(data: List[dict],
                       tokenizer: PreTrainedTokenizer,
                       max_seq_length: int,
                       padding: bool,
                       data_type: Literal['pretraining', 'instruction_tuning'],
                       num_workers: int = 16,
                       **kwargs) -> List[Dict[str, List[int]]]:
        """Tokenize a list of data in parallel.

        Parameters
        ----------
        data : List[dict]
            Data to tokenize.
        tokenizer : PreTrainedTokenizer
            Tokenizer used to tokenize data.
        max_seq_length : int
            Max sequence length.
        padding : bool
            Whether to pad the sequence.
        data_type : Literal[&#39;pretraining&#39;, &#39;instruction_tuning&#39;]
            The type of data.
        num_workers : int, optional
            Number of workers in parallel, by default 16

        Returns
        -------
        List[Dict[str, List[int]]]
            Tokenized data, each element is a dict with keys:
                - index: int, index of the datapoint.
                - input_ids: List[int], tokenized input ids.
                - labels: List[int], tokenized labels.
                - attention_mask: List[int], attention mask.
        """

        futures = []
        tokenized_data = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            for index, datapoint in enumerate(data):
                future = executor.submit(JsonDataset._tokenize_datapoint,
                                         datapoint, index, tokenizer,
                                         max_seq_length, padding, data_type,
                                         **kwargs)
                futures.append(future)
                if index % 10000 == 0:
                    logging_rank(
                        f"Prepare Tokenizing {index} / {len(data)} datapoints."
                    )

            logging_rank(f"Prepare tokenizing {len(data)} datapoints.")

            index = 0
            for future in concurrent.futures.as_completed(futures):
                if index % 10000 == 0:
                    logging_rank(
                        f"Complete Tokenizing {index} / {len(data)} datapoints."
                    )
                result = future.result()
                if result is not None:
                    tokenized_data.append(result)
                index += 1

            logging_rank(f"Finish tokenizing {len(data)} datapoints.")

        tokenized_data = sorted(tokenized_data, key=lambda x: x['index'])
        return tokenized_data

    @staticmethod
    def _tokenize_datapoint(datapoint: List[dict], index: int,
                            tokenizer: PreTrainedTokenizer,
                            max_seq_length: int, padding: bool,
                            data_type: Literal['pretraining',
                                               'instruction_tuning'],
                            **kwargs) -> Dict[str, List[int]]:
        """Tokenize a single datapoint.

        Parameters
        ----------
        datapoint : List[dict]
            Datapoint to tokenize.
        index : int
            Index of the datapoint.
        tokenizer : PreTrainedTokenizer
            Tokenizer used to tokenize data.
        max_seq_length : int
            Max sequence length.
        padding : bool
            Whether to pad the sequence.
        data_type : Literal[&#39;pretraining&#39;, &#39;instruction_tuning&#39;]
            The type of data.

        Returns
        -------
        Dict[str, List[int]]
            Tokenized result of a single datapoint, which is a dict with keys:
                - index: int, index of the datapoint.
                - input_ids: List[int], tokenized input ids.
                - labels: List[int], tokenized labels.
                - attention_mask: List[int], attention mask.

        Raises
        ------
        ValueError
            Data type not supported.
        """

        if data_type == 'instruction_tuning':
            input_ids, labels = JsonDataset._tokenize_datapoint_instruction_tuning(
                datapoint, tokenizer, **kwargs)
        elif data_type == 'pretraining':
            input_ids, labels = JsonDataset._tokenize_datapoint_pretraining(
                datapoint, tokenizer, **kwargs)
        else:
            raise ValueError(
                f"data_type must be either 'instruction_tuning' or 'pretraining, got {data_type}."
            )

        # Skip empty datapoints.
        if len(input_ids) == 0 or len(labels) == 0 or all(x == IGNORE_INDEX
                                                          for x in labels):
            return None

        attention_mask = [1] * len(input_ids)
        max_seq_length = max_seq_length

        if len(input_ids) > max_seq_length:
            input_ids = input_ids[:max_seq_length]
            labels = labels[:max_seq_length]
            attention_mask = attention_mask[:max_seq_length]

        if padding:
            padding_length = max_seq_length - len(input_ids)
            input_ids = input_ids + [tokenizer.pad_token_id] * padding_length
            labels = labels + [IGNORE_INDEX] * padding_length
            attention_mask = attention_mask + [0] * padding_length

        # BUG: Using torch.tensor in Datacolator will be slower than list,
        # so we use list here.
        
        # input_ids = torch.tensor(input_ids, dtype=torch.int)
        # labels = torch.tensor(labels, dtype=torch.int)
        # attention_mask = torch.tensor(attention_mask, dtype=torch.bool)

        tokenized_datapoint = {
            'index': index,
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
        }

        return tokenized_datapoint

    @staticmethod
    def _tokenize_datapoint_instruction_tuning(
            datapoint: List[dict],
            tokenizer: PreTrainedTokenizer,
            train_on_prompt: bool = True,
            **kwargs) -> Tuple[List[int], List[int]]:
        """Tokenize a single datapoint for instruction tuning.

        Parameters
        ----------
        datapoint : List[dict]
            Datapoint to tokenize.
        tokenizer : PreTrainedTokenizer
            Tokenizer used to tokenize data.
        train_on_prompt : bool, optional
            Whether to backward loss and train on dialogues before the last, by default True

        Returns
        -------
        Tuple[List[int], List[int]]
            Input ids and labels.
        """

        assert 'messages' in datapoint, "Datapoint must contain 'messages' key."
        messages = datapoint['messages']

        # Make sure the last message is from assistant
        while messages[-1]["role"] != "assistant":
            _ = messages.pop()

        assert hasattr(tokenizer, 'apply_chat_template'
                       ), "Tokenizer must have .apply_chat_template() method."
        assert hasattr(
            tokenizer,
            'chat_template'), "Tokenizer must have .chat_template attribute."

        # BUG: In extremely rare cases, assistant content begin with '\n',
        # Qwen2 Tokenizer would tokenize <im_start>assistant\n\n into ['<im_start>', 'assistant', '\n\n'] instead of ['<im_start>', 'assistant', '\n', '\n']
        #   So we need to remove the last token in input_ids_without_target when checking
        #   `tq-llm`` has fixed this in 11/10/2024. If you really care about this, you should comment the code
        #   between `== BUG START ==` and `== BUG END ==` and then uncomment the code between `== FIX START ==` and `== FIX END ==`.
        #
        # However, we don't fix this since it should be handled by `transformers` library` instead of us.

        # == BUG START ==

        input_ids = tokenizer.apply_chat_template(messages,
                                                  add_generation_prompt=False,
                                                  tokenize=True)
        labels = [IGNORE_INDEX] * len(input_ids)

        # Use the whole message as the label
        for i, message in enumerate(messages):
            if message["role"] == "assistant":
                input_ids_without_target = tokenizer.apply_chat_template(
                    messages[:i], add_generation_prompt=True, tokenize=True)
                input_ids_with_target = tokenizer.apply_chat_template(
                    messages[:i + 1],
                    add_generation_prompt=False,
                    tokenize=True)

                label_start = len(input_ids_without_target)
                label_end = len(input_ids_with_target)

                # Check if the input_ids without target and input_ids with target match
                assert input_ids_without_target[:
                                                -1] == input_ids[:label_start -
                                                                 1], "input_ids without target mismatch with input_ids with target"
                assert input_ids_with_target == input_ids[:
                                                          label_end], "input_ids of sub messages mismatch with input_ids of all the messages"

                if train_on_prompt or i == len(messages) - 1:
                    labels[label_start:label_end] = input_ids[
                        label_start:label_end]

        # == BUG END ==

        # == FIX START ==

        # input_ids = []
        # labels = []
        # pre_chat_prompt = ""

        # if train_on_prompt:
        #     for i, message in enumerate(messages):
        #         if message["role"] == "assistant":
        #             input_prompt_wo_target = tokenizer.apply_chat_template(messages[:i], add_generation_prompt=True, tokenize=False)
        #             cur_input_prompt = input_prompt_wo_target[len(pre_chat_prompt):]
        #             input_ids_wo_target = tokenizer.encode(cur_input_prompt, add_special_tokens=False)

        #             input_ids += input_ids_wo_target
        #             labels += [IGNORE_INDEX] * len(input_ids_wo_target)

        #             input_prompt_with_target = tokenizer.apply_chat_template(messages[:i+1], add_generation_prompt=False, tokenize=False)
        #             cur_target_prompt = input_prompt_with_target[len(input_prompt_wo_target):]
        #             cur_labels = tokenizer.encode(cur_target_prompt, add_special_tokens=False)

        #             input_ids += cur_labels
        #             if train_on_prompt or i == len(messages) - 1:
        #                 labels += [IGNORE_INDEX] * len(cur_labels)
        #             else:
        #                 labels += cur_labels

        #             pre_chat_prompt = input_prompt_with_target

        # == FIX END ==

        input_ids += [tokenizer.eos_token_id]
        labels += [tokenizer.eos_token_id]

        return input_ids, labels

    @staticmethod
    def _tokenize_datapoint_pretraining(
            datapoint: List[dict], tokenizer: PreTrainedTokenizer,
            **kwargs) -> Tuple[List[int], List[int]]:
        """Tokenize a single datapoint for pretraining.

        Parameters
        ----------
        datapoint : List[dict]
            Datapoint to tokenize.
        tokenizer : PreTrainedTokenizer
            Tokenizer used to tokenize data.

        Returns
        -------
        Tuple[List[int], List[int]]
            Input ids and labels.
        """

        assert 'text' in datapoint, "Datapoint must contain 'text' key."
        text = datapoint['text']

        input_ids = tokenizer(text).input_ids + [tokenizer.eos_token_id]
        labels = input_ids.copy()

        return input_ids, labels
