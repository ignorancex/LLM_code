import sys, os
import glob
import random

from datasets import load_dataset, load_metric
from model.PartialSrcAttnModels import partial_attn_data_processor

from transformers import (
    AutoTokenizer,
)

class DataLoader(object):
    def __init__(self, data_args, model_args, training_args, logger):
        self.data_args = data_args
        self.model_args = model_args
        self.training_args = training_args
        self.logger = logger

        self.do_train = training_args.do_train
        self.do_eval = training_args.do_eval
        self.do_predict = training_args.do_predict

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            do_lower_case=True,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            local_files_only=model_args.local_files_only)

        self.prefix = data_args.source_prefix if data_args.source_prefix is not None else ""
        self.text_column = data_args.text_column
        self.summary_column = data_args.summary_column
        self.preprocessing_num_workers = data_args.preprocessing_num_workers
        self.overwrite_cache = data_args.overwrite_cache
        self.max_source_length = data_args.max_source_length
        self.max_target_length = data_args.max_target_length
        self.batch_size = training_args.per_device_train_batch_size
        self.model_name = model_args.model_name_or_path
        self.partial_attn = model_args.partial_attn

        self.dataset = None
        self.train_dataset = None
        self.eval_dataset = None
        self.test_shards = []


    def _preprocess_function(self, examples):
        padding = "max_length" if self.data_args.pad_to_max_length else False
        ignore_pad_token_for_loss = self.data_args.ignore_pad_token_for_loss

        inputs = examples[self.text_column]
        targets = examples[self.summary_column]
        inputs = [self.prefix + inp for inp in inputs]
        model_inputs = self.tokenizer(inputs, max_length=self.max_source_length, padding=padding, truncation=True)

        # Setup the tokenizer for targets
        with self.tokenizer.as_target_tokenizer():
            if self.do_predict:
                labels = self.tokenizer(targets, max_length=None, padding=padding, truncation=False)
            else:
                labels = self.tokenizer(targets, max_length=self.max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != self.tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]
        model_inputs["labels"] = labels["input_ids"]

        if self.model_name == "allenai/led-base-16384":
            # create 0 global_attention_mask lists
            model_inputs["global_attention_mask"] = len(model_inputs["input_ids"]) * [
                [0 for _ in range(len(model_inputs["input_ids"][0]))]]
            # since above lists are references, the following line changes the 0 index for all samples
            model_inputs["global_attention_mask"][0][0] = 1

        if self.partial_attn:
            model_inputs["main_attn_mask"] = partial_attn_data_processor(inputs, self.tokenizer, self.max_source_length)

        return model_inputs


    def _check_column_name(self, column_names):
        if self.text_column not in column_names:
            raise ValueError(
                f"--text_column' value '{self.text_column}' needs to be one of: {', '.join(column_names)}")
        if self.summary_column not in column_names:
            raise ValueError(
                f"--summary_column' value '{self.summary_column}' needs to be one of: {', '.join(column_names)}")


    def process_data(self):
        if self.do_train:
            column_names = self.datasets["train"].column_names
            self._check_column_name(column_names)
            max_train_samples = self.data_args.max_train_samples
            if "train" not in self.datasets:
                raise ValueError("--do_train requires a train dataset")
            train_dataset = self.datasets["train"]

            if max_train_samples is not None:
                train_dataset = train_dataset.select(range(max_train_samples))
            self.train_dataset = train_dataset.map(
                self._preprocess_function,
                batched=True,
                batch_size=self.batch_size,
                num_proc=self.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not self.overwrite_cache,)

        if self.do_eval:
            column_names = self.datasets["validation"].column_names
            self._check_column_name(column_names)
            max_val_samples = self.data_args.max_val_samples
            if "validation" not in self.datasets:
                raise ValueError("--do_eval requires a validation dataset")

            eval_dataset = self.datasets["validation"]
            if max_val_samples is not None:
                eval_dataset = eval_dataset.select(range(max_val_samples))
            self.eval_dataset = eval_dataset.map(
                self._preprocess_function,
                batched=True,
                batch_size=self.batch_size,
                num_proc=self.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not self.overwrite_cache,)

        if self.do_predict:
            column_names = self.datasets["test_0"].column_names
            self._check_column_name(column_names)
            max_test_samples = self.data_args.max_test_samples

            for file_id in self.datasets:
                if not file_id.startswith("test"):
                    continue
                test_dataset = self.datasets[file_id]
                if max_test_samples is not None:
                    test_dataset = test_dataset.select(range(max_test_samples))
                test_dataset = test_dataset.map(
                    self._preprocess_function,
                    batched=True,
                    batch_size=self.batch_size,
                    num_proc=self.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=not self.overwrite_cache,)
                self.test_shards.append(test_dataset)


    def load_data(self):
        data_files = {}
        data_args = self.data_args

        if data_args.train_path is not None:
            pts = sorted(glob.glob(data_args.train_path + '.train.[0-9]*.json'))
            random.shuffle(pts)
            data_files["train"] = pts

        if data_args.validation_path is not None:
            data_files["validation"] = sorted(glob.glob(data_args.validation_path + '.dev.[0-9]*.json'))

        if data_args.test_path is not None:
            shards_names = sorted(glob.glob(data_args.test_path + '.test.[0-9]*.json'))
            for name in shards_names:
                shard_id = name.split('.')[-2]
                data_files["test_"+shard_id] = [name]

        self.datasets = load_dataset('json', data_files=data_files)

