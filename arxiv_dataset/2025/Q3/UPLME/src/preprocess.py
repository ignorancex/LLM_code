import random
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, DataCollatorWithPadding
from datasets import Dataset
import logging

from utils import log_info, read_newsemp_file

logger = logging.getLogger(__name__)

pd.options.mode.copy_on_write = True # https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy

class NewsEmpPreprocessorFromRaw:
    """
    Preprocess the raw data to the format that can be used in other data processing pipelines here.
    It does the minimum processing required for the data.
    """
    def __init__(
        self,
        noise_level: float,
        delta: float | None
    ):
        self.delta = delta
        
        # keeping them constant for now, can make them arguments if required
        self.label_shift = 3.0
        self.noise_level = noise_level
        self.label_min = 1.0
        self.label_max = 7.0
        self.label_column = "empathy"
        self.llm_column = "llm_empathy"
        self.columns_to_keep = ["essay", "article", self.label_column, self.llm_column, "article_id"]

    def _raw_to_processed(
        self, path: str, sanitise_labels: bool, add_noise: bool
    ) -> pd.DataFrame:
    
        data = read_newsemp_file(path)
        log_info(logger, f"Read {len(data)} samples from {path}")
        
        # if it is val of 2022 and 2023, the labels are separate files
        val_goldstandard_file = None
        if "WASSA23_essay_level_dev" in path:
            val_goldstandard_file = "data/NewsEmp2023/goldstandard_dev.tsv"
        elif "messages_dev_features_ready_for_WS_2022" in path:
            val_goldstandard_file = "data/NewsEmp2022/goldstandard_dev_2022.tsv"
        if val_goldstandard_file is not None:
            assert os.path.exists(val_goldstandard_file), f"File {val_goldstandard_file} does not exist."
            goldstandard = pd.read_csv(
                val_goldstandard_file, 
                sep='\t',
                header=None # had no header in the file
            )
            # first column is empathy
            goldstandard = goldstandard.rename(columns={0: self.label_column})
            data = pd.concat([data, goldstandard], axis=1)

        selected_data = data[[col for col in self.columns_to_keep if col in data.columns]] # keep only the columns that are in the data

        if sanitise_labels:
            log_info(logger, f"Santitising labels of {path} file.\n")
            selected_data = self._label_fix(selected_data)

        if add_noise:
            log_info(logger, f"Flipping labels of {path} file.\n")
            selected_data = self._flip_labels(selected_data)
        
        if selected_data.isna().any().any(): 
            log_info(logger, f"Columns {selected_data.columns[selected_data.isna().any()].tolist()} have {selected_data.isna().sum().sum()} NaN values in total.")
            selected_data.dropna(inplace=True) # drop NaN values; this could be NaN if the essay or label is None, so we drop the whole row
            log_info(logger, f"Removed rows with any NaN values. {len(selected_data)} samples remaining.\n")

        assert not selected_data.isna().any().any(), "There are still NaN values in the data."
        assert not selected_data.isnull().any().any(), "The are still null values in the data"

        return selected_data
    
    def _label_fix(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Only keep the samples with the absolute difference between 'empathy' and 'llm_empathy' less than self.delta
        """
        assert self.label_column in data.columns, f"{self.label_column} column not found in the data"
        assert self.llm_column in data.columns, f"{self.llm_column} column not found in the data"
        
        if self.delta is not None:
            # Calculate the absolute difference between 'empathy' and 'llm_empathy'
            condition = np.abs(data[self.label_column] - data[self.llm_column]) < self.delta
            data = data[condition]
        
        data[self.label_column] = data[[self.label_column, self.llm_column]].mean(axis=1)

        data = data.drop(columns=[self.llm_column])

        return data
    
    def _flip_labels(self, data: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
        np.random.seed(seed)
        num_noisy_samples = int(self.noise_level * len(data))
        data["noise"] = 0.0

        noisy_indices = np.random.choice(data.index, size=num_noisy_samples, replace=False)

        label_middle = (self.label_max + self.label_min) / 2
        for idx in noisy_indices:
            original_label = data.at[idx, self.label_column]
            if original_label > label_middle:
                # high labels are flipped to lower labels
                new_label = max(self.label_min, original_label - self.label_shift)
                noise_amount = original_label - new_label
            else:
                # low labels are flipped to higher labels
                new_label = min(self.label_max, original_label + self.label_shift)
                noise_amount = new_label - original_label
            data.at[idx, self.label_column] = new_label
            data.at[idx, "noise"] = noise_amount

        return data
    
    def process_data(self, data_paths: list[str], sanitise_labels: bool, add_noise: bool):
        # we may combine the data from different versions
        all_data = pd.DataFrame()
        for data_path in data_paths:
            data = self._raw_to_processed(
                path=data_path,
                sanitise_labels=sanitise_labels,
                add_noise=add_noise
            )
            all_data = pd.concat([all_data, data], ignore_index=True) if not all_data.empty else data

        log_info(logger, f"Total number of samples: {len(all_data)}\n")
        assert not all_data.isna().any().any(), "There are still NaN values in the data." # may occur due to the concat
        assert not all_data.isnull().any().any(), "The are still null values in the data"

        if self.llm_column in all_data.columns:
            all_data.drop(columns=[self.llm_column], inplace=True)

        all_data.rename(
            columns={
                self.label_column: "labels", 
                "essay": "text_1"
            },
            inplace=True
        )

        # add article information
        article = pd.read_csv("data/article-summarised.csv", index_col=0)
        all_data = pd.merge(all_data, article, on="article_id", how="left")
        all_data.drop(columns=["article_id", "text"], inplace=True)
        all_data.rename(columns={"summary_text": "text_2"}, inplace=True)

        return all_data

class BiEncoderDataCollator:
    def __init__(self, tokeniser_1, tokeniser_2):
        self.collator_1 = DataCollatorWithPadding(tokenizer=tokeniser_1)
        self.collator_2 = DataCollatorWithPadding(tokenizer=tokeniser_2)

    def __call__(self, batch):
        labels = [example["labels"] for example in batch] if "labels" in batch[0] else None

        batch_1 = [{k.replace("_1", ""): v for k, v in example.items() if "_1" in k} for example in batch]
        batch_2 = [{k.replace("_2", ""): v for k, v in example.items() if "_2" in k} for example in batch]

        batch_1 = self.collator_1(batch_1)
        batch_2 = self.collator_2(batch_2)

        final_batch = {
            "input_ids_1": batch_1["input_ids"],
            "attention_mask_1": batch_1["attention_mask"],
            "input_ids_2": batch_2["input_ids"],
            "attention_mask_2": batch_2["attention_mask"],
        }

        if labels is not None:
            final_batch["labels"] = torch.tensor(labels)

        return final_batch

class PairedTextDataModule:
    def __init__(
            self, 
            noise_level: float,
            delta: float | None, 
            tokeniser_plms: list[str],
            tokenise_paired_texts_each_tokeniser: bool
        ):

        self.noise_level = noise_level
        self.delta = delta
        self.tokenise_paired_texts_each_tokeniser = tokenise_paired_texts_each_tokeniser

        # keeping them constant for now, can make them arguments if required
        self.feature_to_tokenise = ["text_1", "text_2"]
        self.max_length = 512
        self.num_workers = 12

        self.tokeniser_plms = tokeniser_plms

        self.tokeniser = AutoTokenizer.from_pretrained(
            self.tokeniser_plms[0],
            use_fast=True,
            add_prefix_space=False # the first word is tokenised differently if not a prefix space, but it might decrease performance, so False (09/24)
        )

        if len(self.tokeniser_plms) > 1:
            # the second tokeniser, required for bi-encoder and cross-encoder modelling
            self.tokeniser_extra = AutoTokenizer.from_pretrained(
                self.tokeniser_plms[1],
                use_fast=True,
                add_prefix_space=False # the first word is tokenised differently if not a prefix space, but it might decrease performance, so False (09/24)
            ) # TODO: need to check if the parameters need any change depending on the tokeniser plm

            self.data_collator = BiEncoderDataCollator(tokeniser_1=self.tokeniser, tokeniser_2=self.tokeniser_extra)

        else:
            self.data_collator = DataCollatorWithPadding(tokenizer=self.tokeniser)

    def _individual_text_through_two_tokenisers(self, sentence):
        """Process single sentence column through two tokenisers"""
        tokeniser_1 = self.tokeniser(
            sentence[self.feature_to_tokenise[0]],
            truncation=True,
            max_length=self.max_length
        )

        tokeniser_2 = self.tokeniser_extra(
            sentence[self.feature_to_tokenise[1]],
            truncation=True,
            max_length=self.max_length
        )

        return {
            'input_ids_1': tokeniser_1['input_ids'],
            'attention_mask_1': tokeniser_1['attention_mask'],
            'input_ids_2': tokeniser_2['input_ids'],
            'attention_mask_2': tokeniser_2['attention_mask']
        }
        
    def _paired_texts_through_one_tokeniser(self, sentence):
        """Process pair of sentence through the same tokeniser"""
        return self.tokeniser(
            sentence[self.feature_to_tokenise[0]],
            sentence[self.feature_to_tokenise[1]],
            truncation=True,
            max_length=self.max_length
        )

    def _paired_texts_through_two_tokenisers(self, sentence):
        """Process pair of sentence through the two tokenisers"""

        tokeniser_1 = self.tokeniser(
            sentence[self.feature_to_tokenise[0]],
            sentence[self.feature_to_tokenise[1]],
            truncation=True,
            max_length=self.max_length
        )

        tokeniser_2 = self.tokeniser_extra(
            sentence[self.feature_to_tokenise[0]],
            sentence[self.feature_to_tokenise[1]],
            truncation=True,
            max_length=self.max_length
        )

        return {
            'input_ids_1': tokeniser_1['input_ids'],
            'attention_mask_1': tokeniser_1['attention_mask'],
            'input_ids_2': tokeniser_2['input_ids'],
            'attention_mask_2': tokeniser_2['attention_mask']
        }

    def get_hf_data(
            self, data_paths: list[str], sanitise_newsemp_labels: bool, 
            add_noise: bool, is_newsemp: bool = True, do_augment: bool = False
        ):
        if do_augment:
            from textattack.augmentation import (
                WordNetAugmenter
            )
        def _normalise_nullable_bool(x: str) -> bool | np.ndarray:
            if pd.isna(x):
                return np.nan
            try:
                f = float(x)
                if f == 1.0:
                    return True
                elif f == 0.0:
                    return False
                else:
                    raise ValueError 
            except (ValueError, TypeError):
                if str(x).strip().lower() == "true":
                    return True
                elif str(x).strip().lower() == "false":
                    return False
                else:
                    raise ValueError(f"Value {x} is not converted properly to a valid nullable boolean value.")

        def _augment_combine_save(data: pd.DataFrame | None, save_as: str) -> pd.DataFrame:
            """
            save_path to save or load the whole data
            if loaded, we don't need to augment again
            """
            if os.path.exists(save_as):
                data = pd.read_csv(save_as, sep="\t")
                # if all are augmented, then we don't need to do anything
                # While saving, the True/False is converted to string "1.0"/"0.0", so converting back
                data["is_augmented"] = data["is_augmented"].apply(_normalise_nullable_bool)
                if data[data["is_augmented"] == False].empty:
                    log_info(logger, f"All the samples are augmented, so not trying to augment further.")
                    return data
            else:
                data["is_augmented"] = False

            augmenter = WordNetAugmenter(
                pct_words_to_swap=0.1,
                transformations_per_example=1
            )

            augm_data_list = []
            counter = 0
            # for idx, row in data[data["is_augmented"] == False].iterrows():
            for row in tqdm(data[data["is_augmented"] == False].itertuples(), 
                            total=len(data[data["is_augmented"] == False]), 
                            desc="Augmenting"):
                idx = row.Index
                augm_text_1 = augmenter.augment(row.text_1)
                augm_text_2 = augmenter.augment(row.text_2)

                # Note: if transform_per_example > 1, we could also permute to get more samples

                for text_1, text_2 in zip(augm_text_1, augm_text_2):
                    augm_data_list.append({
                        "text_1": text_1,
                        "text_2": text_2,
                        "labels": row.labels,
                        "is_augmented": np.nan # These are nan because we don't want to augment them again
                    })

                data.at[idx, "is_augmented"] = True # mark the original as augmented
                counter += 1

                if counter % 5 == 0:
                    data_augms = pd.DataFrame(augm_data_list)
                    data = pd.concat([data, data_augms], ignore_index=True)
                    data.to_csv(save_as, sep="\t", index=False)
                    log_info(logger, f"Saved the data to {save_as}")
                    augm_data_list = [] # clear the list

            if len(augm_data_list) > 0:
                # save the remaining at the end
                data_augms = pd.DataFrame(augm_data_list)
                data = pd.concat([data, data_augms], ignore_index=True)
                data.to_csv(save_as, sep="\t", index=False)
                log_info(logger, f"Saved the data to {save_as}")

            return data

        if do_augment:
            filenames = [os.path.splitext(os.path.basename(path))[0]
                         for path in data_paths]
            if len(filenames) > 1:
                save_as = os.path.join(
                    os.path.commonpath(data_paths),
                    f"{'_'.join(filenames)}_augmented.tsv"
                )
            else:
                # in that case, common path doesn't make sense
                save_as = os.path.join(os.path.dirname(data_paths[0]), f"{filenames[0]}_augmented.tsv")
            
            if os.path.exists(save_as):
                log_info(logger, f"IMPORTANT: Loading saved augmented data. Hence, sanitising and adding noise is not being done. Whatever was done during augmentation is being used.")
                log_info(logger, f"Reading data from {save_as}")
                # Now it can be either augmentation done or need to be resumed
                # so calling the augmenter again
                all_data = _augment_combine_save(data=None, save_as=save_as)
                log_info(logger, f"Read {len(all_data)} samples from {save_as}")
            else:
                log_info(logger, f"No saved augmented data found as {save_as}. Processing from scratch.")
                if is_newsemp:
                    newsemp_preprocessor = NewsEmpPreprocessorFromRaw(noise_level=self.noise_level, delta=self.delta)
                    all_data = newsemp_preprocessor.process_data(
                        data_paths=data_paths,
                        sanitise_labels=sanitise_newsemp_labels,
                        add_noise=add_noise
                    )
                else:
                    # doesn't require much processing, so done here
                    all_data = pd.DataFrame()
                    for data_path in data_paths:
                        data = pd.read_csv(data_path)
                        log_info(logger, f"Read {len(data)} samples from {data_path}")
                        all_data = pd.concat([all_data, data], ignore_index=True) if not all_data.empty else data
                    all_data["story_A"] = all_data["story_A"].str.replace("\n", "", regex=False)
                    all_data["story_B"] = all_data["story_B"].str.replace("\n", "", regex=False)
                    all_data.rename(
                        columns={
                            "story_A": "text_1",
                            "story_B": "text_2",
                            "similarity_empathy_human_AGG": "labels"
                        },
                        inplace=True
                    )
                    all_data = all_data[["text_1", "text_2", "labels"]] #FIXME: can be removed later as we filter the columns later
                    
                    all_data = _augment_combine_save(all_data, save_as=save_as)
        else:
            log_info(logger, "Processing from scratch without any augmentation.")
            if is_newsemp:
                newsemp_preprocessor = NewsEmpPreprocessorFromRaw(noise_level=self.noise_level, delta=self.delta)
                all_data = newsemp_preprocessor.process_data(
                    data_paths=data_paths,
                    sanitise_labels=sanitise_newsemp_labels,
                    add_noise=add_noise
                )
            else:
                # doesn't require much processing, so done here
                all_data = pd.DataFrame()
                for data_path in data_paths:
                    data = pd.read_csv(data_path)
                    log_info(logger, f"Read {len(data)} samples from {data_path}")
                    all_data = pd.concat([all_data, data], ignore_index=True) if not all_data.empty else data
                all_data["story_A"] = all_data["story_A"].str.replace("\n", "", regex=False)
                all_data["story_B"] = all_data["story_B"].str.replace("\n", "", regex=False)
                all_data.rename(
                    columns={
                        "story_A": "text_1",
                        "story_B": "text_2",
                        "similarity_empathy_human_AGG": "labels"
                    },
                    inplace=True
                )
        
        keep_col = ["text_1", "text_2"]
        if "labels" in all_data.columns: # for non 2024 test set, we don't have labels
            keep_col.append("labels")
        if add_noise:
            keep_col.append("noise")
        all_data = all_data[keep_col]
        log_info(logger, f"Total number of samples: {len(all_data)}\n")

        assert not all_data.isna().any().any(), "There are still NaN values in the data."
        assert not all_data.isnull().any().any(), "The are still null values in the data"
        
        # the remaining processing like convertint to hf

        all_data_hf = Dataset.from_pandas(all_data, preserve_index=False) # convert to huggingface dataset

        # resolve tokeniser function
        if self.tokenise_paired_texts_each_tokeniser:
            tokeniser_fn = self._paired_texts_through_one_tokeniser if len(self.tokeniser_plms) == 1 else self._paired_texts_through_two_tokenisers
        else:
            tokeniser_fn = self._individual_text_through_two_tokenisers if len(self.tokeniser_plms) > 1 else None # Cannot have single text through single tokeniser
        # tokenise
        all_data_hf = all_data_hf.map(
            tokeniser_fn,
            batched=True,
            remove_columns=self.feature_to_tokenise
        )

        all_data_hf.set_format('torch')
        
        return all_data_hf
    
    # taken from https://pytorch.org/docs/stable/notes/randomness.html
    def _seed_worker(self, worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed) 
    
    def _get_dl(
            self, data_paths: list[str], shuffle: bool, batch_size: int, 
            sanitise_newsemp_labels: bool, add_noise: bool, seed: int | None = None,
            is_newsemp: bool = True, do_augment: bool = False, lbl_split: float = 1.0
        ):

        if shuffle:
            # making sure the shuffling is reproducible
            g = torch.Generator()
            g.manual_seed(seed)

        hf_ds = self.get_hf_data(
            data_paths=data_paths,
            sanitise_newsemp_labels=sanitise_newsemp_labels,
            add_noise=add_noise,
            is_newsemp=is_newsemp,
            do_augment=do_augment
        )

        # for baseline (of ssl) experiments, we'd only use x% data for training
        if lbl_split < 1.0:
            split_ds = hf_ds.train_test_split(
                train_size=lbl_split,
                shuffle=True,
                seed=seed
            )
            hf_ds = split_ds["train"]

        return DataLoader(
            hf_ds,
            batch_size=batch_size, 
            shuffle=shuffle,
            collate_fn=self.data_collator,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=self._seed_worker,
            generator=g if shuffle else None
        )
    
    def get_train_dl(
            self, data_path_list: list, batch_size: int,
            sanitise_newsemp_labels: bool = True, add_noise: bool = False, seed: int | None = None,
            is_newsemp: bool = True, do_augment: bool = False, lbl_split: float = 1.0
        ):
        
        return self._get_dl(
            data_path_list, shuffle=True, 
            batch_size=batch_size, sanitise_newsemp_labels=sanitise_newsemp_labels, add_noise=add_noise, seed=seed,
            is_newsemp=is_newsemp, do_augment=do_augment, lbl_split=lbl_split
        )
    
    def get_val_dl(
            self, data_path_list:list, batch_size: int, 
            sanitise_newsemp_labels: bool = True, add_noise: bool = False,
            is_newsemp: bool = True
        ):
        # depending on data_name, the labels can be in different file
        return self._get_dl(
            data_path_list, shuffle=False, 
            batch_size=batch_size, sanitise_newsemp_labels=sanitise_newsemp_labels, add_noise=add_noise,
            is_newsemp=is_newsemp
        )
    
    def get_test_dl(
            self, data_path_list: list, batch_size: int = 32,
            sanitise_newsemp_labels: bool = True, add_noise: bool = False,
            is_newsemp: bool = True
        ):
        return self._get_dl(
            data_path_list, shuffle=False,
            batch_size=batch_size, sanitise_newsemp_labels=sanitise_newsemp_labels, add_noise=add_noise,
            is_newsemp=is_newsemp
        ) # we have labels in 2024 data
