import os
import re
import glob
import torch
import random
import pandas as pd
from p_tqdm import p_umap
import itertools
from pathlib import Path

from dataclasses import dataclass
import transformers

from torch.utils.data import Dataset
import pickle

from abnumber import Chain

IGNORE_INDEX = -100
MAX_LENGTH = 2048

class CloneDataset(Dataset):
    def __init__(
        self,
        csv_dir,
        tokenizer=None,
        format_options={},
        chain_type="light",
        min_clone_size=10,
        skip_small_clones=True,
        use_clone_attention=False,
        task="train",
        oracle_task=None,
    ):
        super().__init__()
        
        if (task == "train") or (task == "generation"):
            self.setup_filenames(chain_type, csv_dir, min_clone_size, skip_small_clones)
        elif task == "oracle_train":
            self.setup_filenames_oracle(chain_type, csv_dir, min_clone_size, skip_small_clones, oracle_task)
        else:
            raise ValueError
        
        self.tokenizer = tokenizer
        self.format_options = format_options
        self.use_clone_attention = use_clone_attention
   
    def process_clone(self, input_df):
        
        # chain_id = input_df["chain_id"].iloc[0]
        # chain_token_id = self.tokenizer.hv_token_id if chain_id == "heavy" else self.tokenizer.lv_token_id

        seqs = input_df["sequence_alignment_aa"].tolist()
        
        random.shuffle(seqs)

        # TODO: check the function for this process_clone function
        # breakpoint() 

        lengths = [len(seq) for seq in seqs]
        if (sum(lengths) + len(seqs)) > MAX_LENGTH:
            num_seqs = MAX_LENGTH // max(lengths)
            seqs = seqs[:num_seqs]

        tokens = [
            self.tokenizer(" ".join(seq), return_tensors="pt").input_ids[0][1:-1] for seq in seqs
        ]

        one_t = torch.ones(1, dtype=torch.long)

        if self.use_clone_attention:
            w_seps = []

            for t in tokens:
                w_seps.append(torch.cat([self.tokenizer.bos_token_id * one_t, t, self.tokenizer.seq_sep_token_id * one_t]))
            
            input_ids = labels = w_seps
            input_ids_lens = labels_lens = [input_id.ne(self.tokenizer.pad_token_id).sum().item() for input_id in input_ids]
        else:
            w_seps = [
                self.tokenizer.bos_token_id * one_t,
                # chain_token_id * one_t,
            ]

            for t in tokens:
                w_seps.append(t)
                w_seps.append(
                    self.tokenizer.seq_sep_token_id * one_t
                )
            
            tokens = torch.cat(w_seps)
            tokens = tokens[:MAX_LENGTH]
            
            input_ids = labels = tokens
            input_ids_lens = labels_lens = tokens.ne(
                self.tokenizer.pad_token_id).sum().item()
                
        return dict(
            input_ids=input_ids,
            labels=labels,
            input_ids_lens=input_ids_lens,
            labels_lens=labels_lens,
        )

    def setup_filenames_oracle(self, chain_type, csv_dir, min_clone_size, skip_small_clones, oracle_task):
        if chain_type=="heavy":
            if oracle_task=="train":
                list_of_csv_files = [
                                    "clone_0_n10195.csv",
                                    "clone_1_n7859.csv",
                                    "clone_2_n5741.csv",
                                    "clone_3_n4364.csv",
                                    "clone_4_n4270.csv",
                                    "clone_5_n4149.csv",
                                    "clone_6_n3927.csv",
                                    "clone_7_n3917.csv",
                                    "clone_8_n3904.csv",
                                    "clone_9_n3533.csv",
                                    "clone_10_n3501.csv",
                                    "clone_11_n3423.csv",
                                    "clone_12_n3365.csv",
                                    "clone_13_n3293.csv",
                                    "clone_14_n3216.csv",
                                    "clone_15_n3095.csv",
                                    "clone_16_n3077.csv",
                                    "clone_17_n3059.csv",
                ]
            elif oracle_task=="val":
                list_of_csv_files = ["clone_18_n2990.csv"]
            elif oracle_task=="test":
                list_of_csv_files = ["clone_19_n2880.csv"]
            else:
                raise ValueError
        elif chain_type=="light":
            if oracle_task=="train":
                list_of_csv_files = [
                                    "clone_0_n342.csv",
                                    "clone_1_n257.csv",
                                    "clone_2_n255.csv",
                                    "clone_3_n240.csv",
                                    "clone_4_n232.csv",
                                    "clone_5_n218.csv",
                                    "clone_6_n209.csv",
                                    "clone_7_n209.csv",
                                    "clone_8_n208.csv",
                                    "clone_9_n207.csv",
                                    "clone_10_n206.csv",
                                    "clone_11_n205.csv",
                                    "clone_12_n205.csv",
                                    "clone_13_n205.csv",
                                    "clone_14_n205.csv",
                                    "clone_15_n202.csv",
                                    "clone_16_n201.csv",
                                    "clone_17_n200.csv",
                                    ]
            elif oracle_task=="val":
                list_of_csv_files = ["clone_18_n200.csv"]
            elif oracle_task=="test":
                list_of_csv_files = ["clone_19_n195.csv"]
            else:
                raise ValueError            
        else:
            raise ValueError
        
        self.filenames = list(map(lambda csv_i: os.path.join(csv_dir, csv_i), list_of_csv_files))
        print(f"len(self.filenames)={len(self.filenames)}")

    def setup_filenames(self, chain_type, csv_dir, min_clone_size, skip_small_clones):        
        if chain_type=="heavy":
            if isinstance(csv_dir, list):
                clones_size_csv = pd.read_csv("/vast/yl9959/heavy_train_size.csv")
            else:
                if "val" in csv_dir:
                    clones_size_csv = pd.read_csv("/vast/yl9959/heavy_val_size.csv")
                elif "test" in csv_dir:
                    clones_size_csv = pd.read_csv("/vast/yl9959/heavy_test_size.csv")
                else:
                    raise ValueError
        elif chain_type=="light":
            if "train" in csv_dir:
                clones_size_csv = pd.read_csv("/vast/yl9959/light_train_size.csv")
            elif "val" in csv_dir:
                clones_size_csv = pd.read_csv("/vast/yl9959/light_val_size.csv")
            elif "test" in csv_dir:
                clones_size_csv = pd.read_csv("/vast/yl9959/light_test_size.csv")
            else:
                raise ValueError
        else:
            raise ValueError

        filtered_clones_size_csv = clones_size_csv[clones_size_csv['num_clones'] >= min_clone_size]
        self.filenames = filtered_clones_size_csv['file_name'].tolist()
        print(f"len(self.filenames)={len(self.filenames)}")
        
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError("Index out of range")

        fn = self.filenames[index]
        df = pd.read_csv(fn)
        return self.process_clone(df)
    

@dataclass
class DataCollatorForSeqLabelsDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    use_clone_attention: bool

    def __call__(self, instances):
        if self.use_clone_attention:
            input_ids, labels = tuple(
                [[elem.clone().detach() for elem in instance[key]] for instance in instances] 
                    for key in ("input_ids", "labels")
            )
            input_ids = list(itertools.chain.from_iterable(input_ids))
            labels = list(itertools.chain.from_iterable(labels))

            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
            )
            labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        else:
            input_ids, labels = tuple(
                [instance[key].clone().detach() for instance in instances] 
                    for key in ("input_ids", "labels")
            )

            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
            )
            labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
            
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
    
def setup_clone_datasets(args, tokenizer):    
    format_options = {}

    if args.chain_type == "light":
        if str(args.data_path) == "/scratch/aa11803/big_hat/oracle_train_data/light_clones":
            datasets = {
                "train": CloneDataset(
                    str(args.data_path),
                    tokenizer,
                    format_options,
                    args.chain_type,
                    args.min_clone_size,
                    skip_small_clones=True,
                    use_clone_attention=args.use_clone_attention,
                    task=args.task,
                    oracle_task="train",
                ),
                "val": CloneDataset(
                    str(args.data_path),
                    tokenizer,
                    format_options,
                    args.chain_type,
                    args.min_clone_size,
                    skip_small_clones=True,
                    use_clone_attention=args.use_clone_attention,
                    task=args.task,
                    oracle_task="val",
                ),
                "test": CloneDataset(
                    str(args.data_path),
                    tokenizer,
                    format_options,
                    args.chain_type,
                    args.min_clone_size,
                    skip_small_clones=True,
                    use_clone_attention=args.use_clone_attention,
                    task=args.task,
                    oracle_task="test",
                ),
            }
        elif str(args.data_path) == "/light_chain":
            datasets = {
                "train": CloneDataset(
                    str(args.data_path / "train"), 
                    tokenizer,
                    format_options,
                    args.chain_type,
                    args.min_clone_size,
                    skip_small_clones=True,
                    use_clone_attention=args.use_clone_attention,
                    task=args.task,
                    oracle_task=None,
                ),
                "val": CloneDataset(
                    str(args.data_path / "val"),
                    tokenizer,
                    format_options,
                    args.chain_type,
                    args.min_clone_size,
                    skip_small_clones=True,
                    use_clone_attention=args.use_clone_attention,
                    task=args.task,
                    oracle_task=None,
                ),
                "test": CloneDataset(
                    str(args.data_path / "test"),
                    tokenizer,
                    format_options,
                    args.chain_type,
                    args.min_clone_size,
                    skip_small_clones=True,
                    use_clone_attention=args.use_clone_attention,
                    task=args.task,
                    oracle_task=None,
                ),
            }
        else:
            raise ValueError
    elif args.chain_type == "heavy":
        if str(args.data_path) == "/scratch/aa11803/big_hat/oracle_train_data/heavy_clones":
            datasets = {
                "train": CloneDataset(
                    str(args.data_path),
                    tokenizer,
                    format_options,
                    args.chain_type,
                    args.min_clone_size,
                    skip_small_clones=True,
                    use_clone_attention=args.use_clone_attention,
                    task=args.task,
                    oracle_task="train",
                ),
                "val": CloneDataset(
                    str(args.data_path),
                    tokenizer,
                    format_options,
                    args.chain_type,
                    args.min_clone_size,
                    skip_small_clones=True,
                    use_clone_attention=args.use_clone_attention,
                    task=args.task,
                    oracle_task="val",
                ),
                "test": CloneDataset(
                    str(args.data_path),
                    tokenizer,
                    format_options,
                    args.chain_type,
                    args.min_clone_size,
                    skip_small_clones=True,
                    use_clone_attention=args.use_clone_attention,
                    task=args.task,
                    oracle_task="test",
                ),
            }
        elif str(args.data_path) == "/":
            datasets = {
                "train": CloneDataset(
                    [str(args.data_path / "train1"), str(args.data_path / "train2"), str(args.data_path / "train3"), str(args.data_path / "train4")], 
                    tokenizer,
                    format_options,
                    args.chain_type,
                    args.min_clone_size,
                    skip_small_clones=True,
                    use_clone_attention=args.use_clone_attention,
                    task=args.task,
                    oracle_task=None,
                ),
                "val": CloneDataset(
                    str(args.data_path / "val"),
                    tokenizer,
                    format_options,
                    args.chain_type,
                    args.min_clone_size,
                    skip_small_clones=True,
                    use_clone_attention=args.use_clone_attention,
                    task=args.task,
                    oracle_task=None,
                ),
                "test": CloneDataset(
                    str(args.data_path / "test"),
                    tokenizer,
                    format_options,
                    args.chain_type,
                    args.min_clone_size,
                    skip_small_clones=True,
                    use_clone_attention=args.use_clone_attention,
                    task=args.task,
                    oracle_task=None,
                ),
            }
        else:
            raise ValueError
    else:
        raise ValueError

    return datasets


AHO_HEAVY_LENGTH = 149
AHO_LIGHT_LENGTH = 148

class LabeledDataset(Dataset):
    def __init__(
        self,
        csv_fn,
        split_name="train",
        tokenizer=None,
        sequence_name=None,
        label_name=None,
        num_labels=1,
        align_seq=True,
    ):
        super().__init__()
        
        self.data = pd.read_csv(csv_fn)
        self.data = self.data.dropna()
        self.data = self.data.sample(frac=1, random_state=0).reset_index(drop=True)

        if sequence_name is None:
            sequence_name = self.data.columns[0]
        if label_name is None:
            label_name = self.data.columns[1]

        if num_labels == 1:
            self.mean = self.data[label_name].mean()
            self.std = self.data[label_name].std()
            self.data[label_name] = (self.data[label_name] - self.mean) / self.std 

        if split_name == "train":
            self.data = self.data[:int(0.8*len(self.data))]
        elif split_name == "val":
            self.data = self.data[int(0.8*len(self.data)):int(0.9*len(self.data))]
        elif split_name == "test":
            self.data = self.data[int(0.9*len(self.data)):]

        self.data = self.data.to_dict(orient="records")
        self.num_labels = num_labels

        self.data = [(d[sequence_name], d[label_name]) for d in self.data]

        if align_seq:
            print("Aligning sequences with aho...")
            self.data = p_umap(lambda t: (self.align(t[0]), t[1]), self.data)

        self.tokenizer = tokenizer
        
    def align(self, seq):
        chain = Chain(seq.replace("-",""), scheme='aho', cdr_definition='chothia')
        chain_dict = {int(re.sub("[^0-9]", "", str(pos)[1:])): aa for pos, aa in chain}
        expect_length = AHO_HEAVY_LENGTH if chain.is_heavy_chain() else AHO_LIGHT_LENGTH
        aligned = [chain_dict.get(i, "-") for i in range(1, expect_length+1)]
        return "".join(aligned)

    def inverse_transform(self, label):
        return label * self.std + self.mean

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError("Index out of range")

        seq, label = self.data[index]
  
        input_ids = self.tokenizer(" ".join(seq), return_tensors="pt").input_ids[0]
        dtype = torch.float if self.num_labels == 1 else torch.long
        labels = torch.tensor(label, dtype=dtype)

        return {
            "input_ids": input_ids,
            "labels": labels,
        }

@dataclass
class DataCollatorForLabeledDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        input_ids, labels = tuple(
            [instance[key].clone().detach() for instance in instances] 
                for key in ("input_ids", "labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.stack(labels)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def setup_labeled_datasets(args, tokenizer):    
    datasets = {
        "train": LabeledDataset(
            args.data_csv_fn,
            "train",
            tokenizer,
            args.data_sequence_name,
            args.data_label_name,
            args.data_num_labels,
        ),
        "val": LabeledDataset(
            args.data_csv_fn,
            "val",
            tokenizer,
            args.data_sequence_name,
            args.data_label_name,
            args.data_num_labels,
        ),
        "test": LabeledDataset(
            args.data_csv_fn,
            "test",
            tokenizer,
            args.data_sequence_name,
            args.data_label_name,
            args.data_num_labels,
        ),
    }
    return datasets