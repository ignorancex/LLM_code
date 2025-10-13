import torch
from torch.utils.data import Dataset, Subset
import numpy as np
import os
from utils.document import SynthBioDoc, SynthBioPerplex, Wiki2023, Wiki2023_EVAL
from utils.question_dataset import QASynthBio
import random
import os

IGNORE_TOKEN_ID = -100


def get_shuffle_idx(seed, size):
    np_rng = np.random.RandomState(seed=seed)
    dtype_ = np.uint32
    if size >= (np.iinfo(np.uint32).max - 1):
        dtype_ = np.int64
    shuffle_idx = np.arange(start=0, stop=size, step=1, dtype=dtype_)
    np_rng.shuffle(shuffle_idx)
    return shuffle_idx


class CustomDataset(Dataset):
    def __init__(self, dataset, pad_token_id) -> None:
        super().__init__()
        self.dataset = dataset
        self.pad_token_id = pad_token_id

    def __len__(self):
        length = len(self.dataset)
        return length

    def __getitem__(self, idx):
        input_ids = self.dataset[idx]["input_ids"]
        end_question = self.dataset[idx].get("end_question", None)
        answer = self.dataset[idx].get("answer", None)
        true_answer = self.dataset[idx].get("true_answer", None)
        index = self.dataset[idx].get("index", None)
        labels = self.dataset[idx].get("labels", input_ids)
        index_list = self.dataset[idx].get("index_list", None)
        attention_mask = self.dataset[idx]["attention_mask"]

        return {
            "end_question": end_question,
            "index_list": index_list,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "answer": answer,
            "true_answer": true_answer,
            "index": index,
        }


class TwoDataset(Dataset):
    def __init__(
        self,
        dataset_first,
        dataset_second,
        p_choose_first,
        noise_ratio=0.0,
        vocab_size=50000,
        concat_dataset=False,
        mask_first=False,
    ) -> None:
        super().__init__()
        self.dataset_first = dataset_first
        self.dataset_second = dataset_second
        self.p_choose_first = p_choose_first
        self.noise_ratio = noise_ratio
        self.vocab_size = vocab_size
        self.concat_dataset = concat_dataset
        self.mask_first = mask_first

    def __len__(self):
        length = (
            len(self.dataset_first)
            if not self.concat_dataset
            else len(self.dataset_first) + len(self.dataset_second)
        )
        return length

    def sample_id(self, idx):
        if self.concat_dataset:
            if idx > len(self.dataset_first) - 1:
                data_pick = self.dataset_second
                idx = idx - len(self.dataset_first)
            else:
                data_pick = self.dataset_first
        else:
            if random.uniform(0, 1) < self.p_choose_first:
                data_pick = self.dataset_first
            else:
                data_pick = self.dataset_second
                random.seed(idx)
                idx = random.randint(0, len(data_pick) - 1)
        return data_pick, idx

    def __getitem__(self, idx):
        data_pick, idx = self.sample_id(idx)
        input_ids = data_pick[idx]["input_ids"]
        end_question = data_pick[idx].get("end_question", None)
        answer = data_pick[idx].get("answer", None)
        true_answer = data_pick[idx].get("true_answer", None)
        labels = data_pick[idx].get("labels", input_ids)
        attention_mask = data_pick[idx]["attention_mask"]
        index = data_pick[idx].get("index", None)

        if data_pick == self.dataset_second and self.noise_ratio > 0:
            queries = data_pick[idx]["input_queries"]
            new_ids = []
            for ind, id in enumerate(input_ids):
                if ind > len(queries) and random.uniform(0, 1) < self.noise_ratio:
                    new_id = random.randint(0, self.vocab_size - 1)
                    new_ids.append(new_id)
                else:
                    new_ids.append(id)
            input_ids = new_ids
        return {
            "end_question": end_question,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "answer": answer,
            "true_answer": true_answer,
            "index": index,
        }


def load_document_dataset(
    name_data,
    file_path,
    tokenizer,
    max_seq_len,
    inst_mode=True,
    shuffle=0,
    val=False,
    task_name=None,
):
    print(name_data, file_path)
    if name_data == "synth_language5":
        sentence_perplex = False
        class_doc = SynthBioDoc
        doc_data = class_doc(
            tokenizer=tokenizer,
            filename=file_path,
            max_seq_len=max_seq_len,
            inst_mode=inst_mode,
            shuffle=shuffle,
            sentence_perplex=sentence_perplex,
            eval=False,
        )
    elif name_data == "wiki_film":
        class_doc = Wiki2023
        sentence_perplex = False
        if val:
            if isinstance(task_name, int):
                file_path = f"film_sent_{task_name}.jsonl"
                sentence_perplex = True
            else:
                file_path = "film_all.jsonl"
                class_doc = Wiki2023_EVAL
        doc_data = class_doc(
            tokenizer=tokenizer,
            filename=file_path,
            max_seq_len=max_seq_len,
            inst_mode=inst_mode,
            shuffle=shuffle,
            eval=False,
            sentence_perplex=sentence_perplex,
        )    
    doc_data.init_dataset()
    doc_data = doc_data.dataset["train"]
    print(
        doc_data["answer"][0],
        doc_data["query"][0],
        tokenizer.decode(doc_data["input_ids"][0]),
    )
    return doc_data


def load_qa_dataset(
    file_path,
    tokenizer,
    max_seq_len,
    inst_mode=False,
    val=False,
    no_answer=False,
    task_name=None,
):
    qa_data = QASynthBio(
        tokenizer=tokenizer,
        filename=file_path,
        max_seq_len=max_seq_len,
        inst_mode=inst_mode,
        val=val,
        no_answer=no_answer,
    )
    qa_data.init_dataset()
    qa_data = qa_data.dataset["train"]
    print(qa_data["answer"][0], qa_data["question"][0], qa_data["query"][0])
    return qa_data


def get_filenames(train_config):
    cache_path = train_config["cache_path"]
    model = train_config["model_name"]
    doc_dataset = train_config["doc_dataset"]
    qa_dataset = train_config["qa_dataset"]
    inst_mode_doc = train_config["inst_mode_doc"]
    inst_mode_qa = train_config["inst_mode_qa"]
    shuffle = train_config["shuffle"]
    max_seq_len = train_config["max_seq_len"]
    doc_fname = f"{doc_dataset}_model_{model}_length_{max_seq_len}_inst_{inst_mode_doc}_shuffle_{shuffle}.pt"
    qa_fname = f"{qa_dataset}_model_{model}_length_{max_seq_len}_inst_{inst_mode_qa}_shuffle_{shuffle}.pt"
    eval_fname = f"{qa_dataset}_evaldata_model_{model}_length_{max_seq_len}_inst_{inst_mode_qa}_shuffle_{shuffle}.pt"
    doc_fname = os.path.join(cache_path, doc_fname)
    qa_fname = os.path.join(cache_path, qa_fname)
    eval_fname = os.path.join(cache_path, eval_fname)
    return doc_fname, qa_fname, eval_fname


def create_dataset(tokenizer, train_config):

    num_eval = 1000
    doc_fname, qa_fname, eval_fname = get_filenames(train_config)
    doc_dataset = train_config["doc_dataset"]
    qa_dataset = train_config["qa_dataset"]
    local_rank = train_config["local_rank"]

    cache_found_doc = os.path.isfile(doc_fname)
    cache_found_qa = os.path.isfile(qa_fname)
    cache_found_eval = os.path.isfile(eval_fname)
    buf_create_cache_doc = torch.ByteTensor([not cache_found_doc]).cuda()
    buf_create_cache_qa = torch.ByteTensor([not cache_found_qa]).cuda()
    buf_create_cache_eval = torch.ByteTensor([not cache_found_eval]).cuda()

    ## evaluation mode, evaluate QA performance.
    if train_config["eval_qa"]:
        torch.distributed.all_reduce(buf_create_cache_doc)
        if local_rank <= 0:
            eval_dataset = load_qa_dataset(
                file_path=train_config["path_qa_eval"],
                tokenizer=tokenizer,
                max_seq_len=train_config["max_seq_len"],
                inst_mode=train_config["inst_mode_qa"],
                val=True,
                no_answer=True,
            )
            eval_dataset = CustomDataset(eval_dataset, tokenizer.pad_token_id)
            torch.save(eval_dataset, eval_fname)
        torch.distributed.barrier()
        eval_dataset = torch.load(eval_fname)
        return eval_dataset

    ## make document training data
    torch.distributed.all_reduce(buf_create_cache_doc)
    doc_dataset = load_document_dataset(
        name_data=doc_dataset,
        file_path=train_config["path_doc_train"],
        tokenizer=tokenizer,
        max_seq_len=train_config["max_seq_len"],
        inst_mode=train_config["inst_mode_doc"],
        shuffle=train_config["shuffle"],
    )
    torch.distributed.barrier()
    torch.distributed.all_reduce(buf_create_cache_qa)
    ## make QA data
    ## validation data
    eval_dataset = load_qa_dataset(
        file_path=train_config["path_qa_eval"],
        tokenizer=tokenizer,
        max_seq_len=train_config["max_seq_len"],
        inst_mode=train_config["inst_mode_qa"],
        val=True,
        no_answer=False,
    )
    eval_dataset = CustomDataset(eval_dataset, tokenizer.pad_token_id)
    shuffle_idx = get_shuffle_idx(train_config["seed"], len(eval_dataset))
    eval_dataset = Subset(eval_dataset, shuffle_idx.tolist()[:num_eval])

    ## training data
    qa_dataset = load_qa_dataset(
        file_path=train_config["path_qa_train"],
        tokenizer=tokenizer,
        max_seq_len=train_config["max_seq_len"],
        inst_mode=train_config["inst_mode_qa"],
    )
    torch.save(qa_dataset, qa_fname)
    torch.distributed.barrier()
    qa_dataset = torch.load(qa_fname)
    ## Merge two datasets.
    ## QA_dataset needs to be the first dataset.
    train_dataset = TwoDataset(
        qa_dataset,
        doc_dataset,
        p_choose_first=train_config["p_choose_qa"],
        noise_ratio=train_config["noise_ratio"],
        vocab_size=tokenizer.vocab_size,
        concat_dataset=train_config["concat_dataset"],
    )
    return train_dataset, eval_dataset
