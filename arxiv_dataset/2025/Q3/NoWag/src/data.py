"""
From https://github.com/IST-DASLab/gptq/blob/main/datautils.py
"""

import numpy as np
import torch
from datasets import load_dataset
from torch import nn
from tqdm import trange
import tqdm
import sys


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def get_wikitext2(nsamples, seed, seqlen, model, train_test: str = "train"):
    assert train_test in ["train", "test"]
    from datasets import load_dataset

    if train_test == "train":
        data = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    else:
        from datasets import load_dataset

        data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    enc = tokenizer("\n\n".join(data["text"]), return_tensors="pt")
    if train_test == "test":
        return enc

    import random

    random.seed(seed)
    trainloader = []
    for _ in tqdm.tqdm(range(nsamples)):
        i = random.randint(0, enc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = enc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader


def get_red_pajama(nsamples, seqlen, model, train_test: str = "train"):
    # modified from AQLM

    assert train_test in ["train"]
    traindata = load_dataset("togethercomputer/RedPajama-Data-1T-Sample", split="train")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    tokenizer.bos_token_id = 1
    tokenizer.eos_token_id = 2
    trainloader = []
    import random

    for _ in tqdm.tqdm(range(nsamples), desc="Loading Red Pajama"):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]["text"], return_tensors="pt")
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        assert inp.shape[1] == seqlen
        trainloader.append((inp, None))  # keep it the same as other datasets
    return trainloader


def get_ptb(nsamples, seed, seqlen, model, train_test: str = "train"):
    assert train_test in ["train", "test"]
    from datasets import load_dataset

    if train_test == "train":
        data = load_dataset(
            "ptb_text_only", "penn_treebank", split="train", trust_remote_code=True
        )
    else:
        data = load_dataset(
            "ptb_text_only", "penn_treebank", split="validation", trust_remote_code=True
        )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    enc = tokenizer("\n\n".join(data["sentence"]), return_tensors="pt")

    if train_test == "test":
        return enc

    import random

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, enc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = enc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader


def get_c4(nsamples, seed, seqlen, model, train_test: str = "train"):
    assert train_test in ["train", "test"]
    from datasets import load_dataset

    if train_test == "train":
        data = load_dataset(
            "allenai/c4",
            data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
            split="train",
        )
    else:
        data = load_dataset(
            "allenai/c4",
            data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
            split="validation",
        )
    # print("---------------data type-----------------")
    # print(type(data))
    # print(data.shape)
    # print("----------------seqlen---------")
    # print(seqlen)
    # print("----------nsamples-----------")
    # print(nsamples)
    # print("---------data sample-----------")
    # print(data[0])

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    if train_test == "train":
        import random

        random.seed(seed)
        trainloader = []
        for _ in range(nsamples):
            while True:
                i = random.randint(0, len(data) - 1)
                trainenc = tokenizer(data[i]["text"], return_tensors="pt")
                if trainenc.input_ids.shape[1] >= seqlen:
                    break
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        return trainloader
    else:
        import random

        random.seed(0)
        valenc = []
        for _ in range(256):
            while True:
                i = random.randint(0, len(data) - 1)
                tmp = tokenizer(data[i]["text"], return_tensors="pt")
                # print("--------LEN---------")
                # print(tmp.input_ids.shape[1] - seqlen - 1)
                if tmp.input_ids.shape[1] >= seqlen:
                    break
            i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            # print("-------------------J-----------------------")
            # print(j)
            valenc.append(tmp.input_ids[:, i:j])
        # print("------------VALENC-----------------")
        # print(f"{sys.getsizeof(valenc)} bytes")
        valenc = torch.hstack(valenc)
        print("loaded valenc")
        print(valenc.shape)

        class TokenizerWrapper:
            def __init__(self, input_ids):
                self.input_ids = input_ids

        valenc = TokenizerWrapper(valenc)
        return valenc


def get_ptb_new(nsamples, seed, seqlen, model):
    from datasets import load_dataset

    traindata = load_dataset("ptb_text_only", "penn_treebank", split="train")
    testdata = load_dataset("ptb_text_only", "penn_treebank", split="test")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    trainenc = tokenizer(" ".join(traindata["sentence"]), return_tensors="pt")
    testenc = tokenizer(" ".join(testdata["sentence"]), return_tensors="pt")

    import random

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_c4_new(nsamples, seed, seqlen, model):
    from datasets import load_dataset

    traindata = load_dataset(
        "allenai/c4",
        data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
        split="train",
    )
    valdata = load_dataset(
        "allenai/c4",
        data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
        split="validation",
    )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    import random

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]["text"], return_tensors="pt")
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valenc = tokenizer(" ".join(valdata[:1100]["text"]), return_tensors="pt")
    valenc = valenc.input_ids[:, : (256 * seqlen)]

    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids

    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc


def get_loaders(
    name, nsamples=128, seed=0, seqlen=2048, model="", train_test: str = "train"
):
    if "pajama" in name:
        return get_red_pajama(nsamples, seqlen, model, train_test)
    if "wikitext2" in name:
        return get_wikitext2(nsamples, seed, seqlen, model, train_test)
    if "ptb" in name:
        if "new" in name:
            raise NotImplementedError
            return get_ptb_new(nsamples, seed, seqlen, model)
        return get_ptb(nsamples, seed, seqlen, model, train_test)
    if "c4" in name:
        if "new" in name:
            raise NotImplementedError
            return get_c4_new(nsamples, seed, seqlen, model, train_test)
        return get_c4(nsamples, seed, seqlen, model, train_test)
