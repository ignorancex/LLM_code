import datasets
from torch.utils.data import Dataset, DataLoader
from transformers import DataCollatorWithPadding
import torch
import os
import joblib
from torch.utils.data import ConcatDataset
from logging import Logger
import logging
logger = logging.getLogger(__name__)

pin_memory = True
split_length = {
    1: (0, 2000),
    2: (2000, 4000),
    3: (4000, 6000),
    4: (6000, 8000),
    5: (8000, 10000),
}

def prepare_dataloaders_split(args, tokenizer, doc_type="new", train=True, split=0):
    global split_length
    docs_list = joblib.load(
        os.path.join(args.base_data_dir, f"{doc_type}_docs", "doc_list.pkl")
    )
    train_docs_list = docs_list[split_length[split][0] : split_length[split][1]]
    val_doc_list = train_docs_list

    print(f"Docs list length: {len(train_docs_list)}")
    doc_class = joblib.load(
        os.path.join(args.base_data_dir, f"{doc_type}_docs", "doc_class.pkl")
    )

    train_length = 0
    train_dataloader = None

    # Train
    if train:
        train_data = load_dataset_helper(
            os.path.join(args.base_data_dir, f"{doc_type}_docs", "trainqueries.json")
        )
        # st()
        train_data = train_data.filter(
            lambda example: example["doc_id"] in train_docs_list
        )
        generated_queries = load_dataset_helper(
            os.path.join(args.base_data_dir, f"{doc_type}_docs", "passages_seen.json")
        )
        generated_queries = generated_queries.filter(
            lambda example: example["doc_id"] in train_docs_list
        )
        train_length = len(train_data) + len(generated_queries)

        natural_queries = get_dataset(
            tokenizer=tokenizer, datadict=train_data, doc_class=doc_class
        )
        gen_queries = get_dataset(
            tokenizer=tokenizer, datadict=generated_queries, doc_class=doc_class
        )
        train_dataset = ConcatDataset([natural_queries, gen_queries])

        train_dataloader = get_dataloader(
            train_dataset,
            args.batch_size,
            tokenizer,
            shuffle=True,
            num_workers=args.num_workers,
        )
        logger.info(
            f"Loaded {args.base_data_dir.split('/')[-1]}_{doc_type}_(train+passages_seen)"
        )

    # Val
    _, val_length, val_dataloader = prepare_data_5task(
        args.base_data_dir,
        doc_type,
        "val",
        tokenizer,
        args.batch_size,
        val_doc_list,
        doc_class,
        logger,
        num_workers=args.num_workers,
    )

    # Test
    _, test_length, test_dataloader = prepare_data_5task(
        args.base_data_dir,
        doc_type,
        "test",
        tokenizer,
        args.batch_size,
        val_doc_list,
        doc_class,
        logger,
        num_workers=args.num_workers,
    )

    logger.info(f"{doc_type} train dataset size: {train_length}")
    logger.info(f"{doc_type} val dataset size: {val_length}")
    logger.info(f"{doc_type} test dataset size: {test_length}")

    class_num = len(train_docs_list)
    return train_dataloader, val_dataloader, test_dataloader, class_num


def prepare_dataloaders(args, tokenizer, doc_type="old", train=True):
    docs_list = joblib.load(
        os.path.join(args.base_data_dir, f"{doc_type}_docs", "doc_list.pkl")
    )
    if doc_type == "new" and "msmarco" in args.base_data_dir:
        docs_list = docs_list[:10000]
    doc_class = joblib.load(
        os.path.join(args.base_data_dir, f"{doc_type}_docs", "doc_class.pkl")
    )

    train_length = 0
    train_dataloader = None

    # Train
    if train:
        train_data = load_dataset_helper(
            os.path.join(args.base_data_dir, f"{doc_type}_docs", "trainqueries.json")
        )
        generated_queries = load_dataset_helper(
            os.path.join(args.base_data_dir, f"{doc_type}_docs", "passages_seen.json")
        )
        if doc_type == "new" and "msmarco" in args.base_data_dir:
            train_data = train_data.filter(
                lambda example: example["doc_id"] in docs_list
            )
            generated_queries = generated_queries.filter(
                lambda example: example["doc_id"] in docs_list
            )
        train_length = len(train_data) + len(generated_queries)

        natural_queries = get_dataset(
            tokenizer=tokenizer,
            datadict=train_data,
            doc_class=doc_class,
        )
        gen_queries = get_dataset(
            tokenizer=tokenizer,
            datadict=generated_queries,
            doc_class=doc_class,
        )
        train_dataset = ConcatDataset([natural_queries, gen_queries])

        train_dataloader = get_dataloader(
            train_dataset,
            args.batch_size,
            tokenizer,
            shuffle=True,
            num_workers=args.num_workers,
        )
        logger.info(
            f"Loaded {args.base_data_dir.split('/')[-1]}_{doc_type}_(train+passages_seen)"
        )

    # Val
    _, val_length, val_dataloader = prepare_data(
        args.base_data_dir,
        doc_type,
        "val",
        tokenizer,
        args.batch_size,
        docs_list,
        doc_class,
        logger,
        num_workers=args.num_workers,
    )

    # Test
    _, test_length, test_dataloader = prepare_data(
        args.base_data_dir,
        doc_type,
        "test",
        tokenizer,
        args.batch_size,
        docs_list,
        doc_class,
        logger,
        num_workers=args.num_workers,
    )

    logger.info(f"{doc_type} train dataset size: {train_length}")
    logger.info(f"{doc_type} val dataset size: {val_length}")
    logger.info(f"{doc_type} test dataset size: {test_length}")

    class_num = len(docs_list)
    return train_dataloader, val_dataloader, test_dataloader, class_num


def prepare_data_5task(
    base_data_dir,
    doc_type,
    split,
    tokenizer,
    batch_size,
    doc_list=None,
    doc_class=None,
    logger=None,
    num_workers=0,
):
    if doc_list is None:
        doc_list = joblib.load(
            os.path.join(base_data_dir, f"{doc_type}_docs", "doc_list.pkl")
        )

    if doc_class is None:
        doc_class = joblib.load(
            os.path.join(base_data_dir, f"{doc_type}_docs", "doc_class.pkl")
        )

    data = load_dataset_helper(
        os.path.join(base_data_dir, f"{doc_type}_docs", f"{split}queries.json")
    )
    data = data.filter(lambda example: example["doc_id"] in doc_list)

    class_num = len(doc_list)
    length = len(data)

    dataset = get_dataset(tokenizer=tokenizer, datadict=data, doc_class=doc_class)
    dataloader = get_dataloader(dataset, batch_size, tokenizer, num_workers=num_workers)

    if logger is None:
        print(f"Loaded {base_data_dir.split('/')[-1]}_{doc_type}_{split}")
    elif isinstance(logger, Logger):
        logger.info(f"Loaded {base_data_dir.split('/')[-1]}_{doc_type}_{split}")

    return class_num, length, dataloader


def prepare_data(
    base_data_dir,
    doc_type,
    split,
    tokenizer,
    batch_size,
    doc_list=None,
    doc_class=None,
    logger=None,
    num_workers=0,
    args=None,
):
    if doc_list is None:
        doc_list = joblib.load(
            os.path.join(base_data_dir, f"{doc_type}_docs", "doc_list.pkl")
        )
        if doc_type == "new" and "msmarco" in base_data_dir:
            doc_list = doc_list[:10000]
        elif doc_type == "new" and args.filter and args.filter_num > 0:
            doc_list = doc_list[:args.filter_num]

    if doc_class is None:
        doc_class = joblib.load(
            os.path.join(base_data_dir, f"{doc_type}_docs", "doc_class.pkl")
        )

    data = load_dataset_helper(
        os.path.join(base_data_dir, f"{doc_type}_docs", f"{split}queries.json")
    )
    if (doc_type == "new" and "msmarco" in base_data_dir) or (doc_type == "new" and args.filter and args.filter_num > 0):
        data = data.filter(lambda example: example["doc_id"] in doc_list)
    # if doc_type == "old" and dev:
    #     data = data.filter(lambda example: example["doc_id"] in doc_list)

    class_num = len(doc_list)
    length = len(data)

    # if doc_type == "new" and "nq320k" in base_data_dir and dev:
    #     dataset = get_dataset(
    #         tokenizer=tokenizer, datadict=data, doc_class=doc_class, dev=dev
    #     )
    # else:
    dataset = get_dataset(tokenizer=tokenizer, datadict=data, doc_class=doc_class)
    dataloader = get_dataloader(dataset, batch_size, tokenizer, num_workers=num_workers)

    if logger is None:
        print(f"Loaded {base_data_dir.split('/')[-1]}_{doc_type}_{split}")
    elif isinstance(logger, Logger):
        logger.info(f"Loaded {base_data_dir.split('/')[-1]}_{doc_type}_{split}")

    return class_num, length, dataloader


class IndexingCollator_old(DataCollatorWithPadding):
    def __call__(self, features):
        input_ids = [{"input_ids": x[0]} for x in features]
        docids = [x[1] for x in features]
        inputs = super().__call__(input_ids)

        inputs["labels"] = torch.Tensor(docids).long()
        return inputs


class IndexingCollator(DataCollatorWithPadding):
    def __call__(self, features):
        input_ids = [x[0] for x in features]
        batch = self.tokenizer(
            input_ids,
            padding="longest",
            truncation="only_first",
            max_length=32,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]

        docids = [x[1] for x in features]
        batch["labels"] = torch.Tensor(docids).long()
        batch["texts"] = input_ids
        return batch


def load_dataset_helper(path):
    data = datasets.load_dataset(
        "json", data_files=path, ignore_verifications=False, cache_dir="cache"
    )["train"]

    return data


class get_dataset(Dataset):
    def __init__(self, tokenizer, datadict, doc_class, dev=False):
        super().__init__()
        self.train_data = datadict

        self.tokenizer = tokenizer
        self.total_len = len(self.train_data)
        self.doc_class = doc_class
        # self.dev = dev

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        data = self.train_data[idx]

        text = [
            data[key]
            for key in ["question", "doc_text", "gen_question"]
            if key in data.keys()
        ]
        assert len(text) == 1, "More than one text field in data"

        # if self.dev:  # Rescaling for dev nq320k
        #     return text[0], self.doc_class[data["doc_id"]] - 98743 + 9874

        return text[0], self.doc_class[data["doc_id"]]


def get_dataloader_old(
    dataset, batch_size, tokenizer, padding="longest", shuffle=False, drop_last=False
):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=IndexingCollator(tokenizer, padding=padding),
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=4,
        pin_memory=pin_memory,
    )


def get_dataloader(
    dataset,
    batch_size,
    tokenizer,
    padding="longest",
    shuffle=False,
    drop_last=False,
    num_workers=0,
):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=IndexingCollator(tokenizer),
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


# ------------------- Using fasttext original model ----------------------
class IndexingCollator_fasttext(DataCollatorWithPadding):
    def __call__(self, features):
        input_ids = [x[0] for x in features]
        batch = self.tokenizer(
            input_ids,
            padding="longest",
            truncation="only_first",
            max_length=32,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]

        docids = [x[1] for x in features]
        batch["labels"] = torch.Tensor(docids).long()
        batch["texts"] = input_ids
        return batch


class get_dataset_fasttext(Dataset):
    def __init__(self, tokenizer, datadict, doc_class):
        super().__init__()
        self.train_data = datadict

        self.tokenizer = tokenizer
        self.total_len = len(self.train_data)
        self.doc_class = doc_class

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        data = self.train_data[idx]

        text = [
            data[key]
            for key in ["question", "doc_text", "gen_question"]
            if key in data.keys()
        ]
        assert len(text) == 1, "More than one text field in data"

        return text[0], self.doc_class[data["doc_id"]]


def prepare_data_fasttext(
    base_data_dir,
    doc_type,
    split,
    tokenizer,
    batch_size,
    doc_list=None,
    doc_class=None,
    logger=None,
    num_workers=0,
):
    if doc_list is None:
        doc_list = joblib.load(
            os.path.join(base_data_dir, f"{doc_type}_docs", "doc_list.pkl")
        )

    if doc_class is None:
        doc_class = joblib.load(
            os.path.join(base_data_dir, f"{doc_type}_docs", "doc_class.pkl")
        )

    data = load_dataset_helper(
        os.path.join(base_data_dir, f"{doc_type}_docs", f"{split}queries.json")
    )

    class_num = len(doc_list)
    length = len(data)

    dataset = get_dataset_fasttext(
        tokenizer=tokenizer, datadict=data, doc_class=doc_class
    )
    dataloader = get_dataloader_fasttext(
        dataset, batch_size, tokenizer, num_workers=num_workers
    )

    if logger is None:
        print(f"Loaded {base_data_dir.split('/')[-1]}_{doc_type}_{split}")
    elif isinstance(logger, Logger):
        logger.info(f"Loaded {base_data_dir.split('/')[-1]}_{doc_type}_{split}")

    return class_num, length, dataloader


def get_dataloader_fasttext(
    dataset,
    batch_size,
    tokenizer,
    padding="longest",
    shuffle=False,
    drop_last=False,
    num_workers=0,
):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=IndexingCollator_fasttext(tokenizer, padding=padding),
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
