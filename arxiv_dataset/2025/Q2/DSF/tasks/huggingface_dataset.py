import numpy as np
from torch.utils.data import DataLoader
from datasets import load_dataset as load_dataset_hf
from transformers import AutoTokenizer


def load_huggingface_dataset(config):
    dataset = load_dataset_hf(config.dataset.name, config.dataset.version, cache_dir=config.dataset.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(config.dataset.tokenizer)

    if tokenizer.pad_token_id is None or not (0 <= tokenizer.pad_token_id < tokenizer.vocab_size):
        tokenizer.pad_token_id = tokenizer.eos_token_id

    num_proc = config.dataset.preprocessing_workers
    text_column = config.dataset.text_column
    
    if config.dataset.max_train_samples is not None:
        # use only a subset of the dataset
        dataset['train'] = dataset['train'].select(range(config.dataset.max_train_samples))

    # split into train/test/validation if missing
    if 'validation' not in dataset:
        print("No validation set, making one...")
        split_dataset = dataset['train'].train_test_split(test_size=config.dataset.generated_val_size, seed=2357, shuffle=True)
        dataset['validation'] = split_dataset['test']
        dataset['train'] = split_dataset['train']

    # tokenize
    def tokenize_function(examples):
        result = tokenizer([x for x in examples[text_column]])
        return result
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True, num_proc=num_proc, remove_columns=dataset['train'].column_names, desc="Tokenizing")

    # contatenate all texts
    ctx_len = config.dataset.context_length
    def group_texts(examples):
        result = dict()
        for k in examples.keys():
            concatenated = np.concatenate(examples[k])
            total_length = len(concatenated)
            total_length = (total_length // ctx_len) * ctx_len  # we drop the last incomplete sequence
            result[k] = [concatenated[i: i + ctx_len] for i in range(0, total_length, ctx_len)]

        return result

    tokenized_dataset = tokenized_dataset.map(group_texts, batched=True, num_proc=num_proc, desc="Grouping texts")

    tokenized_dataset.set_format(type='torch', columns=['input_ids'])

    train_loader = DataLoader(tokenized_dataset['train'], batch_size=config.training.train_batch_size, shuffle=True, num_workers=config.dataset.dataloader_workers, drop_last=True)
    validation_loader = DataLoader(tokenized_dataset['validation'], batch_size=config.training.eval_batch_size, shuffle=False, num_workers=config.dataset.dataloader_workers)
    info = {
        'task_type': 'language_modeling',
        'vocab_size': tokenizer.vocab_size,
        'pad_token_id': tokenizer.pad_token_id,
    }

    return train_loader, validation_loader, info