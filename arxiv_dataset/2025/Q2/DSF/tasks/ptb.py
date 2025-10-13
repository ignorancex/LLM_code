from torch.utils.data import Dataset, DataLoader
import torch

from ptb import data

class PTBDataset(Dataset):
    def __init__(self, corpus, split="train"):
        if split == "train":
            self.data = corpus.train
        elif split == "valid":
            self.data = corpus.valid
        elif split == "test":
            self.data = corpus.test
        else:
            raise ValueError(f"Invalid split: {split}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sentence = self.data[idx]
        return sentence.long()


def get_ptb_dataloaders(config):
    if config.dataset.tokenizer == 'word':
        corpus = data.Corpus("./ptb/", "ptb.train.txt", "ptb.valid.txt", "ptb.test.txt")
    elif config.dataset.tokenizer == 'char':
        corpus = data.Corpus("./ptb/", "ptb.char.train.txt", "ptb.char.valid.txt", "ptb.char.test.txt")
    else:
        raise ValueError(f"Invalid tokenizer: {config.dataset.tokenizer}")
    
    
    PAD_IDX = corpus.dictionary.word2idx["<pad>"]
    SEQ_LEN = config.dataset.context_length
    
    def collate_fn(batch):
        """Pads sequences in batch to the same length"""
        batch = [seq.long()[:SEQ_LEN+1] for seq in batch]
        batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=PAD_IDX)
        return batch[:, :-1], batch[:, 1:]  # X is input, Y is target (next word)
    
    train_loader = DataLoader(PTBDataset(corpus, split="train"), batch_size=config.training.train_batch_size, num_workers=config.dataset.dataloader_workers, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(PTBDataset(corpus, split="valid"), batch_size=config.training.eval_batch_size, num_workers=config.dataset.dataloader_workers, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(PTBDataset(corpus, split="test"), batch_size=config.training.eval_batch_size, num_workers=config.dataset.dataloader_workers, shuffle=False, collate_fn=collate_fn)

    info = {
        'task_type': 'language_modeling',
        "vocab_size": len(corpus.dictionary),
        "pad_token_id": PAD_IDX,
    }

    return train_loader, valid_loader, test_loader, info