from torch.utils.data import Dataset as torchDataset
import torch
from gptq.datautils import get_c4, get_c4_new
class C4_Dataset(torchDataset):
    def __init__(
        self, seed, use_train_split, tokenizer_name, verbose=False, \
        nsamples=128, seqlen=2048, device="cuda"
    ):
        self.seqlen = seqlen
        self.use_train_split = use_train_split
        train_loader, testenc = get_c4(nsamples=nsamples, seed=seed, seqlen=seqlen, model=tokenizer_name)
        if use_train_split:
            self.tokenized_data = train_loader  # This is a list of input id tensors
        else:
            self.tokenized_data = testenc  # This is one dict of input ids, atten, and position ids

    def __len__(self):
        if not self.use_train_split:
            raise Exception("len is not implemented for test split, extract testenc via .testenc")
        return len(self.tokenized_data)
    
    def shuffle(self, seed=42):
        """Empty function, as the dataset is already shuffled upon initialization via serial_number"""
        pass

    def truncate_to_seqlen_n_samples(self, seqlen, n_samples):
        """Empty function, as the dataset is already truncated upon initialization via seqlen and nsamples"""
        pass
    
    def __getitem__(self, idx):
        if not self.use_train_split:
            raise Exception("getitem not implemented for test split, extract testenc via .testenc")
        out = {"input_ids": self.tokenized_data[idx][0].squeeze(0)} # input ids
        return out
        
    
class C4_New_Dataset(torchDataset):
    def __init__(
        self, seed, use_train_split, tokenizer_name, verbose=False, \
        nsamples=128, seqlen=2048, device="cuda"
    ):
        self.seqlen = seqlen
        self.use_train_split = use_train_split
        train_loader, testenc = get_c4_new(nsamples=nsamples, seed=seed, seqlen=seqlen, model=tokenizer_name)
        if use_train_split:
            self.tokenized_data = train_loader  # This is a list of input id tensors
        else:
            self.tokenized_data = testenc  # This is one dict of input ids, atten, and position ids

    def __len__(self):
        if not self.use_train_split:
            raise Exception("len is not implemented for test split, extract testenc via .testenc")
        return len(self.tokenized_data)
    
    def shuffle(self, seed=42):
        """Empty function, as the dataset is already shuffled upon initialization via serial_number"""
        pass

    def truncate_to_seqlen_n_samples(self, seqlen, n_samples):
        """Empty function, as the dataset is already truncated upon initialization via seqlen and nsamples"""
        pass
    
    def __getitem__(self, idx):
        if not self.use_train_split:
            raise Exception("getitem not implemented for test split, extract testenc via .testenc")
        out = {"input_ids": self.tokenized_data[idx][0].squeeze(0)} # input ids
        return out