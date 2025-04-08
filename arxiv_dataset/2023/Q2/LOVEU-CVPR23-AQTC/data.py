import os
import torch
import random
import math
import copy
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule

class EncodedAssistQA(Dataset):
    def __init__(self, samples):
        super().__init__()
        self.samples = samples
        
    def __getitem__(self, index):
        sample = self.samples[index]
        video = torch.load(sample["video"], map_location="cpu")

        #timestamp_script = None
        sents_timestamp, script = None, None

        timestamp_para = torch.load(sample['para'], map_location="cpu")
        paras_timestamp, function_para = timestamp_para

        question = sample["question"]
        actions = sample["answers"] 
        meta = {
            'question': sample['src_question'], 'folder': sample['folder'], 
            'paras_score': sample['paras_score'], 'paras_timestamp': paras_timestamp, 
            'sents_score': sample['sents_score'], 'sents_timestamp': sents_timestamp
        }
        if 'correct' in sample:
            label = torch.tensor(sample['correct']) - 1 # NOTE here, start from 1
        else:
            label = None
        return video, script, question, function_para, actions, label, meta
        
    def __len__(self, ):
        return len(self.samples)

    @staticmethod
    def collate_fn(samples):
        return samples

class EncodedAssistQADataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        root = self.cfg.DATASET.TRAIN
        train_samples = []
        valid_samples = []
        for t in os.listdir(root):
            sample = torch.load(os.path.join(root, t, cfg.INPUT.QA), map_location="cpu")
            for s in sample:
                s["video"] = os.path.join(self.cfg.DATASET.VIDEO, t, cfg.INPUT.VIDEO)
                s["script"] = os.path.join(root, t, cfg.INPUT.SCRIPT)
                s["para"] = os.path.join(root, t, cfg.INPUT.PARA)
            random.shuffle(sample)
            if len(sample)>4:
                split_num = int(len(sample)*self.cfg.DATASET.SPLIT_RATIO)
                train_samples.extend(sample[:split_num])
                valid_samples.extend(sample[split_num:])
            else:
                train_samples.extend(sample)
        self.train_samples = train_samples
        self.valid_samples = valid_samples

        pseudo_samples = copy.deepcopy(self.valid_samples)
        for t in os.listdir(root):
            sample = torch.load(os.path.join(root, t, cfg.INPUT.QATEST), map_location="cpu")
            for s in sample:
                s["video"] = os.path.join(self.cfg.DATASET.VIDEO, t, cfg.INPUT.VIDEO)
                s["script"] = os.path.join(root, t, cfg.INPUT.SCRIPT)
                s["para"] = os.path.join(root, t, cfg.INPUT.PARA)
            pseudo_samples.extend(sample)
        self.pseudo_samples = pseudo_samples


    
    def train_dataloader(self): 
        cfg = self.cfg
        trainset = EncodedAssistQA(self.train_samples)
        pset = EncodedAssistQA(self.pseudo_samples)
        return [DataLoader(trainset, batch_size=cfg.SOLVER.BATCH_SIZE, collate_fn=EncodedAssistQA.collate_fn,
            shuffle=True, drop_last=True, num_workers=cfg.DATALOADER.NUM_WORKERS, pin_memory=True),
            DataLoader(pset, batch_size=cfg.SOLVER.BATCH_SIZE, collate_fn=EncodedAssistQA.collate_fn,
            shuffle=True, drop_last=True, num_workers=cfg.DATALOADER.NUM_WORKERS, pin_memory=True)]

    def val_dataloader(self):
        cfg = self.cfg
        valset = EncodedAssistQA(self.valid_samples)
        return DataLoader(valset, batch_size=cfg.SOLVER.BATCH_SIZE, collate_fn=EncodedAssistQA.collate_fn,
            shuffle=False, drop_last=False, num_workers=cfg.DATALOADER.NUM_WORKERS, pin_memory=True)

class EncodedAssistQATestDataModule(EncodedAssistQADataModule):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        root = self.cfg.DATASET.TRAIN
        train_samples = []
        valid_samples = []
        for t in os.listdir(root):
            sample = torch.load(os.path.join(root, t, cfg.INPUT.QA), map_location="cpu")
            for s in sample:
                s["video"] = os.path.join(self.cfg.DATASET.VIDEO, t, cfg.INPUT.VIDEO)
                s["script"] = os.path.join(root, t, cfg.INPUT.SCRIPT)
                s["para"] = os.path.join(root, t, cfg.INPUT.PARA)
            valid_samples.extend(sample)
        self.train_samples = train_samples
        self.valid_samples = valid_samples

    
    def train_dataloader(self): 
        cfg = self.cfg
        trainset = EncodedAssistQA(self.train_samples)
        return DataLoader(trainset, batch_size=cfg.SOLVER.BATCH_SIZE, collate_fn=EncodedAssistQA.collate_fn,
            shuffle=True, drop_last=True, num_workers=cfg.DATALOADER.NUM_WORKERS, pin_memory=True)

    def val_dataloader(self):
        cfg = self.cfg
        valset = EncodedAssistQA(self.valid_samples)
        return DataLoader(valset, batch_size=cfg.SOLVER.BATCH_SIZE, collate_fn=EncodedAssistQA.collate_fn,
            shuffle=False, drop_last=False, num_workers=cfg.DATALOADER.NUM_WORKERS, pin_memory=True)
    
def build_data(cfg):
    if cfg.DATASET.GT:
        return EncodedAssistQADataModule(cfg)
    else:
        return EncodedAssistQATestDataModule(cfg)