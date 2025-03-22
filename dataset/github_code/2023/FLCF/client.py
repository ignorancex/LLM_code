"""
    @file: device.py
    @desc: federated learning
"""
import numpy as np
import torch
from util import *
from datasets import NCFUserData
from FakeRoast.FedOrchestrator import FedOrchestrator
import pdb

class Client:
    def __init__(self, user_id, compression, device,
                full_train_dataset, full_test_dataset, args):
        super().__init__()
        self.compression = compression
        if device == -1:
            dev = 'cpu'
        else:  
            dev = "cuda:" + str(device)
        self.pdevice = torch.device(dev)
        self.model = None
        self.user_id = user_id
        self.train_dataset = NCFUserData(full_train_dataset, user_id)
        self.test_dataset = NCFUserData(full_test_dataset, user_id)
        self.args = args

    def get_id(self):
        return self.user_id

    def download(self, raw_model, server_params):
        ''' server to client '''
        self.model = raw_model
        FedOrchestrator.set_wts_roast(self.model, server_params, False, 0)


    def train(self):
        self.model = self.model.to(self.pdevice)
        train(self.model, self.train_dataset, self.args.E, self.args.batch_size, self.args, self.pdevice)
        self.model.cpu()

    def eval(self, total_items, full):
        evaluate(self.model, self.train_loader, self.test_loader,
                 total_items, self.pdevice, user_batch=10, full=full)

