import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset, ConcatDataset
import clip
import os

from models import HugeModel
from losses import return_task_loss
from .client import SingleTaskClient
from copy import deepcopy

from .base import SimpleNetwork

def FedProxNetwork(SimpleNetwork):
    
    def __init__(self, sample_dataset, assignments, device, args, save_dir) -> None:
        
        super(self, FedProxNetwork).__init__(sample_dataset, assignments, device, args, save_dir)
        self.mu = args.mu

    def train(self):
        
        # Record global model initial performance on client datasets
        val_loss_m, val_loss_s = self.validate_global_model()
        self.mean_loss_history.append(val_loss_m)
        self.std_loss_history.append(val_loss_s)

        # Loop for the federated training
        for i in range(self.n_rounds):
            for client in self.clients:
                # Downlink
                client.sync_params(self.global_model)
                # Training
                client.train_once(self.global_model)
                # Uplink
                self.global_model = self.fedprox_aggregate(self.global_model, client.model)
            
            # Record performance of global model
            val_loss_m, val_loss_s = self.validate_global_model()
            self.mean_loss_history.append(val_loss_m)
            self.std_loss_history.append(val_loss_s)