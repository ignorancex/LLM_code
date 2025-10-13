import copy
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import wandb

from dataset import DatasetSplit
from utils import init_model
from utils import init_optimizer, model_parameter_vector
from models_dict.wrapper import MoonWrapper
from utils import trainable_params

class Node(object):

    def __init__(self, num_id, local_data, train_set, args):
        # new
        self.mask = None
        self.reverse_mask = None
        self.direc_mask = None
        self.gauss_mask = None

        ######
        self.num_id = num_id
        self.args = args
        self.node_num = self.args.node_num
        if num_id == -1:
            self.valid_ratio = args.server_valid_ratio
        else:
            self.valid_ratio = args.client_valid_ratio

        if self.args.dataset in ['cifar10', 'mnist', 'fmnist']:
            self.num_classes = 10
        elif self.args.dataset == 'cifar100':
            self.num_classes = 100
        elif self.args.dataset == 'tinyimagenet':
            self.num_classes = 200
        elif "glue" in self.args.dataset:
            self.num_classes = len(set(train_set.dst["labels"]))
            print("num_classes: ", self.num_classes)

        if "glue" in args.dataset:
            self.collator = train_set.get_collator()
        else:
            self.collator = None
        
        if args.iid == 1 or num_id == -1:
            # for the server, use the validate_set as the training data, and use local_data for testing
            self.local_data, self.validate_set = self.train_val_split_forServer(local_data.indices, train_set, self.valid_ratio, self.num_classes)
        else:
            self.local_data, self.validate_set = self.train_val_split(local_data, train_set, self.valid_ratio)
            self.sample_per_class = self.generate_sample_per_class(self.local_data)

            # zj balanced loss
            n_class = self.num_classes
            class2data = torch.zeros(n_class)
            ds = self.local_data.dataset
            all_targets = []
            for data, target in self.local_data:
                all_targets.append(target)
            all_targets = torch.cat(all_targets, 0)
            uniq_val, uniq_count = np.unique(all_targets, return_counts=True)
            for i, c in enumerate(uniq_val.tolist()):
                class2data[c] = uniq_count[i]
            self.class2data = class2data.unsqueeze(dim=0).cuda()
        
        print(f"num_id: {self.num_id}, data_size: {len(self.local_data.dataset)}, validate_data_size: {len(self.validate_set.dataset)}")

        self.model = init_model(self.args.local_model, self.args, self.num_classes).cuda()
        if num_id == -1:
            wandb.watch(self.model, log_freq=10, log="all")
        if args.client_method == 'moon':
            self.model: MoonWrapper = MoonWrapper(self.model)
            self.prev_model = copy.deepcopy(self.model)
        if args.sam:
            self.optimizer = init_optimizer(self.num_id, self.model, args, is_sam=True)
        else: 
            self.optimizer = init_optimizer(self.num_id, self.model, args)

        # p_head for fedrod and fedrep
        if args.client_method == 'fedrod':
            try:
                self.p_head = copy.deepcopy(self.model.linear_head)
            except:
                self.p_head = copy.deepcopy(self.model.classifier)

        # cluster_id assignment for swapping label
        if self.num_id != self.node_num:
            if self.args.num_cluster == 4:
                # 4 clusters existed
                if self.num_id  < self.node_num // 4:
                    self.cluster_id = 0
                elif self.num_id  < 2 * self.node_num // 4:
                    self.cluster_id = 1
                elif self.num_id  < 3 * self.node_num // 4:
                    self.cluster_id = 2
                else:
                    self.cluster_id = 3
            elif self.args.num_cluster == 2:
                # 2 clusters existed
                if self.num_id + 1 > self.node_num // 2:
                    self.cluster_id = 1
                else:
                    self.cluster_id = 0
            else:
                raise ValueError('The number of clusters is not well-defined...')
        else:
            self.cluster_id = 0
        
        # node init for feddyn
        if 'feddyn' in  args.client_method:
            self.old_grad = None
            self.old_grad = copy.deepcopy(self.model)
            self.old_grad = model_parameter_vector(args, self.old_grad)
            self.old_grad = torch.zeros_like(self.old_grad)
        if 'feddyn' in args.server_method:
            self.server_state = copy.deepcopy(self.model)
            for param in self.server_state.parameters():
                param.data = torch.zeros_like(param.data)
        
        # node init for scaffold
        if args.server_method == 'scaffold':
            self.c_global = [torch.zeros_like(param) for param in trainable_params(self.model)]
        if args.client_method == 'scaffold':
            self.c_local: Dict[List[torch.Tensor]] = {}
            self.c_global: List[torch.Tensor] = []
            self.y_delta = None
            self.c_delta = None

        if args.client_method == 'prog2_layer':
            self.unfreeze_key = set([name for name in self.model.state_dict().keys() if ('head' in name) or ('first' in name)])
            self.freeze_key = set([key for key in self.model.state_dict().keys()
                                       if key not in self.unfreeze_key])
        

        if args.client_method == 'prog2_layer_firstlayer':
            self.head_key = set([name for name in self.model.state_dict().keys() if 'first' in name])
            self.base_key = set([key for key in self.model.state_dict().keys()
                                       if key not in self.head_key])

    def train_val_split(self, idxs, train_set, valid_ratio): 
        print("idx: ", idxs)
        np.random.shuffle(idxs)
        if self.args.data_ratio is not None:
            idxs = idxs[:int(len(idxs) * self.args.data_ratio)]


        validate_size = valid_ratio * len(idxs)

        idxs_test = idxs[:int(validate_size)]
        idxs_train = idxs[int(validate_size):]

        train_loader = DataLoader(DatasetSplit(train_set, idxs_train),
                                  batch_size=self.args.batchsize, num_workers=0, shuffle=True, collate_fn=self.collator)

        test_loader = DataLoader(DatasetSplit(train_set, idxs_test),
                                 batch_size=self.args.validate_batchsize,  num_workers=0, shuffle=True, collate_fn=self.collator)
        

        return train_loader, test_loader

    def train_val_split_forServer(self, idxs, train_set, valid_ratio, num_classes=10): # local data index, trainset
        np.random.shuffle(idxs)
        if self.args.data_ratio is not None:
            idxs = idxs[:int(len(idxs) * self.args.data_ratio)]


        validate_size = valid_ratio * len(idxs)

        # generate proxy dataset with balanced classes
        idxs_test = []
        test_class_count = [int(validate_size)/num_classes for _ in range(num_classes)]
        k = 0
        while sum(test_class_count) > 0:
            if test_class_count[train_set[idxs[k]][1]] > 0:
                idxs_test.append(idxs[k])
                test_class_count[train_set[idxs[k]][1]] -= 1
            else: 
                pass
            k += 1
        label_list = []
        for k in idxs_test:
            label_list.append(train_set[k][1])

        idxs_train = [idx for idx in idxs if idx not in idxs_test]

        train_loader = DataLoader(DatasetSplit(train_set, idxs_train),
                                  batch_size=self.args.batchsize, num_workers=0, shuffle=True, collate_fn=self.collator)
        test_loader = DataLoader(DatasetSplit(train_set, idxs_test),
                                 batch_size=self.args.validate_batchsize,  num_workers=0, shuffle=True, collate_fn=self.collator)

        return train_loader, test_loader

    def generate_sample_per_class(self, local_data):
        sample_per_class = torch.tensor([0 for _ in range(self.num_classes)])

        for idx, (data, target) in enumerate(local_data):
            sample_per_class += torch.tensor([sum(target==i) for i in range(self.num_classes)])

        sample_per_class = torch.where(sample_per_class > 0, sample_per_class, 1)

        return sample_per_class