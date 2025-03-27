import copy
import math
import random
import sys

import numpy as np
import torch
from Backbones.inc_net import FedPromptVitNet
from Datasets.data_manager import DataManager
from Models.meta_model import BaseLearner
from torch import nn, optim
from torch.nn import functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.toolkit import tensor2numpy


class Learner(BaseLearner):
    def __init__(self, network_global, client_id, args):
        super().__init__(client_id, args)
        self.lamda = args.lamda
        self.infoNCET = args.infoNCET
        if network_global is None:
            self.network = FedPromptVitNet(args)
        else:
            self.network = network_global

        if self.client_id != -1:
            print(f"Client {self.client_id} loaded model")
        else:
            promptfc_params = sum(
                p.numel() for p in self.network.prompt_fc.parameters()
            )
            prompt_params = sum(p.numel() for p in self.network.prompt.parameters())
            fc_params = sum(p.numel() for p in self.network.fc.parameters())
            print(promptfc_params, prompt_params, fc_params)

    def train(self, w_global, train_loader, global_protos):
        print(f"Learning on: {self.cur_classes} (Global task {self.cur_task_global})")
        self.network.load_state_dict(w_global)
        optimizer = self.get_optimizer()
        scheduler = self.get_scheduler(optimizer)

        if self.cur_task > 0:
            if self.cur_task_global not in self.known_tasks:
                print("Prompt transferred from previous task")
                self.init_prompt(optimizer)
            elif self.new_task:
                print("Prompt transferred from other clients")

        if self.cur_task > 0 and self.args.reinit_optimizer:
            optimizer = self.get_optimizer()

        w_local = self.init_train(train_loader, optimizer, scheduler, global_protos)

        agg_protos = self.extract_protos(train_loader)

        return w_local, agg_protos

    def get_optimizer(self):
        params = (
            list(self.network.prompt.parameters())
            + list(self.network.fc.parameters())
            + list(self.network.prompt_fc.parameters())
        )
        if self.args.optimizer == "adam":
            optimizer = optim.Adam(
                params, lr=self.init_lr, weight_decay=self.weight_decay
            )
        return optimizer

    def get_scheduler(self, optimizer):
        if self.args.scheduler == "constant":
            scheduler = None

        return scheduler

    def init_prompt(self, optimizer):
        model = self.network
        # Transfer previous learned prompt params to the new prompt
        prev_start = self.pre_task_global
        prev_end = self.pre_task_global + 1

        cur_start = self.cur_task_global
        cur_end = self.cur_task_global + 1

        cur_idx = slice(cur_start, cur_end)
        prev_idx = slice(prev_start, prev_end)

        with torch.no_grad():
            if model.prompt.prompt.grad is not None:
                model.prompt.prompt.grad.zero_()
            model.prompt.prompt[cur_idx] = model.prompt.prompt[prev_idx]
            optimizer.param_groups[0]["params"] = model.parameters()

    def init_train(self, train_loader, optimizer, scheduler, global_protos):
        if len(global_protos) != 0:
            all_global_protos_keys = np.array(list(global_protos.keys()))
            all_protos = []
            for protos_key in all_global_protos_keys:
                all_protos.append(global_protos[protos_key])
            all_protos = np.vstack(all_protos)

        prog_bar = tqdm(range(self.tuned_epoch))
        for _, epoch in enumerate(prog_bar):
            self.network.train()

            correct, total = 0, 0
            losses_CE, losses_InfoNCE = 0.0, 0.0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                output = self.network(inputs, self.cur_task_global, train=True)
                logits = output["logits"]
                features = output["features"]
                class_mask = [
                    col_index
                    for col_index in range(logits.shape[1])
                    if col_index not in self.cur_classes
                ]
                logits[:, class_mask] = float("-inf")

                loss_CE = F.cross_entropy(logits, targets.long())

                if len(global_protos) == 0 or self.args.wo_unified:
                    loss_InfoNCE = 0 * loss_CE
                else:
                    count = 0
                    loss_InfoNCE = None
                    for i, label in enumerate(targets):
                        if (label.item() in global_protos.keys()) and (
                            label.item() in self.cur_classes
                        ):
                            count += 1
                            feature = features[i].unsqueeze(0)
                            loss_instance = self.calculate_infonce(
                                feature,
                                label.item(),
                                all_protos,
                                all_global_protos_keys,
                            )

                            if loss_InfoNCE is None:
                                loss_InfoNCE = loss_instance
                            else:
                                loss_InfoNCE += loss_instance
                    if count != 0:
                        loss_InfoNCE = loss_InfoNCE / count
                    else:
                        loss_InfoNCE = 0 * loss_CE
                loss_InfoNCE = loss_InfoNCE

                loss = loss_CE + self.lamda * loss_InfoNCE

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses_CE += loss_CE.item()
                losses_InfoNCE += loss_InfoNCE.item() * self.lamda
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            if scheduler:
                scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            info = "Task {}, Epoch {}/{} => CE Loss {:.3f}, InfoNCE Loss {:.3f}, Train_accy {:.2f}".format(
                self.cur_task,
                epoch + 1,
                self.tuned_epoch,
                losses_CE / len(train_loader),
                losses_InfoNCE / len(train_loader),
                train_acc,
            )
            prog_bar.set_description(info)
        print(info)

        net_para = self.network.state_dict()
        w_local = {
            k: copy.deepcopy(v)
            for k, v in net_para.items()
            if not any(except_key in k for except_key in self.except_part)
        }
        return w_local

    def extract_protos(self, train_loader):
        target_list = []
        feature_list = []
        for _, inputs, targets in train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            with torch.no_grad():
                output = self.network(inputs)
            features = output["features"]
            target_list.append(targets)
            feature_list.append(features)
        target_list = torch.cat(target_list, dim=0)
        feature_list = torch.cat(feature_list, dim=0)

        agg_protos = {}
        for class_index in self.cur_classes:
            data_index = (target_list == class_index).nonzero().squeeze(-1)
            if data_index.shape[0] != 0:
                all_features = feature_list[data_index]
                proto = all_features.mean(0).cpu().numpy()
                agg_protos[class_index] = proto

        return agg_protos

    def calculate_infonce(self, feature, label, all_protos, all_global_protos_keys):
        pos_index = np.where(all_global_protos_keys == label)[0]
        neg_index = np.where(
            (all_global_protos_keys != label)
            & np.isin(all_global_protos_keys, self.cur_classes)
        )[0]
        f_pos = torch.from_numpy(all_protos[pos_index]).to(self.device)
        f_neg = torch.from_numpy(all_protos[neg_index]).to(self.device)
        f_proto = torch.cat((f_pos, f_neg), dim=0)
        l = torch.cosine_similarity(feature, f_proto, dim=1)
        l = l / self.infoNCET

        exp_l = torch.exp(l)
        exp_l = exp_l.view(1, -1)
        pos_mask = [1 for _ in range(f_pos.shape[0])] + [
            0 for _ in range(f_neg.shape[0])
        ]
        pos_mask = torch.tensor(pos_mask, dtype=torch.float).to(self.device)
        pos_mask = pos_mask.view(1, -1)
        pos_l = exp_l * pos_mask
        sum_pos_l = pos_l.sum(1)
        sum_exp_l = exp_l.sum(1)
        infonce_loss = -torch.log(sum_pos_l / sum_exp_l)

        return infonce_loss
