import copy

import numpy as np
import torch
from Servers.meta_server import SeverMethod
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from utils.toolkit import tensor2numpy


class Server(SeverMethod):
    def __init__(self, args):
        super().__init__(args)
        self.batch_size = args.batch_size
        self.pin_memory = args.pin_memory
        self.num_workers = args.num_workers
        self.server_tuned_epoch = args.server_tuned_epoch
        self.server_init_lr = args.server_init_lr
        self.server_weight_decay = args.server_weight_decay

    def gen_protos_dataset(self, protos):
        labels = list(protos.keys())
        if self.args.wo_pool:
            labels = labels[-self.args.inc_class :]
        print(labels)
        proto_features, proto_labels = [], []
        for label in labels:
            features = protos[label]
            proto_features.append(features)
            proto_labels.append(np.full((features.shape[0],), label))
        proto_features = np.concatenate(proto_features)
        proto_labels = np.concatenate(proto_labels)
        tensor_proto_features = torch.tensor(proto_features, dtype=torch.float32)
        tensor_proto_labels = torch.tensor(proto_labels, dtype=torch.long)
        dataset = TensorDataset(tensor_proto_features, tensor_proto_labels)
        return dataset, list(labels)

    def debias(self, global_net, clients_list, local_classify_protos):
        dataset, labels = self.gen_protos_dataset(local_classify_protos)

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        params = list(global_net.fc.parameters())
        optimizer = optim.Adam(
            params, lr=self.server_init_lr, weight_decay=self.server_weight_decay
        )

        prog_bar = tqdm(range(self.server_tuned_epoch))
        for _, epoch in enumerate(prog_bar):
            global_net.train()

            losses = 0.0
            correct, total = 0, 0
            for i, (inputs, targets) in enumerate(dataloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                logits = global_net.fc(inputs)
                class_mask = [
                    col_index
                    for col_index in range(logits.shape[1])
                    if col_index not in labels
                ]
                logits[:, class_mask] = float("-inf")

                loss = F.cross_entropy(logits, targets.long())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            info = "Tasks {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                self.global_known_tasks,
                epoch + 1,
                self.server_tuned_epoch,
                losses / len(dataloader),
                train_acc,
            )
            prog_bar.set_description(info)
        print(info)

        fc_w = global_net.fc.state_dict()
        for client in clients_list:
            client.network.fc.load_state_dict(fc_w)

    def proto_aggregation(self, local_protos):
        global_protos = dict()
        local_classify_protos = dict()
        for key, local_proto in local_protos.items():
            for label in local_proto.keys():
                if label in local_classify_protos:
                    local_classify_protos[label].append(local_proto[label])
                else:
                    local_classify_protos[label] = [local_proto[label]]
        for label in local_classify_protos.keys():
            global_protos[label] = np.mean(local_classify_protos[label], axis=0)
            local_classify_protos[label] = np.vstack(local_classify_protos[label])
        return global_protos, local_classify_protos

    def sever_update(
        self,
        fed_aggregation,
        online_clients_list,
        clients_list,
        global_net,
        local_protos,
        w_local_list,
    ):

        self.global_known_tasks = np.unique(
            self.global_known_tasks
            + [clients_list[i].cur_task_global for i in online_clients_list]
        ).tolist()
        print("Global Tasks:", self.global_known_tasks)
        for client in clients_list:
            client.known_tasks = copy.deepcopy(self.global_known_tasks)

        global_protos, local_classify_protos = self.proto_aggregation(local_protos)

        freq = fed_aggregation.weight_calculate(
            clients_list=clients_list, online_clients_list=online_clients_list
        )

        fed_aggregation.agg_parts(
            freq=freq,
            w_local_list=w_local_list,
            global_net=global_net,
            except_part=self.args.freeze if hasattr(self.args, "freeze") else [],
        )

        if not self.args.wo_debias:
            self.debias(
                global_net=global_net,
                clients_list=clients_list,
                local_classify_protos=local_classify_protos,
            )

        w_global = copy.deepcopy(global_net.state_dict())

        return w_global, global_protos
