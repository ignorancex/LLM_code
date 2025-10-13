import numpy as np
from sklearn.mixture import GaussianMixture

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast as autocast
from torch.nn.utils.rnn import pad_sequence
import copy


from .correctors import SelfieCorrector, JointOptimCorrector
# from .nets import get_model
from model_arch.build_model import build_model as get_model

import torchvision
from torchvision import transforms


def dataset_split_collate_fn(batch):
    if len(batch[0]) == 2:  # (data, label)
        data, labels = zip(*batch)
        if isinstance(data[0], torch.Tensor) and data[0].dim() == 1:  
            data_padded = pad_sequence(data, batch_first=True, padding_value=0)  
            labels = torch.tensor(labels, dtype=torch.long)
            return data_padded, labels
        else:  # images
            data = torch.stack(data)
            labels = torch.tensor(labels, dtype=torch.long)
            return data, labels

    elif len(batch[0]) == 3:  # (data, label, item)
        data, labels, items = zip(*batch)
        if isinstance(data[0], torch.Tensor) and data[0].dim() == 1:  
            data_padded = pad_sequence(data, batch_first=True, padding_value=0)
            labels = torch.tensor(labels, dtype=torch.long)
            items = torch.tensor(items, dtype=torch.long)
            return data_padded, labels, items
        else:  # images
            data = torch.stack(data)
            labels = torch.tensor(labels, dtype=torch.long)
            items = torch.tensor(items, dtype=torch.long)
            return data, labels, items

    elif len(batch[0]) == 4:  # (data, label, item, real_idx)
        data, labels, items, real_idxs = zip(*batch)
        if isinstance(data[0], torch.Tensor) and data[0].dim() == 1:  
            data_padded = pad_sequence(data, batch_first=True, padding_value=0)
            labels = torch.tensor(labels, dtype=torch.long)
            items = torch.tensor(items, dtype=torch.long)
            real_idxs = torch.tensor(real_idxs, dtype=torch.long)
            return data_padded, labels, items, real_idxs
        else:  # images
            data = torch.stack(data)
            labels = torch.tensor(labels, dtype=torch.long)
            items = torch.tensor(items, dtype=torch.long)
            real_idxs = torch.tensor(real_idxs, dtype=torch.long)
            return data, labels, items, real_idxs

    else:
        raise ValueError(f"Unsupported batch format with {len(batch[0])} elements.")


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs, idx_return=False, real_idx_return=False):
        self.dataset = dataset
        self.idxs = list(idxs)
        self.idx_return = idx_return
        self.real_idx_return = real_idx_return

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        item = int(item)
        image, label = self.dataset[self.idxs[item]]

        if self.idx_return:
            return image, label, item
        elif self.real_idx_return:
            return image, label, item, self.idxs[item]
        else:
            return image, label


class PairProbDataset(Dataset):
    def __init__(self, dataset, idxs, prob, idx_return=False):
        self.dataset = dataset
        self.idxs = list(idxs)
        self.idx_return = idx_return
        self.prob = prob

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        item = int(item)
        image1, label = self.dataset[self.idxs[item]]
        image2, label = self.dataset[self.idxs[item]]
        prob = self.prob[self.idxs[item]]

        if self.idx_return:
            return image1, image2, label, prob, item
        else:
            return image1, image2, label, prob


class PairDataset(Dataset):
    def __init__(self, dataset, idxs, idx_return=False, label_return=False):
        self.dataset = dataset
        self.idxs = list(idxs)
        self.idx_return = idx_return
        self.label_return = label_return

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        item = int(item)
        image1, label = self.dataset[self.idxs[item]]
        image2, label = self.dataset[self.idxs[item]]
        sample = (image1, image2,)

        if self.label_return:
            sample += (label,)

        if self.idx_return:
            sample += (item,)

        return sample


class DatasetSplitRFL(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        item = int(item)
        image, label = self.dataset[self.idxs[item]]

        return image, label, self.idxs[item]


def mixup(inputs, targets, alpha=1.0):
    l = np.random.beta(alpha, alpha)
    l = max(l, 1 - l)

    idx = torch.randperm(inputs.size(0))

    input_a, input_b = inputs, inputs[idx]
    target_a, target_b = targets, targets[idx]

    mixed_input = l * input_a + (1 - l) * input_b
    mixed_target = l * target_a + (1 - l) * target_b

    return mixed_input, mixed_target


def linear_rampup(current, warm_up, lambda_u, rampup_length=16):
    current = np.clip((current - warm_up) / rampup_length, 0.0, 1.0)
    return lambda_u * float(current)


class SemiLoss:
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, lambda_u, epoch, warm_up):
        probs_u = torch.softmax(outputs_u, dim=1)

        # labeled data loss
        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x,
                         dim=1) * targets_x, dim=1))
        # unlabeled data loss
        Lu = torch.mean((probs_u - targets_u) ** 2)

        lamb = linear_rampup(epoch, warm_up, lambda_u)

        return Lx + lamb * Lu


def get_local_update_objects(args, dataset_train, dict_users=None, noise_rates=None, gaussian_noise=None, glob_centroid=None):
    local_update_objects = []
    for idx, noise_rate in zip(range(args.num_users), noise_rates):
        local_update_args = dict(
            args=args,
            user_idx=idx,
            dataset=dataset_train,
            idxs=dict_users[idx],
        )

        # TODO: original federated learning methods
        if args.method == 'fedavg' or args.method == 'median' or args.method == 'krum' or args.method == 'trimmedMean' or args.method == 'fedexp' or args.method =='RFA':
            local_update_object = BaseLocalUpdate(**local_update_args)


        elif args.method == 'maskedOptim':
            local_update_object = LocalUpdateMaskedOptim(**local_update_args)

        else:
            raise NotImplementedError

        local_update_objects.append(local_update_object)

    return local_update_objects



class FedDecorrLoss(nn.Module):

    def __init__(self):
        super(FedDecorrLoss, self).__init__()
        self.eps = 1e-8
        

    def _off_diagonal(self, mat):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = mat.shape
        assert n == m
        return mat.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, x):
        N, C = x.shape
        if N == 1:
            return 0.0

        x = x - x.mean(dim=0, keepdim=True)
        x = x / torch.sqrt(self.eps + x.var(dim=0, keepdim=True))

        corr_mat = torch.matmul(x.t(), x)

        loss = (self._off_diagonal(corr_mat).pow(2)).mean()
        loss = loss / N

        return loss


class SCELoss(torch.nn.Module):
    def __init__(self, alpha, beta, num_classes=10):
        super(SCELoss, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, pred, labels):
        # CCE
        ce = self.cross_entropy(pred, labels)

        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(
            labels, self.num_classes).float().to(self.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))

        # Loss
        loss = self.alpha * ce + self.beta * rce.mean()
        return loss



class BaseLocalUpdate:
    def __init__(
            self,
            args,
            user_idx=None,
            dataset=None,
            idxs=None,
            idx_return=False,
            real_idx_return=False,
    ):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()

        self.dataset = dataset
        self.idxs = idxs
        self.user_idx = user_idx
        self.update_name = args.method

        self.idx_return = idx_return
        self.real_idx_return = real_idx_return
        
        #TODO: Load custom collate_fn
        collate_fn = dataset_split_collate_fn if args.collate_fn else None

        self.ldr_train = DataLoader(
            DatasetSplit(dataset, idxs, idx_return=idx_return,
                         real_idx_return=real_idx_return),
            batch_size=self.args.local_bs,
            shuffle=True,
            num_workers=self.args.num_workers,
            collate_fn= collate_fn if args.collate_fn else None,
            pin_memory=True,
            drop_last=True,
        )

        self.total_epochs = 0
        self.epoch = 0
        self.batch_idx = 0

        self.net1 = get_model(self.args)
        self.net2 = get_model(self.args)


        self.last_updated = 0


    def train(self, net, net2=None):
        if net2 is None:
            return self.train_single_model(net)
        else:
            return self.train_multiple_models(net, net2)

    def train_single_model(self, net):

        # net.to(self.args.device)
        net.train()

        optimizer = torch.optim.SGD(
            net.parameters(),
            lr=self.args.lr,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay,
        )

        epoch_loss = []
        for epoch in range(self.args.local_ep):
            self.epoch = epoch
            batch_loss = []
            for batch_idx, batch in enumerate(self.ldr_train):
                self.batch_idx = batch_idx

                if(len(batch) == 0):
                    continue

                net.zero_grad()

                # with autocast():
                loss = self.forward_pass(batch, net)
                

                loss.backward()
                optimizer.step()


                batch_loss.append(loss.item())
                self.on_batch_end()

            if(len(batch_loss) > 0):
                epoch_loss.append(sum(batch_loss) / len(batch_loss))

            self.total_epochs += 1
            self.on_epoch_end()

        self.net1.load_state_dict(net.state_dict())
        self.last_updated = self.args.g_epoch

        net.to('cpu')
        # del net

        return self.net1.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def train_multiple_models(self, net1, net2):

        # net1.to(self.args.device)
        # net2.to(self.args.device)

        net1.train()
        net2.train()

        optimizer_args = dict(
            lr=self.args.lr,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay,
        )
        optimizer1 = torch.optim.SGD(net1.parameters(), **optimizer_args)
        optimizer2 = torch.optim.SGD(net2.parameters(), **optimizer_args)

        epoch_loss1 = []
        epoch_loss2 = []
        for epoch in range(self.args.local_ep):
            self.epoch = epoch
            batch_loss1 = []
            batch_loss2 = []
            for batch_idx, batch in enumerate(self.ldr_train):
                self.batch_idx = batch_idx
                net1.zero_grad()
                net2.zero_grad()

                # with autocast():
                loss1, loss2 = self.forward_pass(batch, net1, net2)

                loss1.backward()
                loss2.backward()
                optimizer1.step()
                optimizer2.step()

                if self.args.verbose and batch_idx % 10 == 0:
                    print(f"Epoch: {epoch} [{batch_idx}/{len(self.ldr_train)}"
                          f"({100. * batch_idx / len(self.ldr_train):.0f}%)]\tLoss: {loss1.item():.6f}"
                          f"\tLoss: {loss2.item():.6f}")

                batch_loss1.append(loss1.item())
                batch_loss2.append(loss2.item())
                self.on_batch_end()

            epoch_loss1.append(sum(batch_loss1) / len(batch_loss1))
            epoch_loss2.append(sum(batch_loss2) / len(batch_loss2))
            self.total_epochs += 1
            self.on_epoch_end()

        self.net1.load_state_dict(net1.state_dict())
        self.net2.load_state_dict(net2.state_dict())
        self.last_updated = self.args.g_epoch

        # net1.to('cpu')
        # net2.to('cpu')
        # del net1
        # del net2

        return self.net1.state_dict(), sum(epoch_loss1) / len(epoch_loss1), \
            self.net2.state_dict(), sum(epoch_loss2) / len(epoch_loss2)

    def forward_pass(self, batch, net, net2=None):
        images, labels = batch

        # text
        if isinstance(images, torch.Tensor) and images.dim() == 2:  # text (but named with images), bad name!
            images = images.to(self.args.device)
        else:  # images
            images = images.to(self.args.device).float()

        labels = labels.to(self.args.device)

        log_probs, features = net(images)
        loss = self.loss_func(log_probs, labels)
        


        if net2 is None:
            return loss

        # 处理第二个模型
        log_probs2, features2 = net2(images)
        loss2 = self.loss_func(log_probs2, labels)

        return loss, loss2

    def on_batch_end(self):
        pass

    def on_epoch_end(self):
        pass




class LocalUpdateMaskedOptim(BaseLocalUpdate):
    def __init__(
            self,
            args,
            user_idx=None,
            dataset=None,
            idxs=None,
            idx_return=False,
            real_idx_return=True,
    ):
        self.args = args

        self.dataset = dataset
        self.idxs = idxs
        
        

        self.user_idx = user_idx
        self.update_name = 'maskedOptim'





        self.idx_return = idx_return
        self.real_idx_return = True #Fixed
        self.total_epochs = 0
        
        #TODO: Load custom collate_fn
        self.collate_fn = dataset_split_collate_fn if self.args.collate_fn else None

        self.ldr_train = DataLoader(
            DatasetSplit(dataset, idxs, idx_return=idx_return,
                         real_idx_return=real_idx_return),
            batch_size=self.args.local_bs,
            shuffle=True,
            num_workers=self.args.num_workers,
            collate_fn=self.collate_fn if self.args.collate_fn else None,
            pin_memory=True,
            drop_last=True
        )

        self.ldr_train_infer = DataLoader(
            DatasetSplit(dataset, idxs, idx_return=idx_return,
                         real_idx_return=real_idx_return),
            batch_size=self.args.local_bs,
            shuffle=True,
            num_workers=self.args.num_workers,
            collate_fn=self.collate_fn if self.args.collate_fn else None,
            pin_memory=True,
            drop_last=False
        )

        self.class_sum = np.array([0] * args.num_classes) 
        for idx in self.idxs:
            label = self.dataset.train_labels[idx]
            self.class_sum[label] += 1

        from utils.losses import LogitAdjust, LA_KD, LogitAdjust_soft
        self.loss_func1 = LogitAdjust(cls_num_list=self.class_sum)
        self.loss_func_soft = LogitAdjust_soft(cls_num_list=self.class_sum)
        self.loss_func2 = LA_KD(cls_num_list=self.class_sum)
        self.net1 = get_model(self.args)
        self.last_updated = 0



        self.local_datasize = len(idxs)

        
        self.index_mapper, self.index_mapper_inv = {}, {}

        for i in range(len(self.idxs)):
            self.index_mapper[self.idxs[i]] = i
            self.index_mapper_inv[i] = self.idxs[i]

        self.label_update = torch.index_select(
            args.Soft_labels, 0, torch.tensor(self.idxs))
        # yy = torch.FloatTensor(yy)
        self.label_update = torch.FloatTensor(self.label_update)
        
        self.true_labels_local = torch.index_select(
            args.True_Labels, 0, torch.tensor(self.idxs))

        self.estimated_labels = copy.deepcopy(self.label_update)

        #yield by the local model after E local epochs
        self.final_prediction_labels = copy.deepcopy(self.label_update)

        # self.estimated_labels = F.softmax(self.label_update, dim=1)
        self.lamda = args.lamda_pencil



        for batch_idx, batch in enumerate(self.ldr_train_infer):

            if self.idx_return:
                images, labels, _ = batch
            elif self.real_idx_return:
                images, labels, _, ids = batch
            else:
                images, labels = batch

            indexss = self.indexMapping(ids)
            # self.label_update[indexss].cuda()
            for i in range(len(indexss)):
                self.label_update[indexss[i]][labels[i]] = self.args.K_pencil #fixed 10

        print(f"Initializing the client #{self.user_idx}... Done")



    def logit_clip(self, logits):
        
        delta = 1/self.args.tao
        norms = torch.norm(logits, p=2, dim=-1, keepdim=True) + 1e-7   #p = self.args.lp <- 2
        logits_norm = torch.div(logits, norms) * delta
        clip = (norms > self.args.tao).expand(-1, logits.shape[-1])
        logits_final = torch.where(clip, logits_norm, logits)
        
        return logits_final



    # from overall index to local index
    def indexMapping(self, indexs):
        indexss = indexs.cpu().numpy().tolist()
        target_mapping = []
        for each in indexss:
            target_mapping.append(self.index_mapper[each])
        return target_mapping

    def label_updating(self, labels_grad):
        self.label_update = self.label_update - self.lamda * labels_grad
        self.estimated_labels = F.softmax(self.label_update, dim=1)



    def pencil_loss(self, outputs, labels_update, labels, feat):

        pred = F.softmax(outputs, dim=1)
        
        # masking
        pred_labels = torch.argmax(pred, dim=1)
        mask = (pred_labels == labels).float()

        # 计算每个样本的交叉熵损失
        ce_loss = F.cross_entropy(outputs, labels, reduction='none')
        # 选择前80%损失较小的样本
        k = int(0.8 * len(ce_loss))
        _, indices = torch.topk(ce_loss, k, largest=False)
        ce_mask = torch.zeros_like(ce_loss)
        ce_mask[indices] = 1.0

        # Keep partial predictions for compatibility
        # Lo = -torch.mean(F.log_softmax(labels_update, dim=1)[torch.arange(labels_update.shape[0]),labels] * mask)
        # Lo = -torch.mean(F.log_softmax(labels_update, dim=1)[torch.arange(labels_update.shape[0]),labels])
        Lo = -torch.mean(F.log_softmax(labels_update, dim=1)[torch.arange(labels_update.shape[0]),labels] * ce_mask)
        
        Le = -torch.mean(torch.sum(F.log_softmax(outputs, dim=1) * pred, dim=1))
        # Le = -torch.mean(torch.sum(F.log_softmax(outputs, dim=1) * pred, dim=1) * ce_mask)

        # Lc = -torch.mean(torch.sum(F.log_softmax(labels_update, dim=1) * pred, dim=1) * ce_mask) - Le
        Lc = -torch.mean(torch.sum(F.log_softmax(labels_update, dim=1) * pred, dim=1)) - Le
        
        loss_total = Lc/self.args.num_classes + self.args.alpha_pencil* Lo + self.args.beta_pencil* Le/self.args.num_classes 
        
        return loss_total
    


    def train_stage1(self, net):  # train with LA
        net.train()

        optimizer = torch.optim.SGD(
            net.parameters(),
            lr=self.args.lr,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay,
        )

        epoch_loss = []




        for epoch in range(self.args.local_ep):
            self.epoch = epoch
            batch_loss = []
            


            for batch_idx, batch in enumerate(self.ldr_train):
                self.batch_idx = batch_idx
                net.zero_grad()

                if self.idx_return:
                    images, labels, _ = batch
                elif self.real_idx_return:
                    images, labels, _, ids = batch
                else:
                    images, labels = batch
                images = images.to(self.args.device)
                labels = labels.to(self.args.device)

                logits, feat = net(images) 
                
                # logits = self.logit_clip(logits)   
                loss = self.loss_func1(logits, labels)


                loss.backward()
                optimizer.step()


                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            self.total_epochs += 1

        self.net1.load_state_dict(net.state_dict())
        self.last_updated = self.args.g_epoch

        net.to('cpu')
        # del net

        return self.net1.state_dict(), sum(epoch_loss) / len(epoch_loss)
    


    #TODO: for noisy clients in the second phase
    def train_stage2(self, net, global_net, weight_kd):
        # 保存初始模型参数
        initial_state_dict = copy.deepcopy(net.state_dict())
        
        net.train()

        optimizer = torch.optim.SGD(
            net.parameters(),
            lr=self.args.lr,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay,
        )

        epoch_loss = []

        #TODO: To begin the local training
        for epoch in range(self.args.local_ep):
            self.epoch = epoch
            batch_loss = []

            labels_grad = torch.zeros(self.local_datasize, self.args.num_classes, dtype=torch.float32)

            for batch_idx, batch in enumerate(self.ldr_train):
                self.batch_idx = batch_idx
                net.zero_grad()

                if self.idx_return:
                    images, labels, _ = batch

                #TODO: we use the below one
                elif self.real_idx_return:
                    images, labels, _, ids = batch
                else:
                    images, labels = batch

                images = images.to(self.args.device)
                labels = labels.to(self.args.device)

                with autocast():
                    logits, feat = net(images)

                # logits = self.logit_clip(logits)
                
                indexss = self.indexMapping(ids)

                labels_update = self.label_update[indexss,:].cuda()
                labels_update.requires_grad_()
                # labels_update = torch.autograd.Variable(labels_update,requires_grad = True)

                loss = self.pencil_loss(
                                logits, labels_update, labels, feat)
                
                loss.backward()

                labels_grad[indexss] = labels_update.grad.cpu().detach() #.numpy()

                labels_update = labels_update.to('cpu')
                del labels_update

                optimizer.step()

                batch_loss.append(loss.item())
            
            self.label_updating(labels_grad)
            
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            self.total_epochs += 1
        
        # After E local epochs
        self.net1.load_state_dict(net.state_dict())
        self.last_updated = self.args.g_epoch

        #TODO: traverse dataset
        after_correct_predictions = 0
        for batch_idx, batch in enumerate(self.ldr_train_infer):
            if self.idx_return:
                images, labels, _ = batch
            elif self.real_idx_return:
                images, labels, _, ids = batch
            else:
                images, labels = batch

            images = images.to(self.args.device)
            labels = labels.to(self.args.device)
            local_index = self.indexMapping(ids)

            with torch.no_grad():
                output_final, teacher_feat = net(images)
                output_final = output_final.to('cpu')

                soft_label = torch.softmax(output_final, dim=1) 
                self.final_prediction_labels[local_index]  = soft_label

        net.to('cpu')
        del net

        #TODO: merge the softmax(self.label_update) and the prediction after local training(self.final_prediction_labels)
        self.label_update = self.label_update.to('cpu')

        updated_local_labels_tmp = F.softmax(self.label_update, dim=1)
        final_model_prediction_tmp = self.final_prediction_labels
        # average the above two
        merged_local_labels = (updated_local_labels_tmp + final_model_prediction_tmp) / 2
        # the GT labels is self.true_labels_local
        predicted_classes = torch.argmax(merged_local_labels, dim=1)

        # replace the label_update with the merged_local_labels, and rescale by K_pencil
        self.label_update = merged_local_labels * self.args.K_pencil

        # EMA
        final_state_dict = self.net1.state_dict()
        ema_state_dict = {}
        for key in final_state_dict.keys():
            initial_param = initial_state_dict[key].to('cpu')
            final_param = final_state_dict[key].to('cpu')
            ema_state_dict[key] = 0.2 * initial_param + 0.8 * final_param
        

        self.net1.load_state_dict(ema_state_dict)

        return self.net1.state_dict(), sum(epoch_loss) / len(epoch_loss)
