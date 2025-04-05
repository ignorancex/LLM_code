import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from typing import Tuple, List
import yaml


# TSA
def get_tsa_thresh(schedule, global_step, num_train_steps, start, end, device):
    training_progress = torch.tensor(float(global_step) / float(num_train_steps))
    if schedule == 'linear_schedule':
        threshold = training_progress
    elif schedule == 'exp_schedule':
        scale = 5
        threshold = torch.exp((training_progress - 1) * scale)
    elif schedule == 'log_schedule':
        scale = 5
        threshold = 1 - torch.exp((-training_progress) * scale)
    else:
        raise NotImplementedError('TSA scheduler method %s is not implemented' % schedule)
    output = threshold * (end - start) + start
    return output.to(device)


class SemiSupMultiMLP(nn.Module):
    '''
    Baic one single shared bottom for all tasks across all views for benchmark evaluation for semi-supervised learning
    '''
    def __init__(self, num_tasks, num_views, seq_len, sensor_dim, hidden_dim, output_dim, init_method, temperature, device, dropout_p=0.2):
        super(SemiSupMultiMLP, self).__init__()
        self.num_views = num_views
        self.seq_len = seq_len
        self.sensor_dim = sensor_dim
        self.temperature = temperature
        self.init_method = init_method
        self.device = device
        self.shared_bottom = nn.Linear(sensor_dim, 1)
        # self.specific_tasks_1 = nn.ModuleList([nn.Linear(sensor_dim, 1) for _ in range(num_tasks)])
        self.specific_tasks_2 = nn.ModuleList([nn.Linear(num_views, 1) for _ in range(num_tasks)])
        self.towers = nn.ModuleList([nn.Sequential(
            nn.Linear(seq_len, hidden_dim),
            nn.BatchNorm1d(num_features=hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, output_dim)
        ) for _ in range(num_tasks)])
        self._init_task_logits_(num_tasks, num_views, 3)

    def _init_task_logits_(self, num_candidates_1, num_candidates_2, num_options):
        if self.init_method == 'all':
            task_logits = .8 * torch.ones(num_candidates_1, num_candidates_2, num_options)
            for i in range(1, num_options):
                task_logits[:, :, i] = 0
        elif self.init_method == 'random':
            task_logits = 1e-3 * torch.randn(num_candidates_1, num_candidates_2, num_options)
        elif self.init_method == 'equal':
            task_logits = .5 * torch.ones(num_candidates_1, num_candidates_2, num_options)
        else:
            raise NotImplementedError('Initial Method %s is not implemented' % self.init_method)
        
        self.register_parameter('task_logits', nn.Parameter(task_logits, requires_grad=True))

    def train_sample_policy(self, temperature, hard_sampling):
        
        return F.gumbel_softmax(getattr(self, 'task_logits'), temperature, hard=hard_sampling)
    
    def forward(self, sup_x, unsup_x=None):
        # X shape: [batch_size, num_tasks, num_views, seq_len, sensor_dim] from single task
        if unsup_x is None:
            bsz, num_tasks, num_views, seq_len, sensor_dim = sup_x.size()
            x = sup_x.transpose(2, 3) # >> [batch_size, num_tasks, seq_len, num_views, sensor_dim]
            # x = x.reshape(bsz, num_tasks, seq_len, -1)
            x = self.shared_bottom(x).squeeze(-1) # >> [batch_size, num_tasks, seq_len, num_views]
            # output = [self.specific_tasks_1[task_id](x[:, task_id, ...]).squeeze(-1) for task_id in range(num_tasks)] # >> [batch_size, seq_len, num_views] * num_tasks
            output = [self.specific_tasks_2[task_id](x[:, task_id, ...]).squeeze(-1) for task_id in range(num_tasks)] # >> [batch_size, seq_len] * num_tasks
            # x = torch.stack(output, dim=1) # >> [batch_size, num_tasks, seq_len]
            output = [self.towers[task_id](output[task_id]) for task_id in range(num_tasks)] # >> [batch_size, output_dim] * num_tasks

            return torch.stack(output, dim=0), None
        else:
            # Adaptive Data Augmentation for Unsupervised Part
            bsz, num_tasks, num_views, seq_len, sensor_dim = sup_x.size()
            unsup_bsz = unsup_x.size(0)
            policy = self.train_sample_policy(self.temperature, True) # >> [num_tasks, 2]
            unsup_x += torch.empty(unsup_bsz, num_tasks, num_views, seq_len, sensor_dim, device=self.device).normal_(0, 10) * \
                    policy[:, :, 0][None, :, :, None, None] + \
                    torch.empty(unsup_bsz, num_tasks, num_views, seq_len, sensor_dim, device=self.device).uniform_(0, 10) * \
                    policy[:, :, 1][None, :, :, None, None] + \
                    torch.empty(unsup_bsz, num_tasks, num_views, seq_len, sensor_dim, device=self.device).log_normal_(0, 3) * \
                    policy[:, :, 2][None, :, :, None, None]  # >> [unsup_bsz, ...]
            # unsup_x += .1
            x = torch.cat([sup_x, unsup_x], dim=0) # >> [bsz+unsup_bsz, ...]
            x = x.transpose(2, 3) # >> [batch_size, num_tasks, seq_len, num_views, sensor_dim]
            # x = x.reshape(bsz, num_tasks, seq_len, -1)
            x = self.shared_bottom(x).squeeze(-1) # >> [batch_size, num_tasks, seq_len, num_views]
            # output = [self.specific_tasks_1[task_id](x[:, task_id, ...]).squeeze(-1) for task_id in range(num_tasks)] # >> [batch_size, seq_len, num_views] * num_tasks
            output = [self.specific_tasks_2[task_id](x[:, task_id, ...]).squeeze(-1) for task_id in range(num_tasks)] # >> [batch_size, seq_len] * num_tasks
            # x = torch.stack(output, dim=1) # >> [batch_size, num_tasks, seq_len]
            output = [self.towers[task_id](output[task_id]) for task_id in range(num_tasks)] # >> [batch_size, output_dim] * num_tasks
            output = torch.stack(output, dim=0)

            return output[:, :bsz, :], output[:, bsz:, :] # >> [num_tasks, bsz, output_dim], [num_tasks, unsup_bsz, output_dim]


class SemiSupUncertaintyLossWrapper(nn.Module):
    """Implementation of paper: Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics

    params: num_tasks
    params: model
    return: Wrapped losses of multiple tasks
    """
    def __init__(self, model, num_tasks, sup_criterion, l2_criterion, unsup_criterion, eta, beta, opt, device):
        super(SemiSupUncertaintyLossWrapper, self).__init__()
        self.model = model
        self.num_tasks = num_tasks
        self.sup_criterion = sup_criterion
        self.unsup_criterion = unsup_criterion
        self.l2_criterion = l2_criterion
        if self.training:
            self.tsa = opt['train']['is_tsa']
        else:
            self.tsa = False
        self.sample_ratio = opt['train']['adaption_sample_ratio']
        self.adaption_steps = opt['train']['adaption_steps']
        self.tsa_schedule = opt['train']['tsa_schedule']
        self.total_steps = opt['train']['total_steps']
        self.uda_softmax_temp = opt['train']['uda_softmax_temp']
        self.uda_confidence_thresh = opt['train']['uda_confidence_thresh']
        self.uda_coeff = opt['train']['uda_coefficient']
        self.device = device
        assert len(eta) == num_tasks * 2, "length of eta should be same as number of tasks"
        # variable change for stability >> using eta = 2log\sigma
        self.init_eta(eta)
        # self.init_beta(beta)
        # self.eta = nn.Parameter(torch.Tensor(eta)).to(device)
    
    def init_beta(self, beta):
        self.register_parameter('beta', nn.Parameter(torch.Tensor(beta), requires_grad=True))

    def init_eta(self, eta):
        self.register_parameter('eta', nn.Parameter(torch.Tensor(eta), requires_grad=True))

    def forward(self, sup_inputs, targets, unsup_inputs, global_step):
        # Compute Supervised loss with uncertainty
        targets = targets.transpose(0, 1)
        if unsup_inputs is not None:
            unsup_bsz = unsup_inputs.size(0)
            sample_bsz = int(unsup_bsz * self.sample_ratio)
            aug_unsup_inputs = []
            for k in range(1, self.adaption_steps):
                # adaptation_indices = random.sample(range(unsup_bsz), sample_bsz)
                # aug_unsup_indices.append(adaptation_indices)
                aug_unsup_inputs.append(unsup_inputs[sample_bsz*k:sample_bsz*(k+1)])
            aug_unsup_inputs = torch.cat(aug_unsup_inputs, dim=0)
            # print(aug_unsup_inputs.size())
            # adaptation_indices = np.zeros(unsup_bsz, dtype=bool)
            # adaptation_indices[np.arange(unsup_bsz//2) * 2] = True
            # evaluation_indices = torch.tensor(~adaptation_indices)
            # adaptation_indices = torch.from_numpy(adaptation_indices)
            sup_outputs, unsup_outputs = self.model(sup_inputs, aug_unsup_inputs)
            # print(unsup_outputs.size())
        else:
            sup_outputs, unsup_outputs = self.model(sup_inputs, unsup_inputs)

        # Compute Supervised Cross Entropy Loss
        sup_total_loss = 0
        sup_loss_list = [self.sup_criterion(o, y) for o, y in zip(sup_outputs, targets)]
        if self.tsa:
            tsa_thresh = get_tsa_thresh(self.tsa_schedule, global_step, self.total_steps, start=1./sup_outputs.size(-1), end=1, device=self.device)
            for task_id in range(self.num_tasks):
                sup_loss = sup_loss_list[task_id]
                larger_than_threshold = torch.exp(-sup_loss) > tsa_thresh   # prob = exp(log_prob), prob > tsa_threshold
                # larger_than_threshold = torch.sum(  F.softmax(pred[:sup_size]) * torch.eye(num_labels)[sup_label_ids]  , dim=-1) > tsa_threshold
                loss_mask = torch.ones_like(targets[task_id], dtype=torch.float32) * (1 - larger_than_threshold.type(torch.float32))
                # loss_mask = loss_mask.to(self.device)
                sup_loss = torch.sum(sup_loss * loss_mask, dim=-1) / torch.sum(loss_mask, dim=-1)
                sup_total_loss += torch.sum(sup_loss * torch.exp(-self.eta[task_id]) + self.eta[task_id])
        else:
            # pass
            for i in range(self.num_tasks):
                sup_total_loss += torch.sum(sup_loss_list[i] * torch.exp(-self.eta[i]) + self.eta[i])
                # sup_total_loss += torch.sum(sup_loss_list[i])
        # print(sup_total_loss)

        # Compute l2 loss between view-specific and merged outputs
        # l2_total_loss = 0
        # view_outputs, merged_outputs = F.log_softmax(view_outputs, dim=-1), F.log_softmax(merged_outputs, dim=-1)
        # l2_loss_list = [self.l2_criterion(o, y) for o, y in zip(view_outputs, merged_outputs)]
        # for i in range(self.num_tasks):
        #     l2_total_loss += torch.sum(l2_loss_list[i])
        # print(l2_total_loss)
        
        # Compute Unsupervised loss
        if unsup_outputs is not None:
            unsup_total_loss = 0
            with torch.no_grad():
                # aug
                # softmax temperature controlling
                aug_log_probs_multi = []
                for k in range(self.adaption_steps - 1):
                    uda_softmax_temp = self.uda_softmax_temp if self.uda_softmax_temp > 0 else 1.
                    aug_log_probs = F.log_softmax(unsup_outputs[:, k*sample_bsz:(k+1)*sample_bsz, :] / uda_softmax_temp, dim=-1) # >> [num_tasks, unsup_bsz, output_dim]
                    aug_log_probs_multi.append(aug_log_probs)
                
                # Original
                # evaluation_indices = random.sample(range(unsup_bsz), sample_bsz)
                # print(evaluation_indices)
                ori_outputs, _, = self.model(unsup_inputs[:sample_bsz])
                ori_probs = F.softmax(ori_outputs, dim=-1) # >> [num_tasks, unsup_bsz, output_dim] # KLdiv target
                # print(ori_probs.size())
                # confidence-based masking
                if self.uda_confidence_thresh != -1:
                    unsup_loss_masks = torch.max(ori_probs, dim=-1)[0] > self.uda_confidence_thresh
                    unsup_loss_masks = unsup_loss_masks.type(torch.float32)
                else:
                    unsup_loss_masks = torch.ones(self.num_tasks, unsup_inputs.size(0), dtype=torch.float32)
                unsup_loss_masks = unsup_loss_masks.to(self.device) # >> [num_tasks, unsup_bsz]
    
                # KLdiv loss
                """
                    nn.KLDivLoss (kl_div)
                    input : log_prob (log_softmax)
                    target : prob    (softmax)
                    https://pytorch.org/docs/stable/nn.html
                    unsup_loss is divied by number of unsup_loss_mask
                    it is different from the google UDA official
                    The official unsup_loss is divided by total
                    https://github.com/google-research/uda/blob/master/text/uda.py#L175
                """
                for k in range(self.adaption_steps - 1):
                    aug_log_probs = aug_log_probs_multi[k]
                    # print(aug_log_probs.size())
                    # print(ori_probs.size())
                    unsup_loss_list = [torch.sum(self.unsup_criterion(aug_log_prob, ori_prob), dim=-1)
                                        for aug_log_prob, ori_prob in zip(aug_log_probs, ori_probs)]
                    # unsup_loss = torch.sum(self.unsup_criterion(aug_log_prob, ori_prob), dim=-1)
                    unsup_loss_list = [torch.sum(unsup_loss * unsup_loss_mask, dim=-1) / torch.sum(unsup_loss_mask, dim=-1)
                                        for unsup_loss, unsup_loss_mask in zip(unsup_loss_list, unsup_loss_masks)]
                    for i in range(self.num_tasks):
                        # unsup_total_loss += torch.sum(unsup_loss_list[i] * torch.exp(-self.eta[self.num_tasks+i]) + self.eta[self.num_tasks+i])
                        unsup_total_loss += torch.sum(unsup_loss_list[i])
            # print(unsup_total_loss)
            final_loss = sup_total_loss + self.uda_coeff * unsup_total_loss / (self.adaption_steps - 1)
            # print(final_loss)
            return sup_outputs, final_loss, sup_total_loss, unsup_total_loss


        # total_loss =  torch.Tensor(loss).to(self.device) * torch.exp(-self.eta) + self.eta
        # total_loss = torch.Tensor(loss).to(self.device) / self.num_tasks        
        return sup_outputs, sup_total_loss, None, None # omit 1/2


if __name__ == '__main__':
    with open('./yamls/realworld2016_exp.yaml') as f:
        opt = yaml.safe_load(f)
    sup_criterion = torch.nn.CrossEntropyLoss(reduction='none')
    unsup_criterion = torch.nn.KLDivLoss(reduction='none')
    device = torch.device('cuda:0')
    sensor_dim = opt['num_sensors'] * 3
    model = SemiSupMultiMLP(opt['num_tasks'], opt['num_views'], opt['window_size'], 
                            sensor_dim, 100, opt['num_action_types'], opt['train']['init_method'], 
                            opt['train']['temperature'], device, .5)
    model = model.to(device)
    eta = [0.] * opt['num_tasks']
    wrapper = SemiSupUncertaintyLossWrapper(model, opt['num_tasks'], sup_criterion, unsup_criterion, eta, opt, device)
    sup_x = torch.randn(32, 8, 7, 250, 9)
    unsup_x = torch.randn(16, 8, 7, 250, 9)
    targets = targets = torch.empty(32, 8, dtype=torch.long).random_(7)
    sup_x, targets, unsup_x = sup_x.to(device), targets.to(device), unsup_x.to(device)
    # sup_y, unsup_y = model(sup_x, unsup_x)
    # print(sup_y.size(), unsup_y.size())
    # y = model(x)
    outputs, fina_loss, sup_loss, unsup_loss = wrapper(sup_x, targets, unsup_x, 100)
    # print(y[0].size())
    print(outputs.size())
    print(fina_loss, sup_loss, unsup_loss)