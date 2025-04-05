import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List


class UncertaintyLossWrapper(nn.Module):
    """Implementation of paper: Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics

    params: num_tasks
    params: model
    return: Wrapped losses of multiple tasks
    """
    def __init__(self, model, num_tasks, criterion, eta, device):
        super(UncertaintyLossWrapper, self).__init__()
        self.model = model
        self.num_tasks = num_tasks
        self.criterion = criterion
        self.device = device
        assert len(eta) == num_tasks, "length of eta should be same as number of tasks"
        # variable change for stability >> using eta = 2log\sigma
        self.eta = nn.Parameter(torch.Tensor(eta)).to(device)
    
    def forward(self, inputs, targets):
        # print(self.model.device, targets.device)
        outputs = self.model(inputs)
        total_loss = 0
        loss = [self.criterion(o, y) for o, y in zip(outputs, targets.transpose(0, 1))]
        for i in range(self.num_tasks):
            total_loss += torch.sum(loss[i] * torch.exp(-self.eta[i]) + self.eta[i])
        # total_loss =  torch.Tensor(loss).to(self.device) * torch.exp(-self.eta) + self.eta
        # total_loss = torch.Tensor(loss).to(self.device) / self.num_tasks        
        return outputs, torch.mean(total_loss) # omit 1/2


class MultiTaskModelWrapper(nn.Module):
    """Return a wrapped multi-task model based on specific backbone
    """
    def __init__(self, backbone, num_tasks):
        super(MultiTaskModelWrapper, self).__init__()
        self.nets = nn.ModuleList([backbone for _ in range(num_tasks)])
        self.num_tasks = num_tasks

    def forward(self, input):
        # input shape: [batch_size, num_tasks, num_views, window_size, sensor_dim]
        # input = input.transpose(0, 1)
        return [self.nets[task_id](input[:, task_id, ...]) for task_id in range(self.num_tasks)]


class MultiMLP(nn.Module):
    '''
    Baic one single shared bottom for all tasks across all views for benchmark evaluation
    '''
    def __init__(self, num_tasks, num_views, seq_len, sensor_dim, hidden_dim, output_dim, dropout_p=0.2):
        super(MultiMLP, self).__init__()
        self.num_views = num_views
        self.seq_len = seq_len
        self.sensor_dim = sensor_dim
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

    def forward(self, x):
        # X shape: [batch_size, num_tasks, num_views, seq_len, sensor_dim] from single task
        bsz, num_tasks, num_views, seq_len, sensor_dim = x.size()
        x = x.transpose(2, 3) # >> [batch_size, num_tasks, seq_len, num_views, sensor_dim]
        # x = x.reshape(bsz, num_tasks, seq_len, -1)
        x = self.shared_bottom(x).squeeze(-1) # >> [batch_size, num_tasks, seq_len, num_views]
        # output = [self.specific_tasks_1[task_id](x[:, task_id, ...]).squeeze(-1) for task_id in range(num_tasks)] # >> [batch_size, seq_len, num_views] * num_tasks
        output = [self.specific_tasks_2[task_id](x[:, task_id, ...]).squeeze(-1) for task_id in range(num_tasks)] # >> [batch_size, seq_len] * num_tasks
        # x = torch.stack(output, dim=1) # >> [batch_size, num_tasks, seq_len]
        output = [self.towers[task_id](output[task_id]) for task_id in range(num_tasks)] # >> [batch_size, output_dim] * num_tasks

        return output


class AdaMultiMLP(nn.Module):
    """Adaptively choose whether being packed together and passed through the same shared bottom
    """
    def __init__(self, num_tasks, num_views, seq_len, sensor_dim, hidden_dim, output_dim, init_method='random', temperature=1., dropout_p=0.2):
        super(AdaMultiMLP, self).__init__()
        self.num_tasks = num_tasks
        self.num_views = num_views
        self.seq_len = seq_len
        self.sensor_dim = sensor_dim
        self.init_method = init_method
        self.temperature = temperature
        self.shared_bottoms = nn.ModuleList([nn.Linear(sensor_dim, 1) for _ in range(num_tasks)])
        # self.specific_tasks_1 = nn.ModuleList([nn.Linear(sensor_dim, 1) for _ in range(num_tasks)])
        self.specific_tasks_2 = nn.ModuleList([nn.Linear(num_views, 1) for _ in range(num_tasks)])
        self.towers = nn.ModuleList([nn.Sequential(
            nn.Linear(seq_len, hidden_dim),
            nn.BatchNorm1d(num_features=hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, output_dim)
        ) for _ in range(num_tasks)])
        self._init_task_logits_(num_tasks, num_tasks) # >> register parameter task_logits as gates

    def _init_task_logits_(self, num_candidates, num_options):
        if self.init_method == 'all':
            task_logits = .5 * torch.ones(num_candidates, num_options)
            for i in range(1, num_options):
                task_logits[:, i] = 0
        elif self.init_method == 'random':
            task_logits = 1e-3 * torch.randn(num_candidates, num_options)
        elif self.init_method == 'equal':
            task_logits = .5 * torch.ones(num_candidates, num_options)
        else:
            raise NotImplementedError('Initial Method %s is not implemented' % self.init_method)
        
        self.register_parameter('task_logits', nn.Parameter(task_logits, requires_grad=True))

    def train_sample_policy(self, temperature, hard_sampling):
        
        return F.gumbel_softmax(getattr(self, 'task_logits'), temperature, hard=hard_sampling)

    def forward(self, x):
        # X shape: [batch_size, num_tasks, num_views, seq_len, sensor_dim] from single task
        bsz, num_tasks, num_views, seq_len, sensor_dim = x.size()
        x = x.transpose(2, 3) # >> [bsz, num_tasks, seq_len, num_views, sensor_dim]
        policy = self.train_sample_policy(self.temperature, True) # >> [num_tasks, 2]
        # outputs = []
        y = self.shared_bottoms[0](x).squeeze(-1) * policy[:, 0][None, :, None, None]
        for i in range(1, policy.size(1)):
            y += self.shared_bottoms[i](x).squeeze(-1) * policy[:, i][None, :, None, None]
        # for view_id in range(num_views):
        #     output = self.shared_bottoms[0](x[:, :, view_id, ...]) * policy[view_id][0] + \
        #                 self.shared_bottoms[1](x[:, :, view_id, ...]) * policy[view_id][1]
        #     outputs.append(output)
        # x = torch.stack(outputs, dim=3).squeeze(-1) # >> [bsz, num_tasks, seq_len, num_views]
        # print(x.size())
        # x = self.shared_bottom(x).squeeze(-1) # >> [batch_size, num_tasks, seq_len, num_views]
        # output = [self.specific_tasks_1[task_id](x[:, task_id, ...]).squeeze(-1) 
        #                                 for task_id in range(num_tasks)] # >> [batch_size, seq_len, num_views] * num_tasks
        output = [self.specific_tasks_2[task_id](y[:, task_id, ...]).squeeze(-1) for task_id in range(num_tasks)] # >> [batch_size, seq_len] * num_tasks
        # x = torch.stack(output, dim=1) # >> [batch_size, num_tasks, seq_len]
        output = [self.towers[task_id](output[task_id]) for task_id in range(num_tasks)] # >> [batch_size, output_dim] * num_tasks

        return output


class MultiViewMLP(nn.Module):
    '''
    Different views share different shared bottom
    '''
    def __init__(self, num_tasks, num_views, seq_len, sensor_dim, hidden_dim, output_dim, dropout_p=0.2):
        super(MultiViewMLP, self).__init__()
        self.num_views = num_views
        self.seq_len = seq_len
        self.sensor_dim = sensor_dim
        self.shared_bottoms = nn.ModuleList([nn.Linear(sensor_dim, 1) for _ in range(num_views)])
        self.specific_tasks_1 = nn.ModuleList([nn.Linear(num_views, 1) for _ in range(num_tasks)])
        self.towers = nn.ModuleList([nn.Sequential(
            nn.Linear(seq_len, hidden_dim),
            nn.BatchNorm1d(num_features=hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, output_dim)
        ) for _ in range(num_tasks)])

    def forward(self, x):
        # X shape: [batch_size, num_tasks, num_views, seq_len, sensor_dim] from single task
        bsz, num_tasks, num_views, seq_len, sensor_dim = x.size()
        output = [self.shared_bottoms[view_id](x[:, :, view_id, ...]).squeeze(-1) 
                                            for view_id in range(num_views)] # >> [batch_size, num_tasks, seq_len] * num_views
        x = torch.stack(output, dim=3) # >> [batch_size, num_tasks, seq_len, num_views]
        output = [self.specific_tasks_1[task_id](x[:, task_id, ...]).squeeze(-1) for task_id in range(num_tasks)] # >> [batch_size, seq_len] * num_tasks
        output = [self.towers[task_id](output[task_id]) for task_id in range(num_tasks)] # >> [batch_size, output_dim] * num_tasks

        return output


class ConvMultiView(nn.Module):
    def __init__(self, num_tasks, num_views, seq_len, sensor_dim, hidden_dim, output_dim, device, dropout_p=.2):
        super(ConvMultiView, self).__init__()
        self.num_tasks = num_tasks
        self.num_views = num_views
        self.sensor_dim = sensor_dim
        self.dropout_p = dropout_p
        self.device = device
        self.view_convs = self.build_view_convs()
        self.task_convs = self.build_task_convs()
        self.linears = nn.ModuleList(
            [nn.Linear(hidden_dim, output_dim) for _ in range(num_tasks)]
        )

    def forward(self, x):
        '''Forward
        in : (batch_size, num_tasks, num_views, window_size, sensor_dim)
        out : 
        '''
        x = x.transpose(3, 4) # >> [bsz, num_tasks, num_views, sensor_dim, window_size]
        output = [
            self.shared_cnn(x[:, :, view_id, ...], self.view_convs[0][view_id], self.view_convs[1][view_id])
            for view_id in range(self.num_views)
        ] # >> [batch_size, num_tasks, *] * num_views
        output = self.merge_view(output) # >> [batch_size, num_tasks, 1, num_views, *]
        # print(output.size())
        output = [
            self.specific_cnn(output[:, task_id, ...], self.task_convs[0][task_id], self.task_convs[1][task_id])
            for task_id in range(self.num_tasks)
        ] # >> [batch_size, *] * num_tasks
        # print(output[0].size())
        output = [
            self.linears[task_id](output[task_id]) for task_id in range(self.num_tasks)
        ]

        return output

    def specific_cnn(self, input, conv_4, conv_5_6):
        '''
        in : (batch_size, 1, num_views, *)
        out: (batch_size, *)
        '''
        input = conv_4(input) # >> (batch_size, channel_dim, 1, *)
        input = torch.squeeze(input, dim=2) # >> (batch_size, channel_dim, *)
        # input = conv_5_6(input) # >> (batch_size, channel_dim, *)
        input = torch.flatten(input, start_dim=1) # >> (batch_size, *)

        return input

    def shared_cnn(self, input :torch.Tensor, conv_1 :nn.Module, conv_2_3: nn.Module):
        '''
        in : (batch_size, num_tasks, sensor_dim, window_size) | (batch_size, 1, num_views, *)
        out : (batch_size, *)
        '''
        input = conv_1(input)  # >> (batch_size, num_tasks, 1, *)
        input = torch.squeeze(input, dim=2)  # >> (batch_size, num_tasks,*)
        input = conv_2_3(input)  # (batch_size, num_tasks, *)
        # input = torch.flatten(input, start_dim=1)  # (batch_size, *)

        return input

    def build_view_convs(self, k1=3, k2=4, k3=2, stride=2) -> Tuple[nn.Module, nn.Module]:
        '''Soft sharing across different tasks through each specific view
        in : (batch_size, num_tasks, sensor_dim, window_size)
        out : (batch_size, num_tasks, 1, *)
        '''
        # conv_1 = nn.Conv2d(
        #     self.num_tasks, self.num_tasks, (self.sensor_dim, k1)
        # ) # >> (batch_size, num_tasks, 1, window_size - kernel_size + 1)
        # # squeeze shape to >> (batch_size, num_tasks, *)
        
        # conv_2_3 = nn.Sequential(
        #     nn.BatchNorm1d(self.num_tasks), nn.ReLU(), nn.Dropout(self.dropout_p),
        #     nn.Conv1d(self.num_tasks, self.num_tasks, k2, stride=stride), nn.BatchNorm1d(self.num_tasks), nn.ReLU(), nn.Dropout(self.dropout_p),
        #     nn.Conv1d(self.num_tasks, self.num_tasks, k3, stride=stride), nn.BatchNorm1d(self.num_tasks), nn.ReLU(),
        # ) # (batch_size, num_tasks, *) -> #(batch_size, num_tasks, *)

        conv_1s = nn.ModuleList(
            [
                nn.Conv2d(self.num_tasks, self.num_tasks, (self.sensor_dim, k1)) for _ in range(self.num_views)
            ]
        ).to(self.device)

        conv_2_3s = nn.ModuleList(
            [
                nn.Sequential(
                    nn.BatchNorm1d(self.num_tasks), nn.ReLU(), nn.Dropout(self.dropout_p),
                    # nn.Conv1d(self.num_tasks, self.num_tasks, k2, stride=stride), nn.BatchNorm1d(self.num_tasks), 
                    # nn.ReLU(), nn.Dropout(self.dropout_p),
                    # nn.Conv1d(self.num_tasks, self.num_tasks, k3, stride=stride), nn.BatchNorm1d(self.num_tasks), nn.ReLU(),
                ) for _ in range(self.num_views)
            ]
        ).to(self.device)

        return conv_1s, conv_2_3s

    def build_task_convs(self, k1=5, k2=8, k3=4, channel_dim=1):
        '''Task specific convolution
        in : (batch_size, 1, num_views, *)
        out: (batch_size, channel_size, *)
        '''
        # conv_4 = nn.Conv2d(1, channel_dim, (self.num_views, k1)) # (batch_size, channel_dim, 1, *)

        # # (batch_size, channel_dim, *) -> #(batch_size, channel_dim, *)
        # conv_5_6 = nn.Sequential(
        #     nn.BatchNorm1d(channel_dim), nn.ReLU(), nn.Dropout(self.dropout_p),
        #     nn.Conv1d(channel_dim, channel_dim, k2, k2), nn.BatchNorm1d(channel_dim), nn.ReLU(), nn.Dropout(self.dropout_p),
        #     nn.Conv1d(channel_dim, channel_dim, k3, k3), nn.BatchNorm1d(channel_dim), nn.ReLU(),
        # )

        conv_4s = nn.ModuleList(
            [
                nn.Conv2d(1, channel_dim, (self.num_views, k1)) for _ in range(self.num_tasks)
            ]
        ).to(self.device)

        conv_5_6s = nn.ModuleList(
            [
                nn.Sequential(
                    nn.BatchNorm1d(channel_dim), nn.ReLU(), 
                    nn.Dropout(self.dropout_p),
                    nn.Conv1d(channel_dim, channel_dim, k2, k2), nn.BatchNorm1d(channel_dim), nn.ReLU(), 
                    nn.Dropout(self.dropout_p),
                    nn.Conv1d(channel_dim, channel_dim, k3, k3), nn.BatchNorm1d(channel_dim), nn.ReLU(),
                ) for _ in range(self.num_tasks)
            ]
        ).to(self.device)

        return conv_4s, conv_5_6s

    def merge_view(self, inputs):
        '''Merge data from different views
        in : (batch_size, num_tasks, *) * num_views
        out: (batch_size, num_tasks, num_views, *)
        '''

        return torch.stack(inputs, dim=2).unsqueeze(2) # >> [bsz, num_tasks, 1, num_views, *]


if __name__ == '__main__':
    device = torch.device('cuda:0')
    model = AdaMultiMLP(8, 7, 250, 9, 100, 7)
    model = model.to(device)
    x = torch.randn(32, 8, 7, 250, 9)
    x = x.to(device)
    y = model(x)
    print(y[0].size())