import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseMLP(nn.Module):
    '''
    Simple MLP on single task for benchmark evaluation
    '''
    def __init__(self, num_views, seq_len, sensor_dim, hidden_dim, output_dim, dropout_p=0.2):
        super(BaseMLP, self).__init__()
        self.num_views = num_views
        self.seq_len = seq_len
        self.sensor_dim = sensor_dim
        self.bottom = nn.Linear(sensor_dim, 1)
        self.combine_views = nn.Linear(num_views, 1)
        self.towers = nn.Sequential(
            nn.Linear(seq_len, hidden_dim),
            nn.BatchNorm1d(num_features=hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, output_dim)
        )
        # self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Relu(), nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        # X shape: [batch_size, num_views, seq_len, sensor_dim] from single task
        bsz, num_views, seq_len, sensor_dim = x.size()
        x = x.transpose(1, 2) # >> [batch_size, seq_len, num_views, sensor_dim]
        x = self.bottom(x).squeeze(-1) # >> [bsz, seq_len, num_views]
        x = self.combine_views(x).squeeze(-1)
        output = self.towers(x) # >> [bsz, output_dim]
        
        return output


class SemiSupMultiViewMLP(nn.Module):
    '''
    Baic one single shared bottom for all tasks across all views for benchmark evaluation for semi-supervised learning
    '''
    def __init__(self,
                num_tasks, num_views, seq_len, sensor_dim, hidden_dim_1, hidden_dim_2,
                output_dim, init_method, temperature, device, dropout_p=0.2
    ):
        super(SemiSupMultiViewMLP, self).__init__()
        self.num_views = num_views
        self.seq_len = seq_len
        self.sensor_dim = sensor_dim
        self.temperature = temperature
        self.init_method = init_method
        self.device = device
        self.shared_bottom = nn.ModuleList(
            [
                nn.Linear(sensor_dim, 1) for _ in range(num_views)
            ]
        )
        self.specific_views = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.Linear(seq_len, hidden_dim_1),
                            nn.BatchNorm1d(hidden_dim_1),
                            nn.ReLU(),
                            nn.Dropout(dropout_p),
                            nn.Linear(hidden_dim_1, hidden_dim_2),
                            nn.BatchNorm1d(hidden_dim_2),
                            nn.ReLU(),
                            nn.Dropout(dropout_p),
                        ) for _ in range(num_views)
                    ]
                ) for _ in range(num_tasks)
            ]
        )
        
        # self.specific_tasks = nn.ModuleList(
        #     [
        #         nn.Sequential(
        #             # nn.Linear(hidden_dim_1, hidden_dim_2),
        #             # nn.BatchNorm1d(hidden_dim_3),
        #             # nn.ReLU(),
        #             # nn.Dropout(dropout_p),
        #             nn.Linear(hidden_dim_1, output_dim),
        #             # nn.ReLU(),
        #             # nn.Dropout(dropout_p),
        #         ) for _ in range(num_tasks)
        #     ]
        # )

        self.merge_views = nn.ModuleList(
            [
                nn.Linear(num_views * hidden_dim_2, output_dim) for _ in range(num_tasks)
            ]
        )
        # self.merge_views_output = nn.ModuleList(
        #     [
        #         nn.Linear(hidden_dim_1, output_dim) for _ in range(num_tasks)
        #     ]
        # )

        # self._init_task_logits_(num_tasks, num_views, 3)

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
            # x = sup_x.contiguous().view(bsz, num_tasks, num_views, -1) # >> [batch_size, num_tasks, num_views, seq_len * sensor_dim]
            # x = x.reshape(bsz, num_tasks, seq_len, -1)
            output = [self.shared_bottom[view_id](sup_x[:, :, view_id, ...]).squeeze(-1) for view_id in range(num_views)] # >> [bsz, num_tasks, seq_len] * num_views
            task_output = []
            for task_id in range(num_tasks):
                view_output = []
                for view_id in range(num_views):
                    y = self.specific_views[task_id][view_id](output[view_id][:, task_id, :]) # >> [bsz, hidden_dim_1]
                    view_output.append(y)
                view_output = torch.stack(view_output, dim=1) # >> [bsz, num_views, hidden_dim_1]
                task_output.append(view_output) # >> [bsz, num_views, hidden_dim_1] * num_tasks
            # output = [self.specific_tasks[task_id](task_output[task_id]) for task_id in range(num_tasks)] # >> [bsz, num_views, output_dim] * num_tasks
            # output = [torch.mean(output[task_id], dim=1) for task_id in range(num_tasks)] # >> [bsz, output_dim] * num_tasks
            # output = torch.stack(output, dim=0)
            merged_output = [self.merge_views[task_id](task_output[task_id].view(bsz, -1)) for task_id in range(num_tasks)] # >> [bsz, output_dim] * num_tasks
            merged_output = torch.stack(merged_output, dim=0)
            # final_output = .5 * output + .5 * merged_output

            return merged_output, None
        else:
            # pass
            # Adaptive Data Augmentation for Unsupervised Part
            bsz, num_tasks, num_views, seq_len, sensor_dim = sup_x.size()
            unsup_bsz = unsup_x.size(0)
            # policy = self.train_sample_policy(self.temperature, True) # >> [num_tasks, 3]
            # unsup_x *= torch.empty(unsup_bsz, num_tasks, num_views, seq_len, sensor_dim, device=self.device).normal_(1, 1) * \
            #         policy[:, :, 0][None, :, :, None, None] + \
            #         torch.empty(unsup_bsz, num_tasks, num_views, seq_len, sensor_dim, device=self.device).uniform_(1, 2) * \
            #         policy[:, :, 1][None, :, :, None, None] + \
            #         torch.empty(unsup_bsz, num_tasks, num_views, seq_len, sensor_dim, device=self.device).log_normal_(0.5, 0.1) * \
            #         policy[:, :, 2][None, :, :, None, None]  # >> [unsup_bsz, ...]
            # unsup_x += .1
            x = torch.cat([sup_x, unsup_x], dim=0) # >> [bsz+unsup_bsz, ...]
            # x = x.contiguous().view(bsz+unsup_bsz, num_tasks, num_views, -1) # >> [batch_size, num_tasks, num_views, seq_len * sensor_dim]
            # x = x.reshape(bsz, num_tasks, seq_len, -1)
            output = [self.shared_bottom[view_id](x[:, :, view_id, ...]).squeeze(-1) for view_id in range(num_views)] # >> [bsz+unsup_bsz, num_tasks, seq_len] * num_views
            task_output = []
            for task_id in range(num_tasks):
                view_output = []
                for view_id in range(num_views):
                    y = self.specific_views[task_id][view_id](output[view_id][:, task_id, :]) # >> [bsz+unsup_bsz, hidden_dim_1]
                    view_output.append(y)
                view_output = torch.stack(view_output, dim=1) # >> [bsz+unsup_bsz, num_views, hidden_dim_1]
                task_output.append(view_output) # >> [bsz+unsup_bsz, num_views, hidden_dim_1] * num_tasks
            # output = [self.specific_tasks[task_id](task_output[task_id]) for task_id in range(num_tasks)] # >> [bsz+unsup_bsz, num_views, output_dim] * num_tasks
            # output = [torch.mean(output[task_id], dim=1) for task_id in range(num_tasks)] # >> [bsz+unsup_bsz, output_dim] * num_tasks
            # output = torch.stack(output, dim=0)
            merged_output = [self.merge_views[task_id](task_output[task_id].view(bsz+unsup_bsz, -1)) for task_id in range(num_tasks)] # >> [bsz+unsup_bsz, output_dim] * num_tasks
            merged_output = torch.stack(merged_output, dim=0)
            # final_output = .5 * output + .5 * merged_output

            return merged_output[:, :bsz, :], merged_output[:, bsz:, :] # >> [num_tasks, bsz, output_dim], [num_tasks, unsup_bsz, output_dim]
