
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor



class DenseVanillaBlock(nn.Module):
    def __init__(self, in_features, out_features, bias=True, batch_norm=True, dropout=0., activation=nn.ReLU,
                 w_init_=lambda w: nn.init.kaiming_normal_(w, nonlinearity='relu')):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias)
        if w_init_:
            w_init_(self.linear.weight.data)
        self.activation = activation()
        self.batch_norm = nn.BatchNorm1d(out_features) if batch_norm else None
        self.dropout = nn.Dropout(dropout) if dropout else None

    def forward(self, input):
        
        input = self.linear(input)
        if self.batch_norm:
            input = self.batch_norm(input)
            
        input = self.activation(input)
        if self.dropout:
            input = self.dropout(input)
        return input

class MLPVanilla(nn.Module):
    def __init__(self, in_features, num_nodes, time_step, batch_norm=True, dropout=None, activation=nn.ReLU,
                 output_activation=None, output_bias=True, output_risk = 1,
                 w_init_=lambda w: nn.init.kaiming_normal_(w, nonlinearity='relu')):
        super().__init__()
        self.in_features = in_features
        self.time_step = time_step
        self.output_risk = output_risk
        
        num_nodes = [in_features] + num_nodes
        if not hasattr(dropout, '__iter__'):
            dropout = [dropout for _ in range(len(num_nodes)-1)]

        net = []
        for n_in, n_out, p in zip(num_nodes[:-1], num_nodes[1:], dropout):
            
            net.append(DenseVanillaBlock(n_in, n_out, True, batch_norm, p, activation, w_init_))
       
        net.append(nn.Linear(num_nodes[-1], output_risk, output_bias))
        if output_activation:
            net.append(output_activation)
        self.net = nn.Sequential(*net)
    
    def set_time_step(self, time_step):
        self.time_step = time_step

    def forward(self, input):
        time_step = input.shape[1]

        input = input.view(-1, self.in_features)
        out = self.net(input)
        if self.output_risk == 1:
            return out.reshape(-1, time_step)
        return out.reshape(-1, time_step, self.output_risk)

class MLPTimeEncode(nn.Module):
    def __init__(self, in_features, num_nodes, num_nodes_cs, time_dim, batch_norm=True, dropout=None, activation=nn.ReLU,
                 output_activation=None, output_bias=True, output_risk = 1,
                 w_init_=lambda w: nn.init.kaiming_normal_(w, nonlinearity='relu')):
        super().__init__()
        in_features = in_features + time_dim-1
        self.in_features = in_features
        self.output_risk = output_risk
        
        num_nodes = [in_features] + num_nodes
        if not hasattr(dropout, '__iter__'):
            dropout = [dropout for _ in range(len(num_nodes)-1)]

        net = []
        for n_in, n_out, p in zip(num_nodes[:-1], num_nodes[1:], dropout):
            
            net.append(DenseVanillaBlock(n_in, n_out, True, batch_norm, p, activation, w_init_))
        
        num_nodes_cs = [num_nodes[-1]+in_features] + num_nodes_cs
        net_cs = []
        for n_in, n_out, p in zip(num_nodes_cs[:-1], num_nodes_cs[1:], dropout):
            net_cs.append(DenseVanillaBlock(n_in, n_out, True, batch_norm, p, activation, w_init_))
        self.time_embedding = Time2Vec(1, time_dim)


        self.last_block = nn.Linear(num_nodes_cs[-1] , output_risk, output_bias)

        if output_activation:
            net.append(output_activation)
        self.net = nn.Sequential(*net)
        self.net_cs = nn.Sequential(*net_cs)
    
    def forward(self, input):

        time_step = input.shape[1]
        input = input.view(-1, input.shape[-1])
        time = self.time_embedding(input[:, -1].view(-1, 1))

        input = torch.cat([input[:, :-1], time.view(-1, time.shape[-1])], dim = 1)

        out = self.net(input)
        out = self.net_cs(torch.cat([input, out], dim=1))

        out = self.last_block(out)

        if self.output_risk == 1:

            return out.view(-1, time_step)
        return out.reshape(-1, time_step, self.output_risk)


class Time2Vec(nn.Module):
    def __init__(self, input_dim=6, embed_dim=512, act_function=torch.sin):
        assert embed_dim % input_dim == 0
        super(Time2Vec, self).__init__()
        self.enabled = embed_dim > 0
        if self.enabled:
            self.embed_dim = embed_dim // input_dim
            self.input_dim = input_dim
            self.embed_weight = nn.parameter.Parameter(
                torch.randn(self.input_dim, self.embed_dim)
            )
            self.embed_bias = nn.parameter.Parameter(
                torch.randn(self.input_dim, self.embed_dim)
            )
            self.act_function = act_function

    def forward(self, x):
        if self.enabled:
            x = torch.diag_embed(x)

            x_affine = torch.matmul(x, self.embed_weight) + self.embed_bias

            x_affine_0, x_affine_remain = torch.split(
                x_affine, [1, self.embed_dim - 1], dim=-1
            )
            x_affine_remain = self.act_function(x_affine_remain)
            x_output = torch.cat([x_affine_0, x_affine_remain], dim=-1)
            x_output = x_output.view(x_output.size(0), x_output.size(1), -1)
        else:
            x_output = x
        return x_output
    