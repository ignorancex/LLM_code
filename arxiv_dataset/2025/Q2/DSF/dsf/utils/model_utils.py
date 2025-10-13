import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from ..dsf import RnnCell, StandardRnn, DfaRnn, NoBpttRnn

### Configuration class
class Config:
    def __init__(self, device, epochs, batch_size, max_length, learning_rate, weight_decay, scheduler, scheduler_kwargs, num_layers, hidden_size, activation, log_interval, use_dfa, state_transition, measure):
        
        # Training parameters
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_length = max_length
        
        # Optimizer parameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler = scheduler
        self.scheduler_kwargs = scheduler_kwargs
        
        # Model parameters
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.activation = activation if activation is not None else nn.Tanh()
        
        # Training options
        self.log_interval = log_interval
        self.use_dfa = use_dfa
        self.state_transition = state_transition
        self.measure = measure
        
    def __repr__(self):
        return f"Config(use_dfa={self.use_dfa}, state_transition={self.state_transition}, measure={self.measure})"


### Model Definition
class SimpleCell(RnnCell):
    def __init__(self, input_size, hidden_size):
        super(SimpleCell, self).__init__(input_size, hidden_size)
        self.cell = nn.RNNCell(input_size, hidden_size, nonlinearity="tanh")
        
        self._rnn = nn.RNN(input_size, hidden_size, batch_first=True, nonlinearity="tanh")  # tied equivalent rnn
        self._rnn.weight_ih_l0 = self.cell.weight_ih
        self._rnn.weight_hh_l0 = self.cell.weight_hh
        self._rnn.bias_ih_l0 = self.cell.bias_ih
        self._rnn.bias_hh_l0 = self.cell.bias_hh
        
    def forward(self, x, hidden):
        return self.cell(x, hidden)
    
    def rnn_forward(self, x):
        return self._rnn(x, hx=self.get_initial_state(x).unsqueeze(0))[0]
    
    def get_initial_state(self, x) -> torch.Tensor:
        return torch.zeros(x.size(0), self.hidden_dim, dtype=x.dtype, device=x.device)
    
class GRUCell(RnnCell):
    def __init__(self, input_size, hidden_size):
        super(GRUCell, self).__init__(input_size, hidden_size)
        self.cell = nn.GRUCell(input_size, hidden_size)
        
        self._rnn = nn.GRU(input_size, hidden_size, batch_first=True)  # tied equivalent rnn
        self._rnn.weight_ih_l0 = self.cell.weight_ih
        self._rnn.weight_hh_l0 = self.cell.weight_hh
        self._rnn.bias_ih_l0 = self.cell.bias_ih
        self._rnn.bias_hh_l0 = self.cell.bias_hh
        
    def forward(self, x, hidden):
        return self.cell(x, hidden)
    
    def rnn_forward(self, x):
        return self._rnn(x, hx=self.get_initial_state(x).unsqueeze(0))[0]
    
    def get_initial_state(self, x) -> torch.Tensor:
        return torch.zeros(x.size(0), self.hidden_dim, dtype=x.dtype, device=x.device)


class LSTMCell(RnnCell):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__(input_size, hidden_size)
        self.output_dim = hidden_size * 2
        self.cell = nn.LSTMCell(input_size, hidden_size)
        
    def forward(self, x, hidden):
        h, c = self.cell(x, (hidden[..., :self.hidden_dim], hidden[..., self.hidden_dim:]))
        return torch.concatenate((h, c), dim=-1)
    
    def get_initial_state(self, x) -> torch.Tensor:
        return torch.zeros(x.size(0), self.hidden_dim * 2, dtype=x.dtype, device=x.device)

CELL_TYPES = {
    'simple': SimpleCell,
    'gru': GRUCell,
    'lstm': LSTMCell,
}

class SelfAttentionLayer(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(SelfAttentionLayer, self).__init__()
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        
    def forward(self, x):
        return self.attn(x, x, x)[0]


def get_token_mixing_layer(config):
    if config.model.architecture != 'rnn' and config.model.state_transition != 'BPTT':
        raise ValueError("Only BPTT is supported for non-RNN architectures")

    if config.model.architecture == 'rnn':
        # create cell
        cell_cls = CELL_TYPES[config.model.cell_type]
        cell = cell_cls(config.model.hidden_size, config.model.hidden_size)

        # wrap it with the desired RNN type
        if config.model.use_dfa and config.model.state_transition in ["SSM", "diagonal"]:
            rnn = DfaRnn(cell, config)
        elif config.model.use_dfa and config.model.state_transition == "FT_BPTT":
            rnn = NoBpttRnn(cell) 
        elif not config.model.use_dfa and config.model.state_transition == "BPTT":
            rnn = StandardRnn(cell) 
        else:
            raise ValueError("Invalid configuration")
        
        return rnn, cell.output_dim
        
    elif config.model.architecture == 'attention':
        attn = SelfAttentionLayer(config.model.hidden_size, config.model.num_heads)
        return attn, config.model.hidden_size
    
    else:
        raise ValueError(f"Invalid architecture: {config.model.architecture}")

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.use_skip_connection = config.model.skip_connections
        self.transformer_like = config.model.transformer_like

        self.embedding = nn.Embedding(config.dataset.vocab_size, config.model.hidden_size)

        self.layers = nn.ModuleList()
        for _ in range(config.model.num_layers):

            layer, output_dim = get_token_mixing_layer(config)

            if not self.transformer_like:
                # plain rnn
                self.layers.append(layer)
            
            else:
                # rnn with layernorm and post-projection
                wrapped_layer = [
                    nn.LayerNorm(config.model.hidden_size),
                    layer,
                ]
                if config.model.architecture == 'rnn':
                    # add a linear layer after the rnn
                    wrapped_layer.append(nn.Linear(output_dim, config.model.hidden_size))
                self.layers.append(nn.Sequential(*wrapped_layer))

                # and MLP layer
                self.layers.append(nn.Sequential(
                    nn.LayerNorm(config.model.hidden_size),
                    nn.Linear(config.model.hidden_size, config.model.hidden_size * 4),
                    nn.GELU(),
                    nn.Linear(config.model.hidden_size * 4, config.model.hidden_size),
                ))

        if self.transformer_like:
            self.ln_final = nn.LayerNorm(config.model.hidden_size)

        if config.dataset.task_type == 'language_modeling':
            self.classification_head = nn.Linear(config.model.hidden_size, config.dataset.vocab_size)
            self.output_sequence = True
        elif config.dataset.task_type == 'sequence_classification':
            self.classification_head = nn.Linear(config.model.hidden_size, config.dataset.num_classes)
            self.output_sequence = False
        else:
            raise ValueError(f"Invalid task type: {config.dataset.task_type}")
        
        self.reset_parameters()

    def reset_parameters(self):
        init_std = 0.02

        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=init_std)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=init_std)
            elif isinstance(module, nn.MultiheadAttention):
                module.in_proj_weight.data.normal_(mean=0.0, std=init_std)
                module.out_proj.weight.data.normal_(mean=0.0, std=init_std)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        
        # special initialization like GPT-2: divide the last linear of each layer by sqrt of the number of layers
        for module in self.modules():
            if isinstance(module, nn.Sequential):
                if isinstance(module[-1], nn.Linear):
                    output_proj_weight = module[-1].weight
                elif isinstance(module[-1], SelfAttentionLayer):
                    output_proj_weight = module[-1].attn.out_proj.weight
                else:
                    raise ValueError("Invalid module")

                output_proj_weight.data.normal_(mean=0.0, std=(init_std / len(self.layers) ** 0.5))
        
    def forward(self, x):
        x = self.embedding(x)

        for rnn in self.layers:
            residual = x

            x = rnn(x)

            if self.use_skip_connection:
                x = x + residual
                
        if self.transformer_like:
            x = self.ln_final(x)
        
        if not self.output_sequence:
            x = x[:, -1]  # only take the last state

        x = self.classification_head(x)
        return x









