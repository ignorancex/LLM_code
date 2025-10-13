from dataclasses import dataclass
from .base import Architecture
import torch
import torch.nn as nn
from torch.nn import Embedding, Linear
import math


@dataclass
class LSTM(Architecture):
    layers : int = 2
    embd_dim : int = 48
    hidden_dim: int = 48

    def create(self, input_shape, output_dim):        

        class LstmNetwork(nn.Module):
            def __init__(self, vocab_size: int, embedding_dim : int, hidden_dim: int, layers: int):
                self.vocab_size = vocab_size
                self.embedding_dim = embedding_dim
                self.hidden_dim = hidden_dim

                super().__init__()

                self.embed = Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
                self.lstm = StackedLSTM(input_size=embedding_dim, hidden_size = hidden_dim, num_layers=layers)
                self.linear = Linear(in_features=hidden_dim, out_features=vocab_size)

            def forward(self, x):
                embeddings = self.embed(x)
                features = self.lstm(embeddings)[0]
                return self.linear(features)
            
        return LstmNetwork(vocab_size=output_dim, embedding_dim=self.embd_dim, hidden_dim=self.hidden_dim, layers=self.layers)            
                
            
class StackedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(StackedLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.layers = nn.ModuleList()
        layer_input_size = input_size
        for _ in range(num_layers):
            self.layers.append(LSTMCell(layer_input_size, hidden_size))
            layer_input_size = hidden_size
            
    def forward(self, input, states=None):
        batch_size, seq_len, _ = input.size()
            
        if states is None:
            states = [None] * self.num_layers
            
        current_layer_input = input
        outputs = []
        
        for t in range(seq_len):
            current_timestep = current_layer_input[:, t, :]
            layer_states = []
            
            for layer_idx, layer in enumerate(self.layers):
                layer_state = states[layer_idx]
                out, new_layer_state = layer(current_timestep, layer_state)
                layer_states.append(new_layer_state)
                
                current_timestep = out
                if layer_idx == self.num_layers - 1:
                    outputs.append(current_timestep)
            
            states = layer_states
        
        outputs = torch.stack(outputs, dim=1)
        return outputs, states
  
  
class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.Wih = nn.Parameter(torch.Tensor(input_size, 4 * hidden_size))
        self.Whh = nn.Parameter(torch.Tensor(hidden_size, 4 * hidden_size))
        self.bih = nn.Parameter(torch.Tensor(4 * hidden_size))
        self.bhh = nn.Parameter(torch.Tensor(4 * hidden_size))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
            
    def forward(self, input, state=None):
        batch_size = input.size(0)
        
        if state is None:
            h_0 = torch.zeros(batch_size, self.hidden_size, device=input.device)
            c_0 = torch.zeros(batch_size, self.hidden_size, device=input.device)
        else:
            h_0, c_0 = state
            
        gates = (torch.matmul(input, self.Wih) + self.bih +
                torch.matmul(h_0, self.Whh) + self.bhh)
        
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, dim=1)
        
        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)
        
        c_1 = (forgetgate * c_0) + (ingate * cellgate)
        h_1 = outgate * torch.tanh(c_1)
        
        return h_1, (h_1, c_1)
