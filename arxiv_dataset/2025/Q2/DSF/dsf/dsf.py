import abc
from typing import Tuple

import torch 
from torch import nn 


class RnnCell(nn.Module, abc.ABC):
    """ Abstract class for a RNN cell. """
    def __init__(self, input_dim: int, hidden_dim: int):
        super(RnnCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = hidden_dim

    @abc.abstractmethod
    def forward(self, x: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """ Forward pass of the RNN cell. Computes h_t = f(x_t, h_{t-1}). """
        pass

    @abc.abstractmethod 
    def get_initial_state(self, x: torch.Tensor) -> torch.Tensor:
        """ Returns the initial state of the RNN cell. """
        pass


class StandardRnn(nn.Module):
    """ A standard RNN that computes the hidden states for each time step. """
    def __init__(self, rnn_cell: RnnCell):
        super(StandardRnn, self).__init__()
        self.rnn_cell = rnn_cell

    def forward(self, x):
        if hasattr(self.rnn_cell, 'rnn_forward'):
            return self.rnn_cell.rnn_forward(x)

        batch_size, seq_len, dim = x.shape
        state = self.rnn_cell.get_initial_state(x)
        states = []
        for t in range(seq_len):
            state = self.rnn_cell(x[:, t], state) # x[:, t] = x[:, t, :] PyTorch automatically expands the tensor
            states.append(state)

        return torch.stack(states, dim=1)


class NoBpttRnn(nn.Module):
    """ A standard RNN without backpropagation through time.
        Main point: 
           - detach() at the end of the forward
           - parameter update: up to you. 
    """
    def __init__(self, rnn_cell: RnnCell):
        super(NoBpttRnn, self).__init__()
        self.rnn_cell = rnn_cell

    def forward(self, x):
        batch_size, seq_len, dim = x.shape

        if hasattr(self.rnn_cell, 'rnn_forward'):
            # get hidden states with fast method in no grad
            h0 = self.rnn_cell.get_initial_state(x)
            with torch.no_grad():
                states = self.rnn_cell.rnn_forward(x)
            states = torch.cat([h0.unsqueeze(1), states[:, :-1]], dim=1)

            # second parallel forward to detach the hidden states
            states = self.rnn_cell(x.view(-1, dim), states.view(-1, states.size(2)))
            states = states.view(batch_size, seq_len, -1)
            return states
        
        state = self.rnn_cell.get_initial_state(x)
        states = []
        for t in range(seq_len):
            state = self.rnn_cell(x[:, t], state)
            states.append(state)
            state = state.detach()  # detach the state to prevent backpropagation through time

        return torch.stack(states, dim=1)
        
        
class ParallelBackwardRnn(nn.Module):
    """ An Abstract RNN that computes the hidden states for each time step and approximates backpropagation through time. 
    - detach() at the end of the forward
    - See the registered hook in the forward
    - get_state_gradients: to be implemented
    """
    def __init__(self, rnn_cell: RnnCell):
        super(ParallelBackwardRnn, self).__init__()
        self.rnn_cell = rnn_cell

    def forward(self, x):
        batch_size, seq_len, dim = x.shape
        
        if hasattr(self.rnn_cell, 'rnn_forward'):
            # get hidden states with fast method in no grad
            h0 = self.rnn_cell.get_initial_state(x)
            with torch.no_grad():
                states = self.rnn_cell.rnn_forward(x)
            states = torch.cat([h0.unsqueeze(1), states[:, :-1]], dim=1)

            # second parallel forward to detach the hidden states
            states = self.rnn_cell(x.view(-1, dim), states.view(-1, states.size(2)))
            outputs = states.view(batch_size, seq_len, -1)
        
        else:
            state = self.rnn_cell.get_initial_state(x)
            states = []
            for t in range(seq_len):
                state = self.rnn_cell(x[:, t], state)
                states.append(state)
                state = state.detach()  # detach the state to prevent backpropagation through time
            outputs = torch.stack(states, dim=1)

        # use get_state_gradients to approximate backpropagation through time
        if outputs.requires_grad:
            @torch.no_grad()
            def hook(grad):
                return self.get_state_gradients(grad)
            outputs.register_hook(hook)
        return outputs
    
    @abc.abstractmethod
    def get_state_gradients(self, output_gradients: torch.Tensor) -> torch.Tensor:
        """ Computes the gradients of the loss with respect to the RNN states.
        """
        pass


class DfaRnn(ParallelBackwardRnn):
    """ A RNN that approximates backpropagation through time using DFA. 
    """
    def __init__(self, rnn_cell: RnnCell , config, *args, **kwargs):
        super(DfaRnn, self).__init__(rnn_cell, *args, **kwargs)
        self.rnn_cell = rnn_cell
        self.config = config
        h_dim = self.rnn_cell.output_dim
        
        A = self.initialize_feedback_matrix(h_dim, config)
        self.register_buffer("A", A)  # Store as a tensor buffer

    def get_state_gradients(self, output_gradients: torch.Tensor) -> torch.Tensor:
        """ 
        Gather approximated gradients after the forward pass. 
        returns a Tensor of dims (batch size ,time steps , nb params of the cell)
        """
        seq_len = output_gradients.shape[1]
        gt = output_gradients[:, seq_len-1]
        gradients = [gt] # List of tensors, one tensor per time step
        for t in range(seq_len-2, -1, -1):
            gt = output_gradients[:, t] + torch.matmul(gt, self.A)
            gradients.append(gt)
        return torch.stack(gradients[::-1], dim=1)
    
    @staticmethod
    def initialize_feedback_matrix(h_dim: int, config):
        """
        Initialize the feedback matrix A with different strategies.
        Returns:
            torch.Tensor: Initialized feedback matrix A.
        """
        if config.model.state_transition == 'gaussian':
            A = torch.randn(h_dim, h_dim) / (h_dim ** 0.5)

        elif config.model.state_transition == 'orthogonal':
            A, _ = torch.qr(torch.randn(h_dim, h_dim))

        elif config.model.state_transition == 'diagonal':
            A = torch.diag(torch.linspace(0.0, 1.0, h_dim))
        else:
            raise ValueError(f"Unknown initialization method: {config.model.state_transition}")

        return A  




