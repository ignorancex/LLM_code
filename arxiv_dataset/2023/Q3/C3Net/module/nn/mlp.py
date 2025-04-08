from torch import nn
from module.nn.base import Dense

__all__ = ['MLP']
class MLP(nn.Module):
    def __init__(self, n_in, n_out, n_layers=2, activation=True):
        super(MLP, self).__init__()
        # assign a dense layer (with activation function) to each hidden layer
        layers = []
        n_neurons_in = n_in
        for i in range(n_layers-1):
            n_neurons_out = n_neurons_in//2
            layers.append(Dense(n_neurons_in, n_neurons_out, activation=activation))
            n_neurons_in = n_neurons_out
            
        # assign a Dense layer (without activation function) to the output layer
        layers.append(Dense(n_neurons_in, n_out, activation=None))
        # put all layers together to make the network
        self.out_net = nn.Sequential(*layers)

    
    def forward(self, inputs):
        return self.out_net(inputs)