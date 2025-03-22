import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F

class OWM_FC_block(nn.Module):
    def __init__(self, in_features, out_features):
        super(OWM_FC_block, self).__init__()
        self.W = nn.Linear(in_features, out_features, bias = False)
        self.P = nn.Linear(in_features, in_features, bias = False)
        
        with torch.no_grad():
            nn.init.xavier_uniform_(self.W.weight, gain = nn.init.calculate_gain('relu'))
            self.P.weight.data = torch.eye(in_features)

    def forward(self, x):
        x = self.W(x)
        return x

# define net module
class owm_fc(nn.Module):
    def __init__(self, num_hidden, num_classes, num_in = 784, no_actfun = False):
        super(owm_fc, self).__init__()
        self.layer_1 = OWM_FC_block(num_in, num_hidden)
        self.afun_1 = nn.ReLU()
        self.layer_2 = OWM_FC_block(num_hidden, num_classes)
        self.no_actfun = no_actfun
        self.type = 'fc'
        
    def forward(self, x):
        hidden_1 = self.layer_1(x)

        hidden_2 = self.layer_2(hidden_1) if self.no_actfun else self.layer_2(self.afun_1(hidden_1))

        return hidden_2, hidden_1

__factory = {
    'owm_fc': owm_fc,
}

def create(name, num_hidden, num_classes, num_in = 784, no_actfun = False):
    if name not in __factory.keys():
        raise KeyError("Unknown model: {}".format(name))
    return __factory[name](num_hidden, num_classes, num_in, no_actfun)