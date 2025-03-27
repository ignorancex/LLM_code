import torch, torch.nn as nn, numpy as np
from sklearn.metrics import accuracy_score, f1_score
import random

torch.backends.cudnn.deterministic = True
seed = 1
torch.manual_seed(seed)

def RM(Ytest, max_len, probs): #Random model
    """Random length predictor
    """
    np.random.seed(1)
    Pred = []
    for y in range(len(Ytest)):
        pred = np.random.choice(range(1, max_len+1), p=probs)
        Pred.append(pred)
    F1 = f1_score(Pred, Ytest, average='macro')
    Acc = accuracy_score(Pred, Ytest)
    print("Accuracy: {}, F1 score: {}".format(Acc, F1))

class LengthLearner_LSTM(nn.Module):
    
    def __init__(self, kwargs):
        super().__init__()
        self.name = 'LSTM'
        self.as_classification = kwargs['as_classification']
        self.lstm = nn.LSTM(kwargs['input_size'], kwargs['rnn_hidden'], kwargs['rnn_n_layers'], 
                            dropout=kwargs['dropout_prob'], batch_first=True)
        self.bn = nn.BatchNorm1d(kwargs['linear_hidden'])
        self.dropout = nn.Dropout(kwargs['dropout_prob'])
        self.fc1 = nn.Linear(kwargs['rnn_hidden'], kwargs['linear_hidden']) 
        self.fc2 = nn.Linear(kwargs['linear_hidden'], kwargs['out_size'])  
    
    def forward(self, x):
        ''' Forward pass through the network. 
            These inputs are x, and the hidden/cell state `hidden`. '''
                
        x, _ = self.lstm(x)
        x = x.sum(1).contiguous().view(x.shape[0], -1)
        x = self.fc1(x)
        x = torch.selu(x)
        x = self.bn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        if self.as_classification:
            x = torch.sigmoid(x)
        else:
            x = torch.selu(x)
        return x

class LengthLearner_GRU(nn.Module):
    
    def __init__(self, kwargs):
        super().__init__()
        self.name = 'GRU'
        self.as_classification = kwargs['as_classification']
        self.gru = nn.GRU(kwargs['input_size'], kwargs['rnn_hidden'], kwargs['rnn_n_layers'], dropout = kwargs['dropout_prob'], batch_first=True)
        self.bn = nn.BatchNorm1d(kwargs['linear_hidden'])
        self.dropout = nn.Dropout(kwargs['dropout_prob'])
        self.fc1 = nn.Linear(kwargs['rnn_hidden'], kwargs['linear_hidden'])
        self.fc2 = nn.Linear(kwargs['linear_hidden'], kwargs['out_size'])

    def forward(self, x):
        x, _ = self.gru(x)
        x = x.sum(1).contiguous().view(x.shape[0], -1)
        x = self.fc1(x)
        x = torch.selu(x)
        x = self.bn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        if self.as_classification:
            x = torch.sigmoid(x)
        else:
            x = torch.selu(x)
        return x

class LengthLearner_MLP(nn.Module):
    def __init__(self, kwargs):
        super().__init__()
        self.name = 'MLP'
        self.as_classification = kwargs['as_classification']
        linears = []
        random.seed(kwargs['seed'])
        layer_dims = [kwargs['input_size']] + [kwargs['num_units']//random.choice(range(1,10)) for _ in range(kwargs['mlp_n_layers']-1)]+[kwargs['out_size']]
        self.layer_dims = layer_dims
        self.__architecture__()
        for i in range(kwargs['mlp_n_layers']):
            if i == 0:
                linears.extend([nn.Linear(layer_dims[i], layer_dims[i+1]), nn.ReLU()])
            elif i < kwargs['mlp_n_layers']-1 and i%2==0:
                linears.extend([nn.Linear(layer_dims[i], layer_dims[i+1]), nn.ReLU()])
            elif i < kwargs['mlp_n_layers']-1 and i%2==1:
                linears.extend([nn.Linear(layer_dims[i], layer_dims[i+1]), nn.SELU(), nn.BatchNorm1d(layer_dims[i+1]), nn.Dropout(p=kwargs['dropout_prob'])])
            else:
                linears.extend([nn.Linear(layer_dims[i], layer_dims[i+1])])
        self.linears = nn.ModuleList(linears)
      
    def forward(self, x):
        for l in self.linears:
            x = x.view(x.shape[0], -1)
            x = l(x)
        if self.as_classification:
            x = torch.sigmoid(x)
        else:
            x = torch.selu(x)
        return x
    def __architecture__(self):
        print("MLP architecture:")
        print(self.layer_dims)

class LengthLearner_MLP2(nn.Module):
    def __init__(self, kwargs):
        super().__init__()
        self.name = 'MLP2'
        self.as_classification = kwargs['as_classification']
        linears = []
        random.seed(kwargs['seed'])
        layer_dims = [kwargs['input_size']] + [kwargs['num_units']//random.choice(range(1,10)) for _ in range(kwargs['mlp_n_layers']-1)]+[kwargs['out_size']]
        self.layer_dims = layer_dims
        self.__architecture__()
        for i in range(kwargs['mlp_n_layers']):
            if i == 0:
                linears.extend([nn.Linear(layer_dims[i], layer_dims[i+1]), nn.ReLU()])
            elif i < kwargs['mlp_n_layers']-1 and i%2==0:
                linears.extend([nn.Linear(layer_dims[i], layer_dims[i+1]), nn.ReLU()])
            elif i < kwargs['mlp_n_layers']-1 and i%2==1:
                linears.extend([nn.Linear(layer_dims[i], layer_dims[i+1]), nn.SELU(), nn.Dropout(p=kwargs['dropout_prob'])])
            else:
                linears.extend([nn.Linear(layer_dims[i], layer_dims[i+1])])
        self.linears = nn.ModuleList(linears)
      
    def forward(self, x):
        for l in self.linears:
            x = x.view(x.shape[0], x.shape[1], -1)
            x = l(x)
        if self.as_classification:
            x = torch.sigmoid(x.mean(1))
        else:
            x = torch.selu(x.mean(1))
        return x
    
    def __architecture__(self):
        print("MLP2 architecture:")
        print(self.layer_dims)
        

class LengthLearner_CNN(nn.Module):
    def __init__(self, kwargs):
        super().__init__()
        self.name = 'CNN'
        self.as_classification = kwargs['as_classification']
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(kwargs['kernel_h'],kwargs['kernel_w']), stride=(kwargs['stride_h'],kwargs['stride_w']), padding=(0,0))
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(kwargs['kernel_h']+1,kwargs['kernel_w']), stride=(kwargs['stride_h']+2,kwargs['stride_w']+1), padding=(0,0))
        self.dropout1d = nn.Dropout(kwargs['dropout_prob'])
        self.dropout2d = nn.Dropout2d(kwargs['dropout_prob'])
        self.bn = nn.BatchNorm1d(kwargs['conv_out']//5)
        self.fc1 = nn.Linear(in_features=kwargs['conv_out'], out_features=kwargs['conv_out']//5)
        self.fc2 = nn.Linear(kwargs['conv_out']//5, kwargs['out_size'])
        
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = torch.selu(x)
        x = self.dropout2d(x)
        x = self.conv2(x)
        x = x.view(x.shape[0], -1)
        #print("shape", x.shape)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.bn(x)
        x = self.dropout1d(x)
        x = self.fc2(x)
        if self.as_classification:
            x = torch.sigmoid(x)
        else:
            x = torch.selu(x)
        return x
    
class LengthLearner_Reformer(nn.Module):
    def __init__(self, kwargs):
        super().__init__()
        raise NotImplementedError
    
        
