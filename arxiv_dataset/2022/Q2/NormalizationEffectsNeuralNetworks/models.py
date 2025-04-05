import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Multi-layer perceptron with a single hidden layer
    
    Attributes
    ----------
    hidden_units: int
        the number of nodes in the hidden layer
    gamma: float
        the mean-field scaling parameter
    """
    
    def __init__(self, hidden_units, gamma):
        super(MLP, self).__init__()
        
        # Parameters
        self.hidden_units = hidden_units
        self.gamma = gamma
        
        # Layers
        self.fc1 = nn.Linear(28 * 28, hidden_units)
        nn.init.normal_(self.fc1.weight, mean=0.0, std=1.0)
        self.fc2 = nn.Linear(hidden_units, 10)
        nn.init.uniform_(self.fc2.weight, a=0.0, b=1.0)
    
    def forward(self, x):
        x = x.view(-1, 28 * 28)
        scaling = self.hidden_units ** (-self.gamma)
        x = scaling * torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x


class CNN(nn.Module):
    """Convolutional neural network with 8 convolution layers and a single 
    hidden layer
    
    Attributes
    ----------
    hidden_units: int
        the number of nodes in the hidden layer
    gamma: float
        the mean-field scaling parameter
    
    Methods
    -------
    scale_learning_rates()
        scales the learning rates for all model parameters
    """
    
    def __init__(self, hidden_units, gamma):
        super(CNN, self).__init__()
        
        # Parameters
        self.channels = 64
        self.input_dimension = 32 * 32 * 3
        self.hidden_units = hidden_units
        self.gamma = gamma
        
        # Learning rate scaling parameters
        H0 = float(self.channels)
        H = float(self.hidden_units)
        self.LR_layer_list = [
            1.0, 1.0, 
            H0 / ((H0 ** (1.0-2.0*gamma)) * (H0 ** (2.0-gamma))), H0 / (H0 ** (2.0-gamma)), 
            H0 / ((H0 ** (1.0-2.0*gamma)) * (H0 ** (2.0-gamma))), H0 / (H0 ** (2.0-gamma)), 
            H0 / ((H0 ** (1.0-2.0*gamma)) * (H0 ** (2.0-gamma))), H0 / (H0 ** (2.0-gamma)), 
            H0 / ((H0 ** (1.0-2.0*gamma)) * (H0 ** (2.0-gamma))), H0 / (H0 ** (2.0-gamma)), 
            H0 / ((H0 ** (1.0-2.0*gamma)) * (H0 ** (2.0-gamma))), H0 / (H0 ** (2.0-gamma)), 
            H0 / ((H0 ** (1.0-2.0*gamma)) * (H0 ** (2.0-gamma))), H0 / (H0 ** (2.0-gamma)), 
            (4.0 * 4.0 * H0) / ((H0 ** (1.0-2.0*gamma)) * (H0 ** (2.0-gamma))), (4.0 * 4.0 * H0) / (H0 ** (2.0-gamma)), 
            1.0 / (H0 ** (4.0-4.0*gamma)), 1.0 / (H0 ** (4.0-4.0*gamma)), 1.0 / (H0 ** (4.0-4.0*gamma)), 1.0 / (H0 ** (4.0-4.0*gamma)), 1.0 / (H0 ** (4.0-4.0*gamma)), 1.0 / (H0 ** (4.0-4.0*gamma)), 1.0 / (H0 ** (4.0-4.0*gamma)), 1.0 / (H0 ** (4.0-4.0*gamma)), 1.0 / (H0 ** (4.0-4.0*gamma)), 1.0 / (H0 ** (4.0-4.0*gamma)),
            (4.0 * 4.0 * H0) / ((H ** (1.0-2.0*gamma)) * (H0 ** (2.0-gamma))), H / (H0 ** (2.0-gamma)),
            H / (H0 ** (2.0-gamma)), 1.0 / (H0 ** (2.0-gamma))]
        
        # Layers
        self.conv1 = nn.Conv2d(3, self.channels, 4, padding=(2, 2))
        self.conv2 = nn.Conv2d(self.channels, self.channels, 4, padding=(2, 2))
        self.conv3 = nn.Conv2d(self.channels, self.channels, 4, padding=(2, 2))
        self.conv4 = nn.Conv2d(self.channels, self.channels, 4, padding=(2, 2))
        self.conv5 = nn.Conv2d(self.channels, self.channels, 4, padding=(2, 2))
        self.conv6 = nn.Conv2d(self.channels, self.channels, 3)
        self.conv7 = nn.Conv2d(self.channels, self.channels, 3)
        self.conv8 = nn.Conv2d(self.channels, self.channels, 3)        
        
        self.pool = nn.MaxPool2d(2, 2)
        
        self.dropout1 = torch.nn.Dropout(p=0.5)
        self.dropout2 = torch.nn.Dropout(p=0.5)  
        self.dropout3 = torch.nn.Dropout(p=0.5) 
        self.dropout4 = torch.nn.Dropout(p=0.5)
        self.dropout5 = torch.nn.Dropout(p=0.5) 
        self.dropout6 = torch.nn.Dropout(p=0.5) 
        self.dropout7 = torch.nn.Dropout(p=0.5) 
        self.dropout8 = torch.nn.Dropout(p=0.5) 
        
        self.Bnorm1 = torch.nn.BatchNorm2d(self.channels)
        self.Bnorm2 = torch.nn.BatchNorm2d(self.channels)
        self.Bnorm3 = torch.nn.BatchNorm2d(self.channels)
        self.Bnorm4 = torch.nn.BatchNorm2d(self.channels)
        self.Bnorm5 = torch.nn.BatchNorm2d(self.channels)
        
        self.fc1 = nn.Linear(4 * 4 * self.channels, self.hidden_units)
        self.fc2 = nn.Linear(self.hidden_units, 10)

    def forward(self, x):
        
        x = (self.channels ** (-self.gamma)) * self.Bnorm1(F.relu(self.conv1(x)))
        x = (self.channels ** (-self.gamma)) * F.relu(self.conv2(x)) 
        p = self.dropout1(self.pool(x))
        
        x = (self.channels ** (-self.gamma)) * self.Bnorm2(F.relu(self.conv3(p)))
        x = (self.channels ** (-self.gamma)) * F.relu(self.conv4(x))  
        p = self.dropout2(self.pool(x))
        
        x = (self.channels ** (-self.gamma)) * self.Bnorm3(F.relu(self.conv5(p)))
        x = (self.channels ** (-self.gamma)) * self.dropout3(F.relu(self.conv6(x)))
        x = (self.channels ** (-self.gamma)) * self.Bnorm4(F.relu(self.conv7(x)))
        x = (float(self.channels * 4 * 4) ** (-self.gamma)) * self.Bnorm5(F.relu(self.conv8(x)))
        
        x = self.dropout6(x)
        
        x = x.view(-1, 4 * 4 * self.channels)
        
        x = (self.hidden_units ** (-self.gamma)) * F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
    
    def scale_learning_rates(self):
        for counter, param in enumerate(self.parameters()):
            param.grad.data = param.grad.data * self.LR_layer_list[counter]
        return
    
    def initialize_parameters(self):            
        for param in self.parameters():
            param.data = 1.0 * torch.normal(torch.zeros(param.data.shape))
        return
