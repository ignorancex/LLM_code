import torch
import torch.nn.functional as F

class Convolutional_Neural_Net(torch.nn.Module):
    def __init__(self):
        super(Convolutional_Neural_Net, self).__init__()
        self.conv1 = torch.nn.Conv1d(1, 4, 5)
        self.pool1 = torch.nn.MaxPool1d(5,stride = 1)
        self.lin1 = torch.nn.Linear(504,64)
        self.lin2 = torch.nn.Linear(64,8)
        self.predict = torch.nn.Linear(32,1)   # output layer

    def forward(self, x):
        x = F.relu(self.conv1(x))

        x = self.pool1(x)

        x = F.relu(self.lin1(x))

        x = F.relu(self.lin2(x))
        
        x = x.view((-1,1,32))
        
        x = self.predict(x)
        
        return x