import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from tqdm import tqdm

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

class DiscriminatorLawAw(nn.Module):
    def __init__(self, input_length: int, problem=None):
        super(DiscriminatorLawAw, self).__init__()
        self.problem = problem
        dim1 = 32
        dim2 = 16
        finaldim = 32
        self.hidden = torch.nn.Linear(input_length, dim1)   # hidden layer
        self.hidden1 = torch.nn.Linear(dim1, dim2)   # hidden layer
        self.hidden2 = torch.nn.Linear(dim2, finaldim)   # hidden layer
        self.predict = torch.nn.Linear(finaldim, 1)   # output layer
        self.dropout = nn.Dropout(0.95)
        self.batchnorm = nn.BatchNorm1d(dim1)
        self.batchnorm1 = nn.BatchNorm1d(dim2)
        self.batchnorm2 = nn.BatchNorm1d(finaldim)
        self.laynorm = nn.LayerNorm(1)

    def forward(self, x):
        x = F.leaky_relu(self.hidden(x))
        x = self.batchnorm(x)
        x = self.dropout(x)

        # x = F.leaky_relu(self.hidden1(x))
        # x = self.batchnorm1(x)
        # x = self.dropout(x)

        # x = F.leaky_relu(self.hidden2(x))
        # x = self.batchnorm2(x)
        # x = self.dropout(x)

        x = self.predict(x)
        return x

class DiscriminatorLaw(nn.Module):
    def __init__(self, input_length: int, problem=None):
        super(DiscriminatorLaw, self).__init__()
        self.problem = problem
        dim1 = 256
        dim2 = 128
        finaldim = 128
        self.hidden = torch.nn.Linear(input_length, dim1)   # hidden layer
        self.hidden1 = torch.nn.Linear(dim1, dim2)   # hidden layer
        self.hidden2 = torch.nn.Linear(dim2, finaldim)   # hidden layer
        self.predict = torch.nn.Linear(finaldim, 1)   # output layer
        self.dropout = nn.Dropout(0.5)
        self.batchnorm = nn.BatchNorm1d(dim1)
        self.batchnorm1 = nn.BatchNorm1d(dim2)
        self.batchnorm2 = nn.BatchNorm1d(finaldim)
        self.laynorm = nn.LayerNorm(1)

    def forward(self, x):
        x = F.leaky_relu(self.hidden(x))
        x = self.batchnorm(x)
        x = self.dropout(x)

        x = F.leaky_relu(self.hidden1(x))
        x = self.batchnorm1(x)
        x = self.dropout(x)

        for i in range(2):
            x = F.leaky_relu(self.hidden2(x))
            x = self.batchnorm2(x)
            x = self.dropout(x)

        x = self.predict(x)
        return x

class DiscriminatorAdultAw(nn.Module):
    def __init__(self, input_length: int, problem=None):
        super(DiscriminatorAdultAw, self).__init__()
        self.problem = problem
        dim1 = 128
        dim2 = 64
        finaldim = 64
        self.hidden = torch.nn.Linear(input_length, dim1)   # hidden layer
        self.hidden1 = torch.nn.Linear(dim1, dim2)   # hidden layer
        self.hidden2 = torch.nn.Linear(dim2, finaldim)   # hidden layer
        self.predict = torch.nn.Linear(finaldim, 2)   # output layer
        self.dropout = nn.Dropout(0.5)
        self.soft = nn.Softmax(dim=1)
        self.prelu = nn.PReLU(num_parameters=1, init=0.25)
        self.sig = nn.Sigmoid()
        self.batchnorm1 = nn.BatchNorm1d(128)
        self.batchnorm2 = nn.BatchNorm1d(64)
        self.laynorm = nn.LayerNorm(64)

    def forward(self, x):
        """

        :param x:
        :return:
        """

        x = F.relu(self.hidden(x))
        x = self.dropout(x)
        x = self.batchnorm1(x)

        x = F.relu(self.hidden1(x))
        x = self.dropout(x)
        x = self.batchnorm2(x)

        x = F.relu(self.hidden2(x))
        x = self.dropout(x)
        x = self.batchnorm2(x)

        x = self.predict(x)
        x = self.soft(x)

        return x

class DiscriminatorAdultAg(nn.Module):
    def __init__(self, input_length: int, problem=None):
        super(DiscriminatorAdultAg, self).__init__()
        self.problem = problem
        dim1 = 128
        dim2 = 64
        finaldim = 64
        self.hidden = torch.nn.Linear(input_length, dim1)   # hidden layer
        self.hidden1 = torch.nn.Linear(dim1, dim2)   # hidden layer
        self.hidden2 = torch.nn.Linear(dim2, finaldim)   # hidden layer
        self.predict = torch.nn.Linear(finaldim, 2)   # output layer
        self.dropout = nn.Dropout(0.5)
        self.soft = nn.Softmax(dim=1)
        self.prelu = nn.PReLU(num_parameters=1, init=0.25)
        self.sig = nn.Sigmoid()
        self.batchnorm1 = nn.BatchNorm1d(128)
        self.batchnorm2 = nn.BatchNorm1d(64)

    def forward(self, x):
        x = self.prelu(self.hidden(x))
        x = self.dropout(x)
        x = self.batchnorm1(x)
        x = self.prelu(self.hidden1(x))
        x = self.dropout(x)
        x = self.batchnorm2(x)
        for i in range(10):
            x = self.prelu(self.hidden2(x))
            x = self.dropout(x)
            x = self.batchnorm2(x)
        x = self.predict(x)
        x = self.soft(x)

        return x

class DiscriminatorCompasAw(nn.Module):
    def __init__(self, input_length: int, problem=None):
        super(DiscriminatorCompasAw, self).__init__()
        self.problem = problem
        dim1 = 64
        dim2 = 64
        finaldim = 64
        self.hidden = torch.nn.Linear(input_length, dim1)   # hidden layer
        self.hidden1 = torch.nn.Linear(dim1, dim2)   # hidden layer
        self.hidden2 = torch.nn.Linear(dim2, finaldim)   # hidden layer
        self.predict = torch.nn.Linear(finaldim, 2)   # output layer
        self.dropout = nn.Dropout(0.5)
        self.soft = nn.Softmax(dim=1)
        self.prelu = nn.PReLU(num_parameters=1, init=0.25)
        self.sig = nn.Sigmoid()
        self.batchnorm1 = nn.BatchNorm1d(dim1)
        self.batchnorm2 = nn.BatchNorm1d(dim2)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.dropout(x)
        x = self.batchnorm1(x)

        # x = F.relu(self.hidden1(x))
        # x = self.dropout(x)
        # x = self.batchnorm2(x)

        x = F.relu(self.hidden2(x))
        x = self.dropout(x)
        x = self.batchnorm2(x)

        x = self.predict(x)
        x = self.soft(x)

        return x

class DiscriminatorCompasAg(nn.Module):
    def __init__(self, input_length: int, problem=None):
        super(DiscriminatorCompasAg, self).__init__()
        self.problem = problem
        dim1 = 256
        dim2 = 128
        dim3 = 64
        dim4 = 32
        dim5 = 16
        dim6 = 8
        finaldim = 8
        self.hidden = torch.nn.Linear(input_length, dim1)   # hidden layer
        self.hidden1 = torch.nn.Linear(dim1, dim2)   # hidden layer
        self.hidden2 = torch.nn.Linear(dim2, dim3)   # hidden layer
        self.hidden3 = torch.nn.Linear(dim3, dim4)   # hidden layer
        self.hidden44 = torch.nn.Linear(dim4, dim4)   # hidden layer
        self.hidden4 = torch.nn.Linear(dim4, dim5)   # hidden layer
        self.hidden5 = torch.nn.Linear(dim5, dim6)   # hidden layer
        self.predict = torch.nn.Linear(finaldim, 2)   # output layer
        self.dropout = nn.Dropout(0.5)
        self.soft = nn.Softmax(dim=1)
        self.prelu = nn.PReLU(num_parameters=1, init=0.25)
        self.sig = nn.Sigmoid()
        self.batchnorm = nn.BatchNorm1d(dim1)
        self.batchnorm1 = nn.BatchNorm1d(dim2)
        self.batchnorm2 = nn.BatchNorm1d(dim3)
        self.batchnorm3 = nn.BatchNorm1d(dim4)
        self.batchnorm4 = nn.BatchNorm1d(dim5)
        self.batchnorm5 = nn.BatchNorm1d(dim6)

    def forward(self, x):

        x = self.batchnorm(self.dropout(self.prelu(self.hidden(x))))
        x = self.batchnorm1(self.dropout(self.prelu(self.hidden1(x))))
        x = self.batchnorm2(self.dropout(self.prelu(self.hidden2(x))))
        x = self.batchnorm3(self.dropout(self.prelu(self.hidden3(x))))
        # for i in range(10):
        #     x = self.batchnorm3(self.dropout(self.prelu(self.hidden44(x))))
        x = self.batchnorm4(self.dropout(self.prelu(self.hidden4(x))))
        x = self.batchnorm5(self.dropout(self.prelu(self.hidden5(x))))
        x = self.predict(x)
        x = self.soft(x)

        return x



def train_law(train_x, train_y):
    Net = DiscriminatorLaw(train_x.shape[1])
    data_set = TensorDataset(train_x, train_y)
    train_batches = DataLoader(data_set, batch_size=1024, shuffle=False)

    epochs = 120
    learning_rate = 1e-8

    loss_fn = torch.nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(Net.parameters(), lr=learning_rate)
    for i in tqdm(range(epochs)):
        for x_batch, y_batch in (train_batches):
            optimizer.zero_grad()
            loss = loss_fn(Net(x_batch), y_batch)
            loss.backward()
            optimizer.step()
    return Net

def train_classification(train_x, train_y):
    Net = DiscriminatorCompasAg(train_x.shape[1])
    data_set = TensorDataset(train_x, train_y)
    train_batches = DataLoader(data_set, batch_size=1024, shuffle=False)

    epochs = 120
    learning_rate = 1e-8

    normedWeights = torch.FloatTensor([0.45, 0.55])
    loss_fn = nn.CrossEntropyLoss(normedWeights)

    optimizer = torch.optim.Adam(Net.parameters(), lr=learning_rate)
    for i in tqdm(range(epochs)):
        for x_batch, y_batch in (train_batches):
            optimizer.zero_grad()
            loss = loss_fn(Net(x_batch), y_batch.reshape(-1))
            loss.backward()
            optimizer.step()
    return Net

