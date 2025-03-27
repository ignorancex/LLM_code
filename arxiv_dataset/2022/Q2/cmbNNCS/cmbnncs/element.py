import torch.nn as nn


#%% activation
def relu():
    #here 'inplace=True' is used to save GPU memory
    return nn.ReLU(inplace=True)

def leakyrelu():
    return nn.LeakyReLU(inplace=True)

def prelu():
    return nn.PReLU()

def rrelu():
    return nn.RReLU(inplace=True)

def relu6():
    return nn.ReLU6(inplace=True)

#
def elu():
    return nn.ELU(inplace=True)

def selu():
    return nn.SELU(inplace=True)

#
def sigmoid():
    return nn.Sigmoid()

def tanh():
    return nn.Tanh()

def softsign():
    return nn.Softsign()

def tanhshrink():
    return nn.Tanhshrink()

def softshrink():
    return nn.Softshrink()

def hardtanh():
    return nn.Hardtanh()

def hardshrink():
    return nn.Hardshrink()

def softplus():
    return nn.Softplus()


def activation(active_name='relu'):
    return eval('%s()'%active_name)

#%% Pooling
def maxPool1d():
    return nn.MaxPool1d(kernel_size=2)

def maxPool2d():
    return nn.MaxPool2d(kernel_size=2)

def maxPool3d():
    return nn.MaxPool3d(kernel_size=2)

def avgPool1d():
    return nn.AvgPool1d(kernel_size=2)

def avgPool2d():
    return nn.AvgPool2d(kernel_size=2)

def avgPool3d():
    return nn.AvgPool3d(kernel_size=2)

def maxUnpool2d():
    return nn.MaxUnpool2d(kernel_size=2)

def pooling(pool_name='maxPool2d'):
    return eval('%s()'%pool_name)


#%% Dropout
def dropout():
    return nn.Dropout(inplace=False)

def dropout2d():
    return nn.Dropout2d(inplace=False)

def dropout3d():
    return nn.Dropout3d(inplace=False)

def get_dropout(drouput_name='dropout'):
    return eval('%s()'%drouput_name)

