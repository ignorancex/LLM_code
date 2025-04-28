import torch

def softplus_inv(x,beta=1,threshold=20):
    if x < threshold:
        return 1/beta*torch.log(torch.exp(x)-1)
    else:
        return x

def sigmoid(x, min = -1, max = 1):
    a = (max - min) / 2
    b = (max + min) / 2
    
    return torch.tanh(x)*a + b

def sigmoid_inv(x, min = -1, max = 1):
    a = (max - min) / 2
    b = (max + min) / 2
    
    return torch.arctanh((x-b)/a)