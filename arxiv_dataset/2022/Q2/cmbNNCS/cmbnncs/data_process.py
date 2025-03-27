import torch

def numpy2torch(data):
    dtype = torch.FloatTensor
    data = torch.from_numpy(data).type(dtype)
    return data

def numpy2cuda(data, device=None):
    if device is None:
        dtype = torch.cuda.FloatTensor
        data = torch.from_numpy(data).type(dtype)
    else:
        data = numpy2torch(data)
        data = torch2cuda(data, device=device)
    return data

def torch2cuda(data, device=None):
    return data.cuda(device=device)

def torch2numpy(data):
    return data.numpy()

def cuda2torch(data):
    return data.cpu()

def cuda2numpy(data):
    return data.cpu().numpy()
