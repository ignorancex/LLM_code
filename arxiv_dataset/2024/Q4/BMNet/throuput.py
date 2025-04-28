import os
import numpy as np
import torch
torch.set_default_tensor_type(torch.DoubleTensor)
from data.dotadataset import make_dataset
from torch.utils.data import DataLoader
from network.BMNet import BMNet
from thop import profile


repetitions = 100

model = BMNet(num_stage=10, cs_ratio=(2, 2)).cuda()

# n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print('number of params (M): %.2f' % (n_parameters / 1.e6))

x = torch.rand((1, 1, 1024, 1024)).cuda()
y = torch.rand((1, 1, 1, 512, 512)).cuda()
Phi = torch.rand((1, 4, 1, 512, 512)).cuda()

flops, params = profile(model, inputs=(y, Phi))
print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
print('Params = ' + str(params / 1000 ** 2) + 'M')

starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
timings = np.zeros((repetitions, 1))

torch.cuda.synchronize()
print('testing ...\n')
with torch.no_grad():
    for rep in range(repetitions):
        starter.record()
        _ = model(y, Phi)
        ender.record()
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        timings[rep] = curr_time

avg = timings.sum()/repetitions
print('\navg={}\n'.format(avg))

