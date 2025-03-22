import scipy.io
import numpy as np
import torch
from layer_config_forward import MultiLayerPerceptron_forward
from layer_config_sigma import MultiLayerPerceptron_sigma
import scipy.io as sio
import time
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device: %s' % device)

# Load the forward model
input_size = 8
hidden_size_mu = [100,250,100]
num_classes = 8
# Load the model
mu_models = []
sigma_models = []
for n_net in range(10):
    #load mu models
    mu_model = MultiLayerPerceptron_forward(input_size, hidden_size_mu, num_classes)
    model_name = 'Models/mu_net_joint_%d.ckpt' % (n_net)
    mu_model.load_state_dict(torch.load(model_name))
    mu_model.to(device)
    mu_model.eval()
    mu_models.append(mu_model)



########################################### Ensembel 10 networks ################################
reproduced_Performance_ensembel = torch.empty(10, batchsize, num_classes*2).to(device)
uncertainty = torch.empty((10, batchsize)).to(device)
reproduced_Performance_mu = torch.empty(batchsize, num_classes).to(device)
uncertainty_epistemic = torch.empty((batchsize, num_classes)).to(device)

for net_n in range(10):
    reproduced_mu = mu_models[net_n](design)
    reproduced_sigma,_ = sigma_models[net_n](design)
    reproduced_Performance_ensembel[net_n, :, :] = torch.cat((reproduced_mu, reproduced_sigma), 1)

reproduced_Performance_mu = (1 / 10) * torch.sum(reproduced_Performance_ensembel[:, :, 0:num_classes], 0)
uncertainty_epistemic = (1 / 10) * torch.sum(
    reproduced_Performance_ensembel[:, :, 0:num_classes] ** 2 - reproduced_Performance_mu.repeat(10, 1, 1) ** 2, 0)
