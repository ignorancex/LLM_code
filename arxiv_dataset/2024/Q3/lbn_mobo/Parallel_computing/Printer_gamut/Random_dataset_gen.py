# from My_siren import *
import torch
from layer_config_forward_oracle import MultiLayerPerceptron_forward
import numpy as np



def random_dset_gen(model_name):

    init_samples = 10000
    device = torch.device('cpu')
    print('Using device: %s' % device)
    ##################################################################
    # Load the ground-truth model
    input_size = 44
    hidden_size = [100, 250, 250, 100]
    num_classes = 8
    Models = []
    for n_net in range(0,10):
        Oracle_model = MultiLayerPerceptron_forward(input_size, hidden_size, num_classes)
        Oracle_model.load_state_dict(torch.load(model_name + '/oracle_%d.ckpt' % (n_net), map_location=torch.device('cpu')))
        Models.append(Oracle_model)
    #################################################################
    # Query the ground-truth model
    torch.manual_seed(100)
    random_haftone = torch.rand(init_samples,input_size).float().to(device)
    Performance = torch.empty(10, random_haftone.shape[0], num_classes)
    for net_n in range(10):
        Performance[net_n, :, :] = Models[net_n](random_haftone)

    Performance_ensemble = (1 / 10) * torch.sum(Performance, 0)

    random_haftone = random_haftone.cpu().detach().numpy()
    AB_Data = Performance_ensemble.cpu().detach().numpy()
    data = {'X':random_haftone, 'Y':AB_Data} #to feed the NSM with normalized data
    return data
