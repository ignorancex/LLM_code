import torch
import scipy.io as sio
import torch.nn as nn
import numpy as np
from pymoo.core.problem import Problem
from layer_config_forward import MultiLayerPerceptron_forward

class BO_surrogate_uncertainty(Problem):
    def __init__(self, n_iter):
        super().__init__(n_var=5,
                         n_obj=2,
                         xl=0.1,
                         xu=1)

        
        # Load the forward model
        input_size = 5
        hidden_size_mu = [150, 200, 200 , 150]
        self.num_classes = 2
        # Load the model
        self.mu_models = []
        for n_net in range(1, 11):
            # load mu models
            mu_model = MultiLayerPerceptron_forward(input_size, hidden_size_mu, self.num_classes, n_net)
            model_name = 'Models/iter_%d/mu_net_%d.ckpt' % (n_iter, n_net)
            mu_model.load_state_dict(torch.load(model_name, map_location=torch.device('cpu')))
            mu_model.eval()
            self.mu_models.append(mu_model)


    def _evaluate(self, design, out, *args, **kwargs):
        designs = torch.tensor(design).float()
        batchsize = designs.shape[0]
        reproduced_Performance_ensembel = torch.empty(10, batchsize, self.num_classes)

        reproduced_Performance_mu = torch.empty(batchsize, self.num_classes)
        uncertainty_epistemic = torch.empty((batchsize, self.num_classes))
        for net_n in range(10):
            reproduced_Performance_ensembel[net_n, :, :] = self.mu_models[net_n](designs)

        reproduced_Performance_mu = (1 / 10) * torch.sum(reproduced_Performance_ensembel, 0)
        uncertainty_epistemic = (1 / 10) * torch.sum(
            reproduced_Performance_ensembel ** 2 - reproduced_Performance_mu.repeat(10, 1, 1) ** 2,
            0)

        out["F"] = torch.cat((-reproduced_Performance_mu[:, :2].detach(), -uncertainty_epistemic[:, :2]), 1)
        # out["F"] = -reproduced_Performance_mu[:, :2]
        out["F"] = out["F"].detach().numpy()
