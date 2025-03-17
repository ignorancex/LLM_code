from layer_config_forward_oracle import MultiLayerPerceptron_forward
import torch
import numpy as np

class Oracle_function:
    def __init__(self):
        # Load the forward model
        input_size = 44
        hidden_size = [100, 250, 250, 100]
        self.num_classes = 8
        self.Models = []
        for n_net in range(0,10):
            Oracle_model = MultiLayerPerceptron_forward(input_size, hidden_size, self.num_classes)
            Oracle_model.load_state_dict(torch.load('../Oracle_Relu_ensemble/Models_44_inks/oracle_%d.ckpt' % (n_net), map_location=torch.device('cpu')))
            self.Models.append(Oracle_model)


    def Oracle_eval(self, designs):
        designs = torch.tensor(designs).float()
        Performance = torch.empty(10, designs.shape[0], self.num_classes)
        for net_n in range(10):
            Performance[net_n, :, :] = self.Models[net_n](designs)

        Performance_ensemble = (1 / 10) * torch.sum(Performance, 0)
        return np.array(Performance_ensemble.detach())