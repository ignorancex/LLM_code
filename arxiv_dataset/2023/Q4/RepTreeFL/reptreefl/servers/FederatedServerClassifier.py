import torch
import torch.nn as nn
import copy
import random
from models.ResNet18 import ResNet10
import constants
from utils.utils import *

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("running on GPU")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("running on GPU MPS")
else:
    device = torch.device("cpu")
    print("running on CPU")

# manual_seed = 1773
random.seed(constants.MANUAL_SEED)
np.random.seed(constants.MANUAL_SEED)
torch.manual_seed(constants.MANUAL_SEED)

class FederatedServerClassifier():
    def __init__(self, run_index, no_classes=2):
        self.global_model = ResNet10(no_classes)
        self.global_model.to(device)

        self.global_model_param = get_mdl_params_resnet([self.global_model])[0]

        self.generator_save_path = f'{constants.SAVING_FOLDER_PATH}/{constants.RUN_NAME}/weights/global_weight_run{run_index}.model'
    
    def federated_average_intermodality(self, model_list):
        global_model_state = self.global_model.state_dict()
        for name, param in self.global_model.named_parameters():
            if "conv" in name:
                client_updates = [copy.deepcopy(model.get_generator().state_dict()[name].data) for model in model_list]
                averaged_update = torch.stack(client_updates).mean(dim=0)
                # print(averaged_update)
                global_model_state[name] = averaged_update
                # break

        # Update the global model with the aggregated parameters
        self.global_model.load_state_dict(global_model_state)

        return copy.deepcopy(self.global_model)
    
    def federated_diversity(self, model_list):
        l2_norms_clients = [l2_norm_models(self.global_model, client.get_generator()) for client in model_list]

        # Normalize weights to sum up to 1
        total_weight = sum(l2_norms_clients)
        normalized_weights = [w / total_weight for w in l2_norms_clients]

        # Aggregate model parameters based on the weights
        global_model_state = self.global_model.state_dict()
        for name, param in self.global_model.named_parameters():
            if "conv" in name:
                client_updates = [normalized_weights[i] * copy.deepcopy(client.get_generator().state_dict()[name].data) for i, client in enumerate(model_list)]
                averaged_update = torch.stack(client_updates).sum(dim=0)
                # print(averaged_update)
                global_model_state[name] = averaged_update
                # break

        # Update the global model with the aggregated parameters
        self.global_model.load_state_dict(global_model_state)

        return copy.deepcopy(self.global_model)

    def aggregate_parameters_feddyn(self, model_list):
        for param in self.global_model.parameters():
            param.data = torch.zeros_like(param.data)

        for param_name in model_list[0].get_generator().state_dict():
            if "conv" in param_name:
                self.global_model.state_dict()[param_name].data = torch.stack([model.get_generator().state_dict()[param_name].data.clone() for model in model_list]).sum(0) / len(model_list)

        for server_param, state_param in zip(self.global_model.parameters(), self.server_state.parameters()):
            server_param.data -= (1/constants.ALPHA_COEF) * state_param

    def update_server_state(self, model_list):
        model_delta = copy.deepcopy(self.global_model)
        for param in model_delta.parameters():
            param.data = torch.zeros_like(param.data)

        for client_model in model_list:
            for param_name in model_list[0].get_generator().state_dict():
                if "conv" in param_name:
                    model_delta.state_dict()[param_name].data += (client_model.get_generator().state_dict()[param_name] - self.global_model.state_dict()[param_name]) / constants.NUMBER_CLIENTS

        for state_param, delta_param in zip(self.server_state.parameters(), model_delta.parameters()):
            state_param.data -= constants.ALPHA_COEF * delta_param

    def get_global_model(self):
        return copy.deepcopy(self.global_model)

    def get_global_model_params(self):
        return copy.deepcopy(self.global_model_param)

    def update_global_model_feddyn(self, model_list):
        model_list_params = [get_mdl_params_resnet([model_list[model].get_generator()])[0] for model in range(len(model_list))]
        server_state = np.mean(model_list_params, axis=0)
        old_grad_param_list = [model_list[model].get_old_grad() for model in range(len(model_list))]
        self.global_model_param = server_state + np.mean(old_grad_param_list, axis=0)

        set_model_from_params_resnet(self.global_model, self.global_model_param)

    def save_model(self):
        torch.save(self.global_model.state_dict(), self.generator_save_path)

    def set_generator_saving_path(self, new_path):
        self.generator_save_path = new_path

