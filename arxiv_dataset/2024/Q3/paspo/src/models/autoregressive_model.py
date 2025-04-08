import copy
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from utils.lp_sub_proc import LPSubProc
from utils.uniform_init import uniform_params
from utils.polytope_loader import load_polytope

import torch

def _get_activation_fn(activation):
    if activation == "tanh":
        return torch.nn.Tanh
    elif activation == "relu":
        return torch.nn.ReLU
    else:
        raise ValueError(f"Unknown activation function: {activation}")


class TorchAutoregressiveActionModel(TorchModelV2, torch.nn.Module):
    """PyTorch version of the AutoregressiveActionModel above."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, *args, **kwargs):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        torch.nn.Module.__init__(self)
        
        
        self.use_r_sample = False 
        self.n_dimensions = action_space.shape[0]
        self.add_only_action = model_config.get("custom_model_config").get("add_only_action")
        

        A,b = load_polytope(n_dim=self.n_dimensions,
                            storage_method=model_config.get("custom_model_config").get("polytope_storage_method"), 
                            generation_method=model_config.get("custom_model_config").get("polytope_generation_method"), 
                            polytope_generation_data=model_config.get("custom_model_config").get("polytope_generation_data"))
        
        self.A = A
        self.b = b
        
                
        self.state_encoder_hidden_dim = num_outputs
        
        layers = []
        
        activation_fn = _get_activation_fn(model_config.get("fcnet_activation"))
        

        prev_layer_size = obs_space.shape[0]
        for hidden_size in model_config.get("fcnet_hiddens")[:-1]:
            layers.append(torch.nn.Linear(prev_layer_size, hidden_size))
            layers.append(activation_fn())
            prev_layer_size = hidden_size
            
        layers.append(torch.nn.Linear(prev_layer_size, self.state_encoder_hidden_dim))
        if model_config.get("custom_model_config").get("fcnet_use_activation_function_after_last_layer"):
            layers.append(activation_fn())

        self.state_encoder = torch.nn.Sequential(*layers)
            
        
        self.vf_share = model_config.get("vf_share_layers")
        
        if self.vf_share:
            # V(s)
            self.value_branch = torch.nn.Linear(num_outputs, 1)
        else:
            layers = []            

            prev_layer_size = obs_space.shape[0]
            for hidden_size in model_config["custom_model_config"]["fcnet_critic_hiddens"][:-1]:
                layers.append(torch.nn.Linear(prev_layer_size, hidden_size))
                layers.append(activation_fn())
                prev_layer_size = hidden_size

            layers.append(torch.nn.Linear(prev_layer_size, 1))
            
            self.value_net = torch.nn.Sequential(*layers)
            
                        
            
        self.uniform_bias_init = model_config.get("custom_model_config").get("uniform_bias_init")
        
        self.action_dist_name = model_config.get("custom_model_config").get("action_dist_name")
        
        if self.uniform_bias_init:
            self.bias_params = uniform_params(self.A, self.b, self.n_dimensions, self.action_dist_name, n_points=model_config.get("custom_model_config").get("uniform_bias_number_of_samples_in_polytope"))
            _inversion_bias = torch.log #TODO do not hardcode
            
            
            
        modules = []
        for dim in range(self.n_dimensions - 1):
            
            layers = []
            prev_layer_size = num_outputs + (dim * (1 if self.add_only_action else 5))
            
            activation_fn = _get_activation_fn(model_config.get("custom_model_config").get("fc_hiddens_uniform_autoreg_branches_activation"))
            
            for hidden_size in model_config.get("custom_model_config").get("fc_hiddens_uniform_autoreg_branches"):
                layer = torch.nn.Linear(prev_layer_size, hidden_size)
                
                layers.append(layer)
                layers.append(activation_fn())
            
                prev_layer_size = hidden_size
            
            last_layer = torch.nn.Linear(prev_layer_size, 2)  #TODO maybe do not hardcode number of output params for the action distribution
            
            if self.uniform_bias_init:
                if not self.action_dist_name == "squashed_gaussian":
                    last_layer.bias.data[0] += _inversion_bias(torch.as_tensor(self.bias_params[dim][0], device= last_layer.bias.device, dtype=last_layer.bias.dtype))
                last_layer.bias.data[1] += _inversion_bias(torch.as_tensor(self.bias_params[dim][1], device= last_layer.bias.device, dtype=last_layer.bias.dtype))
            
            layers.append(last_layer)
            
            modules.append(torch.nn.Sequential(*layers))
            

        self.heads = torch.nn.ModuleList(modules)

        #keep evaluator for the polytope bounds in the model!
        self.lp_sub_proc = LPSubProc(model_config.get("custom_model_config").get("num_process_lp_solver"))

    
    def __deepcopy__(self, memo):
        import os
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" #otherwise we get a warning (this is necessary for deterministic)

        cls = self.__class__
        obj = cls.__new__(cls)
        memo[id(self)] = obj
        for k, v in self.__dict__.items():
            if k == "lp_sub_proc":
                setattr(obj, k, LPSubProc(self.model_config.get("custom_model_config").get("num_process_lp_solver")))
            else:
                setattr(obj, k, copy.deepcopy(v, memo))
            pass
        return obj        
    
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs_flat"].float()
        self._last_flat_in = obs.reshape(obs.shape[0], -1)

        self._hidden_state = self.state_encoder(self._last_flat_in)
        if not self.vf_share:
            self._value = torch.reshape(self.value_net(self._last_flat_in), [-1])
        return self._hidden_state, state

    def value_function(self):
        if self.vf_share:
            return torch.reshape(self.value_branch(self._hidden_state), [-1])
        else:        
            return self._value
        