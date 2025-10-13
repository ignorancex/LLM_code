import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.time_utils import DeformNetwork
from utils.cutlass_deform_network import create_deform_network
import os
from utils.system_utils import searchForMaxIteration
from utils.general_utils import get_expon_lr_func


class DeformModel:
    def __init__(self, use_cutlass=False, use_fullyfused=False):
        self.deform = create_deform_network(use_cutlass=use_cutlass, use_fullyfused=use_fullyfused).cuda()
        self.optimizer = None
        self.spatial_lr_scale = 1

    def train_setting(self, training_args):
        # Adjust learning rate for FullyFusedMLP to compensate for smaller network capacity
        is_fullyfused = False
        is_cutlass = False
        
        # Check if this is a tiny-cuda-nn network
        if hasattr(self.deform, 'tcnn_model'):
            # Check network type by looking at the network configuration
            if hasattr(self.deform, 'W') and self.deform.W == 128:
                is_fullyfused = True
            elif hasattr(self.deform, 'W') and self.deform.W == 256:
                is_cutlass = True
        
        if is_fullyfused:
            # Use higher learning rate and longer training for FullyFusedMLP
            lr_init = training_args.position_lr_init * 1.5  # 50% higher initial LR
            lr_final = training_args.position_lr_final * 2.0  # 100% higher final LR
            max_steps = training_args.deform_lr_max_steps  # Use deform max steps instead

            
            # Store custom densification parameters for FullyFusedMLP
            self.custom_densify_grad_threshold = training_args.densify_grad_threshold# * 0.6  # More aggressive densification
            self.custom_densify_until_iter = training_args.densify_until_iter + 5000  # Extend densification period
            self.custom_densification_interval = max(100, training_args.densification_interval // 2)  # More frequent densification
        else:
            lr_init = training_args.position_lr_init * self.spatial_lr_scale
            lr_final = training_args.position_lr_final
            max_steps = training_args.position_lr_max_steps
            self.custom_densify_grad_threshold = None
            self.custom_densify_until_iter = None
            self.custom_densification_interval = None

        l = [
            {'params': list(self.deform.parameters()),
             'lr': lr_init,
             "name": "deform"}
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.deform_scheduler_args = get_expon_lr_func(lr_init=lr_init,
                                                       lr_final=lr_final,
                                                       lr_delay_mult=training_args.position_lr_delay_mult,
                                                       max_steps=max_steps)

    def step(self, x, t):
        return self.deform(x, t)

    def save_weights(self, model_path, iteration):
        out_weights_path = os.path.join(model_path, "deform/iteration_{}".format(iteration))
        os.makedirs(out_weights_path, exist_ok=True)
        torch.save(self.deform.state_dict(), os.path.join(out_weights_path, 'deform.pth'))

    def load_weights(self, model_path, iteration=-1):
        if iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(model_path, "deform"))
        else:
            loaded_iter = iteration
        weights_path = os.path.join(model_path, "deform/iteration_{}/deform.pth".format(loaded_iter))
        self.deform.load_state_dict(torch.load(weights_path))

    def update_learning_rate(self, iteration):
        if self.optimizer is not None:
            for param_group in self.optimizer.param_groups:
                if param_group["name"] == "deform":
                    lr = self.deform_scheduler_args(iteration)
                    param_group['lr'] = lr
                    return lr
        return 0.0
