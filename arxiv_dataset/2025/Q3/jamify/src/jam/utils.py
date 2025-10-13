# Copyright (c) 2025 Project Jamify
#               2025 Declare-Lab
#               2025 AMAAI Lab
#               2025 Renhang Liu (liurenhang0@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

class GradientTracker:
    """
    Class to track and calculate both gradient norms and parameter update norms.
    """
    def __init__(self, model):
        """
        Initialize the gradient tracker.
        
        Args:
            model: PyTorch model whose gradients and updates will be tracked
        """
        self.model = model
        self.params_before = None
        self.grad_norm = None
        self.update_norm = None
    
    def store_params(self):
        """Store current parameter values before optimizer step"""
        self.params_before = {name: param.detach().clone() 
                              for name, param in self.model.named_parameters() 
                              if param.requires_grad}
    
    def compute_grad_norm(self):
        """Calculate L2 norm of all gradients"""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        
        self.grad_norm = total_norm ** 0.5
        return self.grad_norm

    def check_unused_parameters(self):
        """Check if there are any unused parameters"""
        unused_params = []
        for name, param in self.model.named_parameters():
            if param.grad is None and param.requires_grad:
                unused_params.append(name)
        return unused_params
    
    def compute_update_norm(self):
        """Calculate L2 norm of parameter updates after optimizer step"""
        if self.params_before is None:
            raise RuntimeError("You must call store_params() before compute_update_norm()")
        
        total_norm = 0.0
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.params_before:
                param_update = param.detach() - self.params_before[name]
                total_norm += param_update.norm(2).item() ** 2
        
        self.params_before = None
        self.update_norm = total_norm ** 0.5
        return self.update_norm
    
