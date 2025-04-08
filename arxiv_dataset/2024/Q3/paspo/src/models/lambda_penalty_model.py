import torch
from ray.rllib.utils.framework import try_import_torch


torch, nn = try_import_torch()


class LambdaPenaltyModel(torch.nn.Module):
    def __init__(self, config_lambda_model, number_of_constraints: int):
        #This model outputs the summed weightes penalty terms
        super(LambdaPenaltyModel, self).__init__()

        self.availabe_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.learning_rate = config_lambda_model["lambda_model_lr"]

        outputSize = 1 # we sum all the weights up to get a final penalty score
        self.number_of_constraints = number_of_constraints
        
        self.lagrangian_upper_bound = config_lambda_model["lagrangian_upper_bound"]
        
        self.cost_limit = config_lambda_model["cost_limit"] #0.0
        
        self.lagrange_multiplier_init = config_lambda_model["lagrange_multiplier_init"]
        
        self.lagrange_multiplier = torch.nn.Linear(self.number_of_constraints, outputSize, bias=False)
        self.lagrange_multiplier.weight.data.fill_(self.lagrange_multiplier_init)
        
        self.lambda_optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        

    def forward(self, x):
        out = self.lagrange_multiplier(x)
        return out

    def get_lagrange_multiplier(self):
        return self.lagrange_multiplier.weight
    
    def update_lagrange_multiplier(self, constraint_violations):
        self.lambda_optimizer.zero_grad()
        lambda_loss = -self.lagrange_multiplier(constraint_violations - self.cost_limit).sum(dim=-1).mean()
        lambda_loss.backward()
        self.lambda_optimizer.step()
        self.lagrange_multiplier.weight.data.clamp_(min=0.0, max=self.lagrangian_upper_bound) #enforce the bounds
        
        return lambda_loss
        
        