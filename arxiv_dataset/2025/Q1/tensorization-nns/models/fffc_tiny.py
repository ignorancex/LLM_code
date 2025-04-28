#####################################
#          FEED FORWARD NN          #
#####################################

import torch.nn as nn
from ray import tune


expected_accuracy = 0.75  # update after tuning hyperparameters (check tuned_metrics.json)

test_config = {'p_do': 0.5,
               'batch_size': 16,       # mandatory in any config
               'lr': 1e-2,             # mandatory in any config
               'weight_decay': 1e-5}   # mandatory in any config

tuner_config = {'p_do': tune.choice([0., 0.1, 0.25, 0.5]),
                'batch_size': tune.choice([8, 16, 32, 64]),
                'lr': tune.loguniform(1e-5, 1e-1),
                'weight_decay': tune.loguniform(1e-6, 1e-2)}


class Model(nn.Module):
    
    name = 'fffc_tiny'
    
    def __init__(self, config):
        super().__init__()
        
        out_rate = 1000
        p_do = config['p_do']
        
        self.layers = nn.Sequential(
            nn.Linear(out_rate // 2, 50),
            nn.Dropout(p_do),
            nn.ReLU(),
            nn.Linear(50, 2))
        
    def forward(self, x):
        return self.layers(x)
