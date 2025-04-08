import torch.nn as nn

from exceptions.exceptions import InvalidBackboneError


class MLPSimCLR(nn.Module):

    def __init__(self, in_dim, hid_dim=1000, proj_dim=1000, out_dim=1000, n_hidden = 3):
        super(MLPSimCLR, self).__init__()
        
        # Store architecture details for later
        self.arch = {
            'in_dim': in_dim,
            'hid_dim': hid_dim,
            'proj_dim': proj_dim,
            'out_dim': out_dim,
            'n_hidden': n_hidden
        }
        
        # TODO: note that it would be easier to model the backbone and projection head separately
        # but let's stick with this for now
        layers = [nn.Linear(in_dim, hid_dim), nn.ReLU()]
        
        for i in range(0, n_hidden):
            layers.append(nn.Linear(hid_dim, proj_dim if i == n_hidden - 1 else hid_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(proj_dim, out_dim))
        layers.append(nn.ReLU())
        self.backbone = nn.Sequential(*layers)
        # trained backbone is depth-4 MLP, with a depth-1 MLP projection head
        # self.backbone = nn.Sequential(
        #     nn.Linear(in_dim, hid_dim), nn.ReLU(),
        #     nn.Linear(hid_dim, hid_dim), nn.ReLU(),
        #     nn.Linear(hid_dim, hid_dim), nn.ReLU(),
        #     nn.Linear(hid_dim, proj_dim), nn.ReLU(),
        #     nn.Linear(proj_dim, out_dim), nn.ReLU(),
        # )

    def forward(self, x):
        return self.backbone(x)
