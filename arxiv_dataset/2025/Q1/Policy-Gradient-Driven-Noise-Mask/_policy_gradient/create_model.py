import torch
import torch.nn as nn
import timm
import math

class MaskGenerator(nn.Module):
    def __init__(self, input_dim, grid, model_name):
        super(MaskGenerator, self).__init__()
        
        self.alpha = nn.Sequential(nn.Linear(input_dim, input_dim),
                nn.BatchNorm1d(input_dim),
                nn.ReLU(inplace=True),
                nn.Linear(input_dim, input_dim),
                nn.BatchNorm1d(input_dim),
                nn.ReLU(inplace=True),                                         
                nn.Linear(input_dim, grid * grid))
        self.beta = nn.Sequential(nn.Linear(input_dim, input_dim),
                nn.BatchNorm1d(input_dim),
                nn.ReLU(inplace=True),
                nn.Linear(input_dim, input_dim),
                nn.BatchNorm1d(input_dim),
                nn.ReLU(inplace=True),                                  
                nn.Linear(input_dim, grid * grid))
        self.grid = grid
        
        self.fe = timm.create_model(model_name, num_classes=0, pretrained=False)

    def forward(self, x, base_alpha, base_beta, decay = 0.9):
        x = self.fe(x)
        new_alpha = decay * base_alpha + (1 - decay) * self.alpha(x)
        new_beta = decay * base_beta + (1 - decay) * self.beta(x)
        
        base_alpha = 0.99 * base_alpha + (1 - 0.99) * torch.mean(self.alpha(x).detach(), 0, keepdim=True)
        base_beta = 0.99 * base_beta + (1 - 0.99) * torch.mean(self.beta(x).detach(), 0, keepdim=True)
        
        return new_alpha, new_beta, base_alpha, base_beta

def create_model(args, device):
    print("Creating model")
    if args.model == "resnet10t":
        my_model = timm.create_model('resnet10t', num_classes=args.num_classes, pretrained=False)
        my_policy_net = MaskGenerator(512, args.grid, 'resnet10t')
    elif args.model == "resnet18":
        my_model = timm.create_model('resnet18', num_classes=args.num_classes, pretrained=False)
        my_policy_net = MaskGenerator(512, args.grid, 'resnet10t')
    elif args.model == "resnet50":
        my_model = timm.create_model('resnet50', num_classes=args.num_classes, pretrained=False)
        my_policy_net = MaskGenerator(512, args.grid, 'resnet10t')

    my_model.to(device)
    my_policy_net.to(device)
    return my_model, my_policy_net
