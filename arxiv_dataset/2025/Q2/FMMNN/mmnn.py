import torch
import torch.nn as nn

class MMNN(nn.Module):
    def __init__(self, 
                 ranks = [1] + [16]*5 + [1], 
                 widths = [666]*6,
                 device = "cpu", 
                 ResNet = False,
                 fixWb = True,
                 init_scaling = True):
        super().__init__()
        """
        A class to configure the neural network model.
    
        Attributes:
            ranks (list[int]): A list where the i-th element represents the output dimension of the i-th layer.
                               For the j-th layer, ranks[j-1] is the input dimension and ranks[j] is the output dimension.
                               Specifically, ranks[0] represents the input dimension, 
                               while ranks[-1] corresponds to the output dimension of the network.
            
            widths (list[int]): A list where each element specifies the width of the corresponding layer.
            
            device (str): The device (CPU/GPU) on which the PyTorch code will be executed.
            
            ResNet (bool): Indicates whether to use ResNet architecture, which includes identity connections between layers.
            
            fixWb (bool): If True, the weights and biases are not updated during training.

            init_scaling (bool): If True, scales the weight and biase (W0 and b0) in the first affine map to greatly accelerate convergence.
        """
        
        self.ranks = ranks
        self.widths = widths
        self.ResNet = ResNet
        self.depth = len(widths)
        
        fc_sizes = [ ranks[0] ] 
        for j in range(self.depth):
            fc_sizes += [ widths[j], ranks[j+1] ]

        fcs=[]
        for j in range(len(fc_sizes)-1):
            fc = nn.Linear(fc_sizes[j],
                           fc_sizes[j+1], device=device) 
            # setattr(self, f"fc{j}", fc)
            fcs.append(fc)
        self.fcs = nn.ModuleList(fcs)
        
        if fixWb: # Fix W and b in each layer A \sigma( Wx+b ) + c
            for j in range(len(fcs)):
                if j % 2 == 0:
                    self.fcs[j].weight.requires_grad = False
                    self.fcs[j].bias.requires_grad = False
                if j==0 and init_scaling:
                    s = widths[0]/2
                    s = s**(1/ranks[0])*ranks[0]**0.5
                    self.fcs[j].weight.data = s * self.fcs[j].weight.data
                    self.fcs[j].bias.data = s * self.fcs[j].bias.data
 

    def forward(self, x):
        for j in range(self.depth):

            if self.ResNet:
                if 0 < j < self.depth-1:
                    x_id = x + 0

            x = self.fcs[2*j](x)

            x = torch.sin(x)

            x = self.fcs[2*j+1](x) 

            if self.ResNet:
                if 0 < j < self.depth-1:
                    if x.shape[1] <= x_id.shape[1]:
                        x = x + x_id[:,:x.shape[1]]
                    else:
                        x0 = torch.zeros_like(x)
                        x0[:,:x_id.shape[1]] = x_id
                        x = x + x0

        return x



  

