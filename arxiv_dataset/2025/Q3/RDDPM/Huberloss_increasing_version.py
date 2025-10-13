import torch
import numpy as np
class CustomHuberLoss(torch.nn.Module): 
    def __init__(self): 
        super(CustomHuberLoss, self).__init__() 
    def forward(self, noise_pred, noise,t): 
        losses = []
        batch_size = noise.size(0) 
        for i in range(batch_size): 
            delta = 0.2*np.exp((np.log(5))*(t[i].item())*0.001)  # You can adjust how delta is calculated here 
            huber_loss = torch.nn.HuberLoss(delta=delta) 
            loss = huber_loss(noise_pred[i], noise[i]) 
            losses.append(loss) 
        return torch.stack(losses).mean()