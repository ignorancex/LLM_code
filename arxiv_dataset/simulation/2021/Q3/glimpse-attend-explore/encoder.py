import torch as t
import torch.nn as nn
from settings import *
#use first four layers of a pretrained resnet to extract the features from glimpses (and the full image during training)
class Encoder(t.nn.Module):
    def __init__(self):
        super().__init__()
        resnet_list=list(t.hub.load('pytorch/vision:v0.4.0', 'resnet18', pretrained=True).children())[:-3]
        self.layers=nn.ModuleList(resnet_list)
    def forward(self,x):
        for ii,model in enumerate(self.layers):
            x = model(x)
            if ii==2:
                intermediate_1=x.clone()
            if ii==4:
                intermediate_2=x.clone()
            if ii==5:
                intermediate_3=x.clone()
            if ii==6:
                bottleneck=x.clone()
        return bottleneck,intermediate_3,intermediate_2,intermediate_1
