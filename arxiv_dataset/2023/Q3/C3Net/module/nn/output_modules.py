from module.nn import *
from torch import nn as nn

class Atomistic_Model(nn.Module):
    def __init__(self, representation, output_modules):
        super(Atomistic_Model, self).__init__()
        self.representation = representation
        if type(output_modules) not in [list, nn.ModuleList]:
            output_modules = [output_modules]
        if type(output_modules) == list:
            output_modules = nn.ModuleList(output_modules)
        self.output_modules = output_modules
        
    def forward(self, inputs):        
        inputs["representation"] = self.representation(inputs)
        
        outs = {}
        for output_model in self.output_modules:
            outs.update(output_model(inputs))
        return outs


class Atomwise(nn.Module):
    def __init__(self,n_in=32,n_out=1,property="y"):
        super(Atomwise, self).__init__()

        self.property = property

        # build output network
        self.out_net = nn.Sequential(
            GetItem("representation"),
            MLP(n_in, n_out),
        )

        # build aggregation layer
        self.atom_pool = Aggregate(axis=1)
        
    def forward(self, inputs):
        mask = inputs['Mask']

        yi = self.out_net(inputs)
        y = self.atom_pool(yi,mask)

        for i in range(inputs["Solvent_Properties"].size()[0]):
            if inputs['Solvent_ID'][i] in [103,104]: #logP and PAMPA
                y[i] /= 2.303*0.592485 #-RTlnH to log units
        
        # collect results
        result = {self.property: y}

        return result
