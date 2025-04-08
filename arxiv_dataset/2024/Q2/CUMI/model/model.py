import torch
import torch.nn as nn
import torch.nn.functional as F
from math import ceil
from typing import List, Callable, Union, Any, TypeVar, Tuple
import random

class MLP_CU(nn.Module):
    def __init__(self, input_size: int, class_num: int, hidden_dims_C: List = None, hidden_dims_U: List = None):
        super(MLP_CU, self).__init__()
        input_C = input_size 
        # Build Encoder_C
        modules_C = []
        if hidden_dims_C is None:
            hidden_dims_C = [ceil(input_size*1.2), ceil(input_size*0.5), ceil(class_num*10)] #[ceil(input_size*0.52), ceil(input_size*1.2), ceil(class_num*1.2)]
        for h_dim in hidden_dims_C:
                    modules_C.append(
                        nn.Sequential(
                            nn.Linear(input_C,out_features=h_dim),
                            nn.ReLU()
                        )
                    )
                    input_C = h_dim
        self.encoder_C = nn.Sequential(*modules_C)
        
        # Build Encoder_U
        input_U = input_size 
        modules_U = []
        if hidden_dims_U is None:
            hidden_dims_U = [ceil(input_size*1.2), ceil(input_size*0.5), ceil(class_num*5)] #[ceil(input_size*0.52), ceil(input_size*1.2), ceil(class_num*1.2)]
        for h_dim in hidden_dims_U:
            modules_U.append(
                nn.Sequential(
                    nn.Linear(input_U, out_features=h_dim),
                    nn.ReLU()
                )
            )
            input_U = h_dim
        self.encoder_U = nn.Sequential(*modules_U)

        # Build Decoder
        modules_D = []
        output_D = input_size 
        hidden_dims_D = [hidden_dims_C_i + hidden_dims_U_i for hidden_dims_C_i, hidden_dims_U_i in zip(hidden_dims_C, hidden_dims_U)] 
        hidden_dims_D.reverse()
        hidden_dims_D.append(output_D)
        for i in range(len(hidden_dims_D) - 1):
            modules_D.append(
                nn.Sequential(
                    nn.Linear(hidden_dims_D[i],hidden_dims_D[i+1]),
                    nn.ReLU()
                )
            )
        self.decoder = nn.Sequential(*modules_D)
        self.dropout = nn.Dropout(p=0.5)

    
    def forward(self, input):
        z_C = self.encoder_C(input)
        z_C = self.dropout(z_C)
        z_U = self.encoder_U(input)
        z_U = self.dropout(z_U)
        z = torch.cat((z_C, z_U), dim=1)
        input_hat = self.decoder(z)
        return  [input_hat, z_C, z_U]
    
class IBCI(nn.Module):
    def __init__(self, input_dims, class_num) -> None:
        super(IBCI, self).__init__()
        self.n_view = len(input_dims)
        self.X_nets = nn.ModuleList([MLP_CU(input_size=input_dims[i], class_num=class_num) for i in range(self.n_view)])
        self.dropout = nn.Dropout(p=0.5)
        classifier_input_dim = ceil(class_num*10) + len(input_dims)*(ceil(class_num*5))
        self.classifier = nn.Linear(classifier_input_dim, class_num)

    def forward(self, inputs):
        assert len(inputs) == self.n_view, f"Expected {self.n_views} inputs, but got {len(inputs)}"
        com_features, uni_features, inputs_hat = [], [], []
        for i, input_i in enumerate(inputs):
            X_net_i = self.X_nets[i]
            input_i_hat, com_z_i, uni_zi = X_net_i(input_i)
            com_features.append(com_z_i)
            uni_features.append(uni_zi) 
            inputs_hat.append(input_i_hat)
        com_feature = random.choice(com_features)
        uni_feature = torch.cat(uni_features, dim=1)
        feature = torch.cat((com_feature, uni_feature), dim=1)
        feature = self.dropout(feature)
        out    = self.classifier(feature)
        return inputs, inputs_hat, com_feature, uni_features, out