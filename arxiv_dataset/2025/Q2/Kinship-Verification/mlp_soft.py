import torch.nn as nn
import torch
import settings as st

class MLP_Parent_XenterLoss(nn.Module):
    
    def __init__(self, input_dim_parent, output_dim):
        super().__init__()
        
        self.xfeature = nn.Linear(4*input_dim_parent, st.config[st.I]["FC1"])    
        self.fc1 = nn.Linear(st.config[st.I]["FC1"], st.config[st.I]["FC2"])
        self.fc2 = nn.Linear(st.config[st.I]["FC2"], output_dim)
        
    def forward(self, feature1, feature2):
        x1 = torch.pow(feature1, 2) - torch.pow(feature2, 2)
        x2 = torch.pow(feature1 - feature2, 2)
        x3 = feature1 * feature2
        x4 = feature1 + feature2
        x = torch.cat((x3, x2, x1, x4), dim=1)
        x = x.view(x.size(0), -1)        
        x = self.xfeature(x)
        out = self.fc2(self.fc1(x))        
        return out, x
    

class MLP_layer(nn.Module):
    
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)        

        
    def forward(self, x):
        out = self.fc1(x)
        return out

