import torch
import torch.nn as nn
from nets.restormer import *

class L_net(nn.Module):
    def __init__(self, num=64):
        super(L_net, self).__init__()
        self.L_net = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(3, num, 3, 1, 0),
            nn.ReLU(),               
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, num, 3, 1, 0),
            nn.ReLU(), 
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, num, 3, 1, 0),
            nn.ReLU(),               
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, num, 3, 1, 0),
            nn.ReLU(),   
            nn.ReflectionPad2d(1),
            nn.Conv2d(num, 1, 3, 1, 0),
        )

    def forward(self, input):
        return torch.sigmoid(self.L_net(input))

class SRE(nn.Module): # Shared R 
    def __init__(self,
                 out_channels=3,
                 dim=32,
                 num_blocks=3,
                 heads=[8, 8, 8],
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias'):
        super(SRE, self).__init__() 

        self.patch_embed1 = nn.Conv2d(6,dim, kernel_size=3,stride=1, padding=1, bias=bias)

        self.encoder = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                                            bias=bias, LayerNorm_type=LayerNorm_type) for i in range(2)])

        self.output1 = nn.Sequential(
            nn.Conv2d(int(dim),  out_channels, kernel_size=3,
                      stride=1, padding=1, bias=bias),)
        
        self.act = nn.Sigmoid()  

    def forward(self, x1,x2):
        R=self.encoder(self.patch_embed1(torch.concatenate([x1,x2],1)))
        Rhat=self.act(self.output1(R))
        
        return Rhat,R
    

class fusionnet(nn.Module):
    def __init__(self,
                 out_channels=3,
                 dim=64,
                 num_blocks=3,
                 heads=[8, 8, 8],
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias'):
        super(fusionnet, self).__init__() 

        self.SRE=SRE()
        
        self.patch_embed2 = nn.Conv2d(3,dim, kernel_size=3,stride=1, padding=1, bias=bias)
        self.decoder=nn.ModuleList([TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                                            bias=bias, LayerNorm_type=LayerNorm_type,crossatt=True) for i in range(2*num_blocks)])
        

        self.output2 = nn.Sequential(
            nn.Conv2d(int(dim),  out_channels, kernel_size=3,
                      stride=1, padding=1, bias=bias),)

        self.act = nn.Sigmoid()  
                    
          
    def forward(self, x1,x2,L):
        Rhat,_=self.SRE(x1,x2)
        R=self.patch_embed2(Rhat)

        for k in range(len(self.decoder)):
            R=R+self.decoder[k](R)*L

        R = torch.clamp(Rhat+self.output2(R),0,1)
        return R* L, R, Rhat



def unit_test():
    import numpy as np
    x = torch.tensor(np.random.rand(16,3,128,128).astype(np.float32)).cuda()
    x2=torch.tensor(np.random.rand(16,1,128,128).astype(np.float32)).cuda()
    model = fusionnet()
    model.cuda()
    y = model(x,x,x2)[0]
    print('output shape:', y.shape)


if __name__ == '__main__':
    unit_test()