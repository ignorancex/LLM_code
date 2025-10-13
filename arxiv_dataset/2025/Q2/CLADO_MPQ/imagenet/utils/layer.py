import torch
import torch.nn as nn
import torch.nn.functional as F

################################################# Customized Layers for QAVAT #################################################


# LINEAR LAYER
class qLinear(nn.Module):
    def __init__(self,in_features,out_features,nbits_activation=None,nbits_weight=None,init_from=None):
        
        assert (nbits_activation is None or nbits_activation>=2) and (nbits_weight is None or nbits_weight>=2), 'bitwidth should be at least 2'
        assert init_from is None or isinstance(init_from,torch.nn.Linear), 'bad init_from'
        
        super(qLinear,self).__init__()

        if init_from is None:
            layer = nn.Linear(in_features,out_features,bias=False)
        else:
            layer = init_from
            
        self.weight = layer.weight
        self.weight_variation = None

        self.nbits_activation = nbits_activation
        self.nbits_weight = nbits_weight
        self.smoothness = 0.99

        if self.nbits_activation is not None:
            self.input_scale = nn.Parameter(torch.Tensor([0]),requires_grad=False)
            self.intervals_activation = 2**(nbits_activation-1) - 1

        if self.nbits_weight is not None:
            self.weight_scale = nn.Parameter(torch.Tensor([0]),requires_grad=False)
            self.intervals_weight = 2**(nbits_weight-1) - 1
        
        self.bias = layer.bias


    def forward(self,x):
        # activation quantization
        if self.nbits_activation is None:
            Q_x = x
        else:
            new_input_scale = torch.max(x.abs()).detach()/self.intervals_activation
            if self.input_scale == 0:
                self.input_scale += new_input_scale # initalization
            if self.training: # running statistics update for input_scale, disabled in testing
                self.input_scale.data = (self.input_scale * self.smoothness + new_input_scale * (1-self.smoothness)).data
            quant_input = torch.round(x/self.input_scale).clamp(-self.intervals_activation,self.intervals_activation)
            dequant_input = quant_input * self.input_scale
            Q_x =  dequant_input.detach() - x.detach() + x # STE

        # weight quantization
        if self.nbits_weight is None:
            Q_weight = self.weight
        else:            
            # dynamic scale in training
            if self.training or self.weight_scale==0:
                # training, update scale by line search
                nLevel = self.intervals_weight
                gs_n = 10
                init_scale = (self.weight.abs().max()/nLevel).data
                end_scale = (torch.quantile(self.weight.abs(),0.5)/nLevel).data
                gs_interval = (init_scale-end_scale)/(gs_n-1)
                scales = torch.arange(init_scale,end_scale-0.1*gs_interval,-gs_interval)
                scales = scales.unsqueeze(1).unsqueeze(2).to(self.weight.device)
                weights = self.weight.unsqueeze(0)
                Q_weights = torch.round(weights/scales).clamp_(-nLevel,nLevel)
                DQ_weights = Q_weights * scales
                L2errs = ((weights-DQ_weights)**2).sum(dim=[1,2],keepdim=False)
                index = torch.argmin(L2errs)
                self.weight_scale.data = (init_scale - index * gs_interval).data
                Q_weight = DQ_weights[index].squeeze(0).detach() - self.weight.detach() + self.weight # STE
            else:
                # evaluation, scale not updated
                quant_weight = torch.round((self.weight/self.weight_scale)).clamp(-self.intervals_weight,self.intervals_weight)
                dequant_weight = quant_weight * self.weight_scale
                Q_weight =  dequant_weight.detach() - self.weight.detach() + self.weight # STE
            

        if self.weight_variation is not None:
            Q_weight = torch.mul(Q_weight,self.weight_variation) # weight variation
        out = torch.matmul(x,Q_weight.transpose(1,0))
        
        if self.bias is not None:
            out += self.bias
        return out

    def generate_variation(self,noise_std=0):
        device = self.weight.device
        if noise_std == 0:
            self.weight_variation = None
        else:
            if str(device).startswith("cuda"):
                self.weight_variation = torch.cuda.FloatTensor(self.weight.size(),device=device).normal_(1,noise_std)
            else:
                self.weight_variation = torch.FloatTensor(self.weight.size()).normal_(1,noise_std)

# CONV LAYER
class qConv2d(nn.Module):

    def __init__(self,in_channels,out_channels,kernel_size, stride=1,padding=0,
                 nbits_activation=None,nbits_weight=None,init_from=None):

        
        assert (nbits_activation is None or nbits_activation>=2) and (nbits_weight is None or nbits_weight>=2), 'bitwidth should be at least 2'
        assert init_from is None or isinstance(init_from,torch.nn.Conv2d), 'bad init_from'
        
        super(qConv2d,self).__init__()
        
        if init_from is None:
            layer = nn.Conv2d(in_channels = in_channels, out_channels = out_channels,
                                kernel_size = kernel_size, stride = stride, padding = padding,bias=False)
        else:
            layer = init_from
        
        self.weight = layer.weight
        self.padding = layer.padding
        self.stride = layer.stride
        self.shape = (layer.out_channels, layer.in_channels, layer.kernel_size[0], layer.kernel_size[1])
        self.weight_variation = None
        self.nbits_activation = nbits_activation
        self.nbits_weight = nbits_weight
        self.smoothness = 0.99
        if self.nbits_activation is not None:
            self.input_scale = nn.Parameter(torch.Tensor([0]),requires_grad=False)
            self.intervals_activation = 2**(nbits_activation-1) - 1
        if self.nbits_weight is not None:
            self.weight_scale = nn.Parameter(torch.Tensor([0]),requires_grad=False)
            self.intervals_weight = 2**(nbits_weight-1) - 1
        self.bias = layer.bias

    def forward(self,x):
        # activation quantization
        if self.nbits_activation is None:
            Q_x = x
        else:
            new_input_scale = torch.max(x.abs()).detach()/self.intervals_activation
            if self.input_scale == 0:
                self.input_scale += new_input_scale # initalization
            if self.training: # running statistics update for input_scale, disabled in testing
                self.input_scale.data = (self.input_scale * self.smoothness + new_input_scale * (1-self.smoothness)).data
            quant_input = torch.round(x/self.input_scale).clamp(-self.intervals_activation,self.intervals_activation)
            dequant_input = quant_input * self.input_scale
            Q_x =  dequant_input.detach() - x.detach() + x # STE

        # weight quantization
        if self.nbits_weight is None:
            Q_weight = self.weight
        else:
            # dynamic scale in training
            if self.training or self.weight_scale==0:
                # training, update scale by line search
                nLevel = self.intervals_weight
                gs_n = 10
                init_scale = (self.weight.abs().max()/nLevel).data
                end_scale = (torch.quantile(self.weight.abs(),0.5)/nLevel).data
                gs_interval = (init_scale-end_scale)/(gs_n-1)
                scales = torch.arange(init_scale,end_scale-0.1*gs_interval,-gs_interval)
                scales = scales.unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4).to(self.weight.device)
                weights = self.weight.unsqueeze(0)
                Q_weights = torch.round(weights/scales).clamp_(-nLevel,nLevel)
                DQ_weights = Q_weights * scales
                L2errs = ((weights-DQ_weights)**2).sum(dim=[1,2,3,4],keepdim=False)
                index = torch.argmin(L2errs)
                self.weight_scale.data = (init_scale - index * gs_interval).data
                Q_weight = DQ_weights[index].squeeze(0).detach() - self.weight.detach() + self.weight # STE
            else:
                # evaluation, scale not updated
                quant_weight = torch.round((self.weight/self.weight_scale)).clamp(-self.intervals_weight,self.intervals_weight)
                dequant_weight = quant_weight * self.weight_scale
                Q_weight =  dequant_weight.detach() - self.weight.detach() + self.weight # STE

        if self.weight_variation is not None:
            Q_weight = torch.mul(Q_weight,self.weight_variation) # weight variation

        out = F.conv2d(Q_x,Q_weight,bias=None,padding=self.padding, stride=self.stride)
        if self.bias is not None:
            out += self.bias
        return out

    def generate_variation(self,noise_std=0):
        device = self.weight.device
        if noise_std == 0:
            self.weight_variation = None
        else:
            if str(device).startswith("cuda"):
                self.weight_variation = torch.cuda.FloatTensor(self.weight.size(),device=device).normal_(1,noise_std)
            else:
                self.weight_variation = torch.FloatTensor(self.weight.size()).normal_(1,noise_std)