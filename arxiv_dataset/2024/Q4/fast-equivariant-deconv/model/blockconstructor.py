from .graphconv import Conv
import torch.nn as nn
import torch


class Block(nn.Module):
    """GCNN Unet block.
    """
    def __init__(self, channels, lap, kernel_sizeSph, kernel_sizeSpa, conv_name, isoSpa, patch_size, vec=None):
        """Initialization.
        Args:
            channels (list int): Number of channels
            lap (list): Laplacian
            kernel_sizeSph (int): Size of the spherical kernel (i.e. Order of the Chebyshev polynomials + 1)
            kernel_sizeSpa (int): Size of the spatial kernel
            conv_name (str): Name of the convolution (spherical or mixed)
        """
        super(Block, self).__init__()
        #kernel_sizeSpa = min(kernel_sizeSpa, patch_size)
        if conv_name=='mixed' and patch_size==1:
            conv_name = 'spherical'
        if conv_name in ['spatial', 'spatial_vec', 'spatial_sh']:
            if patch_size==1:
                kernel_sizeSpa = 1
            self.block = BlockSpatial(channels, lap, kernel_sizeSph, kernel_sizeSpa, conv_name, isoSpa)
        elif conv_name in ['spherical', 'mixed', 'bekkers']:
            self.block = BlockProposed(channels, lap, kernel_sizeSph, kernel_sizeSpa, conv_name, isoSpa, vec=vec)
        else:
            raise NotImplementedError
            
    def forward(self, x):
        """Forward Pass.
        Args:
            x (:obj:`torch.Tensor`): input to be forwarded. [B x F_in_ch x V x X x Y x Z] or [B x F_in_ch x X x Y x Z]
        Returns:
            :obj:`torch.Tensor`: output [B x F_out_ch x V x X x Y x Z] or [B x F_in_ch x X x Y x Z]
        """
        x = self.block(x) # B x F_int_ch x V x X x Y x Z or B x F_int_ch x X x Y x Z
        return x
            
class SubBlock(nn.Module):
    """GCNN Unet block.
    """
    def __init__(self, channels, lap, kernel_sizeSph, kernel_sizeSpa, conv_name, isoSpa, vec=None):
        """Initialization.
        Args:
            channels (list int): Number of channels
            lap (list): Laplacian
            kernel_sizeSph (int): Size of the spherical kernel (i.e. Order of the Chebyshev polynomials + 1)
            kernel_sizeSpa (int): Size of the spatial kernel
            conv_name (str): Name of the convolution (spherical or mixed)
        """
        super(SubBlock, self).__init__()
        assert isinstance(channels, list)
        assert len(channels)>=2
        self.conv = []
        self.bn = []
        #if conv_name in ['spatial', 'spatial_vec', 'spatial_sh']:
        #    bn_mul = lap.shape[0]
        #else:
        #    bn_mul = 1
        bn_mul = 1
        if vec is None:
            vec = [None]*len(channels)
        for i in range(len(channels)-1):
            conv_name_sel = conv_name
            #if conv_name!='mixed':
            #    conv_name_sel = conv_name
            #elif conv_name=='mixed':
            #    if i>0 or lap.shape[0]>3500:
            #        conv_name_sel = 'spherical'
            #    else:
            #        conv_name_sel = 'mixed'
                    #conv_name*(conv_name!='mixed')+('mixed'*(laps[-1].shape[0]<500) + 'spherical'*(laps[-1].shape[0]>=500))*(conv_name=='mixed')
            self.conv += [Conv(channels[i], channels[i+1], lap, kernel_sizeSph, kernel_sizeSpa, conv_name=conv_name_sel, isoSpa=isoSpa, vec=vec[i])]
            self.bn += [nn.BatchNorm3d(channels[i+1]*bn_mul)]
        self.conv = nn.ModuleList(self.conv)
        self.bn = nn.ModuleList(self.bn)
        # Activation
        self.activation = nn.ReLU()

    
class BlockSpatial(SubBlock):
    """GCNN Unet block.
    """
    def __init__(self, channels, lap, kernel_sizeSph, kernel_sizeSpa, conv_name, isoSpa):
        """Initialization.
        Args:
            channels (list int): Number of channels
            lap (list): Laplacian
            kernel_sizeSph (int): Size of the spherical kernel (i.e. Order of the Chebyshev polynomials + 1)
            kernel_sizeSpa (int): Size of the spatial kernel
            conv_name (str): Name of the convolution (spherical or mixed)
        """
        super(BlockSpatial, self).__init__(channels, lap, kernel_sizeSph, kernel_sizeSpa, conv_name, isoSpa)
        assert conv_name in ['spatial', 'spatial_vec', 'spatial_sh']
    
    def forward(self, x):
        """Forward Pass.
        Args:
            x (:obj:`torch.Tensor`): input to be forwarded. [B x F_in_ch x V x X x Y x Z]
        Returns:
            :obj:`torch.Tensor`: output [B x F_out_ch x V x X x Y x Z]
        """
        B, F, V, X, Y, Z = x.shape
        for i in range(len(self.conv)):
            x = self.activation(self.bn[i](self.conv[i](x).view(B, -1, X, Y, Z)).view(B, -1, V, X, Y, Z)) # B x (F_int_ch x V) x X x Y x Z
        return x

class BlockProposed(SubBlock):
    """GCNN Unet block.
    """
    def __init__(self, channels, lap, kernel_sizeSph, kernel_sizeSpa, conv_name, isoSpa, vec=None):
        """Initialization.
        Args:
            channels (list int): Number of channels
            lap (list): Laplacian
            kernel_sizeSph (int): Size of the spherical kernel (i.e. Order of the Chebyshev polynomials + 1)
            kernel_sizeSpa (int): Size of the spatial kernel
            conv_name (str): Name of the convolution (spherical or mixed)
        """
        super(BlockProposed, self).__init__(channels, lap, kernel_sizeSph, kernel_sizeSpa, conv_name, isoSpa, vec=vec)
        assert conv_name in ['spherical', 'mixed', 'bekkers']
    
    def forward(self, x):
        """Forward Pass.
        Args:
            x (:obj:`torch.Tensor`): input to be forwarded. [B x F_in_ch x V x X x Y x Z]
        Returns:
            :obj:`torch.Tensor`: output [B x F_out_ch x V x X x Y x Z]
        """
        B, _, V, X, Y, Z = x.shape
        for i in range(len(self.conv)):
            x = self.activation(self.bn[i](self.conv[i](x).view(B, -1, V * X, Y, Z)).view(B, -1, V, X, Y, Z)) # B x F_int_ch x V x X x Y x Z
        return x


class BlockHead(nn.Module):
    """GCNN Unet block.
    """
    def __init__(self, channels, lap, kernel_sizeSph, kernel_sizeSpa, conv_name, isoSpa, patch_size, keepSphericalDim=False, vec=None, n_vec=None):
        """Initialization.
        Args:
            channels (list int): Number of channels
            lap (list): Laplacian
            kernel_sizeSph (int): Size of the spherical kernel (i.e. Order of the Chebyshev polynomials + 1)
            kernel_sizeSpa (int): Size of the spatial kernel
            conv_name (str): Name of the convolution (spherical or mixed)
        """
        super(BlockHead, self).__init__()
        self.keepSphericalDim = keepSphericalDim
        self.n_vec = n_vec # If we keep spherical dim, we need this information
        #if conv_name!='mixed':
        #    conv_name = conv_name
        #elif conv_name=='mixed':
        #    if lap.shape[0]>3500:
        #        conv_name = 'spherical'
        #    else:
        #        conv_name = 'mixed'
        #kernel_sizeSpa = min(kernel_sizeSpa, patch_size)
        
        #kernel_sizeSpa = min(kernel_sizeSpa, patch_size)
        if conv_name=='mixed' and patch_size==1:
            conv_name = 'spherical'
        
        self.conv_name = conv_name
        if keepSphericalDim:
            if conv_name=='MLP_head':
                self.conv = torch.nn.Linear(channels[0], channels[1])
            else:
                self.conv = Conv(channels[0], channels[1], lap, kernel_sizeSph, kernel_sizeSpa, conv_name=conv_name, isoSpa=isoSpa, vec=vec[0])
        else:
            if conv_name=='MLP_head':
                self.conv = torch.nn.Linear(channels[0], channels[1])
            else:
                if conv_name=='spherical':
                    kernel_sizeSpa = 1
                self.conv = Conv(channels[0], channels[1], torch.ones(1, 1), kernel_sizeSph, kernel_sizeSpa, conv_name='spatial', isoSpa=isoSpa, vec=vec[0])
        
    def forward(self, x):
        """Forward Pass.
        Args:
            x (:obj:`torch.Tensor`): input to be forwarded. [B x F_in_ch x V x X x Y x Z]
        Returns:
            :obj:`torch.Tensor`: output [B x F_out_ch x V x X x Y x Z] or [B x F_out_ch x 1 x X x Y x Z]
        """
        if not self.keepSphericalDim:
            x = torch.mean(x, dim=2, keepdim=True)
        if self.conv_name=='MLP_head':
            x = x.permute(0, 2, 3, 4, 5, 1).contiguous()
        x = self.conv(x) # B x F_int_ch x V x X x Y x Z or B x F_int_ch x 1 x X x Y x Z
        if self.conv_name=='MLP_head':
            x = x.permute(0, 5, 1, 2, 3, 4).contiguous()
        if self.keepSphericalDim and (self.conv_name in ['spatial_vec', 'spatial_sh', 'spatial']):
            X, Y, Z = x.shape[-3:]
            B = x.shape[0] 
            x = x.view(B, -1, self.n_vec, X, Y, Z)
        return x
