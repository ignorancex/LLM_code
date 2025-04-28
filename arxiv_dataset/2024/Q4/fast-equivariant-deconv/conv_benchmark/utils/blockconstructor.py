from .graphconv import ConvPrecomputed
from .graphconv import Conv 
import torch.nn as nn
import torch
import math


class Block(nn.Module):
    """GCNN Unet block.
    """
    def __init__(self, channels, lap, kernel_sizeSph, kernel_sizeSpa, conv_name, isoSpa, patch_size, vec=None, old_conv=False, dense=True, precompute=False, einsum=False, repeat_interleave=True):
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
            self.block = BlockSpatial(channels, lap, kernel_sizeSph, kernel_sizeSpa, conv_name, isoSpa, old_conv=old_conv, dense=dense, precompute=precompute, einsum=einsum, repeat_interleave=repeat_interleave)
        elif conv_name in ['spherical', 'mixed', 'bekkers']:
            self.block = BlockProposed(channels, lap, kernel_sizeSph, kernel_sizeSpa, conv_name, isoSpa, old_conv=old_conv, dense=dense, precompute=precompute, einsum=einsum, repeat_interleave=repeat_interleave, vec=vec)
        elif conv_name=='muller':
            self.block = MullerBlock(channels, kernel_sizeSpa, isoSpa, vec)
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
    def __init__(self, channels, lap, kernel_sizeSph, kernel_sizeSpa, conv_name, isoSpa, old_conv=False, dense=True, precompute=True, einsum=False, repeat_interleave=True, vec=None):
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
        use_bn = True
        if conv_name=='bekkers':
            old_conv = False
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
            if old_conv:
                self.conv += [Conv(channels[i], channels[i+1], lap, kernel_sizeSph, kernel_sizeSpa, conv_name=conv_name_sel, isoSpa=isoSpa, dense=dense, precompute=precompute, einsum=einsum, repeat_interleave=repeat_interleave)]
            else:
                self.conv += [ConvPrecomputed(channels[i], channels[i+1], lap, kernel_sizeSph, kernel_sizeSpa, conv_name=conv_name_sel, isoSpa=isoSpa, vec=vec[i])]
            if use_bn:
                self.bn += [nn.BatchNorm3d(channels[i+1]*bn_mul)]
            else:
                self.bn += [nn.Identity()]
        self.conv = nn.ModuleList(self.conv)
        self.bn = nn.ModuleList(self.bn)
        # Activation
        self.activation = nn.ReLU()

    
class BlockSpatial(SubBlock):
    """GCNN Unet block.
    """
    def __init__(self, channels, lap, kernel_sizeSph, kernel_sizeSpa, conv_name, isoSpa, old_conv=False, dense=True, precompute=False, einsum=False, repeat_interleave=True):
        """Initialization.
        Args:
            channels (list int): Number of channels
            lap (list): Laplacian
            kernel_sizeSph (int): Size of the spherical kernel (i.e. Order of the Chebyshev polynomials + 1)
            kernel_sizeSpa (int): Size of the spatial kernel
            conv_name (str): Name of the convolution (spherical or mixed)
        """
        super(BlockSpatial, self).__init__(channels, lap, kernel_sizeSph, kernel_sizeSpa, conv_name, isoSpa, old_conv=old_conv, dense=dense, precompute=precompute, einsum=einsum, repeat_interleave=repeat_interleave)
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
    def __init__(self, channels, lap, kernel_sizeSph, kernel_sizeSpa, conv_name, isoSpa, old_conv=False, dense=True, precompute=False, einsum=False, repeat_interleave=True, vec=None):
        """Initialization.
        Args:
            channels (list int): Number of channels
            lap (list): Laplacian
            kernel_sizeSph (int): Size of the spherical kernel (i.e. Order of the Chebyshev polynomials + 1)
            kernel_sizeSpa (int): Size of the spatial kernel
            conv_name (str): Name of the convolution (spherical or mixed)
        """
        super(BlockProposed, self).__init__(channels, lap, kernel_sizeSph, kernel_sizeSpa, conv_name, isoSpa, old_conv=old_conv, dense=dense, precompute=precompute, einsum=einsum, repeat_interleave=repeat_interleave, vec=vec)
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
    def __init__(self, channels, lap, kernel_sizeSph, kernel_sizeSpa, conv_name, isoSpa, patch_size, keepSphericalDim=False, vec=None, n_vec=None, old_conv=False, dense=True, precompute=False, einsum=False, repeat_interleave=True):
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
        if conv_name =='bekkers':
            old_conv = False
        
        self.conv_name = conv_name
        if keepSphericalDim:
            if conv_name=='muller':
                from equideepdmri.layers.layer_builders import build_pq_layer
                out_filter = [channels[1]]
                if isoSpa:
                    in_filter = [channels[0]]
                else:
                    j = 0
                    in_filter = [math.ceil(channels[0]/(2**j))]
                    j = 2
                    while math.ceil(channels[0]/(2**j))>0:
                        in_filter += [math.ceil(channels[0]/(2**j))]
                        j += 1
                self.conv = build_pq_layer(in_filter, out_filter, #[1], [16, 4],
                                p_kernel_size=kernel_sizeSpa,
                                kernel="pq_TP",
                                q_sampling_schema_in=vec[0],
                                q_sampling_schema_out=vec[1],
                                use_batch_norm=False,
                                transposed=False,
                                non_linearity_config={"tensor_non_lin":"gated", "scalar_non_lin":"relu"},
                                use_non_linearity=False,
                                p_radial_basis_type="cosine",
                                p_radial_basis_params={"num_layers": 3, "num_units": 50},
                                sub_kernel_selection_rule={"l_diff_to_out_max": 1})
            elif conv_name=='MLP_head':
                self.conv = torch.nn.Linear(channels[0], channels[1])
            else:
                if old_conv:
                    self.conv = Conv(channels[0], channels[1], lap, kernel_sizeSph, kernel_sizeSpa, conv_name=conv_name, isoSpa=isoSpa, dense=dense, precompute=precompute, einsum=einsum, repeat_interleave=repeat_interleave)
                else:
                    self.conv = ConvPrecomputed(channels[0], channels[1], lap, kernel_sizeSph, kernel_sizeSpa, conv_name=conv_name, isoSpa=isoSpa, vec=vec[0])
        else:
            if conv_name=='muller':
                from equideepdmri.layers.layer_builders import build_p_layer
                out_filter = [channels[1]]
                if isoSpa:
                    in_filter = [channels[0]]
                else:
                    j = 0
                    in_filter = [math.ceil(channels[0]/(2**j))]
                    j = 2
                    while math.ceil(channels[0]/(2**j))>0:
                        in_filter += [math.ceil(channels[0]/(2**j))]
                        j += 1
                self.conv = build_p_layer(in_filter, out_filter,#[64, 8, 2], [10],
                            kernel_size=kernel_sizeSpa,
                            non_linearity_config={"tensor_non_lin":"gated", "scalar_non_lin":"relu"},
                            use_non_linearity=False,
                            use_batch_norm=False,
                            transposed=False,
                            p_radial_basis_type="cosine",
                            p_radial_basis_params={"num_layers": 3, "num_units": 50})
            elif conv_name=='MLP_head':
                self.conv = torch.nn.Linear(channels[0], channels[1])
            else:
                if conv_name=='spherical':
                    kernel_sizeSpa = 1
                if old_conv:
                    self.conv = Conv(channels[0], channels[1], torch.ones(1, 1), kernel_sizeSph, kernel_sizeSpa, conv_name='spatial', isoSpa=isoSpa, dense=dense, precompute=precompute, einsum=einsum, repeat_interleave=repeat_interleave)
                else:
                    self.conv = ConvPrecomputed(channels[0], channels[1], torch.ones(1, 1), kernel_sizeSph, kernel_sizeSpa, conv_name='spatial', isoSpa=isoSpa, vec=vec[0])
        
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


class MullerBlock(nn.Module):
    """GCNN Unet block.
    """
    def __init__(self, channels, kernel_sizeSpa, isoSpa, vec):
        """Initialization.
        Args:
            channels (list int): Number of channels
            lap (list): Laplacian
            kernel_sizeSph (int): Size of the spherical kernel (i.e. Order of the Chebyshev polynomials + 1)
            kernel_sizeSpa (int): Size of the spatial kernel
            conv_name (str): Name of the convolution (spherical or mixed)
        """
        super(MullerBlock, self).__init__()
        assert isinstance(channels, list)
        assert len(channels)>=2
        from equideepdmri.layers.layer_builders import build_pq_layer
        self.conv = []
        for i in range(len(channels)-1):
            if isoSpa:
                in_filter = [channels[i]]
                out_filter = [channels[i+1]]
            else:
                j = 0
                in_filter, out_filter = [math.ceil(channels[i]/(2**j))], [math.ceil(channels[i+1]/(2**j))]
                j = 2
                while math.ceil(channels[i]/(2**j))>0:
                    in_filter += [math.ceil(channels[i]/(2**j))]
                    j += 1
                j = 2
                while math.ceil(channels[i+1]/(2**j))>0:
                    out_filter += [math.ceil(channels[i+1]/(2**j))]
                    j += 1
            conv = build_pq_layer(in_filter, out_filter, #[1], [16, 4],
                p_kernel_size=kernel_sizeSpa,
                kernel="pq_TP",
                q_sampling_schema_in=vec[i],
                q_sampling_schema_out=vec[i+1],
                use_batch_norm=True,
                transposed=False,
                non_linearity_config={"tensor_non_lin":"gated", "scalar_non_lin":"relu"},
                use_non_linearity=True,
                p_radial_basis_type="cosine",
                p_radial_basis_params={"num_layers": 3, "num_units": 50},
                sub_kernel_selection_rule={"l_diff_to_out_max": 1})
            self.conv += [conv]
        self.conv = nn.ModuleList(self.conv)

    def forward(self, x):
        """Forward Pass.
        Args:
            x (:obj:`torch.Tensor`): input to be forwarded. [B x F_in_ch x X x Y x Z]
        Returns:
            :obj:`torch.Tensor`: output [B x F_out_ch x X x Y x Z]
        """
        for i in range(len(self.conv)):
            x = self.conv[i](x) # B x F_int_ch x X x Y x Z
        return x