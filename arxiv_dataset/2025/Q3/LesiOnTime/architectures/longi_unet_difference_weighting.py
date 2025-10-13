import pydoc
import warnings
from os.path import join
from torch import nn
import torch

import dynamic_network_architectures

from difference_weighting.utils import recursive_find_python_class
from difference_weighting.building_blocks.difference_weighting_block import DifferenceWeightingBlock
import math
from monai.networks.nets import VoxelMorphUNet, VoxelMorph
use_gpu = torch.cuda.is_available()

# class VoxelMorph3D_MONAI(nn.Module):
#     def __init__(self, in_channels=2, use_gpu=False):
#         super(VoxelMorph3D_MONAI, self).__init__()
#         # First, a backbone network is constructed. In this case, we use a VoxelMorphUNet as the backbone network.
#         self.backbone = VoxelMorphUNet(
#                         spatial_dims=3,
#                         in_channels=2,
#                         unet_out_channels=32,
#                         channels=(16, 32, 32, 32, 32, 32),  # this indicates the down block at the top takes 16 channels as
#                                                             # input, the corresponding up block at the top produces 32
#                                                             # channels as output, the second down block takes 32 channels as
#                                                             # input, and the corresponding up block at the same level
#                                                             # produces 32 channels as output, etc.
#                         final_conv_channels=(16, 16)
#                     )

#         # Then, a full VoxelMorph network is constructed using the specified backbone network.
#         self.net = VoxelMorph(
#                     backbone=self.backbone,
#                     integration_steps=7,
#                     half_res=False
#                     )
        
#         self.use_gpu=use_gpu
#         if use_gpu:
#             self.backbone = self.backbone.cuda()
#             self.net = self.net.cuda()
    
#     def forward(self, moving_image, fixed_image): # (b,c,d,h,w)
#         #print(moving_image.dtype,fixed_image.dtype)
#         #self.net=self.net.cuda()
#         #self.net=self.net.half()
#         #moving_image=moving_image.half()
#         #fixed_image=fixed_image.half()

#         dtypes = set(param.dtype for param in self.net.parameters())
#         #print("Parameter dtypes in model:", dtypes)
#         #print(moving_image.shape,fixed_image.shape)
#         if self.use_gpu:
#             moving_image=moving_image[:,0,:,:,:].cuda()
#             fixed_image=fixed_image[:,0,:,:,:].cuda()
#         else:
#             moving_image=moving_image[:,0,:,:,:].cpu()
#             fixed_image=fixed_image[:,0,:,:,:].cpu()      
#         moving_image=moving_image.unsqueeze(1)
#         fixed_image=fixed_image.unsqueeze(1)
#         #print(moving_image.type(),fixed_image.type())
#         #print(moving_image.shape,fixed_image.shape)
#         regsisterd_image,ddf = self.net(moving_image,fixed_image)

#         return regsisterd_image,ddf


class TPALayer6D(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        t = int(abs((math.log(channels, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: [B, C, T, D, H, W]
        B, C, T, D, H, W = x.shape

        # Global average pooling over T, D, H, W -> [B, C]
        y = x.mean(dim=[2, 3, 4, 5])  # -> [B, C]

        # Conv1d expects [B, 1, C]
        y = y.unsqueeze(1)  # -> [B, 1, C]
        y = self.conv(y)    # -> [B, 1, C]
        y = self.sigmoid(y)

        # Reduce over channels -> [B, 1]
        y = y.squeeze(1).mean(dim=1, keepdim=True)

        return y  # Shape: [B, 1]
    
class LongiUNetDiffWeighting(nn.Module):
    
    def __init__(self, input_channels, num_classes, backbone_class_name, **architecture_kwargs):
        print("Class LongiUNetDiffWeighting")
        super().__init__()
        backbone_class = pydoc.locate(backbone_class_name)
        # sometimes things move around, this makes it so that we can at least recover some of that
        if backbone_class is None:
            warnings.warn(f'Network class {backbone_class_name} not found. Attempting to locate it within '
                        f'dynamic_network_architectures.architectures...')
            backbone_class = recursive_find_python_class(join(dynamic_network_architectures.__path__[0], "architectures"),
                                                backbone_class_name.split(".")[-1],
                                                'dynamic_network_architectures.architectures')
            if backbone_class is not None:
                print(f'FOUND IT: {backbone_class}')
            else:
                raise ImportError('Network class could not be found, please check/correct your plans file')

        # basic channel concatenation
        self.backbone = backbone_class(
            input_channels=input_channels,
            num_classes=num_classes,
            **architecture_kwargs
        )

        self.skip_diff_weighting = DifferenceWeightingBlock(architecture_kwargs['features_per_stage'], architecture_kwargs['conv_op'])

    def forward(self, d_c, d_p=None,is_tpa=False):
        # allow for concatenation at different points in the code
        if d_p is None:
            d_c, d_p = torch.tensor_split(d_c, 2, dim=1)
     
        skips_current = self.backbone.encoder(d_c)
        skips_prior = self.backbone.encoder(d_p)

        skips_weights=[]
        if is_tpa:
            skips_concat=[]
            for DIM in range(len(skips_current)): ## iterate at every layer of UNet

                ## Apply the TPA module at every layer and save the skip_weights
                skips_concat.append(torch.cat([skips_current[DIM].unsqueeze(0), skips_prior[DIM].unsqueeze(0)], dim=0)) 
                skips_weights.append(self.tpa(skips_concat[DIM]))
        else:

            for DIM in range(len(skips_current)):
                skips_weights.append([1,1]) ## 1-vector if no tpa applied

        skips = self.skip_diff_weighting(skips_current, skips_prior,skips_weights)

        x = self.backbone.decoder(skips)
        return x,skips_current,skips_prior
    
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            if hasattr(self.backbone, name):
                return getattr(self.backbone, name)
            raise

    def __setattr__(self, name, value):
        if name != 'backbone' and hasattr(self, 'backbone') and hasattr(self.backbone, name):
            setattr(self.backbone, name, value)
        else:
            super().__setattr__(name, value)


class LesiOnTime(nn.Module):
    
    def __init__(self, input_channels, num_classes, backbone_class_name, **architecture_kwargs):
        super().__init__()
        backbone_class = pydoc.locate(backbone_class_name)
        # sometimes things move around, this makes it so that we can at least recover some of that
        if backbone_class is None:
            warnings.warn(f'Network class {backbone_class_name} not found. Attempting to locate it within '
                        f'dynamic_network_architectures.architectures...')
            backbone_class = recursive_find_python_class(join(dynamic_network_architectures.__path__[0], "architectures"),
                                                backbone_class_name.split(".")[-1],
                                                'dynamic_network_architectures.architectures')
            if backbone_class is not None:
                print(f'FOUND IT: {backbone_class}')
            else:
                raise ImportError('Network class could not be found, please check/correct your plans file')

        # basic channel concatenation
        self.backbone = backbone_class(
            input_channels=input_channels,
            num_classes=num_classes,
            **architecture_kwargs
        )

        self.skip_diff_weighting = DifferenceWeightingBlock(architecture_kwargs['features_per_stage'], architecture_kwargs['conv_op'])
        self.tpa=TPALayer6D(channels=2)
    def forward(self, d_c, d_p=None,is_tpa=False):
        # allow for concatenation at different points in the code
        if d_p is None:
            if d_c.ndim==4:
                d_c=d_c.unsqueeze(0)
            d_c, d_p = torch.tensor_split(d_c, 2, dim=1)

        skips_current = self.backbone.encoder(d_c)
        skips_prior = self.backbone.encoder(d_p)
 
        skips_weights=[]
        if is_tpa:
            skips_concat=[]
            for DIM in range(len(skips_current)): ## iterate at every layer of UNet
                ## Apply the TPA module at every layer and save the skip_weights
                skips_concat.append(torch.cat([skips_current[DIM].unsqueeze(0), skips_prior[DIM].unsqueeze(0)], dim=0)) 
                skips_weights.append(self.tpa(skips_concat[DIM]))
        
<<<<<<< HEAD
        else:
            for DIM in range(len(skips_current)):
                skips_weights.append([1,1]) ## 1-vector if no tpa applied

        skips = self.skip_diff_weighting(skips_current,skips_prior,skips_weights) ## need to change skips_weights for standard longi-baseline

        x = self.backbone.decoder(skips)
        return x,skips_current,skips_prior
    
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            if hasattr(self.backbone, name):
                return getattr(self.backbone, name)
            raise

    def __setattr__(self, name, value):
        if name != 'backbone' and hasattr(self, 'backbone') and hasattr(self.backbone, name):
            setattr(self.backbone, name, value)
        else:
            super().__setattr__(name, value)


import pydoc
import warnings
from os.path import join
from torch import nn
import torch

import dynamic_network_architectures

from difference_weighting.utils import recursive_find_python_class
from difference_weighting.building_blocks.difference_weighting_block import DifferenceWeightingBlock
import math
from monai.networks.nets import VoxelMorphUNet, VoxelMorph
use_gpu = torch.cuda.is_available()

# class VoxelMorph3D_MONAI(nn.Module):
#     def __init__(self, in_channels=2, use_gpu=False):
#         super(VoxelMorph3D_MONAI, self).__init__()
#         # First, a backbone network is constructed. In this case, we use a VoxelMorphUNet as the backbone network.
#         self.backbone = VoxelMorphUNet(
#                         spatial_dims=3,
#                         in_channels=2,
#                         unet_out_channels=32,
#                         channels=(16, 32, 32, 32, 32, 32),  # this indicates the down block at the top takes 16 channels as
#                                                             # input, the corresponding up block at the top produces 32
#                                                             # channels as output, the second down block takes 32 channels as
#                                                             # input, and the corresponding up block at the same level
#                                                             # produces 32 channels as output, etc.
#                         final_conv_channels=(16, 16)
#                     )

#         # Then, a full VoxelMorph network is constructed using the specified backbone network.
#         self.net = VoxelMorph(
#                     backbone=self.backbone,
#                     integration_steps=7,
#                     half_res=False
#                     )
        
#         self.use_gpu=use_gpu
#         if use_gpu:
#             self.backbone = self.backbone.cuda()
#             self.net = self.net.cuda()
    
#     def forward(self, moving_image, fixed_image): # (b,c,d,h,w)
#         #print(moving_image.dtype,fixed_image.dtype)
#         #self.net=self.net.cuda()
#         #self.net=self.net.half()
#         #moving_image=moving_image.half()
#         #fixed_image=fixed_image.half()

#         dtypes = set(param.dtype for param in self.net.parameters())
#         #print("Parameter dtypes in model:", dtypes)
#         #print(moving_image.shape,fixed_image.shape)
#         if self.use_gpu:
#             moving_image=moving_image[:,0,:,:,:].cuda()
#             fixed_image=fixed_image[:,0,:,:,:].cuda()
#         else:
#             moving_image=moving_image[:,0,:,:,:].cpu()
#             fixed_image=fixed_image[:,0,:,:,:].cpu()      
#         moving_image=moving_image.unsqueeze(1)
#         fixed_image=fixed_image.unsqueeze(1)
#         #print(moving_image.type(),fixed_image.type())
#         #print(moving_image.shape,fixed_image.shape)
#         regsisterd_image,ddf = self.net(moving_image,fixed_image)

#         return regsisterd_image,ddf


class TPALayer6D(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        t = int(abs((math.log(channels, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: [B, C, T, D, H, W]
        B, C, T, D, H, W = x.shape

        # Global average pooling over T, D, H, W -> [B, C]
        y = x.mean(dim=[2, 3, 4, 5])  # -> [B, C]

        # Conv1d expects [B, 1, C]
        y = y.unsqueeze(1)  # -> [B, 1, C]
        y = self.conv(y)    # -> [B, 1, C]
        y = self.sigmoid(y)

        # Reduce over channels -> [B, 1]
        y = y.squeeze(1).mean(dim=1, keepdim=True)

        return y  # Shape: [B, 1]
    
class LongiUNetDiffWeighting(nn.Module):
    
    def __init__(self, input_channels, num_classes, backbone_class_name, **architecture_kwargs):
        print("Class LongiUNetDiffWeighting")
        super().__init__()
        backbone_class = pydoc.locate(backbone_class_name)
        # sometimes things move around, this makes it so that we can at least recover some of that
        if backbone_class is None:
            warnings.warn(f'Network class {backbone_class_name} not found. Attempting to locate it within '
                        f'dynamic_network_architectures.architectures...')
            backbone_class = recursive_find_python_class(join(dynamic_network_architectures.__path__[0], "architectures"),
                                                backbone_class_name.split(".")[-1],
                                                'dynamic_network_architectures.architectures')
            if backbone_class is not None:
                print(f'FOUND IT: {backbone_class}')
            else:
                raise ImportError('Network class could not be found, please check/correct your plans file')

        # basic channel concatenation
        self.backbone = backbone_class(
            input_channels=input_channels,
            num_classes=num_classes,
            **architecture_kwargs
        )

        self.skip_diff_weighting = DifferenceWeightingBlock(architecture_kwargs['features_per_stage'], architecture_kwargs['conv_op'])

    def forward(self, d_c, d_p=None,is_tpa=False):
        # allow for concatenation at different points in the code
        if d_p is None:
            d_c, d_p = torch.tensor_split(d_c, 2, dim=1)
     
        skips_current = self.backbone.encoder(d_c)
        skips_prior = self.backbone.encoder(d_p)

        skips_weights=[]
        if is_tpa:
            skips_concat=[]
            for DIM in range(len(skips_current)): ## iterate at every layer of UNet

                ## Apply the TPA module at every layer and save the skip_weights
                skips_concat.append(torch.cat([skips_current[DIM].unsqueeze(0), skips_prior[DIM].unsqueeze(0)], dim=0)) 
                skips_weights.append(self.tpa(skips_concat[DIM]))
        else:

            for DIM in range(len(skips_current)):
                skips_weights.append([1,1]) ## 1-vector if no tpa applied

        skips = self.skip_diff_weighting(skips_current, skips_prior,skips_weights)

        x = self.backbone.decoder(skips)
        return x,skips_current,skips_prior
    
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            if hasattr(self.backbone, name):
                return getattr(self.backbone, name)
            raise

    def __setattr__(self, name, value):
        if name != 'backbone' and hasattr(self, 'backbone') and hasattr(self.backbone, name):
            setattr(self.backbone, name, value)
        else:
            super().__setattr__(name, value)


class LesiOnTime(nn.Module):
    
    def __init__(self, input_channels, num_classes, backbone_class_name, **architecture_kwargs):
        super().__init__()
        backbone_class = pydoc.locate(backbone_class_name)
        # sometimes things move around, this makes it so that we can at least recover some of that
        if backbone_class is None:
            warnings.warn(f'Network class {backbone_class_name} not found. Attempting to locate it within '
                        f'dynamic_network_architectures.architectures...')
            backbone_class = recursive_find_python_class(join(dynamic_network_architectures.__path__[0], "architectures"),
                                                backbone_class_name.split(".")[-1],
                                                'dynamic_network_architectures.architectures')
            if backbone_class is not None:
                print(f'FOUND IT: {backbone_class}')
            else:
                raise ImportError('Network class could not be found, please check/correct your plans file')

        # basic channel concatenation
        self.backbone = backbone_class(
            input_channels=input_channels,
            num_classes=num_classes,
            **architecture_kwargs
        )

        self.skip_diff_weighting = DifferenceWeightingBlock(architecture_kwargs['features_per_stage'], architecture_kwargs['conv_op'])
        self.tpa=TPALayer6D(channels=2)
    def forward(self, d_c, d_p=None,is_tpa=False):
        # allow for concatenation at different points in the code
        if d_p is None:
            if d_c.ndim==4:
                d_c=d_c.unsqueeze(0)
            d_c, d_p = torch.tensor_split(d_c, 2, dim=1)

        skips_current = self.backbone.encoder(d_c)
        skips_prior = self.backbone.encoder(d_p)
 
        skips_weights=[]
        if is_tpa:
            skips_concat=[]
            for DIM in range(len(skips_current)): ## iterate at every layer of UNet
                ## Apply the TPA module at every layer and save the skip_weights
                skips_concat.append(torch.cat([skips_current[DIM].unsqueeze(0), skips_prior[DIM].unsqueeze(0)], dim=0)) 
                skips_weights.append(self.tpa(skips_concat[DIM]))
        
        else:
            for DIM in range(len(skips_current)):
                skips_weights.append([1,1]) ## 1-vector if no tpa applied

        skips = self.skip_diff_weighting(skips_current,skips_prior,skips_weights) ## need to change skips_weights for standard longi-baseline

        x = self.backbone.decoder(skips)
        return x,skips_current,skips_prior
=======
        else:
            for DIM in range(len(skips_current)):
                skips_weights.append([1,1]) ## 1-vector if no tpa applied

        skips = self.skip_diff_weighting(skips_current,skips_prior,skips_weights) ## need to change skips_weights for standard longi-baseline

        x = self.backbone.decoder(skips)
        return x,skips_current,skips_prior
>>>>>>> f769cfa9edb6504ece08427987c3f5b03d0d36bb
    
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            if hasattr(self.backbone, name):
                return getattr(self.backbone, name)
            raise

    def __setattr__(self, name, value):
        if name != 'backbone' and hasattr(self, 'backbone') and hasattr(self.backbone, name):
            setattr(self.backbone, name, value)
        else:
            super().__setattr__(name, value)


