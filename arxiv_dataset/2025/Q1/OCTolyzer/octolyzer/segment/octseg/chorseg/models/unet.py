import torch
import torch.nn as nn
import numpy as np
import timm.models as timm_models


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class ConvNormAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding: [int, str] = 'same', conv_kwargs=None,
                 norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU):
        super().__init__()
        if conv_kwargs is None:
            conv_kwargs = {}
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, **conv_kwargs)
        self.norm = norm_layer(out_channels)
        self.act = act_layer()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, conv_kwargs=None,
                 norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU, pool_layer=nn.MaxPool2d, down_size=2,
                 use_resid_connection=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if conv_kwargs is None:
            conv_kwargs = {}
        self.conv1 = ConvNormAct(in_channels, out_channels, kernel_size, padding=1, conv_kwargs=conv_kwargs,
                                 norm_layer=norm_layer, act_layer=act_layer)
        self.conv2 = ConvNormAct(out_channels, out_channels, kernel_size, padding=1, conv_kwargs=conv_kwargs,
                                 norm_layer=norm_layer, act_layer=act_layer)
        self.pool = pool_layer(kernel_size=down_size)

        self.use_resid_connection = use_resid_connection
        if use_resid_connection:
            self.resid_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.pool(x)
        x_out = self.conv1(x)
        x_out = self.conv2(x_out)
        if self.use_resid_connection:
            x_out += self.resid_connection(x)
        return x_out

    def __repr__(self):
        return f'DownBlock({self.in_channels}->{self.out_channels})'


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, conv_kwargs=None,
                 x_skip_channels=None, up_size = 2, norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU, 
                 up_type='interpolate', use_resid_connection=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.x_skip_channels = x_skip_channels or in_channels // up_size
        self.up_type = up_type
        if conv_kwargs is None:
            conv_kwargs = {}

        # upsample will double the number of channels, so we need to halve the number of channels in the input
        conv1_in_channels = in_channels // up_size + self.x_skip_channels

        self.conv1 = ConvNormAct(conv1_in_channels, out_channels, kernel_size, padding=1, conv_kwargs=conv_kwargs,
                                 norm_layer=norm_layer, act_layer=act_layer)
        self.conv2 = ConvNormAct(out_channels, out_channels, kernel_size, padding=1, conv_kwargs=conv_kwargs,
                                 norm_layer=norm_layer, act_layer=act_layer)
        if up_type == 'interpolate':
            self.upsample = nn.Sequential(nn.Upsample(scale_factor=up_size, mode='bilinear', align_corners=False),
                                          nn.Conv2d(in_channels, in_channels // up_size, kernel_size=1, stride=1))
        elif up_type == 'convtranspose':
            self.upsample = nn.ConvTranspose2d(in_channels, in_channels // up_size, kernel_size=up_size, stride=2)
        elif up_type == 'conv_then_interpolate':
            self.upsample = nn.Sequential(nn.Conv2d(in_channels, in_channels // up_size, kernel_size=1, stride=1),
                                          nn.Upsample(scale_factor=up_size, mode='bilinear', align_corners=False))
        else:
            raise ValueError(
                f'Unknown up_type: {up_type}, must be "interpolate", "convtranspose", "conv_then_interpolate"')

        self.use_resid_connection = use_resid_connection
        if use_resid_connection:
            self.resid_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x, x_skip):
        x = self.upsample(x)
        x = torch.cat([x, x_skip], dim=1)
        x_out = self.conv1(x)
        x_out = self.conv2(x_out)
        if self.use_resid_connection:
            x_out += self.resid_connection(x)
        return x_out

    def __repr__(self):
        return f'UpBlock({self.up_type}, {self.in_channels}->{self.out_channels})'


class PadIfNecessary(nn.Module):
    """Pad input to make it divisible by 2^depth. Has .pad() and .unpad() methods"""

    # TODO: fix
    def __init__(self, depth):
        super().__init__()
        self.depth = depth
        self.two_to_depth = 2 ** depth
        self.pad_amt = None
        self.unpad_loc = None

    def get_pad_amt(self, x):
        b, c, h, w = x.shape
        pad_h = (self.two_to_depth - h % self.two_to_depth) % self.two_to_depth
        pad_w = (self.two_to_depth - w % self.two_to_depth) % self.two_to_depth
        # pad_amt = [pad_left, pad_right, pad_top, pad_bottom]
        pad_amt = [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
        return pad_amt

    def get_unpad_loc(self, x):
        b, c, h, w = x.shape
        # unpad wil deal with padded inputs, so we need to account for the padding here
        h += self.pad_amt[2] + self.pad_amt[3]
        w += self.pad_amt[0] + self.pad_amt[1]

        # all elements in batch, all channels, top to bottom, left to right
        unpad_loc = [slice(None), slice(None),
                     slice(self.pad_amt[2], h - self.pad_amt[3]),
                     slice(self.pad_amt[0], w - self.pad_amt[1])]
        return unpad_loc

    def pad(self, x):
        if self.pad_amt is None:
            self.pad_amt = self.get_pad_amt(x)
            self.unpad_loc = self.get_unpad_loc(x)
        return nn.functional.pad(x, self.pad_amt)

    def unpad(self, x):
        if self.pad_amt is None:
            raise ValueError('Must call .pad() before .unpad()')
        return x[self.unpad_loc]
    

class SelfAttention(nn.Module):
    """
    Self-attention with equal input-output dimensions
    """
    def __init__(self, patch_size, input_dim, hidden_dim=None, feat_size=(12,12), 
                 up_type="interpolate", num_heads=8, attn_dropout=0.0, proj_dropout=0.0):
        super().__init__()

        self.patch_size = patch_size
        self.patch_flatten = patch_size**2
        self.feat_size = feat_size
        self.H, self.W = feat_size
        self.N_patches = int(self.H*self.W / self.patch_flatten)
        
        self.up_type = up_type
        self.input_dim = input_dim
        self.embed_dim = input_dim *  self.patch_flatten
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.proj_dropout = proj_dropout

        self.attention = timm_models.tnt.Attention(dim=self.embed_dim, hidden_dim=self.embed_dim, num_heads=num_heads, 
                                                  attn_drop=attn_dropout, proj_drop=proj_dropout)
        self.align_deep_res = self.get_resizer(feat_size)
       
    # 31/01/2024 - Removing conv option as the conv layers are trainable and are only ever used when dims != (768,768), so
    # won't be used every batch, and will observe dims of *higher* and *lower* resolution (problematic?).
    def get_resizer(self, feat_size):
        if self.up_type == "interpolate":
            align_resolution = nn.UpsamplingBilinear2d(size=feat_size)
        elif self.up_type == 'interpolate_then_conv':
            align_resolution = nn.Sequential(nn.UpsamplingBilinear2d(size=feat_size),
                                             nn.Conv2d(self.input_dim, self.input_dim, kernel_size=1, stride=1))
        elif self.up_type == 'conv_then_interpolate':
            align_resolution = nn.Sequential(nn.Conv2d(self.input_dim, self.input_dim, kernel_size=1, stride=1),
                                             nn.UpsamplingBilinear2d(size=feat_size))
        return align_resolution
        

    def patchify(self, x):
        '''
        Given an input image, extract overlapping patches given a custom stride
        '''
        # Patching stride and patch
        N = x.shape[0]
        shape = (N, self.input_dim, # Batch, filters
                 self.H - self.patch_size + 1, self.W - self.patch_size + 1, # N_strides vertically, horizontally
                 self.patch_size, self.patch_size) # Patch size vertically, horizontally
        H_str, W_str = x.stride(dim=-2), x.stride(dim=-1)
        strides = (1, 1, H_str, W_str, H_str, W_str)

        # Patch the input
        x = torch.as_strided(x, size=shape, stride=strides)
        x = x[:, :, ::self.patch_size, ::self.patch_size] # x is (batch_size, filters, patchH, patchW, patch_size, patch_size)

        # Reshape for attention input - put into a function
        # x = x.reshape(-1, self.input_dim, self.N_patches, self.patch_flatten) #N_patches=patchH*patchW, patch_flatten=patch_size**2
        # x = torch.transpose(x, 1, 2) # swap round filters and N_patches 
        # x = x.reshape(self.batch_size, self.N_patches, -1) # This is so we can flatten filters-patch_flatten
        
        return x

    
    def _reshape_input(self, x, reversed=False):
        """
        Reshape input for attenuating / upblocks
        """
        # x is (batch_size, filters, patchH, patchW, patch_size, patch_size)
        if not reversed:
            x = x.reshape(-1, self.input_dim, self.N_patches, self.patch_flatten) # (batch_size, filters, patchH*patchW, patch_size*patch_size) 
            x = torch.transpose(x, 1, 2) # (batch_size, patchH*patchW, filters, patch_size*patch_size) 
            x = x.reshape(-1, self.N_patches, self.input_dim*self.patch_flatten) # (batch_size, patchH*patchW, filters*patch_size*patch_size)

        # x is (batch_size, patchH*patchW, filters*patch_size*patch_size)
        else:
            x = x.reshape(-1, self.N_patches, self.input_dim, self.patch_flatten) # (batch_size, patchH*patchW, filters, patch_size*patch_size)
            x = torch.transpose(x, 1, 2) # (batch_size, filters, patchH*patchW, patch_size*patch_size)
            x = x.reshape(-1, self.input_dim, self.N_patches, self.patch_size, self.patch_size) # (batch_size, filters, patchH*patchW, patch_size, patch_size)
            x = x.reshape(-1, self.input_dim, *self.feat_size)  # (batch_size, filters, H, W)

        return x
            

    def forward(self, x):
        """
        ensure deepest point of network has (H,W) of self.feat_size
        extract patches and attenuate
        """
        H, W = x.shape[-2:]
        #print("input", x.shape)
        if (H, W) != self.feat_size: # handling multi-resolution
            x = self.align_deep_res(x) # Resizing to common res for self-attention
            correct_resolution = self.get_resizer((H, W)) # For outputting back to batch res
        #print("possible input resize", x.shape)
        x = self.patchify(x)
        #print("patch dims", x.shape)
        x = self._reshape_input(x)
        #print("attention reshape", x.shape)
        x = self.attention(x)
        #print("attention shape", x.shape)
        x = self._reshape_input(x, reversed=True)
        #print("output reshape", x.shape)
        if x.shape[-2:] != (H, W):
           x = correct_resolution(x)
        #print("possible output resize", x.shape)
            
        return x

    def __repr__(self):
        return f'SelfAttention({self.num_heads}, {self.input_dim}->{self.input_dim})'



class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, depth=4, channels: [int, str, list] = 32,
                 kernel_size=3, conv_kwargs=None, norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU, 
                 pool_layer=nn.MaxPool2d, block_factor=2, up_type='interpolate', self_attention=False,
                 extra_out_conv=False, use_resid_connection=False, dynamic_padding=False, verbose=False):
        super().__init__()

        self.depth = depth
        self.channels = channels
        self.kernel_size = kernel_size
        self.conv_kwargs = conv_kwargs
        self.norm_layer = norm_layer
        self.act_layer = act_layer
        self.up_type = up_type
        self.extra_out_conv = extra_out_conv
        self.use_resid_connection = use_resid_connection
        self.self_attention = self_attention
        self.verbose = verbose

        if isinstance(block_factor, int):
            self.block_factor = depth*[block_factor]
        elif isinstance(block_factor, list):
            if len(block_factor) != depth:
                self.block_factor = depth*[block_factor[0]]
            else:
                self.block_factor = block_factor
        self.depth_range = np.arange(depth)

        if isinstance(channels, int):
            # default to doubling channels at each layer
            channels = [channels * 2 ** i for i in range(depth + 1)]
        elif isinstance(channels, str):
            initial_channels, strategy = channels.split('_')
            initial_channels = int(initial_channels)
            if strategy == 'same':
                channels = [initial_channels] * (depth + 1)
                
            elif strategy == 'double':
                channels = [initial_channels * 2 ** i for i in range(depth + 1)]
            elif strategy.startswith('doublemax'):
                max_channels = int(strategy.split('-')[1])
                channels = [min(initial_channels * 2 ** i, max_channels) for i in range(depth + 1)]
                
            elif strategy == 'quadruple':
                channels = [initial_channels * 4 ** i for i in range(depth + 1)]
            elif strategy.startswith('quadruplemax'):
                max_channels = int(strategy.split('-')[1])
                channels = [min(initial_channels * 4 ** i, max_channels) for i in range(depth + 1)]
            else:
                raise ValueError(f'Unknown strategy: {strategy}')
                
        elif isinstance(channels, list):
            assert len(channels) == (depth + 1), f'channels must be a list of length {depth + 1}'

        self._unet_channels = channels

        if conv_kwargs is None:
            conv_kwargs = {}

        self.dynamic_padding = dynamic_padding
        if dynamic_padding:
            self.pad_if_necessary = PadIfNecessary(depth)

        self.in_conv = ConvNormAct(in_channels, channels[0], kernel_size, padding=1, conv_kwargs=conv_kwargs,
                                   norm_layer=norm_layer, act_layer=act_layer)

        # encoder
        self.down_blocks = nn.ModuleList()
        for bf, d in zip(self.block_factor, self.depth_range):
            self.down_blocks.append(DownBlock(channels[d], channels[d + 1], kernel_size, conv_kwargs=conv_kwargs,
                                              norm_layer=norm_layer, act_layer=act_layer, pool_layer=pool_layer,
                                              down_size=bf, use_resid_connection=use_resid_connection))

        # # self-attention at deepest layer
        # if self.self_attention:
        #     patch_size = 3
        #     embed_dim = channels[-1] # number of filters at deepest point of network
        #     feat_size = int((768 // 2**np.cumsum(np.log2(self.block_factor)))[-1]) # resolution of deepest point of network
        #     self.attention = SelfAttention(patch_size, embed_dim, feat_size=(feat_size, feat_size),
        #                                    num_heads=8, attn_dropout=0.0, proj_dropout=0.0, up_type="interpolate")

        # decoder
        self.up_blocks = nn.ModuleList()
        for bf, d in zip(self.block_factor[::-1], self.depth_range[::-1]):
            self.up_blocks.append(UpBlock(channels[d + 1], channels[d], kernel_size, conv_kwargs=conv_kwargs,
                                          x_skip_channels=channels[d], norm_layer=norm_layer, act_layer=act_layer, 
                                          up_size=bf, up_type=up_type, use_resid_connection=use_resid_connection))
            

        if not extra_out_conv:
            self.out_conv = nn.Conv2d(channels[0], out_channels, kernel_size=1, stride=1)
        else:
            self.out_conv = nn.Sequential(
                ConvNormAct(channels[0], channels[0], kernel_size, padding=1, conv_kwargs=conv_kwargs,
                            norm_layer=norm_layer, act_layer=act_layer),
                nn.Conv2d(channels[0], out_channels, kernel_size=1, stride=1)
            )

    def forward(self, x):
        if self.dynamic_padding:
            x = self.pad_if_necessary.pad(x)

        x_skip = []
        x_shapes = []

        # input convolution
        x = self.in_conv(x)
        x_shapes.append(x.shape)
        x_skip.append(x)

        # encoder
        for down_block in self.down_blocks:
            x = down_block(x)
            x_shapes.append(x.shape)
            x_skip.append(x)

        # # apply self-attention at deepest point of network
        # if self.self_attention:
        #     x = self.attention(x)

        # remove last element of x_skip, which is the last output of the down_blocks
        x_skip.pop()

        # decoder
        for up_block in self.up_blocks:
            x = up_block(x, x_skip.pop())
            x_shapes.append(x.shape)

        # final convolution
        x = self.out_conv(x)
        x_shapes.append(x.shape)

        if self.dynamic_padding:
            x = self.pad_if_necessary.unpad(x)
        #if self.verbose:
        #    return x, x_shapes
        return x


class WNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, depth=4, channels: [int, str, list] = 32,
                 kernel_size=3, conv_kwargs=None,
                 scale_factor_init=1e-5,
                 norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU, pool_layer=nn.MaxPool2d, up_type='interpolate',
                 use_resid_connection=False, dynamic_padding=False):
        super().__init__()

        self.unet1 = UNet(in_channels, out_channels, depth, channels, kernel_size, conv_kwargs,
                          norm_layer, act_layer, pool_layer, up_type,
                          use_resid_connection, dynamic_padding)
        self.unet2 = UNet(in_channels + out_channels, out_channels, depth, channels, kernel_size, conv_kwargs,
                          norm_layer, act_layer, pool_layer, up_type,
                          use_resid_connection, dynamic_padding)
        self.unet2_scale = nn.Parameter(torch.tensor(scale_factor_init))

    def forward(self, x):
        x1 = self.unet1(x)
        x2 = self.unet2(torch.cat([x, x1], dim=1))
        return x1 + self.unet2_scale * x2
