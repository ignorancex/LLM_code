import torch
import torch.nn as nn
from einops import rearrange
import math



class DepMHSA(nn.Module):
    def __init__(self, n_dims, n_frames=2, width=14, height=14, heads=4, pos_enc = True, q_kernel='133',k_kernel='311',v_kernel='311133'):
        super(DepMHSA, self).__init__()
        self.scale = (n_dims // heads) ** -0.5
        self.heads = heads
        self.pos_enc = pos_enc
        q_kernel = [int(x) for x in list(q_kernel)]
        k_kernel = [int(x) for x in list(k_kernel)]
        v_kernel = [int(x) for x in list(v_kernel)]

        self.query = nn.Conv3d(in_channels=n_dims, out_channels=n_dims, kernel_size=(q_kernel[0], q_kernel[1], q_kernel[2]), stride=1, padding='same',
                            bias=False)
        self.key = nn.Conv3d(in_channels=n_dims, out_channels=n_dims, kernel_size=(k_kernel[0], k_kernel[1], k_kernel[2]), stride=1, padding='same',
                            bias=False)
        self.value = nn.Sequential(nn.Conv3d(in_channels=n_dims, out_channels=n_dims, kernel_size=(v_kernel[0], v_kernel[1], v_kernel[2]), stride=1, padding='same',
                            bias=False),
                                nn.Conv3d(in_channels=n_dims, out_channels=n_dims, kernel_size=(v_kernel[3], v_kernel[4], v_kernel[5]), stride=1, padding='same',
                            bias=False))
        
        # 1, h, dim, 1, height, 1
        self.rel_h = nn.Parameter(torch.randn([1, heads, n_dims // heads, 1, height, 1]), requires_grad=True)
        # 1, h, dim, 1, 1, w
        self.rel_w = nn.Parameter(torch.randn([1, heads, n_dims // heads, 1, 1, width]), requires_grad=True)
        # 1, h, dim , f, 1, 1
        self.rel_t = nn.Parameter(torch.randn([1, heads, n_dims // heads, n_frames, 1, 1]), requires_grad=True)
        
        # 1, h, dim, 1, height, 1
        self.rel_h_2 = nn.Parameter(torch.randn([1, n_dims, 1, height, 1]), requires_grad=True)
        # 1, h, dim, 1, 1, w
        self.rel_w_2 = nn.Parameter(torch.randn([1, n_dims, 1, 1, width]), requires_grad=True)
        # 1, h, dim , f, 1, 1
        self.rel_t_2 = nn.Parameter(torch.randn([1, n_dims, n_frames, 1, 1]), requires_grad=True)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        n_batch, C, F, height, width = x.size()        
        q = self.query(x).reshape(n_batch, self.heads, F, C // self.heads, -1) # B, Head, F, C, HW
        k = self.key(x).reshape(n_batch, self.heads, F, C // self.heads, -1)
        v = self.value(x).reshape(n_batch, self.heads, F, C // self.heads, -1)
        
        qk_weights = torch.matmul(q.permute(0, 1, 2, 4, 3), k) # B Head F C HW -> B Head F H*W C x B Head F C H*W -> B Head F H*W * H*W
        if self.pos_enc:
            # 1, Head, dim, F, H, W = 1, 4, 128, 2, 12, 12
            relative_positions = self.rel_h + self.rel_w + self.rel_t
            
            # 1, Head, dim, F, H*W -> 1, Head, F, H*W, dim | 1, 4, 128, 2, 1024 -> 1, 4, 2, 144, 128
            multihead_relative_positions = relative_positions.reshape(1, self.heads, C // self.heads, F, -1).permute(0, 1, 3, 4, 2)
            multihead_q_times_pos = torch.matmul(multihead_relative_positions, q)

            # # content_content: B, h, F, H, W 4, 4, 2, 256, 256
            energy = (qk_weights + multihead_q_times_pos) * self.scale
        else:
            energy = qk_weights * self.scale
            
        attention = self.softmax(energy)
        mhsa_out = torch.matmul(v, attention.permute(0, 1, 2, 4, 3)) # B Head F C H*W x B head F H*W H*W -> B Head F C H*W
        # out = out.reshape(n_batch, C, F, height, width)
        mhsa_out = mhsa_out.reshape(n_batch, C, F, height, width)
        out = mhsa_out + (self.rel_h_2 + self.rel_t_2 + self.rel_w_2)
        return out
    
class MHSA3D(nn.Module):
    "modified from https://github.com/SanoScience/BabyNet"
    def __init__(self, n_dims, n_frames=2, width=14, height=14, heads=4, pos_enc = True):
        super(MHSA3D, self).__init__()
        self.scale = (n_dims // heads) ** -0.5
        self.heads = heads
        self.pos_enc = pos_enc
        self.query = nn.Conv3d(in_channels=n_dims, out_channels=n_dims, kernel_size=(1, 1, 1), stride=1, padding=0,
                               bias=False)
        self.key = nn.Conv3d(in_channels=n_dims, out_channels=n_dims, kernel_size=(1, 1, 1), stride=1, padding=0,
                             bias=False)
        self.value = nn.Conv3d(in_channels=n_dims, out_channels=n_dims, kernel_size=(1, 1, 1), stride=1, padding=0,
                               bias=False)
        # 1, h, dim, 1, height, 1
        self.rel_h = nn.Parameter(torch.randn([1, heads, n_dims // heads, 1, height, 1]), requires_grad=True)
        # 1, h, dim, 1, 1, w
        self.rel_w = nn.Parameter(torch.randn([1, heads, n_dims // heads, 1, 1, width]), requires_grad=True)
        # 1, h, dim , f, 1, 1
        self.rel_t = nn.Parameter(torch.randn([1, heads, n_dims // heads, n_frames, 1, 1]), requires_grad=True)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # cab_output = self.cab(x)
        n_batch, C, F, height, width = x.size()
        q = self.query(x).reshape(n_batch, self.heads, F, C // self.heads, -1) # B, Head, F, C, H*W
        k = self.key(x).reshape(n_batch, self.heads, F, C // self.heads, -1) # B, Head, F, C, H*W
        v = self.value(x).reshape(n_batch, self.heads, F, C // self.heads, -1) # B, Head, F, C, H*W

        spatio_attention_QK = torch.matmul(q.permute(0, 1, 2, 4, 3), k) # B Head F C HW -> B Head F H*W C x B Head F C H*W -> B Head F H*W * H*W
        
        if self.pos_enc:
            content_rel_pos = self.rel_h + self.rel_w + self.rel_t
            content_position = content_rel_pos.reshape(1, self.heads, C // self.heads, F, -1).permute(0, 1, 3, 4, 2)
            content_position2 = torch.matmul(content_position, q)
            energy = (spatio_attention_QK + content_position2) * self.scale
        else:
            energy = spatio_attention_QK * self.scale
        attention = self.softmax(energy)
        mhsa_out = torch.matmul(v, attention.permute(0, 1, 2, 4, 3)) # B Head F C H*W x B head F H*W H*W -> B Head F C H*W
        mhsa_out = mhsa_out.reshape(n_batch, C, F, height, width)
        out = mhsa_out
        return out
    
    
