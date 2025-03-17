import torch
import torch.nn as nn
import torch.nn.functional as F


class FFN(nn.Module):
    """ A two-feed-forward-layer module """

    def __init__(self, channels, dropout=0.1):
        super(FFN, self).__init__()
        self.lin1 = nn.Linear(channels, channels)  # position-wise
        self.lin2 = nn.Linear(channels, channels)  # position-wise
        self.layer_norm = nn.LayerNorm(channels, eps=1e-6)
        self.Dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x = self.Dropout(x)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        x = x + residual
        return x

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.layer_norm.reset_parameters()


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        # self.label_same_matrix = torch.load('analysis/label_same_matrix_citeseer.pt').float()

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        # self.label_same_matrix = self.label_same_matrix.to(attn.device)
        # attn = attn * self.label_same_matrix * 2 + attn * (1-self.label_same_matrix)
        attn = self.dropout(F.softmax(attn, dim=-1))
        # attn = self.dropout(attn)

        output = torch.matmul(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention module """

    def __init__(self, n_head, channels, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.channels = channels
        d_q = d_k = d_v = channels // n_head

        self.w_qs = nn.Linear(channels, channels, bias=False)
        self.w_ks = nn.Linear(channels, channels, bias=False)
        self.w_vs = nn.Linear(channels, channels, bias=False)
        self.fc = nn.Linear(channels, channels, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        n_head = self.n_head
        d_q = d_k = d_v = self.channels // n_head
        B_q = q.size(0)
        N_q = q.size(1)
        B_k = k.size(0)
        N_k = k.size(1)
        B_v = v.size(0)
        N_v = v.size(1)

        residual = q
        # x = self.dropout(q)

        # Pass through the pre-attention projection: B * N x (h*dv)
        # Separate different heads: B * N x h x dv
        q = self.w_qs(q).view(B_q, N_q, n_head, d_q)
        k = self.w_ks(k).view(B_k, N_k, n_head, d_k)
        v = self.w_vs(v).view(B_v, N_v, n_head, d_v)

        # Transpose for attention dot product: B * h x N x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # For head axis broadcasting.
        if mask is not None:
            mask = mask.unsqueeze(1)

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: B x N x h x dv
        # Combine the last two dimensions to concatenate all the heads together: B x N x (h*dv)
        q = q.transpose(1, 2).contiguous().view(B_q, N_q, -1)
        q = self.fc(q)
        q = q + residual

        return q, attn

    def reset_parameters(self):
        self.w_qs.reset_parameters()
        self.w_ks.reset_parameters()
        self.w_vs.reset_parameters()
        self.fc.reset_parameters()


class TFormer(nn.Module):
    def __init__(self, n_head, channels, use_patch_attn=True, dropout=0.1):
        super(TFormer, self).__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.transformer1 = MultiHeadAttention(n_head, channels, dropout)
        self.ffn1 = FFN(channels, dropout)

    def forward(self, x, patch, attn_mask=None, need_attn=False):
        x = self.norm1(x)
        x, attn = self.transformer1(x, x, x, attn_mask)
        x = self.ffn1(x)
        return x

    def reset_parameters(self):
        self.norm1.reset_parameters()
        self.transformer1.reset_parameters()
        self.ffn1.reset_parameters()
