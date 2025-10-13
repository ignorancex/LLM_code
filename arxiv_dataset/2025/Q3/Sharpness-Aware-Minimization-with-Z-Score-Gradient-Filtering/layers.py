import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary


class TransformerEncoder(nn.Module):
    def __init__(self, feats:int, mlp_hidden:int, head:int=8, dropout:float=0., znorm_enabled=False):
        super(TransformerEncoder, self).__init__()
        self.la1 = nn.LayerNorm(feats)
        self.msa = MultiHeadSelfAttention(feats, head=head, dropout=dropout, znorm_enabled=znorm_enabled)  # ZNorm 적용 여부 전달
        self.la2 = nn.LayerNorm(feats)
        self.mlp = nn.Sequential(
            nn.Linear(feats, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, feats),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.znorm_enabled = znorm_enabled  # ZNorm 활성화 여부

    def forward(self, x):
        out = self.msa(self.la1(x)) + x
        out = self.mlp(self.la2(out)) + out
        return out

    def apply_znorm(self):
        """MultiHeadSelfAttention의 기울기를 정규화"""
        if self.znorm_enabled:
            self.msa.apply_znorm()



class MultiHeadSelfAttention(nn.Module):
    def __init__(self, feats:int, head:int=8, dropout:float=0., znorm_enabled=False):
        super(MultiHeadSelfAttention, self).__init__()
        self.head = head
        self.feats = feats
        self.sqrt_d = self.feats**0.5
        self.znorm_enabled = znorm_enabled  # ZNorm 활성화 여부

        self.q = nn.Linear(feats, feats)
        self.k = nn.Linear(feats, feats)
        self.v = nn.Linear(feats, feats)

        self.o = nn.Linear(feats, feats)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, n, f = x.size()
        q = self.q(x).view(b, n, self.head, self.feats//self.head).transpose(1,2)
        k = self.k(x).view(b, n, self.head, self.feats//self.head).transpose(1,2)
        v = self.v(x).view(b, n, self.head, self.feats//self.head).transpose(1,2)

        score = F.softmax(torch.einsum("bhif, bhjf->bhij", q, k)/self.sqrt_d, dim=-1) #(b,h,n,n)
        attn = torch.einsum("bhij, bhjf->bihf", score, v) #(b,n,h,f//h)
        o = self.dropout(self.o(attn.flatten(2)))

        return o

    def apply_znorm(self):
        """ZNorm을 적용하여 Q, K, V, O의 기울기를 정규화"""
        if not self.znorm_enabled:
            return
        
        for linear_layer in [self.q, self.k, self.v, self.o]:  # Query, Key, Value, Output
            if linear_layer.weight.grad is not None:
                grad_mean = linear_layer.weight.grad.mean(dim=list(range(1, linear_layer.weight.grad.dim())), keepdim=True)
                grad_std = linear_layer.weight.grad.std(dim=list(range(1, linear_layer.weight.grad.dim())), keepdim=True) + 1e-8
                linear_layer.weight.grad = (linear_layer.weight.grad - grad_mean) / grad_std


class MultiHeadDepthwiseSelfAttention(nn.Module):
    def __init__(self, feats:int, head:int=8, dropout:float=0):
        super(MultiHeadDepthwiseSelfAttention, self).__init__()
        ...

    def forward(self, x):
        
        ...

if __name__=="__main__":
    b,n,f = 4, 16, 128
    x = torch.randn(b,n,f)
    # net = MultiHeadSelfAttention(f)
    net = TransformerEncoder(f)
    torchsummary.summary(net, (n,f))
    # out = net(x)
    # print(out.shape)



