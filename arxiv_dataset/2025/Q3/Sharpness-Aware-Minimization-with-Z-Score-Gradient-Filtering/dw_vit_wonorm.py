import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary


class TransformerEncoderNoNorm(nn.Module):
    def __init__(self, feats:int, mlp_hidden:int, head:int=8, dropout:float=0., znorm_enabled=False):
        super(TransformerEncoderNoNorm, self).__init__()
        self.msa = MultiHeadSelfAttentionNoNorm(feats, head=head, dropout=dropout, znorm_enabled=znorm_enabled)
        self.mlp = nn.Sequential(
            nn.Linear(feats, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, feats),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.znorm_enabled = znorm_enabled

    def forward(self, x):
        # LayerNorm이 없으므로 바로 잔차 연결
        out = self.msa(x) + x
        out = self.mlp(out) + out
        return out

    def apply_znorm(self):
        """MultiHeadSelfAttention의 기울기를 정규화(Z-Norm)"""
        if self.znorm_enabled:
            self.msa.apply_znorm()


class MultiHeadSelfAttentionNoNorm(nn.Module):
    def __init__(self, feats:int, head:int=8, dropout:float=0., znorm_enabled=False):
        super(MultiHeadSelfAttentionNoNorm, self).__init__()
        self.head = head
        self.feats = feats
        self.sqrt_d = self.feats**0.5
        self.znorm_enabled = znorm_enabled

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

        score = F.softmax(torch.einsum("bhif, bhjf->bhij", q, k) / self.sqrt_d, dim=-1)  # (b,h,n,n)
        attn = torch.einsum("bhij, bhjf->bihf", score, v)                               # (b,n,h,f//h)
        o = self.dropout(self.o(attn.flatten(2)))
        return o

    def apply_znorm(self):
        """Z-Norm을 적용하여 Q, K, V, O 가중치의 기울기를 정규화"""
        if not self.znorm_enabled:
            return
        
        for linear_layer in [self.q, self.k, self.v, self.o]:
            if linear_layer.weight.grad is not None:
                grad_mean = linear_layer.weight.grad.mean(dim=list(range(1, linear_layer.weight.grad.dim())), keepdim=True)
                grad_std = linear_layer.weight.grad.std(dim=list(range(1, linear_layer.weight.grad.dim())), keepdim=True) + 1e-8
                linear_layer.weight.grad = (linear_layer.weight.grad - grad_mean) / grad_std

##################################################
# DepthwiseViT에서 LayerNorm 제거
##################################################
class DepthwiseViTNoNorm(nn.Module):
    def __init__(self, in_c:int=3, num_classes:int=10, img_size:int=32, patch:int=8, 
                 dropout:float=0., num_layers:int=7, hidden:int=384, 
                 mlp_hidden:int=384*4, head:int=8):
        super(DepthwiseViTNoNorm, self).__init__()
        # 예시로 hidden=384, patch=8 등을 사용.
        # LayerNorm이 없으므로 TransformerEncoderNoNorm 사용
        self.patch = patch
        self.patch_size = img_size // self.patch
        f = self.patch_size**2  # 패치 1개의 픽셀 개수

        self.emb = nn.Linear(f, hidden)  # (b, c, n, f)
        self.cls_token = nn.Parameter(torch.randn(1, 1, 1, hidden))
        self.pos_emb = nn.Parameter(torch.randn(1, 1, (self.patch**2) + 1, hidden))

        enc_list = [TransformerEncoderNoNorm(hidden, mlp_hidden=mlp_hidden, dropout=dropout, head=head)
                    for _ in range(num_layers)]
        self.enc = nn.Sequential(*enc_list)

        # 최종 분류 모듈에서 LayerNorm 제거
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x):
        # (b, c, h, w) -> (b, c, n, hw/n)
        out = self._to_words(x)
        # cls_token 추가
        out = torch.cat([self.cls_token.repeat(out.size(0), 1, 1, 1), self.emb(out)], dim=1)
        # 위치 임베딩 추가
        out = out + self.pos_emb
        # 트랜스포머 인코더
        out = self.enc(out)
        # 첫 번째(cls) 토큰을 분류
        out = out[:, 0]
        out = self.fc(out)
        return out

    def _to_words(self, x):
        """
        (b, c, h, w) -> (b, c, n, patch_size^2)
        patch 단위로 나누고, 각 patch를 1D 벡터로 만듦
        """
        b, c, _, _ = x.size()
        out = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        # out shape: (b, c, patch, patch, patch_size, patch_size)
        out = out.reshape(b, c, self.patch**2, -1)  # => (b, c, patch^2, patch_size^2)
        return out


if __name__ == "__main__":
    b, c, h, w = 4, 3, 32, 32
    x = torch.randn(b, c, h, w)

    # LayerNorm이 전혀 없는 DepthwiseViT
    net = DepthwiseViTNoNorm(
        in_c=c, 
        num_classes=10, 
        img_size=h, 
        patch=16, 
        dropout=0.1, 
        num_layers=7, 
        hidden=384//3,   # 예시로 384//3 사용
        head=12, 
        mlp_hidden=384
    )

    torchsummary.summary(net, (c, h, w))
    
    # 예시 forward
    out = net(x)
    print(f"Output shape: {out.shape}")
