import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary


##################################################
# LayerNorm이 없는 Transformer Encoder
##################################################
class TransformerEncoder(nn.Module):
    def __init__(self, feats:int, mlp_hidden:int, head:int=8, dropout:float=0., znorm_enabled=False):
        super(TransformerEncoder, self).__init__()
        self.msa = MultiHeadSelfAttention(feats, head=head, dropout=dropout, znorm_enabled=znorm_enabled)
        # 기존 LayerNorm + MLP -> LayerNorm 제거 후 간단히 잔차 연결
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
        # LayerNorm이 없으므로 그대로 잔차 연결
        out = self.msa(x) + x
        out = self.mlp(out) + out
        return out

    def apply_znorm(self):
        """MultiHeadSelfAttention의 기울기에 대한 Z-Norm 적용"""
        if self.znorm_enabled:
            self.msa.apply_znorm()


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, feats:int, head:int=8, dropout:float=0., znorm_enabled=False):
        super(MultiHeadSelfAttention, self).__init__()
        self.head = head
        self.feats = feats
        self.sqrt_d = (self.feats ** 0.5)
        self.znorm_enabled = znorm_enabled

        self.q = nn.Linear(feats, feats)
        self.k = nn.Linear(feats, feats)
        self.v = nn.Linear(feats, feats)
        self.o = nn.Linear(feats, feats)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, n, f = x.size()
        q = self.q(x).view(b, n, self.head, self.feats // self.head).transpose(1, 2)
        k = self.k(x).view(b, n, self.head, self.feats // self.head).transpose(1, 2)
        v = self.v(x).view(b, n, self.head, self.feats // self.head).transpose(1, 2)

        score = F.softmax(torch.einsum("bhif, bhjf->bhij", q, k) / self.sqrt_d, dim=-1)  # (b,h,n,n)
        attn = torch.einsum("bhij, bhjf->bihf", score, v)                               # (b,n,h, feats//head)
        out = self.dropout(self.o(attn.flatten(2)))                                     # (b,n, feats)
        return out

    def apply_znorm(self):
        """Q, K, V, O 가중치의 기울기에 Z-Norm 적용"""
        if not self.znorm_enabled:
            return
        
        for linear_layer in [self.q, self.k, self.v, self.o]:
            if linear_layer.weight.grad is not None:
                grad_mean = linear_layer.weight.grad.mean(
                    dim=list(range(1, linear_layer.weight.grad.dim())), keepdim=True
                )
                grad_std = linear_layer.weight.grad.std(
                    dim=list(range(1, linear_layer.weight.grad.dim())), keepdim=True
                ) + 1e-8
                linear_layer.weight.grad = (linear_layer.weight.grad - grad_mean) / grad_std


##################################################
# LayerNorm이 없는 ViT
##################################################
class ViT(nn.Module):
    def __init__(
        self, 
        in_c:int=3, 
        num_classes:int=10, 
        img_size:int=32, 
        patch:int=8, 
        dropout:float=0., 
        num_layers:int=7, 
        hidden:int=384, 
        mlp_hidden:int=384*4, 
        head:int=8, 
        is_cls_token:bool=True, 
        znorm_enabled:bool=False
    ):
        super(ViT, self).__init__()

        self.patch = patch
        self.is_cls_token = is_cls_token
        self.patch_size = img_size // self.patch
        f = (img_size // self.patch) ** 2 * in_c  # 패치 하나가 갖는 픽셀 수(Flatten)
        num_tokens = (patch**2) + 1 if is_cls_token else (patch**2)

        # Patch Embedding
        self.emb = nn.Linear(f, hidden)

        # CLS Token (선택)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden)) if is_cls_token else None

        # 위치 임베딩
        self.pos_emb = nn.Parameter(torch.randn(1, num_tokens, hidden))

        # LayerNorm 없는 TransformerEncoder를 여러 개 쌓음
        enc_list = [
            TransformerEncoder(hidden, mlp_hidden=mlp_hidden, dropout=dropout, head=head, znorm_enabled=znorm_enabled) 
            for _ in range(num_layers)
        ]
        self.enc = nn.Sequential(*enc_list)

        # 최종 분류 모듈에서 LayerNorm 제거
        self.fc = nn.Linear(hidden, num_classes)

        self.znorm_enabled = znorm_enabled

    def forward(self, x):
        # (b, c, h, w) -> (b, n, f)
        out = self._to_words(x)        # 패치 분할 + Flatten
        out = self.emb(out)            # Patch Embedding
        if self.is_cls_token:
            # CLS 토큰 추가
            cls_token = self.cls_token.repeat(out.size(0), 1, 1)  # (b, 1, hidden)
            out = torch.cat([cls_token, out], dim=1)
        
        # 위치 임베딩
        out = out + self.pos_emb

        # Transformer Encoders
        out = self.enc(out)

        # 분류를 위한 pooling (CLS 토큰 또는 mean pooling)
        if self.is_cls_token:
            out = out[:, 0]  # CLS 토큰
        else:
            out = out.mean(dim=1)  # mean pooling

        # 최종 분류 모듈
        out = self.fc(out)
        return out

    def apply_znorm(self):
        """모든 TransformerEncoder 블록에서 Z-Norm 수행"""
        if self.znorm_enabled:
            for encoder in self.enc:
                encoder.apply_znorm()

    def _to_words(self, x):
        """
        (b, c, h, w) -> (b, patch^2, patch_size^2 * c)
        """
        # (b, c, h, w)
        # unfold -> (b, c, patch, patch, patch_size, patch_size)
        out = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        # 차원 순서를 (b, patch, patch, c, patch_size, patch_size)와 같이 맞추고 Flatten
        # unfold 결과: shape = (b, c, patch, patch, patch_size, patch_size)
        # reshape => (b, patch*patch, patch_size*patch_size*c)
        # permute 필요 여부는 batch 축과 patch 축만 신경 쓰면 됨
        b, c, ph, pw, hh, ww = out.size()
        out = out.permute(0, 2, 3, 1, 4, 5).reshape(b, ph*pw, c*hh*ww)
        return out


##################################################
# 테스트 코드
##################################################
if __name__ == "__main__":
    b,c,h,w = 4,3,32,32
    x = torch.randn(b, c, h, w)

    # LayerNorm이 완전히 제거된 ViT
    net = ViT(
        in_c=c, 
        num_classes=10, 
        img_size=h, 
        patch=16, 
        dropout=0.1, 
        num_layers=7, 
        hidden=384, 
        head=12, 
        mlp_hidden=384, 
        is_cls_token=False,
        znorm_enabled=False
    )

    torchsummary.summary(net, (c, h, w))

    out = net(x)
    print(f"Output shape: {out.shape}")
