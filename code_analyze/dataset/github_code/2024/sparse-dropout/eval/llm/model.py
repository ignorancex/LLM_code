import torch
import torch.nn as nn
from torch.nn import functional as F

from flash_dropout.layers import DropoutMM


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embed: int, n_head: int, dropout: dict, batch_first: bool):
        assert n_embed % n_head == 0

        super().__init__()
        self.n_head = n_head
        self.n_embed = n_embed
        self.batch_first = batch_first

        self.q_proj = DropoutMM(n_embed, n_embed, **dropout)
        self.k_proj = DropoutMM(n_embed, n_embed, **dropout)
        self.v_proj = DropoutMM(n_embed, n_embed, **dropout)
        self.out_proj = DropoutMM(n_embed, n_embed, **dropout)

    def forward(self, x):
        # ... d -> ... 3d
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        if not self.batch_first:
            q = q.transpose(0, 1)
            k = k.transpose(0, 1)
            v = v.transpose(0, 1)

        b, s, d = q.shape

        # b s d -> b nh s hd
        q = q.view(b, s, self.n_head, d // self.n_head).transpose(1, 2)
        k = k.view(b, s, self.n_head, d // self.n_head).transpose(1, 2)
        v = v.view(b, s, self.n_head, d // self.n_head).transpose(1, 2)

        # Attn((b nh s hd), (b nh s hd), (b nh s hd)) -> (b nh s hd)
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)

        # b nh s hd -> b s d
        y = y.transpose(1, 2).contiguous().view(b, s, d)
        y = self.out_proj(y)

        if not self.batch_first:
            y = y.transpose(0, 1)

        return y


class MLP(nn.Module):
    def __init__(self, n_embed: int, dropout: dict):
        super().__init__()
        self.n_embed = n_embed

        self.c_fc = DropoutMM(n_embed, 4 * n_embed, **dropout)
        self.gelu = nn.GELU()
        self.c_proj = DropoutMM(4 * n_embed, n_embed, **dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, n_embed: int, n_head: int, dropout: dict, batch_first: bool):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embed, bias=False)
        self.attn = CausalSelfAttention(n_embed, n_head, dropout, batch_first)
        self.ln_2 = nn.LayerNorm(n_embed, bias=False)
        self.mlp = MLP(n_embed, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        n_layer: int,
        n_head: int,
        n_embed: int,
        dropout: dict,
        batch_first: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embed = n_embed
        self.dropout = dropout
        self.batch_first = batch_first

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(vocab_size, n_embed),
                wpe=nn.Embedding(context_length, n_embed),
                h=nn.ModuleList(
                    [
                        Block(n_embed, n_head, dropout, batch_first)
                        for _ in range(n_layer)
                    ]
                ),
                ln_f=nn.LayerNorm(n_embed, bias=False),
            )
        )
        self.lm_head = nn.Linear(n_embed, vocab_size, bias=False)

        print(f"Num parameters: {self.get_num_params():,}")

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def forward(self, idx: torch.Tensor, calc_loss: bool = True):
        if self.batch_first:
            b, s = idx.shape
        else:
            s, b = idx.shape

        assert s <= self.context_length

        tok_emb = self.transformer.wte(idx)
        pos = torch.arange(0, s, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        pos_emb = pos_emb.unsqueeze(0) if self.batch_first else pos_emb.unsqueeze(1)

        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if self.batch_first:
            logits = self.lm_head(x)[:, :-1, :].contiguous()
        else:
            logits = self.lm_head(x)[:-1, :, :]

        if calc_loss:
            labels = idx[:, 1:].contiguous() if self.batch_first else idx[1:, :]
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), labels.view(-1))
        else:
            loss = None

        return logits, loss

    def configure_optimizers(self, weight_decay: float, lr: float):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"Num decayed parameters {num_decay_params:,}")
        print(f"Num non-decayed parameters {num_nodecay_params:,}")

        optimizer = torch.optim.AdamW(optim_groups, lr=lr, fused=True)
        return optimizer

    def estimate_mfu(self, batch_size: int, dt: float):
        """
        Args:
            dt: time per forward-backward pass in seconds
        """
        N = self.get_num_params()
        L = self.n_layer
        H = self.n_head
        Q = self.n_embed // self.n_head
        T = self.context_length

        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_achieved = batch_size * flops_per_fwdbwd * (1.0 / dt)  # per second
        flops_promised = 20e12  # RTX2060 GPU peak flops is about 20 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # If the sequence context is growing too long we must crop it at block_size
            idx_cond = idx[:, -self.context_length :]
            logits, _ = self(idx_cond, calc_loss=False)
            # Greedy decoding
            idx_next = torch.argmax(logits, dim=-1).unsqueeze(-1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
