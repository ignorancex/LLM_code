import torch
import torch.nn as nn


class QFormerProjModel(nn.Module):
    def __init__(self,
                 cross_attention_dim=4096,
                 id_embeddings_dim=1152,
                 num_tokens=128,
                 num_heads=8,
                 num_query_tokens=32):
        super().__init__()
        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens

        self.query_embeds = nn.Parameter(torch.randn(num_tokens, cross_attention_dim))

        self.id_proj = nn.Sequential(
            nn.Linear(id_embeddings_dim, id_embeddings_dim * 2),
            nn.GELU(),
            nn.Linear(id_embeddings_dim * 2, cross_attention_dim * num_query_tokens)
        )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=cross_attention_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.cross_attn_norm = nn.LayerNorm(cross_attention_dim)

        self.norm = nn.LayerNorm(cross_attention_dim)

    def forward(self, id_embeds):
        batch_size = id_embeds.size(0)

        projected = self.id_proj(id_embeds)
        kv = projected.view(batch_size, -1, self.cross_attention_dim)

        queries = self.query_embeds.unsqueeze(0).expand(batch_size, -1, -1)

        attn_output, _ = self.cross_attn(
            query=queries,
            key=kv,
            value=kv
        )
        attn_output = self.cross_attn_norm(attn_output + queries)

        return self.norm(attn_output)


class MLPProjModel(torch.nn.Module):
    def __init__(self,
                 cross_attention_dim=768,
                 id_embeddings_dim=512,
                 num_tokens=4):
        super().__init__()
        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens

        self.proj = torch.nn.Sequential(
            torch.nn.Linear(id_embeddings_dim, id_embeddings_dim * 2),
            torch.nn.GELU(),
            torch.nn.Linear(id_embeddings_dim * 2, cross_attention_dim * num_tokens),
        )
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, id_embeds):
        x = self.proj(id_embeds)
        x = x.reshape(-1, self.num_tokens, self.cross_attention_dim)
        x = self.norm(x)
        return x
