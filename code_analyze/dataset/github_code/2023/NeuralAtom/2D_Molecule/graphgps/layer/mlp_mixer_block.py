from torch import nn

# adapt from https://github.com/lucidrains/mlp-mixer-pytorch/tree/main


class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x


def FeedForward(dim, expansion_factor=4, dropout=0.0, dense=nn.Linear):
    inner_dim = int(dim * expansion_factor)
    return nn.Sequential(
        dense(dim, inner_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(inner_dim, dim),
        nn.Dropout(dropout),
    )


# def MLPMixer(
#     *,
#     # topk,
#     in_dim,
#     dim,
#     out_dim,
#     depth,
#     expansion_factor=4,
#     expansion_factor_token=0.5,
#     dropout=0.0
# ):
#     chan_first, chan_last = partial(nn.Conv1d, kernel_size=1), nn.Linear

#     return nn.Sequential(
#         nn.Linear(in_dim, dim),
#         *[
#             nn.Sequential(
#                 # PreNormResidual(
#                 #     dim, FeedForward(topk, expansion_factor, dropout, chan_first)
#                 # ),
#                 PreNormResidual(
#                     dim, FeedForward(
#                         dim, expansion_factor_token, dropout, chan_last)
#                 ),
#             )
#             for _ in range(depth)
#         ],
#         nn.LayerNorm(dim),
#         # Reduce("n k d -> n d", "mean"),
#         nn.Linear(dim, out_dim)  # * [N, k, D] -> [N, D]
#     )

# def MLPMixer(
#     *,
#     in_dim,
#     dim,
#     out_dim,
#     depth,
#     expansion_factor_token=0.5,
#     dropout=0.0
# ):
#     chan_last = nn.Linear

#     return nn.Sequential(
#         nn.Linear(in_dim, dim),
#         *[
#             nn.Sequential(
#                 PreNormResidual(
#                     dim, FeedForward(
#                         dim, expansion_factor_token, dropout, chan_last)
#                 ),
#             )
#             for _ in range(depth)
#         ],
#         nn.LayerNorm(dim),
#         nn.Linear(dim, out_dim)
#     )

def MLPMixer(
    *,
    in_dim,
    dim,
    depth,
    expansion_factor_token=0.5,
    dropout=0.0
):
    chan_last = nn.Linear

    return nn.Sequential(
        nn.Linear(in_dim, dim),
        *[
            nn.Sequential(
                PreNormResidual(
                    dim, FeedForward(
                        dim, expansion_factor_token, dropout, chan_last)
                ),
            )
            for _ in range(depth)
        ],
        nn.LayerNorm(dim),
        nn.Linear(dim, in_dim),
    )