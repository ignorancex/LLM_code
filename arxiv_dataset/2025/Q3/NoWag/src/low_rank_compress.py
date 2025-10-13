import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import math
import numpy as np
import os
import sys

if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from typing import Tuple, Optional, Union, List, Literal
from src.utils.normalizer import Normalizer
from src.compression_parent import CompressedLinear


class LowRankLinear(CompressedLinear):
    name = "LowRankLinear"

    def low_rank(
        self,
        rank: int,
        normalizer_kwargs: dict = {},
        normalizer: Normalizer = None,
    ):

        normalized_weight = self.initialize_normalizer(
            normalizer=normalizer, normalizer_kwargs=normalizer_kwargs
        )

        hessianDiag = self.get_hessianDiag()

        U, S, V = torch.svd(normalized_weight @ torch.diag(torch.sqrt(hessianDiag)))

        # select the top rank singular values

        self.A = nn.Parameter(U[:, :rank] @ torch.diag(S[:rank]))

        self.B = nn.Parameter(V[:, :rank].T @ torch.diag(1 / torch.sqrt(hessianDiag)))

        # raise Exception("Stop here")
        self.compressed = True

    def compress(
        self, rank: int, normalizer_kwargs: dict = {}, normalizer: Normalizer = None
    ):
        return self.low_rank(
            rank=rank, normalizer_kwargs=normalizer_kwargs, normalizer=normalizer
        )

    def blank_recreate(
        self, rank, normalizer_kwargs: dict = {}, normalizer: Normalizer = None
    ):
        if normalizer is not None:
            self.normalizer = normalizer

        else:
            self.normalizer = Normalizer.blank_recreate(
                self.original_weight, **normalizer_kwargs
            )

        self.A = nn.Parameter(torch.randn(self.out_features, rank))
        self.B = nn.Parameter(torch.randn(rank, self.in_features))

    def reconstruct_(self, denormalize: bool = True):
        if denormalize:
            return self.normalizer.denormalize(self.A @ self.B)
        else:
            return self.A @ self.B

    def _no_checkpoint_forward(self, x):

        if self.forward_method == "oft":
            if self.denormalization_method == "otf":
                return self.normalizer.denormalize_otf_out(
                    F.linear(
                        F.linear(self.normalizer.denormalize_otf_in(x), self.B),
                        self.A,
                        self.bias,
                    )
                )
            elif self.denormalization_method == "ignore":
                return F.linear(F.linear(x, self.B), self.A, self.bias)
            else:
                raise NotImplementedError(
                    f"denormalization method {self.denormalization_method} not implemented"
                )

        else:
            if self.denormalization_method == "otf":
                return self.normalizer.denormalize_otf_out(
                    F.linear(x, self.reconstruct(denormalize=False), self.bias)
                )
            elif self.denormalization_method == "ignore":
                return F.linear(x, self.reconstruct(denormalize=False), self.bias)
            elif self.denormalization_method == "reconstruct":
                return F.linear(x, self.reconstruct(denormalize=True), self.bias)
            else:
                raise NotImplementedError(
                    f"denormalization method {self.denormalization_method} not implemented"
                )

    def get_n_bits(self):
        return self.A.numel() * 16 + self.B.numel() * 16 + self.normalizer.get_n_bits()


if __name__ == "__main__":

    test_weight = "/data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/original_weights/layer_1/mlp.up_proj.pt"
    test_hessian = "/data/lliu/huffman/models/meta-llama/Llama-2-7b-hf/hessians/pajama/128/layer_0/mlp.up_proj.pt"

    device = torch.device("cuda:0")
    weight = torch.load(test_weight, map_location=device, weights_only=True)[
        "weight"
    ].float()
    hessian = torch.load(test_hessian, map_location=device, weights_only=True)[
        "hessian"
    ].float()

    low_rank = LowRankLinear(weight=weight)
    print(low_rank)
    low_rank.hessian = hessian

    low_rank.low_rank(
        rank=256, normalizer_kwargs=dict(norm_order=[0, 1], zero=[False, False])
    )

    low_rank.change_denormalization_method("otf")

    x = torch.randn(100, weight.shape[1]).to(device)

    y = low_rank(x)

    # print(y.shape)
    print("loss=", low_rank.get_reconstruction_error(hessian))
    print("bpv=", low_rank.get_n_bits() / low_rank.get_n_original_parameters())
