import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import math
import numpy as np
import os
import tqdm
from typing import Tuple, Optional, Union, List, Literal
import src.utils.sparse as sparse_utils
import src.utils.normalizer as normalize
import src.compression_parent as compression_parent


class SparseLinear(compression_parent.CompressedLinear):
    name = "SparseLinear"

    def initalize_sparse(
        self,
        sparse_types: List[Literal["dim_0", "dim_1", "unstructured"]],
        frac_sparse: Union[float, List[float]] = 0.1,
        **kwargs,
    ):
        """create a sparse compensator

        Args:
            sparse_type (List[Literal[&quot;hessian&quot;,&quot;dim_0&quot;,&quot;dim_1&quot;,&quot;unstructed&quot;]]): the types of sparsity to use
            frac_sparse (Union[float,List[float]], optional): how much sparsity to use, if a list is given, its
            length should be the same as sparse_type, and it will be the fraction of sparsity for each type. If float, then we equally split the sparsity among the types. Defaults to 0.1.
        """

        if isinstance(frac_sparse, float):
            frac_sparse = (
                [frac_sparse / len(sparse_types)] * len(sparse_types)
                if len(sparse_types) > 0
                else []
            )

        sparse_modules = []
        for i, sparse_type in enumerate(sparse_types):
            # print("sparse_type", sparse_type)
            if frac_sparse[0] <= 0.0:
                new_sparse_module = None
            elif sparse_type == "dim_0":
                new_sparse_module = sparse_utils.Dim0_StructuredSparse(
                    self.out_features,
                    self.in_features,
                    frac_sparse[i],
                    self.original_weight.device,
                )
            elif sparse_type == "dim_1":
                new_sparse_module = sparse_utils.Dim1_StructuredSparse(
                    self.out_features,
                    self.in_features,
                    frac_sparse[i],
                    self.original_weight.device,
                )
            elif sparse_type == "unstructured":
                new_sparse_module = sparse_utils.UnstructuredSparse(
                    self.out_features,
                    self.in_features,
                    frac_sparse[i],
                    self.original_weight.device,
                    pattern=kwargs.get("pattern", None),
                    sparse_group=kwargs.get("group", -1),
                    stochastic=kwargs.get("stochastic", False),
                    temp=kwargs.get("temp", 1.0),
                )
            else:
                raise NotImplementedError

            sparse_modules.append(new_sparse_module)

        self.sparse_modules: nn.ModuleList[sparse_utils.SparseParent] = nn.ModuleList(
            sparse_modules
        )
        self.sparsed = True
        self.sparse_after_norm = False

    @torch.no_grad()
    def sparsify(
        self,
        sparse_types: List[Literal["dim_0", "dim_1", "unstructured"]] = [
            "unstructured"
        ],
        sparse_criterions: Union[
            Literal["wanda", "hessian", "norm_1", "norm_2", "norm_inf"],
            List[Literal["wanda", "hessian", "norm_1", "norm_2", "norm_inf"]],
        ] = "wanda",
        frac_sparse: Union[float, List[float]] = 0.1,
        normalizer_kwargs: Optional[dict] = None,
        normalizer: Optional[normalize.Normalizer] = None,
        **kwargs,
    ):
        """create a sparse compensator

        Args:
            sparse_type (List[Literal[&quot;hessian&quot;,&quot;dim_0&quot;,&quot;dim_1&quot;,&quot;unstructed&quot;]]): the types of sparsity to use
            sparse_criterion (Union[Literal[&quot;wanda&quot;,&quot;hessian&quot;,&quot;norm_1&quot, &quot;norm_2&quot, &quot;norm_inf&quot], List[Literal[&quot;wanda&quot;,&quot;hessian&quot;,&quot;norm_1&quot, &quot;norm_2&quot, &quot;norm_inf&quot]]): the criterion to use for sparsity

            frac_sparse (Union[float,List[float]], optional): how much sparsity to use, if a list is given, its
            length should be the same as sparse_type, and it will be the fraction of sparsity for each type. If float, then we equally split the sparsity among the types. Defaults to 0.1.
        """

        normalized_weight = self.initialize_normalizer(
            normalizer=normalizer, normalizer_kwargs=normalizer_kwargs
        )

        self.initalize_sparse(sparse_types, frac_sparse, **kwargs)

        if isinstance(sparse_criterions, str):
            sparse_criterions = [sparse_criterions] * len(sparse_types)

        for i, sparse_type in enumerate(sparse_types):
            sparse_criterion = sparse_criterions[i]
            if self.sparse_modules[i] is None:
                print("skipping because of 0 sparsity")
                continue
            if sparse_criterion == "hessian":
                assert (
                    sparse_type == "dim_1"
                ), "hessian sparsity is only makes sense for dim_1 sparsity"
                self.sparse_modules[i].update(
                    self.hessian, normalized_weight, lambda x: torch.norm(x, dim=0)
                )
            elif "norm" in sparse_criterion:
                if "dim" in sparse_type:
                    self.sparse_modules[i].sparse(
                        normalized_weight,
                        normalized_weight,
                        lambda x: torch.norm(
                            x,
                            dim=int(sparse_type.split("_")[-1]),
                            p=(
                                int(sparse_criterion.split("_")[-1])
                                if sparse_criterion != "norm_inf"
                                else float("inf")
                            ),
                        ),
                    )
                else:
                    raise NotImplementedError(
                        f"sparse_type {sparse_type} not implemented for {sparse_criterion}"
                    )

            elif sparse_criterion == "wanda":

                # compute the importances
                hessianDiag = self.get_hessianDiag()
                # tqdm.tqdm.write(f"device of normalized_weight {normalized_weight.device}, device of hessianDiag {hessianDiag.device}")
                importances = normalized_weight**2 * hessianDiag.unsqueeze(0)

                if "dim" in sparse_type:
                    self.sparse_modules[i].sparse(
                        importances,
                        normalized_weight,
                        lambda x: torch.norm(x, dim=int(sparse_type.split("_")[-1])),
                    )
                elif sparse_type == "unstructured":
                    # print("here")
                    self.sparse_modules[i].sparse(importances, normalized_weight)
                    # raise Exception("Stop here")
                else:
                    raise NotImplementedError(
                        f"sparse_type {sparse_type} not implemented"
                    )

            # print(self.sparse_modules[i].reconstruct()[:,self.sparse_modules[i].sparse_mask])

            # print(remaining_error[:,self.sparse_modules[i].sparse_mask])
            normalized_weight = normalized_weight - self.sparse_modules[i].reconstruct()

    def compress(
        self,
        sparse_types: List[Literal["dim_0", "dim_1", "unstructured"]] = [
            "unstructured"
        ],
        sparse_criterions: Union[
            Literal["wanda", "hessian", "norm_1", "norm_2", "norm_inf"],
            List[Literal["wanda", "hessian", "norm_1", "norm_2", "norm_inf"]],
        ] = "wanda",
        frac_sparse: Union[float, List[float]] = 0.1,
        normalizer_kwargs: Optional[dict] = None,
        normalizer: Optional[normalize.Normalizer] = None,
        **kwargs,
    ):
        self.compressed = True
        return self.sparsify(
            sparse_types=sparse_types,
            sparse_criterions=sparse_criterions,
            frac_sparse=frac_sparse,
            normalizer_kwargs=normalizer_kwargs,
            normalizer=normalizer,
            **kwargs,
        )

    def _no_checkpoint_forward(self, x: torch.FloatTensor):
        if self.forward_method == "reconstruct":
            if self.denormalization_method == "otf":
                y = F.linear(
                    self.normalizer.denormalize_otf_in(x),
                    self.reconstruct(denormalize=False),
                )
                y = self.normalizer.denormalize_otf_out(y) + (
                    self.bias if self.bias is not None else 0
                )
            else:
                # tqdm.tqdm.write(f"x dtype {x.dtype}, denormalize dtype {self.reconstruct(denormalize = self.denormalization_method == 'reconstruct').dtype}")
                y = F.linear(
                    x,
                    self.reconstruct(
                        denormalize=self.denormalization_method == "reconstruct"
                    ),
                    self.bias,
                )
        else:
            assert (
                self.denormalization_method == "otf"
            ), "on the fly denormalization is only supported for on the fly sparsity"
            x = self.normalizer.denormalize_otf_in(x)
            y = torch.zeros(list(x.shape[:-1]) + [self.out_features], device=x.device)
            for sparse_module in self.sparse_modules:
                if sparse_module is not None:
                    y = y + sparse_module(x)
            y = self.normalizer.denormalize_otf_out(y) + (
                self.bias if self.bias is not None else 0
            )
        return y

    def get_n_bits(self):
        n_bits = 0
        if self.compressed:
            for sparse_module in self.sparse_modules:
                n_bits += sparse_module.get_n_bits()
        return n_bits

    def blank_recreate(
        self,
        sparse_types: List[Literal["dim_0", "dim_1", "unstructed"]] = ["unstructured"],
        frac_sparse: Union[float, List[float]] = 0.1,
        normalizer_kwargs: Optional[dict] = None,
        normalizer: Optional[normalize.Normalizer] = None,
        **kwargs,
    ):

        if normalizer is not None:
            self.normalizer = normalizer
        else:
            self.normalizer = normalize.Normalizer.blank_recreate(
                self.original_weight, **normalizer_kwargs
            )

        self.initalize_sparse(sparse_types, frac_sparse, **kwargs)
        self.compressed = True

    def reconstruct_(self, denormalize: bool = True) -> torch.FloatTensor:
        """reconstructs the weigth matrix from the quantized version"""
        # print("reconstructing")
        reconstructed = self.sparse_modules[0].reconstruct()
        for sparse_module in self.sparse_modules[1:]:
            if sparse_module is not None:
                reconstructed = reconstructed + sparse_module.reconstruct()

        if denormalize:
            reconstructed = self.normalizer.denormalize(reconstructed)

        return reconstructed

    def get_n_nonzero(self) -> int:

        recon = self.reconstruct(denormalize=False)

        n_nonzero = torch.sum(recon != 0).item()
        return n_nonzero
