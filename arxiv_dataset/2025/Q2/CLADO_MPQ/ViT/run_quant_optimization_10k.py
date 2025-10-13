#!/usr/bin/env python

import torch
import pickle
import numpy as np

from utils import (
    format_space,
    clado_proxy_post_process,
    qip_optimizer
)

def feintLady(n_samples=1024):

    num_runs = 1
    for run in range(num_runs):
        proxy = torch.load(f"./clado_mpqco_results/CLADO/Clado_proxy/sample_size{n_samples}/clado_proxy_run{run}.pt")
        layers_to_quant = torch.load("./intermediate_files/layers_to_quant.pt")
        layer_size = torch.load("./intermediate_files/layer_size_W-4-3-2.pt")
        layer_bitops = torch.load("./intermediate_files/layer_bitops_W-4-3-2.pt")

        proxy_matrix, _ = clado_proxy_post_process(
            proxy=proxy,
            layers_to_quant=layers_to_quant,
            format_space=format_space,
        )

        size_bounds = np.append(np.linspace(20.25, 29.84210526, 19), np.linspace(30.375, 40.5, 20))
        
        seed = 43
        
        clado_res, clado_objective = [], []
        for size_bound in size_bounds:
            print(f'Set size bound to {size_bound} MB')
            # clado
            v1, objective_val = qip_optimizer(
                                                cached_proxy=proxy_matrix,
                                                layer_bitops=layer_bitops,
                                                layer_size=layer_size,
                                                schemes_per_layer=3,
                                                bitops_bound=np.inf,
                                                size_bound=size_bound,
                                                diagonal=False,
                                                printInfo=True
                                            )
            clado_res.append(v1)
            clado_objective.append(objective_val)

        with open(f'./clado_mpqco_results/CLADO/Clado_optimal_decisions_ViT/sample_size{n_samples}/clado_a8_w4-3-2_seed{seed}_optimization_run{run}.pkl','wb') as f:
            pickle.dump({'clado_res': clado_res}, f)
        with open(f'./clado_mpqco_results/CLADO/Clado_optimal_objectives_ViT/sample_size{n_samples}/clado_a8_w4-3-2_seed{seed}_optimization_objective_run{run}.pkl','wb') as f:
            pickle.dump({'clado_objectives': clado_objective}, f)

def main():
    feintLady(n_samples=10240)


if __name__ == "__main__":
    main()
