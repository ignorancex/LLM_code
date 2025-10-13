import torch
import pickle
import numpy as np

from utils import qip_optimizer

def optimization():
    proxy_matrix = torch.load(f"./clado_proxy_matrix_wikitext2.pt")
    layer_size = torch.load("./layer_size.pt")
    layer_bitops = torch.load("./layer_bitops.pt")
    constraints = torch.load("./eval_constraints.pt")

    optimal_decisions, optimal_objective_vals = [], []
    optimal_decisions_file = "./clado_optimal_decisions.pt"
    optimal_objective_vals_file = "./clado_optimal_objective_values.pt"
    for constraint in constraints:
        (size_bound, bitops_bound) = (constraint, np.inf) 
        print(f"constraint value: {constraint}")    
        mpq_decision, mpq_objective_val = qip_optimizer(
                                                        cached_proxy=proxy_matrix,
                                                        layer_bitops=layer_bitops,
                                                        layer_size=layer_size,
                                                        schemes_per_layer=3,
                                                        bitops_bound=bitops_bound,
                                                        size_bound=size_bound,
                                                        diagonal=False,
                                                        printInfo=True
                                            )
        optimal_decisions.append(mpq_decision)
        optimal_objective_vals.append(mpq_objective_val)
    torch.save(optimal_decisions, optimal_decisions_file)
    torch.save(optimal_objective_vals, optimal_objective_vals_file)

def main():
    optimization()


if __name__ == "__main__":
    main()
