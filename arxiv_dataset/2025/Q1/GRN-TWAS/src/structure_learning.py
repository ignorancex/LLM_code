import os
import pandas as pd
import numpy as np
import yaml
import findr
import networkx as nx
import json

# Function to calculate p-values using Findr
def calculate_p_values(expression_A, expression_ALL, genotype, method, n=None):
    """
    Calculate posterior probabilities using Findr.
    """
    p0_results = method.pij_rank(dt=expression_A, dt2=expression_ALL, nodiag=True)
    p0 = p0_results['p'][:, :n] if n else p0_results['p']

    p_other_results = method.pijs_gassist(dg=genotype, dt=expression_A, dt2=expression_ALL, nodiag=True)
    p2 = p_other_results['p2'][:, :n] if n else p_other_results['p2']
    p3 = p_other_results['p3'][:, :n] if n else p_other_results['p3']
    p4 = p_other_results['p4'][:, :n] if n else p_other_results['p4']
    p5 = p_other_results['p5'][:, :n] if n else p_other_results['p5']

    p2p3 = p2 * p3
    p2p5 = p2 * p5
    p = 0.5 * (p2p5 + p4)

    return {
        'p0': p0,
        'p2': p2,
        'p3': p3,
        'p4': p4,
        'p5': p5,
        'p2p3': p2p3,
        'p2p5': p2p5,
        'p': p,
    }

# Function to reconstruct GRN
def reconstruct_grn(input_file, output_folder, findr_path, posterior_threshold=0.75):
    """
    Reconstruct a gene regulatory network (GRN) for a single tissue dataset.
    """
    print(f"Reconstructing GRN for dataset: {input_file}")

    # Load the dataset
    data = pd.read_csv(input_file, compression='gzip')

    # Extract expression and genotype data
    sample_ids = [col for col in data.columns if col != 'id']
    expression_A = data[sample_ids].to_numpy(dtype=np.float64)
    expression_ALL = expression_A  # For simplicity, use the same expression matrix
    genotype = data[sample_ids].to_numpy(dtype=np.float64)

    # Initialize Findr
    findr_lib = findr.lib(path=findr_path, loglv=6, rs=0, nth=0)

    # Calculate p-values
    posteriors = calculate_p_values(expression_A, expression_ALL, genotype, findr_lib)

    # Filter posterior probabilities based on the threshold
    filtered_posteriors = np.where(posteriors['p'] >= posterior_threshold, posteriors['p'], 0)

    # Save the filtered results
    output_file = os.path.join(output_folder, "grn_posteriors.csv.gz")
    pd.DataFrame(filtered_posteriors, columns=sample_ids).to_csv(output_file, compression='gzip', index=False)
    print(f"Filtered posteriors saved to {output_file}")

    # Create adjacency matrix
    adjacency_matrix = pd.DataFrame(filtered_posteriors, columns=sample_ids, index=sample_ids)

    # Create graph using NetworkX
    graph = nx.from_pandas_adjacency(adjacency_matrix, create_using=nx.DiGraph)

    # Save the graph
    graph_file = os.path.join(output_folder, "grn_graph.gpickle")
    nx.write_gpickle(graph, graph_file)
    print(f"Graph saved to {graph_file}")

    # Save graph information
    graph_info = {
        "total_nodes": graph.number_of_nodes(),
        "total_edges": graph.number_of_edges(),
    }
    graph_info_file = os.path.join(output_folder, "graph_info.json")
    with open(graph_info_file, "w") as f:
        json.dump(graph_info, f, indent=4)
    print(f"Graph info saved to {graph_info_file}")

if __name__ == "__main__":
    import argparse

    # Command-line arguments
    parser = argparse.ArgumentParser(description="Step 1: Reconstruct GRNs using Findr.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input dataset.")
    parser.add_argument("--output_folder", type=str, required=True, help="Folder to save GRN reconstruction results.")
    parser.add_argument("--findr_path", type=str, required=True, help="Path to the Findr library.")
    parser.add_argument("--posterior_threshold", type=float, default=0.75, help="Threshold for posterior probabilities.")

    args = parser.parse_args()

    # Run network reconstruction
    reconstruct_grn(
        input_file=args.input_file,
        output_folder=args.output_folder,
        findr_path=args.findr_path,
        posterior_threshold=args.posterior_threshold
    )
