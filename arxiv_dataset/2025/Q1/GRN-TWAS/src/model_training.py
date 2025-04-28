import os
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
import networkx as nx

def load_graph(graph_file):
    """
    Load a directed acyclic graph (DAG) from a pickle file.
    """
    with open(graph_file, 'rb') as f:
        graph = pickle.load(f)
    print(f"Loaded graph from {graph_file}")
    return graph

def get_y_data(data, gene_id, samples):
    """
    Extract gene expression data for a specific gene and samples.
    """
    y = data[data['id'] == gene_id][samples].to_numpy(dtype='float64')
    return y.flatten()

def get_x_data(data, snp_ids, samples):
    """
    Extract genotype data for specific SNPs and samples.
    """
    x = data[data['rs_id'].isin(snp_ids)][samples].to_numpy(dtype='float64')
    return x.T

def train_ridge(X_cis, X_trans, y):
    """
    Train Ridge regression models for cis and trans components.
    """
    cis_model = RidgeCV(alphas=np.logspace(-3, 3, 10)).fit(X_cis, y)
    residuals = y - cis_model.predict(X_cis)

    if X_trans is not None:
        trans_model = RidgeCV(alphas=np.logspace(-3, 3, 10)).fit(X_trans, residuals)
    else:
        trans_model = None

    return cis_model, trans_model

def optimize_weights(y, X_cis, X_trans, cis_model, trans_model):
    """
    Optimize weights for combining cis and trans predictions.
    """
    def objective(weights):
        w_cis, w_trans = weights
        y_pred = w_cis * cis_model.predict(X_cis)
        if trans_model is not None:
            y_pred += w_trans * trans_model.predict(X_trans)
        return -r2_score(y, y_pred)

    initial_weights = [0.5, 0.5]
    bounds = [(0, 1), (0, 1)]
    result = minimize(objective, initial_weights, bounds=bounds)
    return result.x

def process_dataset(expression_file, genotype_file, graph_file, output_folder):
    """
    Process a single dataset to train Ridge regression models for gene expression prediction.
    """
    print(f"Processing dataset: {expression_file}, {genotype_file}")

    # Load data
    expression_data = pd.read_csv(expression_file)
    genotype_data = pd.read_csv(genotype_file)
    graph = load_graph(graph_file)

    samples = [col for col in expression_data.columns if col != 'id']
    genes = list(nx.topological_sort(graph))

    results = {}
    for gene in genes:
        # Prepare data
        y = get_y_data(expression_data, gene, samples)
        cis_snp_ids = list(graph.nodes[gene].get('cis_snps', []))
        X_cis = get_x_data(genotype_data, cis_snp_ids, samples)

        # Prepare trans data if applicable
        parents = list(graph.predecessors(gene))
        if parents:
            trans_snp_ids = [snp for parent in parents for snp in graph.nodes[parent].get('cis_snps', [])]
            X_trans = get_x_data(genotype_data, trans_snp_ids, samples)
        else:
            X_trans = None

        # Train models
        cis_model, trans_model = train_ridge(X_cis, X_trans, y)

        # Optimize weights
        if X_trans is not None:
            w_cis, w_trans = optimize_weights(y, X_cis, X_trans, cis_model, trans_model)
        else:
            w_cis, w_trans = 1.0, 0.0

        # Store results
        results[gene] = {
            'cis_model': cis_model,
            'trans_model': trans_model,
            'weights': {'w_cis': w_cis, 'w_trans': w_trans},
        }

    # Save results
    output_file = os.path.join(output_folder, "model_results.pkl")
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    import argparse

    # Command-line arguments
    parser = argparse.ArgumentParser(description="Train Ridge regression models for gene expression prediction.")
    parser.add_argument("--expression_file", type=str, required=True, help="Path to the gene expression file.")
    parser.add_argument("--genotype_file", type=str, required=True, help="Path to the genotype file.")
    parser.add_argument("--graph_file", type=str, required=True, help="Path to the DAG pickle file.")
    parser.add_argument("--output_folder", type=str, required=True, help="Folder to save the results.")

    args = parser.parse_args()

    # Ensure output folder exists
    os.makedirs(args.output_folder, exist_ok=True)

    # Process the dataset
    process_dataset(
        expression_file=args.expression_file,
        genotype_file=args.genotype_file,
        graph_file=args.graph_file,
        output_folder=args.output_folder
    )
