import os
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import json

def load_graph(graph_file):
    """
    Load a directed acyclic graph (DAG) from a pickle file.
    """
    with open(graph_file, 'rb') as f:
        graph = pickle.load(f)
    print(f"Loaded graph from {graph_file}")
    return graph

def get_x_data(data, snp_ids, samples):
    """
    Extract genotype data for specific SNPs and samples.
    """
    x = data[data['rs_id'].isin(snp_ids)][samples].to_numpy(dtype='float64')
    return x.T

def calculate_z_score(alpha, rho, se, X, epsilon=1e-6):
    """
    Calculate Z-score for a gene based on genotype and GWAS data.
    """
    X_standardized = StandardScaler().fit_transform(X)
    if X.shape[1] == 1:
        gene_var = np.var(X_standardized, ddof=1) * alpha[0]**2
    else:
        snp_cov = np.cov(X_standardized, rowvar=False) + np.eye(X.shape[1]) * epsilon
        gene_var = alpha.reshape(1, -1) @ snp_cov @ alpha.reshape(-1, 1)
        gene_var = gene_var[0, 0]

    ratio = np.std(X_standardized, axis=0) / np.sqrt(gene_var + epsilon)
    return np.sum(alpha * ratio * rho / np.maximum(se, epsilon))

def process_association(expression_file, genotype_file, graph_file, gwas_file, output_file):
    """
    Perform gene-disease association analysis for a single dataset.
    """
    print("Starting gene-disease association analysis")

    # Load data
    expression_data = pd.read_csv(expression_file)
    genotype_data = pd.read_csv(genotype_file)
    gwas_data = pd.read_csv(gwas_file, sep='\t')
    graph = load_graph(graph_file)

    samples = [col for col in expression_data.columns if col != 'id']
    genes = list(graph.nodes)

    results = {}
    for gene in genes:
        cis_snp_ids = graph.nodes[gene].get('cis_snps', [])
        X_cis = get_x_data(genotype_data, cis_snp_ids, samples)

        rho_cis = gwas_data[gwas_data['snpid'].isin(cis_snp_ids)].set_index('snpid')['logOR'].to_dict()
        se_cis = gwas_data[gwas_data['snpid'].isin(cis_snp_ids)].set_index('snpid')['se_gc'].to_dict()

        if not rho_cis or not se_cis:
            results[gene] = {'z_score_cis': 0.0, 'z_score_trans': 0.0, 'z_score': 0.0}
            continue

        alpha_cis = np.random.rand(len(cis_snp_ids))  # Placeholder for actual alpha values
        rho_values_cis = np.array([rho_cis[snp] for snp in cis_snp_ids if snp in rho_cis])
        se_values_cis = np.array([se_cis[snp] for snp in cis_snp_ids if snp in se_cis])

        z_score_cis = calculate_z_score(alpha_cis, rho_values_cis, se_values_cis, X_cis)

        # Handle trans effects
        parents = list(graph.predecessors(gene))
        trans_snp_ids = [snp for parent in parents for snp in graph.nodes[parent].get('cis_snps', [])]
        if trans_snp_ids:
            X_trans = get_x_data(genotype_data, trans_snp_ids, samples)
            rho_trans = gwas_data[gwas_data['snpid'].isin(trans_snp_ids)].set_index('snpid')['logOR'].to_dict()
            se_trans = gwas_data[gwas_data['snpid'].isin(trans_snp_ids)].set_index('snpid')['se_gc'].to_dict()

            alpha_trans = np.random.rand(len(trans_snp_ids))  # Placeholder for actual alpha values
            rho_values_trans = np.array([rho_trans[snp] for snp in trans_snp_ids if snp in rho_trans])
            se_values_trans = np.array([se_trans[snp] for snp in trans_snp_ids if snp in se_trans])

            z_score_trans = calculate_z_score(alpha_trans, rho_values_trans, se_values_trans, X_trans)
        else:
            z_score_trans = 0.0

        z_score = z_score_cis + z_score_trans

        results[gene] = {
            'z_score_cis': float(z_score_cis),
            'z_score_trans': float(z_score_trans),
            'z_score': float(z_score)
        }

    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Gene-disease association analysis.")
    parser.add_argument("--expression_file", type=str, required=True, help="Path to the gene expression file.")
    parser.add_argument("--genotype_file", type=str, required=True, help="Path to the genotype file.")
    parser.add_argument("--graph_file", type=str, required=True, help="Path to the DAG pickle file.")
    parser.add_argument("--gwas_file", type=str, required=True, help="Path to the GWAS summary statistics file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the association results.")

    args = parser.parse_args()

    process_association(
        expression_file=args.expression_file,
        genotype_file=args.genotype_file,
        graph_file=args.graph_file,
        gwas_file=args.gwas_file,
        output_file=args.output_file
    )
