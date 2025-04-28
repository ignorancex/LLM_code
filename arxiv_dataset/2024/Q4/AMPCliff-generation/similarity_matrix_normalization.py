import json
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
import numpy as np
from scipy.stats import spearmanr,pearsonr
import seaborn as sns
import matplotlib.pyplot as plt
from Bio.Align import substitution_matrices
import pandas as pd
import ipdb
import math

blosum62 = substitution_matrices.load("BLOSUM62")
# Get the list of amino acids from the matrix (excluding gaps)
amino_acids = list(blosum62.alphabet)[:20]  # Assuming the gap character is the last in the alphabet list

amino_acids_with_gap = amino_acids + [blosum62.alphabet[-1]]
# Initialize a symmetric numpy array
size = len(amino_acids_with_gap)
blosum62_array = np.zeros((size, size), dtype=int)

# Fill in the array with scores
for i, aa1 in enumerate(amino_acids_with_gap):
    for j, aa2 in enumerate(amino_acids_with_gap):
        blosum62_array[i, j] = blosum62[(aa1, aa2)]

blosum62_df = pd.DataFrame(blosum62_array,columns = amino_acids_with_gap, index = amino_acids_with_gap)     


# bit score normalization
'''
lambda_value=0.251
K=0.031
blosum62_df_normalized = (lambda_value * blosum62_df - math.log(K)) / math.log(2)

'''
# Find the minimum and the range of the non-diagonal values

def normalize(blosum62_df):
  min_non_diag = blosum62_df.min(0)
  max_non_diag = blosum62_df.max(0)
  range_non_diag = max_non_diag - min_non_diag
  
  # Normalize non-diagonal values to range from 0 to 1
  blosum62_df_normalized_ = (blosum62_df - min_non_diag) / range_non_diag
  blosum62_df_normalized = (blosum62_df_normalized_ + blosum62_df_normalized_.T)/2
  return blosum62_df_normalized

blosum62_df_normalized = normalize(blosum62_df)
# '''
# ipdb.set_trace()

# Assume the JSON file is named "amino_acids_smiles.json" and has the format:
# {"A": "CC(C(=O)O)N", "R": "N[C@@H](CCCNC(=N)N)C(=O)O", ...}

# Load the SMILES strings from the JSON file
with open("cpp_smiles.json", 'r') as file:
    amino_acids_smiles = json.load(file)

# Convert SMILES strings to RDKit molecules and then to fingerprints
fingerprints = {}

for aa in amino_acids:
    mol = Chem.MolFromSmiles(amino_acids_smiles[aa])
    fingerprints[aa] = AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=2048) # 3, 2048

fingerprints['-'] = DataStructs.ExplicitBitVect(2048)

amino_acids_gap = amino_acids + ["-"]
# Calculate the Tanimoto similarity matrix
similarity_matrix = np.zeros((len(amino_acids_gap), len(amino_acids_gap)))

for i, aa1 in enumerate(amino_acids_gap):
    for j, aa2 in enumerate(amino_acids_gap):
        similarity_matrix[i, j] = DataStructs.FingerprintSimilarity(fingerprints[aa1], fingerprints[aa2])

similarity_matrix_df = pd.DataFrame(similarity_matrix,columns = amino_acids_gap, index = amino_acids_gap)

# normalize tanimoto
similarity_matrix_df_normalized = normalize(similarity_matrix_df)
# blosum62_df = blosum62_df[similarity_matrix_df.columns]
# Now similarity_matrix contains the Tanimoto similarities between the amino acids
# Flatten the matrices to compute Spearman correlation
similarity_matrix_flat = similarity_matrix_df_normalized.values.flatten()
similarity_matrix_flat_org = similarity_matrix_df.values.flatten()
blosum62_matrix_flat = blosum62_df_normalized.values.flatten()
blosum62_matrix_flat_org = blosum62_df.values.flatten()

spearman_corr, _ = spearmanr(blosum62_matrix_flat_org, blosum62_matrix_flat)
print(f"Spearman correlation coefficient blosum62 and normalized bolsum62: {spearman_corr}")

spearman_corr, _ = spearmanr(similarity_matrix_flat_org, similarity_matrix_flat)
print(f"Spearman correlation coefficient tanimoto and normalized tanimoto: {spearman_corr}")

pearson_corr, _ = pearsonr(blosum62_matrix_flat_org, blosum62_matrix_flat)
print(f"Pearson correlation coefficient blosum62 and normalized bolsum62: {pearson_corr}")

pearson_corr, _ = pearsonr(similarity_matrix_flat_org, similarity_matrix_flat)
print(f"Pearson correlation coefficient tanimoto and normalized tanimoto: {pearson_corr}")

spearman_corr, _ = spearmanr(similarity_matrix_flat, blosum62_matrix_flat)
print(f"Spearman correlation coefficient normalized tanimoto and normalized blosum62: {spearman_corr}")

spearman_corr, _ = spearmanr(similarity_matrix_flat_org, blosum62_matrix_flat_org)
print(f"Spearman correlation coefficient tanimoto and blosum62: {spearman_corr}")

# Heatmap of the Tanimoto similarity matrix
plt.figure(figsize=(15, 10))
sns.heatmap(similarity_matrix_df, annot=True, cmap='viridis', square=True)
plt.title("Tanimoto Similarity Matrix Heatmap")
plt.savefig("./figures/Tanimoto Similarity Matrix.png")
similarity_matrix_df.to_csv('./data/tanimoto.csv')

# Heatmap of the BLOSUM62 matrix
plt.figure(figsize=(15, 10))
sns.heatmap(blosum62_df, annot=True, cmap='viridis', square=True)
plt.title("BLOSUM62 Matrix Heatmap")
plt.savefig("BLOSUM62 Matrix.png")
blosum62_df.rename(columns={'*': '-'}, inplace=True)
blosum62_df.to_csv('./data/blosum62.csv')


# Heatmap of the normalized BLOSUM62 matrix
plt.figure(figsize=(15, 10))
sns.heatmap(blosum62_df_normalized, annot=True, cmap='viridis', square=True)
plt.title("Normalized BLOSUM62 Matrix Heatmap")
plt.savefig("./figures/BLOSUM62 Matrix Normalized.png")
blosum62_df_normalized.rename(columns={'*': '-'}, inplace=True)
blosum62_df_normalized.to_csv('./data/blosum62_normalized.csv')


# Heatmap of the normalized tanimoto matrix
plt.figure(figsize=(15, 10))
sns.heatmap(similarity_matrix_df_normalized, annot=True, cmap='viridis', square=True)
plt.title("Normalized Tanimoto Matrix Heatmap")
plt.savefig("./figures/Tanimoto Matrix Normalized.png")
blosum62_df_normalized.to_csv('./data/tanimoto_normalized.csv')