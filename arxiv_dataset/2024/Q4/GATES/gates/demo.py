import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import os
from sklearn.metrics.cluster import adjusted_rand_score
import sys
sys.path.append(r'D:\Pycharm\pytorch\spatial_transcriptomics')
from Train_GATES import train_GATES
from utils import Cal_Spatial_Net, Stats_Spatial_Net, mclust_R, Cal_Gene_Similarity_Net

# the location of R (used for the mclust clustering)
os.environ['R_HOME'] = r'C:\Program Files\R\R-4.4.1'    # 'D:\Program Files\R\R-4.0.3'
os.environ['R_USER'] = r'C:\Users\anaconda3\Lib\site-packages\rpy2'

seed = 151676
section_id = '151675'  # 151507 - 151510 & 151669 - 151676
data_root = 'D:\datasets\spatial_transcriptomics\DLPFC'

input_dir = os.path.join(data_root, section_id)
adata = sc.read_visium(path=input_dir, count_file=section_id+'_filtered_feature_bc_matrix.h5')
adata.var_names_make_unique()

#Normalization
sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# read the annotation
Ann_df = pd.read_csv(os.path.join(data_root,
                                  section_id, section_id+'_truth.txt'),
                     sep='\t', header=None, index_col=0)

Ann_df.columns = ['Ground Truth']

adata.obs['Ground Truth'] = Ann_df.loc[adata.obs_names, 'Ground Truth']

Cal_Spatial_Net(adata, model="Radius", rad_cutoff=150)  # model="KNN", k_cutoff=6
Cal_Gene_Similarity_Net(adata, k_neighbors=6, metric='cosine', verbose=True)  # cosine, euclidean, manhattan
Stats_Spatial_Net(adata, save=f"./saved_images/DLPFC_{section_id}_NumberOfNeighbors.png", plt_show=False)

adata = train_GATES(adata, mod='spatial-similarity', alpha=0, n_epochs=1, verbose=False, random_seed=seed)

sc.pp.neighbors(adata, use_rep='GATES')
sc.tl.umap(adata)
adata = mclust_R(adata, used_obsm='GATES', num_cluster=7)
obs_df = adata.obs.dropna()
ARI = adjusted_rand_score(obs_df['mclust'], obs_df['Ground Truth'])
print('Adjusted rand index = %.4f' % ARI)
