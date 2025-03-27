import apoNN.src.data as apoData
import apoNN.src.utils as apoUtils
import apoNN.src.vectors as vectors
import apoNN.src.fitters as fitters
import apoNN.src.evaluators as evaluators
import apoNN.src.occam as occam_utils
import numpy as np
import random
import pathlib
import pickle
from ppca import PPCA
import apogee.tools.path as apogee_path
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
apogee_path.change_dr(16)

###Setup

root_path = pathlib.Path(__file__).resolve().parents[2]/"outputs"/"data"
#root_path = pathlib.Path("/share/splinter/ddm/taggingProject/tidyPCA/apoNN/scripts").parents[1]/"outputs"/"data"

save_path = root_path.parents[0]/"figures"/"datasize"
save_path.mkdir(parents=True, exist_ok=True)



def standard_fitter(z,z_occam):
        """This fitter performs a change-of-basis to a more appropriate basis for scaling"""
        return fitters.StandardFitter(z,z_occam,use_relative_scaling=True,is_pooled=True,is_robust=True)

def simple_fitter(z,z_occam):
        """This is a simple fitter that just scales the dimensions of the inputed representation. Which is used as a baseline"""
        return fitters.SimpleFitter(z,z_occam,use_relative_scaling=True,is_pooled=True,is_robust=True)

def get_n_random_clusters(vector_occam,n_clusters):
    cluster_list = random.sample(list(vector_occam.registry),n_clusters)
    cluster_idxs = [val for cluster in cluster_list for val in Z_occam.registry[cluster]]
    return vector_occam.only(cluster_list),cluster_idxs


def make_doppelganger_vs_clusters(n_clusters_considered,X,X_occam,n_repeats):
    """
    Calculate the average doppelganger rate for a given number of clusters
    -------------------------------
    n_clusters_considered: list
            cluster sizes to calculate for
    X: vector.Vector
        X dataset
    X_occam:vector.OccamVector
        X_dataset
    """
    res = []
    for n_clusters in n_clusters_considered:
        res.append([])
        for _ in range(n_repeats):
            X_restricted,restricted_idxs = get_n_random_clusters(X_occam,n_clusters)
            print(X.val.shape)
            print(X_restricted.val.shape)
            evaluator_X = evaluators.EvaluatorWithFiltering(X,X_restricted,leave_out=True,fitter_class=standard_fitter,valid_idxs=valid_idxs[restricted_idxs])
            res[-1].append(evaluator_X.weighted_average)  
    return res

###Hyperparameters

z_dim = 30 #PCA dimensionality
#commands fonds for setting figure size of plots


###
###

with open(root_path/"spectra"/"without_interstellar"/"cluster.p","rb") as f:
    Z_occam = pickle.load(f)    

with open(root_path/"spectra"/"without_interstellar"/"pop.p","rb") as f:
    Z = pickle.load(f)    


###
###

valid_idxs = apoUtils.get_valid_intercluster_idxs()
   


### Calculations


n_repeats = 50 #How many different combinations of clusters to sample for each size
n_clusters_considered = [12,16,21] #How many clusters to preserve
#n_dims = [10,20,30]
n_dims=[30,20,10]

res_clusters = [make_doppelganger_vs_clusters(n_clusters_considered,Z[:,:n_dim],Z_occam[:,:n_dim],n_repeats) for n_dim in n_dims]


import matplotlib
cmap = matplotlib.cm.get_cmap('viridis')
colors = [cmap(0.10),cmap(0.6),cmap(0.95)]




plt.style.use('default') 
try: 
    plt.style.use("tex")
except:
    print("tex style not implemented (https://jwalton.info/Embed-Publication-Matplotlib-Latex/)")
    
#plt.style.use('seaborn')




plt.figure(figsize=apoUtils.set_size(apoUtils.column_width))
for i,res_cluster in enumerate(res_clusters):
    plt.plot(np.array(n_clusters_considered)-1,[np.mean(res_i) for res_i in res_cluster],color=colors[i],label=f"{n_dims[i]} dims",marker='o',markersize=9,markeredgecolor="black")

plt.xlabel("Number of clusters")
plt.ylabel("Doppelganger rate")
plt.ylim(0,0.04)
plt.legend()
plt.savefig(save_path/"datasize.pdf",format="pdf",bbox_inches='tight')
