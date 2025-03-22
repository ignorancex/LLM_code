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

#################
z_dim = 30 #number of dimensions to use
root_path = pathlib.Path(__file__).resolve().parents[2]/"outputs"/"data"
tol = 1E-4 #the tol of the pickled ppca being loaded
dim = 30 #the number of dimensions of the pickled ppca being loaded 
num_models = 10 #the number of pickled models to use for statistics
N = list(range(1,8))+[30]
#root_path = pathlib.Path("/share/splinter/ddm/taggingProject/tidyPCA/apoNN/scripts").parents[1]/"outputs"/"data"


##########
valid_idxs = apoUtils.get_valid_intercluster_idxs()

def get_average_n(Z,Z_occam,valid_idxs,N=range(1,3)):
    """
    Returns an array containing the doppelganger rates of the topN fitters
    Parameters:
        N: dimension sizes to loop over for topN fitter
    """
    averages = []
    for n in N:
        def topn_fitter(z,z_occam):
                """This fitter performs a change-of-basis to a more appropriate basis for scaling"""
                return fitters.TopNFitter(z,z_occam,use_relative_scaling=True,is_pooled=True,is_robust=True,N=n)
        ev = evaluators.EvaluatorWithFiltering(Z[:,:z_dim],Z_occam[:,:z_dim],leave_out=True,fitter_class=topn_fitter,valid_idxs=valid_idxs)
        averages.append(ev.weighted_average)
    return averages

def load_path(n,tol,d):
    """"Load a set of Z_occam and Z"""
    with open(root_path/"spectra"/"without_interstellar"/f"clusterD{d}N{n}tol{tol}.p","rb") as f:
        Z_occam = pickle.load(f)

    with open(root_path/"spectra"/"without_interstellar"/f"popD{d}N{n}tol{tol}.p","rb") as f:
        Z = pickle.load(f)
    return Z,Z_occam



per_sim_averages = []
for n in range(num_models):
    Z,Z_occam = load_path(n,tol,dim)
    per_sim_averages.append(get_average_n(Z,Z_occam,valid_idxs,N))


averages = np.mean(per_sim_averages,axis=0) # the average doppelganger rates of the num_models different pickled models
averages_err = np.std(per_sim_averages,axis=0)

for idx,i in enumerate(range(len(averages))):
    print(f"{N[idx]} & ${averages[i]:.4f} \pm {averages_err[i]:.4f}$ \\\\")

    
