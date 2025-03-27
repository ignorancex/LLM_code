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
apogee_path.change_dr(16)


print(pathlib.Path(__file__))
root_path = pathlib.Path(__file__).resolve().parents[2]/"outputs"/"data"
print(root_path)

#### Hyperparameters

n_start = 0
#n_stars = 100000
d = 60 #number of dimensions to use for compression
tol = 0.001 # tolerance to use for PPCA. Larger means faster but less accurate





#### Generate and prune allStar

allStar = apoUtils.load("shuffled_allStar")

upper_temp_cut = allStar["Teff"]<5000
lower_temp_cut = allStar["Teff"]>4000
lower_g_cut = allStar["logg"]>1.5
upper_g_cut = allStar["logg"]<3.

full_params =  ["C_H",  "CI_H",  "N_H",  "O_H",  "Na_H",  "Mg_H",  "Al_H",  "Si_H",   "S_H",  "K_H","Ca_H", "Ti_H", "TiII_H", "V_H", "Cr_H", "Mn_H", "Fe_H", "Co_H", "Ni_H", "Cu_H", "Ce_H"]
Y_full_all = vectors.AllStarVector(allStar,full_params)
bad_abundance_idxs = list(set(np.where(Y_full_all.val==-9999.99)[0])) #get indexes of all stars containing at least one -999.99 (undefined value)
defined_abundances_cut = np.ones(Y_full_all.val.shape[0]).astype(bool)
defined_abundances_cut[bad_abundance_idxs] = False #convert bad_abundances_idx into mask

occamlike_cut = lower_g_cut & upper_g_cut & lower_temp_cut & upper_temp_cut & defined_abundances_cut
allStar_occamlike =  allStar[np.where(occamlike_cut)]


occam = occam_utils.Occam()
occam_kept = occam.cg_prob>0.8
allStar_occam,cluster_idxs = occam_utils.prepare_occam_allStar(occam_kept,allStar_occamlike)


### Remove a few bad apples

bad_apogee_id = ['2M02123870+4942289', '2M18051909-3214413', '2M06134865+5518282']
good_ids = [apogee_id not in bad_apogee_id for apogee_id in allStar_occamlike["Apogee_id"]]
allStar_occamlike = allStar_occamlike[good_ids]

### Remove orphan stars from cluster dataset

    
def get_orphan_idxs(cluster_names):
    """return index of elements within list only appearing once"""
    #repeated code. I should probably merge the two functions.
    clusters_to_exclude = []
    registry = vectors.OccamVector.make_registry(cluster_names)
    for cluster_name in registry:
        if len(registry[cluster_name])==1:
            clusters_to_exclude.append(cluster_name)
    repeated_idxs = []
    for cluster in clusters_to_exclude:
        repeated_idxs.extend(registry[cluster])
    return repeated_idxs

def get_nonorphan_idxs(cluster_names):
    """return indexes of all nonorphan stars"""
    return np.delete(np.arange(len(cluster_names)),get_orphan_idxs(cluster_names))    

nonorphan_idxs = get_nonorphan_idxs(cluster_idxs)
allStar_occam = allStar_occam[nonorphan_idxs]
cluster_idxs = cluster_idxs[nonorphan_idxs]

n_stars = len(allStar_occamlike)



### Remove occamstars from allStar_occamlike

def get_idxs_occam(allStar_occamlike,allStar_occam):
    occam_apoid = allStar_occam["APOGEE_ID"]
    occamlike_apoid = allStar_occamlike["APOGEE_ID"]
    idxs = []
    for apoid in occam_apoid:
        idxs.append(np.where(apoid==occamlike_apoid)[0][0])
    return np.array(idxs)


def remove_occam_from_allStar(allStar_occamlike,allStar_occam):
    """delete occam stars from allStar_occamlike. Useful for not having them appear as doppelganger"""
    idxs_occam_in_allStar = get_idxs_occam(allStar_occamlike,allStar_occam)
    nonoccam_idxs = np.delete(np.array(range(len(allStar_occamlike))),idxs_occam_in_allStar)
    return allStar_occamlike[nonoccam_idxs]

allStar_occamlike = remove_occam_from_allStar(allStar_occamlike,allStar_occam)






### save allStar

with open(root_path/"allStar.p","wb") as f:
    pickle.dump(allStar_occamlike[n_start:n_start+n_stars],f)    

with open(root_path/"allStar_occam.p","wb") as f:
    pickle.dump(allStar_occam,f)    

print("saved the allStars")




### Load stellar spectra

data_occamlike = apoData.Dataset(allStar_occamlike[n_start:n_start+n_stars])
#data_occamlike = apoData.Dataset(allStar_occamlike)
data_occam = apoData.Dataset(allStar_occam)



### Create Z with interstellar features masked

mask_interstellar, interstellar_locs = apoUtils.get_interstellar_bands()
z,z_occam,ppca = fitters.compress_masked_spectra(data_occamlike.masked_spectra[:,mask_interstellar],data_occam.masked_spectra[:,mask_interstellar],d)
Z_occam = vectors.OccamVector(val = z_occam,cluster_names=cluster_idxs).remove_orphans()
Z = vectors.Vector(val = z)


with open(root_path/"spectra"/"without_interstellar"/"cluster.p","wb") as f:
    pickle.dump(Z_occam,f)    

with open(root_path/"spectra"/"without_interstellar"/"pop.p","wb") as f:
    pickle.dump(Z,f)    



### Create Z without interstellar features masked

z_with,z_occam_with,ppca = fitters.compress_masked_spectra(data_occamlike.masked_spectra,data_occam.masked_spectra,d)
Z_occam_with = vectors.OccamVector(val = z_occam_with,cluster_names=cluster_idxs).remove_orphans()
Z_with = vectors.Vector(val = z_with)


with open(root_path/"spectra"/"with_interstellar"/"cluster.p","wb") as f:
    pickle.dump(Z_occam_with,f)    

with open(root_path/"spectra"/"with_interstellar"/"pop.p","wb") as f:
    pickle.dump(Z_with,f)    



### Create Y_full with all well-behaved APOGEE species

Y_occam_full = vectors.AllStarVector(allStar_occam,full_params)
Y_occam_full = vectors.OccamVector(val = Y_occam_full.val,cluster_names = cluster_idxs).remove_orphans()
Y_full = vectors.AllStarVector(allStar_occamlike[n_start:n_start+n_stars],full_params)


with open(root_path/"labels"/"full"/"cluster.p","wb") as f:
    pickle.dump(Y_occam_full,f)    

with open(root_path/"labels"/"full"/"pop.p","wb") as f:
    pickle.dump(Y_full,f)    



### Create Y_core containing core set of abundances

core_params =["Fe_H","Mg_H", "Ni_H", "Si_H","Al_H","C_H","N_H"]
Y_occam_core = vectors.AllStarVector(allStar_occam,core_params)
Y_occam_core = vectors.OccamVector(val = Y_occam_core.val,cluster_names = cluster_idxs).remove_orphans()
Y_core = vectors.AllStarVector(allStar_occamlike[n_start:n_start+n_stars],core_params)


with open(root_path/"labels"/"core"/"cluster.p","wb") as f:
    pickle.dump(Y_occam_core,f)    

with open(root_path/"labels"/"core"/"pop.p","wb") as f:
    pickle.dump(Y_core,f)    



