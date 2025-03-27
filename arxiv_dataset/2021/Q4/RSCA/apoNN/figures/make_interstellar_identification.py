"""Script for identifying wavelength regions containing interstellar features"""

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
from apogee.tools import air2vac, atomic_number,apStarWavegrid
import matplotlib.pyplot as plt 
apogee_path.change_dr(16)

###Setup

root_path = pathlib.Path(__file__).resolve().parents[2]/"outputs"/"data"
#root_path = pathlib.Path("/share/splinter/ddm/taggingProject/tidyPCA/apoNN/scripts").parents[1]/"outputs"/"data"


def standard_fitter(z,z_occam):
        """This fitter performs a change-of-basis to a more appropriate basis for scaling"""
        return fitters.StandardFitter(z,z_occam,use_relative_scaling=True,is_pooled=True,is_robust=True)

def simple_fitter(z,z_occam):
        """This is a simple fitter that just scales the dimensions of the inputed representation. Which is used as a baseline"""
        return fitters.SimpleFitter(z,z_occam,use_relative_scaling=True,is_pooled=True,is_robust=True)


###Hyperparameters

z_dim = 30 #PCA dimensionality


###
with open(root_path/"spectra"/"without_interstellar"/"cluster.p","rb") as f:
    Z_occam = pickle.load(f)    

with open(root_path/"spectra"/"without_interstellar"/"pop.p","rb") as f:
    Z = pickle.load(f)



with open(root_path/"spectra"/"with_interstellar"/"cluster.p","rb") as f:
    Z_occam_interstellar = pickle.load(f)    

with open(root_path/"spectra"/"with_interstellar"/"pop.p","rb") as f:
    Z_interstellar = pickle.load(f)



with open(root_path/"labels"/"core"/"cluster.p","rb") as f:
    Y_occam = pickle.load(f)    

with open(root_path/"labels"/"core"/"pop.p","rb") as f:
    Y = pickle.load(f)    

with open(root_path/"allStar.p","rb") as f:
    allStar = pickle.load(f)    

### generate the two datasets

low_extinction_cut = allStar["AK_TARG"]<0.005
high_extinction_cut = allStar["AK_TARG"]>0.5

data_low = apoData.Dataset(allStar[np.where(low_extinction_cut)])
nonzero_idxs = np.where(np.sum(~(data_low.masked_spectra.data==0),axis=0)!=0)[0]
spectra_low = apoData.infill_masked_spectra(data_low.masked_spectra[:,nonzero_idxs]
,data_low.masked_spectra[0:500,nonzero_idxs])


data_high = apoData.Dataset(allStar[np.where(high_extinction_cut)])
spectra_high = apoData.infill_masked_spectra(data_high.masked_spectra[:,nonzero_idxs]
,data_low.masked_spectra[0:500,nonzero_idxs])


### Apply PCA using the low extinction spectra and find the residuals on the high extinction residuals

from sklearn.decomposition import PCA
pca = PCA(n_components=30)
pca.fit(spectra_low)


rec_spec_low =  pca.inverse_transform(pca.transform(spectra_low))
rec_spec_high =  pca.inverse_transform(pca.transform(spectra_high))

#diff_res = np.mean((np.abs(spectra_high-rec_spec_high)),axis=0)-np.mean((np.abs(spectra_low-rec_spec_low)),axis=0)
diff_res = np.mean((spectra_high-rec_spec_high),axis=0)

### Create figure

mask_interstellar, interstellar_locs = apoUtils.get_interstellar_bands()


diff_res_full = np.zeros(apStarWavegrid().shape) #diff_res_full adds the nan values in nonzero_idxs to 
diff_res_full[nonzero_idxs] = diff_res

save_path = root_path.parents[0]/"figures"/"interstellar"
save_path.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=[14,3])
plt.plot(apStarWavegrid(),diff_res_full,color="black",linewidth=0.5)
alpha_loc = 0.2
for loc in interstellar_locs:
    plt.axvspan(apStarWavegrid()[loc[0]],apStarWavegrid()[loc[1]],color="orange",alpha = alpha_loc)
plt.ylabel("residuals",fontsize=14)
plt.xlabel(r"wavelength $\lambda$ $(\AA)$",fontsize=14)
plt.gcf().subplots_adjust(bottom=0.25)
plt.savefig(save_path/"interstellar_bands.pdf",format="pdf",bbox_inches='tight')

