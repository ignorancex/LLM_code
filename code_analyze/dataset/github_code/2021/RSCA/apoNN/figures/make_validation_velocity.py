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

def get_similarity(X,Y,n_repeats=50000,n_max=10000,use_delta=True):
    """
    OUTPUTS
    -------
    similarity_list: 
        contains the chemical similarity for random pairs of stars
    delta_list:
        contains the difference in variable of interest for these same stars
    use_delta: boolean
        if true give the difference between two varialbles. If false give the average.
    """
    assert len(X)==len(Y)
    similarity_list = []
    delta_list = []
    for _ in range(n_repeats):
        i,j = np.random.choice(n_max,2)
        if  (Y[i]>-999) and (Y[j]>-999): #hack to prevent pairs containing -9999 from showing up
            similarity_list.append(similarity_ij(i,j,X))
            if use_delta is True:
                delta_list.append(np.abs(Y[i]-Y[j]))
            else:
                delta_list.append(np.mean([Y[i],Y[j]]))
    return np.array(similarity_list),delta_list

def similarity_ij(i,j,v):
    return np.linalg.norm(v[i]-v[j])



###Hyperparameters

z_dim = 30 #PCA dimensionality


###
with open(root_path/"spectra"/"without_interstellar"/"cluster.p","rb") as f:
    Z_occam = pickle.load(f)    

with open(root_path/"spectra"/"without_interstellar"/"pop.p","rb") as f:
    Z = pickle.load(f)



with open(root_path/"labels"/"core"/"cluster.p","rb") as f:
    Y_occam = pickle.load(f)    

with open(root_path/"labels"/"core"/"pop.p","rb") as f:
    Y = pickle.load(f)    

with open(root_path/"allStar.p","rb") as f:
    allStar = pickle.load(f)    


#### Apply metric learing #####


Z_fitter = standard_fitter(Z[:,:z_dim],Z_occam[:,:z_dim])
V = Z_fitter.transform(Z_fitter.z.centered(Z_occam[:,:z_dim]))


Y_fitter = simple_fitter(Y,Y_occam)
V_Y = Y_fitter.transform(Y.centered(Y_occam))


y_interest = allStar["VHELIO_AVG"]

similarities,velocity_diffs = get_similarity(V.val,y_interest)
similarities_y,velocity_diffs_y = get_similarity(V_Y.val,y_interest)

### Normalize similarities for sake of comparison
similarities = similarities/np.mean(similarities)
similarities_y = similarities_y/np.mean(similarities_y)

save_path = root_path.parents[0]/"figures"/"validation"/"velocity"
save_path.mkdir(parents=True, exist_ok=True)

try: 
    plt.style.use("tex")
except:
    print("tex style not implemented (https://jwalton.info/Embed-Publication-Matplotlib-Latex/)")
            
import matplotlib
cmap = matplotlib.cm.get_cmap('viridis')
color1 = cmap(0.15)
color2 = cmap(0.75)


figsize = list(apoUtils.set_size(apoUtils.column_width))
figsize[0]=figsize[0]
figsize[1]=figsize[1]/2

fig, ax = plt.subplots(1,2,sharey="row",figsize=figsize,gridspec_kw={'hspace': 0, 'wspace': 0})


ax[0].hist(similarities[velocity_diffs<np.median(velocity_diffs)],bins = 40,color=color1,density=True,label="similar velocity",linewidth=2,histtype="step")
ax[0].hist(similarities[velocity_diffs>np.median(velocity_diffs)],bins = 40,color=color2,density=True,label="dissimilar velocity",linewidth=2,histtype="step")
ax[0].axvline(x=np.mean(similarities[velocity_diffs<np.median(velocity_diffs)]),color=color1,linestyle="--")
ax[0].axvline(x=np.mean(similarities[velocity_diffs>np.median(velocity_diffs)]),color=color2,linestyle="--")
#ax[0].set_xlabel("similarity")
ax[0].set_ylabel("$p$")
ax[0].set_title("From masked spectra")
ax[0].set_xlim([0,3])
ax[0].set_ylim([0,1.25])
#ax[0].legend()

ax[1].hist(similarities_y[velocity_diffs_y<np.median(velocity_diffs_y)],bins = 20,color=color1,density=True,label="similar velocity",linewidth=2,histtype="step")
ax[1].hist(similarities_y[velocity_diffs_y>np.median(velocity_diffs_y)],bins = 20,color=color2,density=True,label="dissimilar velocity",linewidth=2,histtype="step")
ax[1].axvline(x=np.mean(similarities_y[velocity_diffs_y<np.median(velocity_diffs_y)]),color=color1,linestyle="--")
ax[1].axvline(x=np.mean(similarities_y[velocity_diffs_y>np.median(velocity_diffs_y)]),color=color2,linestyle="--")

#ax[1].set_xlabel("similarity")
#ax[1].set_ylabel("$p$")
ax[1].set_title("From stellar abundances")
ax[1].set_xlim([0,3])
ax[0].set_ylim([0,1.25])
ax[1].legend()

ax[0].xaxis.get_major_ticks()[-1].set_visible(False)


fig.add_subplot(111, frameon=False)
# hide tick and tick label of the big axis
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.xlabel("Similarity")


plt.savefig(save_path/"validation_velocity.pdf",format="pdf",bbox_inches='tight')
