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
 
def make_intracluster_similarity_trends(V_occam,get_y):
    """Measures the similarities for all stellar sibling pairs in the dataset and for a y-parameter.
    Parameters
    ----------
    V_occam: Vectors.OccamVector
        OccamVector containing the final transformed representation on which metric learning is applied.
    get_y: function
        Function which takes idx1,idx2 - the indexes of a pair of stars -  and returns the y quantity of interest
    Outputs
    -------
    all_similarities: np.array
        Contains the similarities for all stars in the dataset
    all_y: np.array
        Contains the associated y values for every pair in all_similarities
    """
    all_similarities = []
    all_ys = []
    for cluster in V_occam.registry:
    #for cluster in ['NGC 7789']:
        clust_size = len(V_occam.registry[cluster])
        if clust_size>1:
            combinations = evaluators.BaseEvaluator.get_combinations(clust_size)
            pairings = np.array(V_occam.registry[cluster][np.array(combinations)])
            v1 = V_occam.val[pairings[:,0]]
            v2 = V_occam.val[pairings[:,1]]
            similarities = np.linalg.norm(v1-v2,axis=1)
            ys = np.array([get_y(pair[0],pair[1]) for pair in pairings])
            #overlaps = np.array([apoUtils.get_overlap(mjds,pair[1],pair[0]) for pair in pairings])
            #overlaps = np.array([np.min([allStar_occam["SNR"][pair[0]],allStar_occam["SNR"][pair[1]]]) for pair in pairings])
            #overlaps = np.array([allStar_occam["VHELIO_AVG"][pair[0]]-allStar_occam["VHELIO_AVG"][pair[1]] for pair in pairings])
            all_similarities.append(similarities)
            all_ys.append(ys)
    
    return  np.concatenate(all_similarities), np.concatenate(all_ys)


###Hyperparameters

z_dim = 30 #PCA dimensionality



#commands fonds for setting figure size of plots
text_width = 513.11743
column_width =242.26653


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

with open(root_path/"allStar_occam.p","rb") as f:
    allStar_occam = pickle.load(f)    


#### Apply metric learing #####


Z_fitter = standard_fitter(Z[:,:z_dim],Z_occam[:,:z_dim])
V_occam = Z_fitter.transform(Z_occam[:,:z_dim].centered(Z_occam[:,:z_dim]))

Y_fitter = simple_fitter(Y,Y_occam)
V_Y_occam = Y_fitter.transform(Y_occam.centered(Y_occam))

mjds_occam = np.array(apoUtils.allStar_to_calendar(allStar_occam))


assert(len(allStar_occam)==len(V_Y_occam.val))

def get_overlap(idx1,idx2):
    return apoUtils.get_overlap(mjds_occam,idx1,idx2)
similarities, overlaps = make_intracluster_similarity_trends(V_occam,get_overlap)
similarities_y, overlaps_y = make_intracluster_similarity_trends(V_Y_occam,get_overlap)

### Normalize similarities for sake of comparison

similarities = similarities/np.mean(similarities)
similarities_y = similarities_y/np.mean(similarities_y)


try:
    plt.style.use("tex")
except:
    print("tex style not implemented (https://jwalton.info/Embed-Publication-Matplotlib-Latex/)")

save_path = root_path.parents[0]/"figures"/"validation"/"visits"
save_path.mkdir(parents=True, exist_ok=True)

import matplotlib
cmap = matplotlib.cm.get_cmap('viridis')
color1 = cmap(0.15)
color2 = cmap(0.75)


figsize = list(apoUtils.set_size(apoUtils.column_width))
figsize[0]=figsize[0]*2
figsize[1]=figsize[1]/2*2

fig, ax = plt.subplots(1,2,sharey="row",figsize=figsize,gridspec_kw={'hspace': 0, 'wspace': 0})


ax[0].hist(similarities[overlaps==0],bins = 20,color=color2,density=True,label="observed separately",linewidth=2,histtype="step")
ax[0].hist(similarities[overlaps==1],bins = 20,color=color1,density=True,label="observed together",linewidth=2,histtype="step")
ax[0].axvline(x=np.mean(similarities[overlaps==0]),color=color2,linestyle="--")
ax[0].axvline(x=np.mean(similarities[overlaps==1]),color=color1,linestyle="--")
#ax[0].set_xlabel("similarity")
ax[0].set_ylabel("$p$")
ax[0].set_title("From spectra")
ax[0].set_xlim([0,3])
ax[0].set_ylim([0,2])
#ax[0].legend()

ax[1].hist(similarities_y[overlaps_y==1],bins = 20,color=color1,density=True,label="observed together",linewidth=2,histtype="step")
ax[1].hist(similarities_y[overlaps_y==0],bins = 20,color=color2,density=True,label="observed separately",linewidth=2,histtype="step")
ax[1].axvline(x=np.mean(similarities_y[overlaps_y==0]),color=color2,linestyle="--")
ax[1].axvline(x=np.mean(similarities_y[overlaps_y==1]),color=color1,linestyle="--")

#ax[1].set_xlabel("similarity")
#ax[1].set_ylabel("$p$")
ax[1].set_title("From stellar abundances")
ax[1].set_xlim([0,3])
ax[0].set_ylim([0,2.5])
ax[1].legend()


ax[0].xaxis.get_major_ticks()[-1].set_visible(False)



fig.add_subplot(111, frameon=False)
# hide tick and tick label of the big axis
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.xlabel("Similarity")



plt.savefig(save_path/"validation_visits.pdf",format="pdf",bbox_inches='tight')
