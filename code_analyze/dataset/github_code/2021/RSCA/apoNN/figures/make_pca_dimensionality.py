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


def standard_fitter(z,z_occam):
        """This fitter performs a change-of-basis to a more appropriate basis for scaling"""
        return fitters.StandardFitter(z,z_occam,use_relative_scaling=True,is_pooled=True,is_robust=True)

def simple_fitter(z,z_occam):
        """This is a simple fitter that just scales the dimensions of the inputed representation. Which is used as a baseline"""
        return fitters.SimpleFitter(z,z_occam,use_relative_scaling=True,is_pooled=True,is_robust=True)



###Hyperparameters

z_dim = 30 #PCA dimensionality


###
###

with open(root_path/"spectra"/"without_interstellar"/"clusterD60N0tol1e-05.p","rb") as f:
    Z_occam = pickle.load(f)    

with open(root_path/"spectra"/"without_interstellar"/"popD60N0tol1e-05.p","rb") as f:
    Z = pickle.load(f)    


###
###

with open(root_path/"labels"/"core"/"cluster.p","rb") as f:
    Y_occam = pickle.load(f)

with open(root_path/"labels"/"core"/"pop.p","rb") as f:
    Y = pickle.load(f)


#### Plotting ##############
valid_idxs = apoUtils.get_valid_intercluster_idxs()

evaluator_Y = evaluators.EvaluatorWithFiltering(Y,Y_occam,leave_out=True,fitter_class=standard_fitter,valid_idxs=valid_idxs)
evaluator_Y.weighted_average

evaluator_Y_overfit = evaluators.EvaluatorWithFiltering(Y,Y_occam,leave_out=False,fitter_class=standard_fitter,valid_idxs=valid_idxs)
evaluator_Y_overfit.weighted_average


n_components = [5,10,15,20,25,30,35,40,45,50,55,60]
evaluators_X = [evaluators.EvaluatorWithFiltering(Z[:,:n_component],Z_occam[:,:n_component],leave_out=True,fitter_class=standard_fitter,valid_idxs=valid_idxs) for n_component in n_components]


evaluators_X_overfit = [evaluators.EvaluatorWithFiltering(Z[:,:n_component],Z_occam[:,:n_component],leave_out=False,fitter_class=standard_fitter,valid_idxs=valid_idxs) for n_component in n_components]



try: 
    plt.style.use("tex")
except:
    print("tex style not implemented (https://jwalton.info/Embed-Publication-Matplotlib-Latex/)")


save_path = root_path.parents[0]/"figures"/"pca"
save_path.mkdir(parents=True, exist_ok=True)



plt.figure(figsize=apoUtils.set_size(apoUtils.column_width))

plt.plot(n_components,np.array([i.weighted_average for i in evaluators_X]),label="with cross-validation",color=apoUtils.color1,marker='o',markersize=9,markeredgecolor="black")
plt.plot(n_components,np.array([i.weighted_average for i in evaluators_X_overfit]),label="without cross-validation",color=apoUtils.color2,marker='o',markersize=9,markeredgecolor="black")
#plt.axhline(y=evaluator_Y.weighted_average,c="blue",linestyle  = "--",label="stellar labels")
#plt.axhline(y=evaluator_Y_overfit.weighted_average,c="orange",linestyle  = "--",label="from stellar labels")
plt.ylabel("Doppelganger rate")
plt.xlabel("PCA dimensionality")
plt.minorticks_on()

#dashed_line = mlines.Line2D([], [], color="black",linestyle="--",
#                          markersize=15, label='from stellar labels')
#full_line = mlines.Line2D([], [], color="black",linestyle="-",
#                          markersize=15, label='from spectra')
blue_patch = mpatches.Patch(color=apoUtils.color1, label='With cross-validation')
orange_patch = mpatches.Patch(color=apoUtils.color2, label='Without cross-validation')


#plt.legend(handles=[full_line,dashed_line,blue_patch,orange_patch],frameon=False)
plt.legend(handles=[blue_patch,orange_patch],frameon=False)
plt.ylim(0,0.06)
plt.savefig(save_path/"pca_dimensionality.pdf",format="pdf",bbox_inches='tight')
