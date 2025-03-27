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


def ablation_performance(Z,Z_occam,valid_idxs):
    ev_standard = evaluators.EvaluatorWithFiltering(Z,Z_occam,leave_out=True,fitter_class=standard_fitter,valid_idxs=valid_idxs)
    ev_simple = evaluators.EvaluatorWithFiltering(Z,Z_occam,leave_out=True,fitter_class=simple_fitter,valid_idxs=valid_idxs)
    ev_empty = evaluators.EvaluatorWithFiltering(Z,Z_occam,leave_out=True,fitter_class=fitters.EmptyFitter,valid_idxs=valid_idxs)
    return ev_empty.weighted_average, ev_simple.weighted_average,ev_standard.weighted_average

def autolabel(rects,rects_err=None):
    """
    Attach a text label above each bar displaying its height
    """
    if rects_err is None:
        rects_err = [0]*len(rects)
        
    y_locs = []
    x_locs = []
    text = []
    for rect,rect_err in zip(rects,rects_err):
        x_locs.append(rect.get_x() + rect.get_width()/2.)
        y_locs.append(rect.get_height()+rect_err)
        text.append(rect.get_height())
    return label_locs(x_locs,y_locs,text)

def label_locs(x_locs,y_locs,texts):
    """
    Add text at desired locations
    """
    for (x_loc,y_loc,text) in zip(x_locs,y_locs,texts):
        ax.text(x_loc, y_loc,
                f"{text:#.2g}",
                ha='center', va='bottom',fontsize=5.5)


        
def load_path(n,tol,d):
    """"Load a set of Z_occam and Z"""
    with open(root_path/"spectra"/"without_interstellar"/f"clusterD{d}N{n}tol{tol}.p","rb") as f:
        Z_occam = pickle.load(f)    

    with open(root_path/"spectra"/"without_interstellar"/f"popD{d}N{n}tol{tol}.p","rb") as f:
        Z = pickle.load(f)    
    return Z,Z_occam





###Hyperparameters

z_dim = 30 #PCA dimensionality
tol = 1E-4 #the tol of the pickled ppca being loaded
dim = 30 #the number of dimensions of the pickled ppca being loaded 
num_models = 10 #the number of pickled models to use for statistics
valid_idxs = apoUtils.get_valid_intercluster_idxs()


ablations_Z = [] #take average of multiple runs because of stochasticity of ppca
for n in range(10):
    Z,Z_occam = load_path(n,tol,dim)
    ablation_Z = ablation_performance(Z[:,:z_dim],Z_occam[:,:z_dim],valid_idxs)
    ablations_Z.append(ablation_Z)
    
ablation_Z = np.mean(ablations_Z,axis=0)
ablation_Z_err = np.std(ablations_Z,axis=0)



###
###

with open(root_path/"labels"/"full"/"cluster.p","rb") as f:
    Y_occam_full = pickle.load(f)    

with open(root_path/"labels"/"full"/"pop.p","rb") as f:
    Y_full = pickle.load(f)    

ablation_Y_full = ablation_performance(Y_full,Y_occam_full,valid_idxs)

###
###

with open(root_path/"labels"/"core"/"cluster.p","rb") as f:
    Y_occam_core = pickle.load(f)    

with open(root_path/"labels"/"core"/"pop.p","rb") as f:
    Y_core = pickle.load(f)    

ablation_Y_core = ablation_performance(Y_core,Y_occam_core,valid_idxs)


no_transformation = [ablation_Z[0],ablation_Y_full[0],ablation_Y_core[0]]
only_scaling = [ablation_Z[1],ablation_Y_full[1],ablation_Y_core[1]]
full_algorithm = [ablation_Z[2],ablation_Y_full[2],ablation_Y_core[2]]

no_transformation_err = np.array([ablation_Z_err[0],0,0])
only_scaling_err = np.array([ablation_Z_err[1],0,0])
full_algorithm_err = np.array([ablation_Z_err[2],0,0])



labels = ["Spectra","All abundances","Abundance subset"]
x = np.arange(len(labels))  # the label locations

width = 0.29  # the width of the bars
capsize=2.5

x_err = [-width,0,0+width]
y_err = ablation_Z
y_err_val=ablation_Z_err



save_path = root_path.parents[0]/"figures"/"ablation"
save_path.mkdir(parents=True, exist_ok=True)



plt.style.use('seaborn-colorblind')
plt.style.use('tex')


fig, ax = plt.subplots(figsize =apoUtils.set_size(apoUtils.column_width))
rects1 = ax.bar(x-width, no_transformation, width, label='On raw')
rects2 = ax.bar(x, only_scaling, width, label='On scaled')
rects3 = ax.bar(x +width, full_algorithm, width, label='On transformed')
autolabel(rects1,no_transformation_err)
autolabel(rects2,only_scaling_err)
autolabel(rects3,full_algorithm_err)

plt.errorbar(x=x_err,y=y_err,yerr=y_err_val,linestyle=" ",c="black",capsize=2.2,elinewidth=0.5)

ax.set_ylabel('Doppelganger rate',fontsize=8)
ax.set_xticks(x)
ax.set_ylim([0,0.17])
ax.set_xticklabels(labels, fontsize=7)
ax.legend(frameon=True)
plt.savefig(save_path/"ablation.pdf",format="pdf",bbox_inches='tight')
