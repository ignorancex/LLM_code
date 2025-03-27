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
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
apogee_path.change_dr(16)

###Setup

root_path = pathlib.Path(__file__).resolve().parents[2]/"outputs"/"data"
#root_path = pathlib.Path("/share/splinter/ddm/taggingProject/tidyPCA/apoNN/scripts").parents[1]/"outputs"/"data"

save_path = root_path.parents[0]/"figures"/"local"
save_path.mkdir(parents=True, exist_ok=True)



def standard_fitter(z,z_occam):
        """This fitter performs a change-of-basis to a more appropriate basis for scaling"""
        return fitters.StandardFitter(z,z_occam,use_relative_scaling=True,is_pooled=True,is_robust=True)

def simple_fitter(z,z_occam):
        """This is a simple fitter that just scales the dimensions of the inputed representation. Which is used as a abundances"""
        return fitters.SimpleFitter(z,z_occam,use_relative_scaling=True,is_pooled=True,is_robust=True)


###Hyperparameters

z_dim = 30 #PCA dimensionality
cmap = matplotlib.cm.get_cmap('viridis')
color1 = cmap(0.15)
color2 = cmap(0.75)


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

###calculate representations being visualized

valid_idxs = apoUtils.get_valid_intercluster_idxs()

evaluator_X = evaluators.EvaluatorWithFiltering(Z[:,:z_dim],Z_occam[:,:z_dim],leave_out=True,fitter_class=standard_fitter,valid_idxs=valid_idxs)
evaluator_X.weighted_average

evaluator_Y = evaluators.EvaluatorWithFiltering(Y,Y_occam,leave_out=True,fitter_class=standard_fitter,valid_idxs=valid_idxs)
evaluator_Y.weighted_average

### Define plotting function

def plot_cluster(evaluator,cluster_name,ax1=None,x_max=30,color1="orange",color2="blue",use_annotation=True,title=None,cutoff_percentile=80):
    """function for visualizing the doppelganger rate of a chosen cluster.
    INPUTS
    ------
    cluster_name: string
        the name of the cluster (as found in the registry) to be plotted.
    ax1: matplotlib axis
        an axis on which to plot the cluster. Useful when combining subplots of individual clusters into one large plot.
    x_max: The cut-off to use on the x-axis of ds
    cutoff_percentile:
        number between 0-100 controlling the cutoff xlim. Dictates the fraction of pdf to include within bounds."""
    if ax1 is None:
        ax1 = plt.gca()
    index_cluster = sorted(evaluator.registry).index(cluster_name)
    
    if title is not None:
        ax1.set_title(title,fontsize=14)

    ax1.set_xlabel('d')
    ax1.set_ylabel('p', color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    x_max = np.percentile(evaluator.random_distances[index_cluster],cutoff_percentile)
    ax1.set_xlim(0,x_max)
    ax1.hist(evaluator.distances[index_cluster],bins=evaluator.stars_per_cluster[index_cluster],density=True,label="intercluster",color=color1,linewidth=2,histtype="step")

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.hist(evaluator.random_distances[index_cluster],bins=200,density=True,label="intracluster",color=color2,linewidth=2,histtype="step")
    ax2.set_ylabel('p', color=color2)  # we already handled the x-label with ax1
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.axvline(x=np.median(evaluator.distances[index_cluster]),color=color1,linestyle  = "--",linewidth=2)
    extra = matplotlib.patches.Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
    
    ax2.legend([extra], [f"rate: {evaluator.doppelganger_rates[index_cluster]:5.4f}"],frameon=False)
    
    pad=0.5
    if use_annotation:
        ax1.annotate(f"{cluster_name} ({evaluator.stars_per_cluster[index_cluster]} stars)",fontsize=10, xy=(0, 0.5), xytext=(-ax1.yaxis.labelpad - pad, 0),
                    xycoords=ax1.yaxis.label, textcoords='offset points',
                    size='large', ha='right', va='center',rotation=90)


### Make plots


#plot1
n_cols = 2
n_rows=6
start_idx = 0
fig = plt.figure(constrained_layout=True,figsize=[4*n_cols,2.5*n_rows])
gspec = gridspec.GridSpec(ncols=n_cols, nrows=n_rows, figure=fig)
#for i in range(len(sorted(spectra_evaluator.registry))):
for i in range(n_rows):
    spec_ax = fig.add_subplot(gspec[i,0])
    if i==0:  
        plot_cluster(evaluator_X,sorted(evaluator_X.registry)[i+start_idx],spec_ax,x_max=30,color1=color1,color2=color2,title="spectra")
        abund_ax = fig.add_subplot(gspec[i,1])
        #abund_ax.set_xlabel("d",fontsize=20)
        plot_cluster(evaluator_Y,sorted(evaluator_Y.registry)[i+start_idx],abund_ax,x_max=20,color1=color1,color2=color2,use_annotation=False,title="abundances")

    if i!=0:  
        plot_cluster(evaluator_X,sorted(evaluator_X.registry)[i+start_idx],spec_ax,x_max=30,color1=color1,color2=color2)
        abund_ax = fig.add_subplot(gspec[i,1])
        #abund_ax.set_xlabel("d",fontsize=20)
        plot_cluster(evaluator_Y,sorted(evaluator_Y.registry)[i+start_idx],abund_ax,x_max=20,color1=color1,color2=color2,use_annotation=False)

plt.savefig(save_path/"loc1.pdf",format="pdf")

#plot2
n_cols = 2
n_rows=6
start_idx = 6
fig = plt.figure(constrained_layout=True,figsize=[4*n_cols,2.5*n_rows])
gspec = gridspec.GridSpec(ncols=n_cols, nrows=n_rows, figure=fig)
#for i in range(len(sorted(spectra_evaluator.registry))):
for i in range(n_rows):
    spec_ax = fig.add_subplot(gspec[i,0])
    if i==0:  
        plot_cluster(evaluator_X,sorted(evaluator_X.registry)[i+start_idx],spec_ax,x_max=30,color1=color1,color2=color2,title="spectra")
        abund_ax = fig.add_subplot(gspec[i,1])
        #abund_ax.set_xlabel("d",fontsize=20)
        plot_cluster(evaluator_Y,sorted(evaluator_Y.registry)[i+start_idx],abund_ax,x_max=20,color1=color1,color2=color2,use_annotation=False,title="abundances")

    if i!=0:  
        plot_cluster(evaluator_X,sorted(evaluator_X.registry)[i+start_idx],spec_ax,x_max=30,color1=color1,color2=color2)
        abund_ax = fig.add_subplot(gspec[i,1])
        #abund_ax.set_xlabel("d",fontsize=20)
        plot_cluster(evaluator_Y,sorted(evaluator_Y.registry)[i+start_idx],abund_ax,x_max=20,color1=color1,color2=color2,use_annotation=False)
plt.savefig(save_path/"loc2.pdf",format="pdf")

#plot3
n_cols = 2
n_rows=5
start_idx = 12
fig = plt.figure(constrained_layout=True,figsize=[4*n_cols,2.5*n_rows])
gspec = gridspec.GridSpec(ncols=n_cols, nrows=n_rows, figure=fig)
#for i in range(len(sorted(spectra_evaluator.registry))):
for i in range(n_rows):
    spec_ax = fig.add_subplot(gspec[i,0])
    if i==0:  
        plot_cluster(evaluator_X,sorted(evaluator_X.registry)[i+start_idx],spec_ax,x_max=30,color1=color1,color2=color2,title="spectra")
        abund_ax = fig.add_subplot(gspec[i,1])
        #abund_ax.set_xlabel("d",fontsize=20)
        plot_cluster(evaluator_Y,sorted(evaluator_Y.registry)[i+start_idx],abund_ax,x_max=20,color1=color1,color2=color2,use_annotation=False,title="abundances")

    if i!=0:  
        plot_cluster(evaluator_X,sorted(evaluator_X.registry)[i+start_idx],spec_ax,x_max=30,color1=color1,color2=color2)
        abund_ax = fig.add_subplot(gspec[i,1])
        #abund_ax.set_xlabel("d",fontsize=20)
        plot_cluster(evaluator_Y,sorted(evaluator_Y.registry)[i+start_idx],abund_ax,x_max=20,color1=color1,color2=color2,use_annotation=False)
plt.savefig(save_path/"loc3.pdf",format="pdf")


#plot4
n_cols = 2
n_rows=5
start_idx = 17
fig = plt.figure(constrained_layout=True,figsize=[4*n_cols,2.5*n_rows])
gspec = gridspec.GridSpec(ncols=n_cols, nrows=n_rows, figure=fig)
#for i in range(len(sorted(spectra_evaluator.registry))):
for i in range(n_rows):
    spec_ax = fig.add_subplot(gspec[i,0])
    if i==0:  
        plot_cluster(evaluator_X,sorted(evaluator_X.registry)[i+start_idx],spec_ax,x_max=30,color1=color1,color2=color2,title="spectra")
        abund_ax = fig.add_subplot(gspec[i,1])
        #abund_ax.set_xlabel("d",fontsize=20)
        plot_cluster(evaluator_Y,sorted(evaluator_Y.registry)[i+start_idx],abund_ax,x_max=20,color1=color1,color2=color2,use_annotation=False,title="abundances")

    if i!=0:  
        plot_cluster(evaluator_X,sorted(evaluator_X.registry)[i+start_idx],spec_ax,x_max=30,color1=color1,color2=color2)
        abund_ax = fig.add_subplot(gspec[i,1])
        #abund_ax.set_xlabel("d",fontsize=20)
        plot_cluster(evaluator_Y,sorted(evaluator_Y.registry)[i+start_idx],abund_ax,x_max=20,color1=color1,color2=color2,use_annotation=False)
plt.savefig(save_path/"loc4.pdf",format="pdf")
    
