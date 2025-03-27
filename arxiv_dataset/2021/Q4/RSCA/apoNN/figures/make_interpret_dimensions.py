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
import mpl_scatter_density  # adds projection='scatter_density'
from matplotlib.colors import LinearSegmentedColormap

# "Viridis-like" colormap with white background
white_viridis = LinearSegmentedColormap.from_list(
    "white_viridis",
    [
        (0, "#ffffff"),
        (1e-20, "#440053"),
        (0.2, "#404388"),
        (0.4, "#2a788e"),
        (0.6, "#21a784"),
        (0.8, "#78d151"),
        (1, "#fde624"),
    ],
    N=256,
)


def using_mpl_scatter_density(fig, x, y):
    ax = fig.add_subplot(1, 1, 1, projection="scatter_density")
    density = ax.scatter_density(x, y, cmap=white_viridis)
    # fig.colorbar(density, label='Stars per pixel')


apogee_path.change_dr(16)

###Setup

root_path = pathlib.Path(__file__).resolve().parents[2] / "outputs" / "data"
# root_path = pathlib.Path("/share/splinter/ddm/taggingProject/tidyPCA/apoNN/scripts").parents[1]/"outputs"/"data"


def standard_fitter(z, z_occam):
    """This fitter performs a change-of-basis to a more appropriate basis for scaling"""
    return fitters.StandardFitter(
        z, z_occam, use_relative_scaling=True, is_pooled=True, is_robust=True
    )


def simple_fitter(z, z_occam):
    """This is a simple fitter that just scales the dimensions of the inputed representation. Which is used as a baseline"""
    return fitters.SimpleFitter(
        z, z_occam, use_relative_scaling=True, is_pooled=True, is_robust=True
    )


###Hyperparameters

z_dim = 30  # PCA dimensionality
# commands fonds for setting figure size of plots
text_width = 513.11743
column_width = 242.26653


###
###

with open(root_path / "spectra" / "without_interstellar" / "cluster.p", "rb") as f:
    Z_occam = pickle.load(f)

with open(root_path / "spectra" / "without_interstellar" / "pop.p", "rb") as f:
    Z = pickle.load(f)


###
###

with open(root_path / "labels" / "full" / "cluster.p", "rb") as f:
    Y_occam = pickle.load(f)

with open(root_path / "labels" / "full" / "pop.p", "rb") as f:
    Y = pickle.load(f)


###
###

with open(root_path / "allStar.p", "rb") as f:
    allStar_occamlike = pickle.load(f)

with open(root_path / "allStar_occam.p", "rb") as f:
    allStar_occam = pickle.load(f)


###

Z_fitter = standard_fitter(Z[:, :z_dim], Z_occam[:, :z_dim])
V = Z_fitter.transform(Z_fitter.z.centered(Z_occam[:, :z_dim]))
V_occam = Z_fitter.transform(Z_occam[:, :z_dim].centered())

###

figsize = np.array(apoUtils.set_size(apoUtils.text_width))
figsize[0] = figsize[0] / 3
figsize[1] = figsize[1] * 4 / 9


save_path = root_path.parents[0] / "figures" / "interpretation"
save_path.mkdir(parents=True, exist_ok=True)

###
plt.style.use("tex")

fig = plt.figure(figsize=figsize)
using_mpl_scatter_density(fig, allStar_occamlike["Fe_H"], V.val[:, -1])
plt.scatter(
    allStar_occam["Fe_H"], V_occam.val[:, -1], s=0.2, color="orange", label="clusters"
)
# plt.legend(frameon=False,loc="upper right", markerscale=3)
plt.xlim(-2.0, 0.7)
plt.ylim(-35, 35)
plt.ylabel("Feature \#1")
plt.xlabel("[Fe/H]")
plt.tight_layout()
plt.savefig(save_path / "feature1.pdf", bbox_inches="tight", format="pdf")


fig = plt.figure(figsize=figsize)
using_mpl_scatter_density(fig, allStar_occamlike["Fe_H"], V.val[:, -2])
plt.scatter(
    allStar_occam["Fe_H"], V_occam.val[:, -2], s=0.2, color="orange", label="clusters"
)
plt.xlim(-2.0, 0.7)
plt.ylim(-30, 40)
plt.ylabel("Feature \#2")
plt.xlabel("[Fe/H]")
plt.tight_layout()
plt.savefig(save_path / "feature2.pdf", bbox_inches="tight", format="pdf")


fig = plt.figure(figsize=figsize)
using_mpl_scatter_density(fig, allStar_occamlike["Fe_H"], V.val[:, -3])
plt.scatter(
    allStar_occam["Fe_H"], V_occam.val[:, -3], s=0.2, color="orange", label="clusters"
)
plt.xlim(-2.0, 0.7)
plt.ylim(-20, 50)
plt.ylabel("Feature \#3")
plt.xlabel("[Fe/H]")
plt.tight_layout()
plt.savefig(save_path / "feature3.pdf", bbox_inches="tight", format="pdf")


fig = plt.figure(figsize=figsize)
using_mpl_scatter_density(fig, allStar_occamlike["Fe_H"], allStar_occamlike["Mg_Fe"])
plt.scatter(
    allStar_occam["Fe_H"],
    allStar_occam["Mg_Fe"],
    s=0.2,
    color="orange",
    label="cluster stars",
)
plt.xlim(-2.0, 0.85)
plt.ylim(-0.2, 0.5)
plt.xlabel("[Fe/H]")
plt.ylabel("[Mg/Fe]")
plt.tight_layout()
plt.savefig(save_path / "mg.pdf", bbox_inches="tight", format="pdf")


fig = plt.figure(figsize=figsize)
using_mpl_scatter_density(fig,allStar_occamlike["VHELIO_AVG"],V.val[:,-4])
plt.scatter(
    allStar_occam["VHELIO_AVG"], V_occam.val[:, -4], s=0.2, color="orange", label="clusters"
)
plt.xlim(-300,150)
plt.ylim(-30,20)
plt.xlabel("VHELIO\_AVG")
plt.ylabel("Feature \#4")
plt.savefig(save_path / "vrad.pdf", bbox_inches="tight", format="pdf")



