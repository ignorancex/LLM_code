from __future__ import division
import refnx
from refnx.reflect import structure, ReflectModel, SLD
from refnx.dataset import ReflectDataset
from refnx.analysis import (
    Transform,
    CurveFitter,
    Objective,
    GlobalObjective,
    Parameter,
)

import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

sns.set(palette="colorblind")
import corner

import sys

import mol_vol as mv
import ref_help as rh

import matplotlib as mpl

mpl.rcParams["xtick.labelsize"] = 8
mpl.rcParams["ytick.labelsize"] = 8
mpl.rcParams["axes.facecolor"] = "w"
mpl.rcParams["lines.linewidth"] = 2
mpl.rcParams["xtick.top"] = False
mpl.rcParams["xtick.bottom"] = True
mpl.rcParams["ytick.left"] = True
mpl.rcParams["grid.linestyle"] = "--"
mpl.rcParams["legend.fontsize"] = 8
mpl.rcParams["legend.facecolor"] = [1, 1, 1]
mpl.rcParams["legend.framealpha"] = 0.75
mpl.rcParams["axes.labelsize"] = 8
mpl.rcParams["axes.linewidth"] = 1
mpl.rcParams["axes.edgecolor"] = "k"


# The lipid to be investigated
lipid = 'dspc'
# Number of carbon atoms in tail group
t_length = 17


# The surface pressures to probe
sp = sys.argv[1]
cont = ['d13acmw', 'd13d2o', 'hd2o', 'd70acmw',
        'd70d2o', 'd83acmw', 'd83d2o']
apm = sys.argv[2]
neutron = 1


# A label for the output files
label = "{}_{}".format(lipid, sp)


# Relative directory locations
data_dir = "../../data/experimental/surf_pres_{}".format(sp)
fig_dir = "../../reports/figures/{}".format(label)
anal_dir = "../../output/traditional/{}".format(label)



# For the analysis to be reproduced exactly, the same versions of `refnx`, `scipy`, and `numpy`, must be present.
# The versions used in the original version are:
#
# refnx.version.full_version == 0.1.2
# scipy.version.version == 1.1.0
# np.__version__ == 1.15.4


datasets = []
for i in range(len(cont)):
    datasets.append(
        ReflectDataset(
            "{}/{}{}.dat".format(
                data_dir, cont[i], sp
            )
        )
    )

b_head = []
b_tail = []
for i in range(len(cont)):
    if cont[i][:3] == 'd13':
        head = {"C": 10, "D": 13, "H": 5, "O": 8, "N": 1, "P": 1}
        tail = {"C": t_length * 2, "H": t_length * 4 + 2}
    elif cont[i][:3] == 'd70':
        head = {"C": 10, "D": 5, "H": 13, "O": 8, "N": 1, "P": 1}
        tail = {"C": t_length * 2, "D": t_length * 4 + 2}
    elif cont[i][:3] == 'd83':
        head = {"C": 10, "D": 18, "O": 8, "N": 1, "P": 1}
        tail = {"C": t_length * 2, "D": t_length * 4 + 2}
    elif cont[i][:1] == 'h':
        head = {"C": 10, "H": 18, "O": 8, "N": 1, "P": 1}
        tail = {"C": t_length * 2, "H": t_length * 4 + 2}
    b_head.append(rh.get_scattering_length(head, 1))
    b_tail.append(rh.get_scattering_length(tail, 1))



def get_value(file):
    f = open(file, "r")
    for line in f:
        k = line
    if "^" in k:
        l = k.split("$")[1].split("^")[0]
    else:
        l = k.split("$")[1].split("\\pm")[0]
    return float(l)

d_h = 8.5
V_h = 339.5
V_t = 1100.0
min_d_t = 9


lipids = []
for i in range(len(cont)):
    lipids.append(
        mv.VolMono(
            [V_h, V_t], [b_head[i], b_tail[i]], d_h, t_length, name=label
        )
    )


air = SLD(0, "air")
structures = []
for i in range(len(cont)):
    if cont[i][-3:] == 'd2o':
        water = SLD(6.35, "d2o")
    elif cont[i][-4:] == 'acmw':
        water = SLD(0.0, "acmw")
    structures.append(air(0, 0) | lipids[i] | water(0, 3.3))



for i in range(len(cont)):
    lipids[i].vol[0].setp(
        vary=False
    )
    lipids[i].vol[1].setp(
        vary=True, bounds=(V_t * 0.8, V_t * 1.2)
    )
    lipids[i].d[0].setp(vary=True, bounds=(5, 15))
    max_d_t = 1.54 + 1.265 * t_length
    lipids[i].d[1].setp(vary=True, bounds=(min_d_t, max_d_t))
    lipids[i].vol[1].constraint = lipids[i].d[1] * float(apm)
    structures[i][-1].rough.setp(vary=True, bounds=(3., 8))

lipids = rh.set_constraints(
    lipids,
    structures,
    hold_tails=True,
    hold_rough=True,
    hold_phih=True,
)


models = []
t = len(cont)

for i in range(t):
    models.append(ReflectModel(structures[i]))
    models[i].scale.setp(vary=True, bounds=(0.005, 10))
    models[i].bkg.setp(datasets[i].y[-1], vary=True, bounds=(1e-4, 1e-10))

objectives = []
t = len(cont)

for i in range(t):
    objectives.append(
        Objective(models[i], datasets[i], transform=Transform("YX4"))
    )

global_objective = GlobalObjective(objectives)

fitter = CurveFitter(global_objective)
np.random.seed(1)
res = fitter.fit("differential_evolution")

print(global_objective)


fitter.sample(200, random_state=1)
fitter.sampler.reset()
res = fitter.sample(
    1000,
    nthin=1,
    random_state=1,
    f="{}_chain.txt".format(anal_dir),
)
