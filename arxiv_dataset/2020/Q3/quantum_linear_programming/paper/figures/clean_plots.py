import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from qlpdb.data.models import Data as data_Data
import pickle
import numpy as np
from scipy.linalg import logm
from numpy.linalg import eigh

from django.conf import settings
from qlpdb.tdse.models import Tdse
from qlp.tdse import convert_params, embed_qubo_example
from qlp.mds import graph_tools as gt
from qlp.mds.mds_qlpdb import graph_summary

p = dict()
p["figsize"] = (7, 4)
p["aspect_ratio"] = [0.15, 0.15, 0.8, 0.8]
p["textargs"] = {"fontsize": 10}

colorbar_label_all = ["-0.05w", "-0.04w", "-0.02w", "0.0", "-0.02s", "-0.04s"]
colorbar_label = ["-0.05w", "0.0", "-0.05s"]


red = "#E8606A"
green = "#61C552"
blue = "#4B689E"
yellow = "#EEBD63"
purple = "#80479F"

os_color = dict()
os_color["dwave"] = red
os_color["sim"] = blue

os_alpha = dict()
os_alpha[-0.07] = 0.2
os_alpha[-0.06] = 0.4
os_alpha[-0.05] = 1.0
os_alpha[-0.04] = 0.8
os_alpha[-0.03] = 0.6
os_alpha[-0.02] = 0.4
os_alpha[-0.01] = 0.2
os_alpha[0.0] = 0.5
os_alpha[0.01] = 0.2
os_alpha[0.02] = 0.4
os_alpha[0.03] = 0.6
os_alpha[0.04] = 0.8
os_alpha[0.05] = 1.0
os_alpha[0.06] = 0.4
os_alpha[0.07] = 0.2

os_alpha = {key: 1.0 for key in [-0.07, -0.06, -0.05, -0.04, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]}

OFFSETS = 0.01 * (np.arange(11) - 5)
OFFSET_COLORS = sns.diverging_palette(0, 255, l=50, center="dark", n=11)
OFFSET_COLORS2 = sns.diverging_palette(51, 204, l=50, center="dark", n=11)
OFFSET_COLORS3 = sns.diverging_palette(102, 153, l=50, center="dark", n=11)

OFFSET_MAP = dict(zip(OFFSETS, OFFSET_COLORS))
OFFSET_MAP2 = dict(zip(OFFSETS, OFFSET_COLORS2))
OFFSET_MAP3 = dict(zip(OFFSETS, OFFSET_COLORS3))

OFFSET_CMAP = ListedColormap(OFFSET_COLORS)
sns.palplot(OFFSET_COLORS)

"""
###################################
#####  simulation Data Class  #####
###################################
"""


class Sim:
    def __init__(self):
        self.params = self.parameters()

    def parameters(self):
        # offset params
        offset_params = dict()
        offset_params["annealing_time"] = 1
        offset_params["normalized_time"] = "[0, 1]"
        offset_params["offset"] = "single_sided_binary"
        offset_params["offset_min"] = 0
        offset_params["anneal_curve"] = "dwave"
        print(offset_params)
        # wave params
        wave_params = dict()
        wave_params["type"] = "mixed"
        """==============================="""
        # wave_params["temp"] = 0.015
        # wave_params["gamma"] = 1 / 1
        # wave_params["gamma_local"] = 1 / 10
        """==============================="""
        # wave_params["temp"] = 0.03
        # wave_params["gamma"] = 1 / 1
        # wave_params["gamma_local"] = 1 / 30
        """==============================="""
        wave_params["temp"] = 0.0225
        wave_params["gamma"] = 1 / 1
        wave_params["gamma_local"] = 1 / 15  # 1/15 or 1/20
        """==============================="""
        # wave_params["temp"] = 0.04
        # wave_params["gamma"] = 1 / 10
        # wave_params["gamma_local"] = 1 / 10 # 1/15 or 1/20
        """==============================="""
        wave_params["initial_wavefunction"] = "transverse"

        # graph params
        nvertices = 2
        graph, tag = gt.generate_nn_graph(nvertices)
        qubo, embedding = embed_qubo_example(nvertices)
        graph_params = graph_summary(tag, graph, qubo)

        # solver params
        solver_params = dict()
        solver_params["method"] = "RK45"
        solver_params["rtol"] = 1e-6
        solver_params["atol"] = 1e-7

        params = {
            "offset": offset_params,
            "wave": wave_params,
            "graph": graph_params,
            "solver": solver_params,
        }
        return params

    def get_data(self, offset, normalized_time = [0.0, 1.0]):
        self.params["offset"]["offset_min"] = offset
        self.params["offset"]["normalized_time"] = normalized_time
        print(self.params["graph"]["tag"])
        print(convert_params(self.params["offset"]))
        print(self.params["solver"])
        print(self.params["wave"])
        query = Tdse.objects.filter(
            graph__tag=self.params["graph"]["tag"],
            offset__contains=convert_params(self.params["offset"]),
            solver__contains=self.params["solver"],
            wave__contains=self.params["wave"],
        ).first()
        from django.db import connection
        print(connection.queries)
        with open(f"{settings.MEDIA_ROOT}/{query.solution}", "rb") as file:
            sol = pickle.load(file)
        with open(f"{settings.MEDIA_ROOT}/{query.instance}", "rb") as file:
            tdse = pickle.load(file)
        return query, sol, tdse


"""
###########################################
#####  Plot final state distribution  #####
###########################################
"""


def getallspinconfig(anneal_time=1, offset=-0.05):
    if True:
        dist = {-0.05: {((1, 0, 0, 1, 0), 2.90909090909091): 0.62797, ((1, 1, 1, 1, 1), 3.27272727272727): 0.19094,
                        ((0, 1, 0, 0, 0), 2.90909090909091): 0.16842, ((1, 0, 1, 1, 0), 3.63636363636364): 0.00629,
                        ((1, 0, 0, 1, 1), 3.63636363636364): 0.00489, ((0, 1, 0, 0, 1), 3.63636363636364): 0.00049,
                        ((0, 1, 1, 0, 0), 3.63636363636364): 0.0004, ((1, 1, 0, 1, 1), 4.0): 0.00018,
                        ((1, 1, 1, 1, 0), 4.0): 0.00022, ((0, 0, 0, 0, 0), 4.0): 6e-05,
                        ((1, 1, 1, 0, 1), 3.81818181818182): 0.00011, ((1, 0, 1, 1, 1), 4.36363636363636): 3e-05},
                -0.04: {((1, 0, 0, 1, 0), 2.90909090909091): 0.59694, ((0, 1, 0, 0, 0), 2.90909090909091): 0.22433,
                        ((1, 1, 1, 1, 1), 3.27272727272727): 0.16573, ((1, 0, 1, 1, 0), 3.63636363636364): 0.00633,
                        ((0, 1, 0, 0, 1), 3.63636363636364): 0.00056, ((1, 0, 0, 1, 1), 3.63636363636364): 0.00477,
                        ((1, 1, 1, 0, 1), 3.81818181818182): 0.00012, ((0, 1, 1, 0, 0), 3.63636363636364): 0.00064,
                        ((0, 0, 1, 0, 0), 6.18181818181818): 1e-05, ((1, 1, 1, 1, 0), 4.0): 0.00027,
                        ((1, 0, 1, 1, 1), 4.36363636363636): 7e-05, ((0, 0, 0, 0, 0), 4.0): 4e-05,
                        ((1, 1, 0, 1, 1), 4.0): 0.00016, ((0, 1, 1, 0, 1), 4.36363636363636): 2e-05,
                        ((1, 1, 0, 1, 0), 4.72727272727273): 1e-05},
                -0.03: {((1, 0, 0, 1, 0), 2.90909090909091): 0.57956, ((0, 1, 0, 0, 0), 2.90909090909091): 0.26118,
                        ((1, 0, 0, 1, 1), 3.63636363636364): 0.00474, ((1, 1, 1, 1, 1), 3.27272727272727): 0.14611,
                        ((1, 0, 1, 1, 0), 3.63636363636364): 0.00624, ((0, 1, 1, 0, 0), 3.63636363636364): 0.00089,
                        ((0, 1, 0, 0, 1), 3.63636363636364): 0.0007, ((1, 1, 0, 1, 1), 4.0): 0.00012,
                        ((0, 0, 0, 0, 0), 4.0): 0.00015, ((1, 1, 1, 1, 0), 4.0): 0.00018,
                        ((1, 0, 1, 1, 1), 4.36363636363636): 3e-05, ((1, 1, 1, 0, 1), 3.81818181818182): 8e-05,
                        ((1, 0, 0, 0, 0), 6.36363636363636): 2e-05},
                -0.02: {((1, 1, 1, 1, 1), 3.27272727272727): 0.1422, ((0, 1, 0, 0, 0), 2.90909090909091): 0.29355,
                        ((1, 0, 0, 1, 0), 2.90909090909091): 0.55019, ((1, 0, 1, 1, 0), 3.63636363636364): 0.00649,
                        ((0, 1, 1, 0, 0), 3.63636363636364): 0.00104, ((1, 0, 0, 1, 1), 3.63636363636364): 0.00481,
                        ((0, 1, 0, 0, 1), 3.63636363636364): 0.00113, ((0, 0, 0, 0, 0), 4.0): 0.00017,
                        ((1, 1, 1, 0, 1), 3.81818181818182): 0.0001, ((1, 1, 0, 1, 1), 4.0): 0.0001,
                        ((1, 1, 1, 1, 0), 4.0): 0.00012, ((1, 0, 1, 1, 1), 4.36363636363636): 7e-05,
                        ((1, 1, 0, 1, 0), 4.72727272727273): 1e-05, ((0, 1, 1, 0, 1), 4.36363636363636): 1e-05,
                        ((0, 0, 0, 1, 0), 6.36363636363636): 1e-05},
                -0.01: {((1, 0, 0, 1, 0), 2.90909090909091): 0.53033, ((0, 1, 0, 0, 0), 2.90909090909091): 0.35464,
                        ((1, 1, 1, 1, 1), 3.27272727272727): 0.10024, ((1, 0, 1, 1, 0), 3.63636363636364): 0.00579,
                        ((1, 0, 0, 1, 1), 3.63636363636364): 0.0044, ((0, 1, 0, 0, 1), 3.63636363636364): 0.0017,
                        ((0, 0, 0, 0, 0), 4.0): 0.00037, ((0, 1, 1, 0, 0), 3.63636363636364): 0.00205,
                        ((1, 1, 1, 0, 1), 3.81818181818182): 0.00014, ((1, 1, 1, 1, 0), 4.0): 0.00016,
                        ((1, 1, 0, 1, 1), 4.0): 0.00012, ((1, 0, 1, 1, 1), 4.36363636363636): 4e-05,
                        ((1, 1, 0, 1, 0), 4.72727272727273): 1e-05, ((0, 0, 0, 1, 0), 6.36363636363636): 1e-05},
                0.0: {((0, 1, 0, 0, 0), 2.90909090909091): 0.42569, ((1, 0, 0, 1, 0), 2.90909090909091): 0.47878,
                      ((1, 1, 1, 1, 1), 3.27272727272727): 0.07999, ((0, 1, 1, 0, 0), 3.63636363636364): 0.00292,
                      ((1, 0, 0, 1, 1), 3.63636363636364): 0.00422, ((1, 0, 1, 1, 0), 3.63636363636364): 0.00532,
                      ((0, 1, 0, 0, 1), 3.63636363636364): 0.00255, ((1, 1, 0, 1, 1), 4.0): 0.0001,
                      ((1, 1, 1, 1, 0), 4.0): 6e-05, ((1, 1, 1, 0, 1), 3.81818181818182): 5e-05,
                      ((0, 0, 0, 0, 0), 4.0): 0.00024, ((1, 0, 1, 1, 1), 4.36363636363636): 3e-05,
                      ((0, 1, 1, 0, 1), 4.36363636363636): 1e-05, ((1, 1, 0, 1, 0), 4.72727272727273): 2e-05,
                      ((1, 0, 0, 0, 0), 6.36363636363636): 1e-05, ((1, 1, 1, 0, 0), 4.54545454545454): 1e-05},
                0.01: {((1, 1, 1, 1, 1), 3.27272727272727): 0.06519, ((1, 0, 1, 1, 0), 3.63636363636364): 0.00495,
                       ((0, 1, 0, 0, 0), 2.90909090909091): 0.44892, ((1, 0, 0, 1, 0), 2.90909090909091): 0.46998,
                       ((0, 1, 1, 0, 0), 3.63636363636364): 0.00365, ((1, 0, 0, 1, 1), 3.63636363636364): 0.00355,
                       ((0, 1, 0, 0, 1), 3.63636363636364): 0.003, ((0, 0, 0, 0, 0), 4.0): 0.00037,
                       ((1, 1, 1, 1, 0), 4.0): 5e-05, ((1, 0, 1, 1, 1), 4.36363636363636): 5e-05,
                       ((1, 1, 1, 0, 1), 3.81818181818182): 0.0002, ((1, 1, 0, 1, 1), 4.0): 5e-05,
                       ((0, 1, 1, 0, 1), 4.36363636363636): 1e-05, ((1, 1, 0, 0, 0), 5.27272727272727): 1e-05,
                       ((1, 1, 0, 1, 0), 4.72727272727273): 1e-05, ((1, 1, 1, 0, 0), 4.54545454545454): 1e-05},
                0.02: {((0, 1, 0, 0, 0), 2.90909090909091): 0.4664, ((1, 0, 0, 1, 0), 2.90909090909091): 0.46304,
                       ((1, 1, 1, 1, 1), 3.27272727272727): 0.05255, ((0, 1, 1, 0, 0), 3.63636363636364): 0.0047,
                       ((1, 1, 1, 0, 1), 3.81818181818182): 0.00021, ((0, 1, 0, 0, 1), 3.63636363636364): 0.00384,
                       ((1, 0, 1, 1, 0), 3.63636363636364): 0.0051, ((1, 0, 0, 1, 1), 3.63636363636364): 0.00357,
                       ((0, 0, 0, 0, 0), 4.0): 0.00035, ((1, 1, 1, 1, 0), 4.0): 9e-05, ((1, 1, 0, 1, 1), 4.0): 0.00011,
                       ((0, 0, 0, 1, 0), 6.36363636363636): 1e-05, ((0, 1, 1, 0, 1), 4.36363636363636): 1e-05,
                       ((1, 0, 1, 1, 1), 4.36363636363636): 1e-05, ((1, 1, 0, 1, 0), 4.72727272727273): 1e-05},
                0.03: {((0, 1, 0, 0, 0), 2.90909090909091): 0.49664, ((1, 0, 0, 1, 0), 2.90909090909091): 0.44672,
                       ((1, 1, 1, 1, 1), 3.27272727272727): 0.03604, ((0, 1, 0, 0, 1), 3.63636363636364): 0.00578,
                       ((1, 0, 1, 1, 0), 3.63636363636364): 0.00426, ((1, 0, 0, 1, 1), 3.63636363636364): 0.00313,
                       ((0, 1, 1, 0, 0), 3.63636363636364): 0.00653, ((1, 0, 1, 1, 1), 4.36363636363636): 5e-05,
                       ((0, 0, 0, 0, 0), 4.0): 0.0004, ((0, 0, 0, 1, 0), 6.36363636363636): 2e-05,
                       ((1, 1, 1, 0, 1), 3.81818181818182): 0.00027, ((1, 1, 1, 1, 0), 4.0): 3e-05,
                       ((1, 1, 0, 0, 0), 5.27272727272727): 2e-05, ((0, 1, 1, 0, 1), 4.36363636363636): 6e-05,
                       ((1, 1, 0, 1, 1), 4.0): 3e-05, ((0, 0, 0, 0, 1), 6.18181818181818): 1e-05,
                       ((1, 1, 0, 0, 1), 4.54545454545454): 1e-05},
                0.04: {((0, 1, 0, 0, 0), 2.90909090909091): 0.54121, ((1, 0, 0, 1, 0), 2.90909090909091): 0.40743,
                       ((1, 1, 1, 1, 1), 3.27272727272727): 0.02785, ((0, 1, 0, 0, 1), 3.63636363636364): 0.00764,
                       ((0, 1, 1, 0, 0), 3.63636363636364): 0.0083, ((1, 0, 0, 1, 1), 3.63636363636364): 0.00288,
                       ((1, 0, 1, 1, 0), 3.63636363636364): 0.00373, ((1, 1, 1, 0, 1), 3.81818181818182): 0.00046,
                       ((0, 0, 0, 0, 0), 4.0): 0.00039, ((1, 1, 1, 1, 0), 4.0): 1e-05, ((1, 1, 0, 1, 1), 4.0): 4e-05,
                       ((0, 1, 1, 0, 1), 4.36363636363636): 2e-05, ((1, 0, 1, 1, 1), 4.36363636363636): 3e-05,
                       ((1, 1, 0, 1, 0), 4.72727272727273): 1e-05},
                0.05: {((1, 0, 0, 1, 0), 2.90909090909091): 0.39635, ((0, 1, 0, 0, 0), 2.90909090909091): 0.55616,
                       ((0, 1, 1, 0, 0), 3.63636363636364): 0.0107, ((1, 1, 1, 1, 1), 3.27272727272727): 0.02126,
                       ((0, 1, 0, 0, 1), 3.63636363636364): 0.00902, ((1, 0, 1, 1, 0), 3.63636363636364): 0.00302,
                       ((1, 1, 1, 0, 1), 3.81818181818182): 0.00074, ((0, 0, 0, 0, 0), 4.0): 0.00035,
                       ((1, 0, 0, 1, 1), 3.63636363636364): 0.00214, ((1, 1, 0, 1, 0), 4.72727272727273): 3e-05,
                       ((0, 1, 1, 0, 1), 4.36363636363636): 7e-05, ((1, 1, 0, 1, 1), 4.0): 8e-05,
                       ((1, 1, 0, 0, 0), 5.27272727272727): 2e-05, ((1, 1, 1, 1, 0), 4.0): 4e-05,
                       ((1, 0, 1, 1, 1), 4.36363636363636): 1e-05, ((1, 1, 0, 0, 1), 4.54545454545454): 1e-05}}
        if offset in dist:
            return dist[offset]

    data = data_Data.objects.filter(
        experiment__graph__tag=f"NN(2)",
        experiment__tag=f"FixEmbedding_Single_Sided_Binary_{offset}_z3",
        experiment__settings__annealing_time=anneal_time,
        experiment__settings__num_spin_reversal_transforms=0,
    ).to_dataframe()

    spin = data["spin_config"]
    energy = data["energy"]
    spin_energy_count = dict()
    for idx in range(len(energy)):
        tag = (tuple(spin[idx]), energy[idx])
        if tag in spin_energy_count.keys():
            spin_energy_count[tag] += 1
        else:
            spin_energy_count[tag] = 1
    total = sum([spin_energy_count[tag] for tag in spin_energy_count])
    spin_energy_count = {tag: spin_energy_count[tag] / total for tag in spin_energy_count}
    return spin_energy_count


def getallspintheory(offset=0.0):
    sim = Sim()
    query, sol, tdse = sim.get_data(offset)
    rhodim = 2 ** 5
    state_prob = np.diagonal(sol.y[:, -1].reshape((rhodim, rhodim))).real
    states = [tuple([int(j) for j in list("{0:05b}".format(i))]) for i in range(2 ** 5)]
    prob = {states[idx]: state_prob[idx] for idx in range(2 ** 5)}
    return prob

def plot_distribution(offset=OFFSETS):
    count_energy = {oi: getallspinconfig(offset=oi) for oi in offset}
    count = {
        oi: {
            spin_energy[0]: count_energy[oi][spin_energy]
            for spin_energy in count_energy[oi]
        }
        for oi in count_energy
    }
    energy = {
        oi: {spin_energy[0]: spin_energy[1] for spin_energy in count_energy[oi]}
        for oi in count_energy
    }
    # Xlabel = [tuple([int(q) for q in list(format(i, '05b'))]) for i in range(2 ** 5)]
    # X = np.arange(2 ** 5)
    Xlabel = [(0, 1, 0, 0, 0), (1, 0, 0, 1, 0), (1, 1, 1, 1, 1)]
    X = np.array(range(len(Xlabel)))

    fig = plt.figure(figsize=p["figsize"])
    ax = plt.axes(p["aspect_ratio"])
    for idx, os in enumerate(offset):
        """set X"""
        xdraw = X - 0.5 + idx / (len(offset) + 1) + 0.5 / (len(offset) + 1)
        width = 1 / (len(offset) + 1)
        """plot dwave result
        """
        height = []
        for Xi in Xlabel:
            if Xi in count[os]:
                height.append(count[os][Xi])
            else:
                height.append(0)
        ax.bar(
            x=xdraw,
            height=height,
            width=width,
            color=OFFSET_COLORS[idx],
            label="experiment" if idx == 6 else None,
            zorder=1,
            alpha=0.9,
            align="edge",
        )
        """plot simulation result
        """

        prob = getallspintheory(os)
        height = [prob[Xi] for Xi in Xlabel]
        if idx == len(offset) - 1:
            label = "simulation"
        else:
            label = None
        ax.bar(
            x=xdraw,
            height=height,
            width=width,
            linewidth=1,
            color="none",
            edgecolor="k",
            align="edge",
            label=label,
            zorder=2,
        )

    """labels
    """
    ax.set_xlabel("final state distribution", p["textargs"])
    ax.set_ylabel("ground state probability", p["textargs"])
    ax.set_xticks(X)
    ax.set_xticklabels(Xlabel, rotation="0")

    cax = fig.add_axes([0.7, 0.77, 0.21, 0.02])
    cbar = mpl.colorbar.ColorbarBase(
        cax,
        cmap=OFFSET_CMAP,
        norm=mpl.colors.Normalize(vmin=-0.05, vmax=0.05),
        orientation="horizontal",
        label="offset",
    )
    cax.set_xticklabels(colorbar_label)
    ax.legend(bbox_to_anchor=(0.9, 0.98, 0.045, 0.01), fancybox=True, shadow=False, frameon=True, framealpha=0.3)

    fig.savefig("../new_figures/final_state_distribution.pdf", transparent=True, )


"""
######################################
#####  plot prob vs anneal time  #####
######################################
"""


def getallannealtime(anneal_time=1, offset_tag="FixEmbedding_Single_Sided_Binary_0_z3"):
    data = data_Data.objects.filter(
        experiment__graph__tag=f"NN(6)",
        experiment__tag=offset_tag,
        experiment__settings__annealing_time=anneal_time,
        experiment__settings__num_spin_reversal_transforms=0,
    ).to_dataframe()

    ground_state_count = data.groupby("energy").count()["id"].sort_index().iloc[0]
    total_count = data.count()["id"]
    prob = ground_state_count / total_count
    print(anneal_time, prob)
    return prob


def plot_anneal_time():
    anneal_time = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700,
                   800, 900, 1000, 2000]
    prob = [getallannealtime(anneal_time=at) for at in anneal_time]

    plt.figure(figsize=p["figsize"])
    ax = plt.axes(p["aspect_ratio"])

    ax.errorbar(x=anneal_time, y=prob, ls="-", marker="o", color="k")

    ax.set_xscale("log")

    plt.draw()
    plt.show()


"""
################################################
#####  plot prob vs offset and graph size  #####
################################################
"""


def getall(offset=-0.05, graphsize=2):
    """
    z3 uses qubits with offset range from -0.05 to 0.05
    z4 uses qubits with offset range from -0.08 to 0.03
    """
    if True:
        prob = {
            0.05: [0.9730875, 0.49125, 0.7733, 0.861175, 0.4954375, 0.445675, 0.10225, 0.0489125, 0.2658125, 0.0861875,
                   0.0009875],
            0.04: [0.9704375, 0.4289625, 0.7738, 0.8369625, 0.4561875, 0.4599875, 0.0942, 0.052275, 0.2599625,
                   0.0871625, 0.0014125],
            0.03: [0.97095, 0.4319375, 0.7673875, 0.8057625, 0.4305875, 0.497775, 0.095, 0.054175, 0.2614125, 0.0884375,
                   0.00135],
            0.02: [0.9637375, 0.4123875, 0.7577875, 0.767425, 0.3957125, 0.4703875, 0.0889102564102564, 0.0537125,
                   0.2728625, 0.0788875, 0.0017625],
            0.01: [0.962475, 0.4271, 0.749475, 0.731925, 0.364775, 0.4652375, 0.0925375, 0.0458875, 0.244, 0.0653125,
                   0.00125],
            0.0: [0.95615, 0.41495, 0.7429875, 0.657625, 0.3045875, 0.436725, 0.084625, 0.0288625, 0.22375, 0.0555875,
                  0.0017],
            -0.01: [0.9559, 0.38915, 0.7507875, 0.59275, 0.304275, 0.4103375, 0.0728625, 0.0317, 0.1949, 0.0475875,
                    0.001675],
            -0.02: [0.9461375, 0.38595, 0.7510875, 0.4678875, 0.2430875, 0.3330625, 0.05745, 0.020325, 0.1754, 0.041575,
                    0.0015125],
            -0.03: [0.9414375, 0.4112125, 0.7368875, 0.4222875, 0.2222, 0.2960375, 0.0439875, 0.01975, 0.142625,
                    0.0323875, 0.001275],
            -0.04: [0.942, 0.4114625, 0.7426875, 0.3423, 0.1701125, 0.23095714285714286, 0.0314125, 0.025775, 0.103325,
                    0.024375, 0.0010875],
            -0.05: [0.943775, 0.3779625, 0.7455375, 0.2804, 0.1375625, 0.174, 0.019525, 0.0149125, 0.089575, 0.0175875,
                    0.000775]}

        if len(prob[offset]) + 1 >= graphsize:
            return prob[offset][graphsize - 2]
        else:
            pass
    data = data_Data.objects.filter(
        experiment__graph__tag=f"NN({graphsize})",
        experiment__tag=f"FixEmbedding_Single_Sided_Binary_{offset}_z4",
        experiment__settings__annealing_time=500,
        experiment__settings__num_spin_reversal_transforms=0,
    ).to_dataframe()
    print(data.groupby("energy").count()["id"].sort_index())
    ground_state_count = data.groupby("energy").count()["id"].sort_index().iloc[0]
    total_count = data.count()["id"]
    prob = ground_state_count / total_count
    print(offset, f"NN({graphsize})", prob, total_count)
    return prob


def degenfcn(n):
    if n % 3 == 2:
        return n // 3 + 2
    if n % 3 == 1:
        return 2 * (n // 3 + 1)
    if n % 3 == 0:
        return 1


def plot_all():
    fig = plt.figure(figsize=p["figsize"])
    ax = plt.axes(p["aspect_ratio"])
    """dwave result
    """
    offset = np.array(list(0.01 * (np.arange(11) - 5)))[::-1]
    graphsize = list(np.arange(2, 13))
    prob = {os: [getall(os, gs) for gs in graphsize] for os in offset}
    for os in offset:
        color = OFFSET_MAP[os]
        if os == 0.0:
            ax.errorbar(x=graphsize, y=prob[os], ls="-", marker="o", color=color, alpha=os_alpha[os], label="experiment")
        else:
            ax.errorbar(x=graphsize, y=prob[os], ls="-", marker="o", color=color, alpha=os_alpha[os])
    """random guessing
    """
    degeneracy = np.array([degenfcn(Xi) for Xi in graphsize])
    rguess = np.array([1 / 2 ** xi for xi in graphsize]) * degeneracy
    ax.errorbar(x=graphsize, y=rguess, ls="--", marker="o", color=OFFSET_MAP2[0.05], label="random")
    """labels
    """
    ax.set_xlabel("graph size", p["textargs"])
    ax.set_ylabel("probability", p["textargs"])

    cax = fig.add_axes([0.7, 0.77, 0.21, 0.02])
    cbar = mpl.colorbar.ColorbarBase(
        cax,
        cmap=OFFSET_CMAP,
        norm=mpl.colors.Normalize(vmin=-0.05, vmax=0.05),
        orientation="horizontal",
        label="offset",
    )
    cax.set_xticklabels(colorbar_label)
    ax.legend(bbox_to_anchor=(0.9, 0.98, 0.045, 0.01), fancybox=True, shadow=False, frameon=True, framealpha=0.3)

    fig.savefig("../new_figures/DWave_scaling.pdf", transparent=True, )

def plot_random_ratio():
    plt.figure(figsize=p["figsize"])
    ax = plt.axes(p["aspect_ratio"])
    graphsize = list(np.arange(2, 13))
    degeneracy = np.array([degenfcn(x) for x in graphsize])
    rprob = np.array([1 / 2 ** xi for xi in graphsize]) * degeneracy
    for offset in [0.0, 0.05]:
        if offset == 0.0:
            color = "k"
        else:
            color = yellow
        prob = np.array([getall(offset, gs) for gs in graphsize])
        ratio = prob / rprob
        ax.errorbar(x=graphsize, y=ratio, ls="-", marker="o", color=color, alpha=os_alpha[0.0], label=offset)
    ax.set_xlabel("graph size", p["textargs"])
    ax.set_ylabel("P(DWave) / P(random)")
    plt.savefig(f"../new_figures/random_ratio_compare.pdf", transparent=True)


"""
######################################
#####  data for tdse simulation  #####
######################################
"""


def gettdse(offset=0.0):
    if True:
        prob = {-0.07: 0.70107, -0.06: 0.72948, -0.05: 0.79639, -0.04: 0.82127, -0.03: 0.84074, -0.02: 0.84374,
                -0.01: 0.88497, 0.0: 0.90447, 0.01: 0.9189, 0.02: 0.92944, 0.03: 0.94336, 0.04: 0.94864, 0.05: 0.95251,
                0.06: 0.93402, 0.07: 0.92907}
        if offset in prob.keys():
            return prob[offset]
        else:
            pass
    data = data_Data.objects.filter(
        experiment__graph__tag=f"NN(2)",
        experiment__tag=f"FixEmbedding_Single_Sided_Binary_{offset}_z3",
        experiment__settings__annealing_time=1,
        experiment__settings__num_spin_reversal_transforms=0,
    ).to_dataframe()
    print(offset)
    energy_count = data.groupby("energy").count()["id"].sort_index().iloc[0]
    total_count = data.count()["id"]
    prob = energy_count / total_count
    print(total_count, prob)
    return prob


def gettdsetheory(offset=0.0, normalized_time = "[0, 1]", gamma_local = None, gamma=None):
    sim = Sim()
    if gamma_local is not None:
        sim.params["wave"]["gamma_local"] = gamma_local
    if gamma is not None:
        sim.params["wave"]["gamma"] = gamma
    query, sol, tdse = sim.get_data(offset, normalized_time)
    H = tdse._constructIsingH(np.array(tdse.ising["Jij"]), np.array(tdse.ising["hi"])).todense().tolist()
    eval, evec = eigh(H)
    project = sum([np.kron(evec[:, idx], np.conj(evec[:, idx])) for idx in [0, 1]])
    prob = np.asarray([np.absolute((np.dot(np.conj(project), sol.y[:, i]))) for i in range(sol.t.size)])
    return prob[-1]


def plot_tdse():
    fig = plt.figure(figsize=p["figsize"])
    ax = plt.axes(p["aspect_ratio"])
    """dwave result
    """
    # offset = list(0.01 * (np.arange(15) - 7))
    offset = list(0.01 * (np.arange(11) - 5))
    prob = [gettdse(os) for os in offset]
    ax.errorbar(x=offset, y=prob, ls="-", marker="o", color=OFFSET_MAP2[0.0], label="experiment")
    """simulation result
    """
    offset = list(0.01 * (np.arange(11) - 5))
    prob = [gettdsetheory(os, normalized_time=[0.0, 1.0]) for os in offset]
    ax.errorbar(x=offset, y=prob, ls="--", marker="o", color=OFFSET_MAP2[-0.05], label="simulation")
    """labels
    """
    ax.set_xlabel("offset", p["textargs"])
    ax.set_ylabel("probability", p["textargs"])

    ax.set_xticklabels(colorbar_label_all, p["textargs"])


    ax.legend(loc="best", fancybox=True, shadow=False, frameon=True, framealpha=0.3)
    fig.savefig("../new_figures/NN2_offset_scaling.pdf", transparent=True)

def plot_tdse_extended():
    fig = plt.figure(figsize=p["figsize"])
    ax = plt.axes(p["aspect_ratio"])
    """simulation result
    """
    offset = list(0.01 * (np.arange(11) - 5))

    prob = [gettdsetheory(os, normalized_time = [-0.1, 1.1], gamma_local=0) for os in offset]
    ax.errorbar(x=offset, y=prob, ls="-", marker="o", color=OFFSET_MAP[0.05], label="extended f.c. only")

    prob = [gettdsetheory(os, normalized_time = [-0.1, 1.1]) for os in offset]
    ax.errorbar(x=offset, y=prob, ls="-", marker="o", color=OFFSET_MAP[-0.05], label="extended")

    prob = [gettdsetheory(os, normalized_time = [0.0, 1.0]) for os in offset]
    ax.errorbar(x=offset, y=prob, ls="--", marker="o", color=OFFSET_MAP2[-0.05], label="default")

    prob = [gettdsetheory(os, normalized_time = [-0.1, 1.1], gamma=0) for os in offset]
    ax.errorbar(x=offset, y=prob, ls="-", marker="o", color=OFFSET_MAP2[0.05], label="extended local only")

    """labels
    """
    ax.set_xlabel("offset", p["textargs"])
    ax.set_ylabel("probability", p["textargs"])

    ax.set_xticklabels(colorbar_label_all, p["textargs"])

    ax.legend(loc="best", fancybox=True, shadow=False, frameon=True, framealpha=0.3)
    fig.savefig("../new_figures/NN2_offset_scaling_extended.pdf", transparent=True)

"""
##################################
#####  plot annealing curve  #####
##################################
"""


def getannealcurve(offset=0.05, normalized_time=[0.0, 1.0]):
    sim = Sim()
    query, sol, tdse = sim.get_data(offset, normalized_time)
    X = np.linspace(*query.offset["normalized_time"], 1000)
    yA = np.array([tdse.AS.A(Xi) for Xi in X])[:, 0]
    yB = np.array([tdse.AS.B(Xi) for Xi in X])[:, 0]
    return X, yA, yB

def plot_annealcurve():
    offset = [0.0, 0.05]
    fig = plt.figure(figsize=p["figsize"])
    ax = plt.axes(p["aspect_ratio"])
    for os in offset:
        if os == 0:
            colorA = OFFSET_MAP3[0.0]
            colorB = OFFSET_MAP3[0.0]
        else:
            colorA = purple #OFFSET_MAP3[os]
            colorB = purple #OFFSET_MAP3[-os]
        s, yA, yB = getannealcurve(os, normalized_time=[0, 1])
        if os == 0.05:
            ax.errorbar(x=s, y=yA, color=colorA, marker="None", ls="--", label="$A(s)$")
            ax.errorbar(x=s, y=yB, color=colorB, marker="None", ls="-", label=("$B(s)$"))
        else:
            ax.errorbar(x=s, y=yA, color=colorA, marker="None", ls="--")
            ax.errorbar(x=s, y=yB, color=colorB, marker="None", ls="-")
    """labels
       """
    ax.set_xlabel("annealing time $s$ [$\mu s$]", p["textargs"])
    ax.set_ylabel("coefficient strength [GHz]", p["textargs"])

    ax.legend(fancybox=True, shadow=False, frameon=True, framealpha=0.3, bbox_to_anchor=(0.8, 0.1, 0.045, 0.01))
    #cax = fig.add_axes([0.7, 0.77, 0.21, 0.02])
    #cbar = mpl.colorbar.ColorbarBase(
    #    cax,
    #    cmap=OFFSET_CMAP,
    #    norm=mpl.colors.Normalize(vmin=-0.05, vmax=0.05),
    #    orientation="horizontal",
    #    label="offset",
    #)
    #cax.set_xticklabels([-0.05, 0.0])

    fig.savefig("../new_figures/anneal_schedule.pdf", transparent=True)

def plot_annealcurve_extended():
    offset = [0.0, 0.05]
    fig = plt.figure(figsize=p["figsize"])
    ax = plt.axes(p["aspect_ratio"])
    for os in offset:
        if os == 0:
            colorA = OFFSET_MAP3[0.0]
            colorB = OFFSET_MAP3[0.0]
        else:
            colorA = purple #OFFSET_MAP3[os]
            colorB = purple #OFFSET_MAP3[-os]
        s, yA, yB = getannealcurve(os, normalized_time=[-0.1, 1.1])
        if os == 0.05:
            ax.errorbar(x=s, y=yA, color=colorA, marker="None", ls="--", label="$A(s)$")
            ax.errorbar(x=s, y=yB, color=colorB, marker="None", ls="-", label=("$B(s)$"))
        else:
            ax.errorbar(x=s, y=yA, color=colorA, marker="None", ls="--")
            ax.errorbar(x=s, y=yB, color=colorB, marker="None", ls="-")
    """labels
       """
    ax.set_xlabel("annealing time $s$ [$\mu s$]", p["textargs"])
    ax.set_ylabel("coefficient strength [GHz]", p["textargs"])

    ax.legend(fancybox=True, shadow=False, frameon=True, framealpha=0.3, bbox_to_anchor=(0.8, 0.1, 0.045, 0.01))
    #cax = fig.add_axes([0.7, 0.77, 0.21, 0.02])
    #cbar = mpl.colorbar.ColorbarBase(
    #    cax,
    #    cmap=OFFSET_CMAP,
    #    norm=mpl.colors.Normalize(vmin=-0.05, vmax=0.05),
    #    orientation="horizontal",
    #    label="offset",
    #)
    #cax.set_xticklabels([-0.05, 0.0])

    fig.savefig("../new_figures/anneal_schedule_extended.pdf", transparent=True)

"""
########################################
#####  time dependent probability  #####
########################################
"""


def gettimedependentprobability(offset=0.05):
    sim = Sim()
    query, sol, tdse = sim.get_data(offset)
    H = tdse._constructIsingH(np.array(tdse.ising["Jij"]), np.array(tdse.ising["hi"])).todense().tolist()
    eval, evec = eigh(H)
    project = sum([np.kron(evec[:, idx], np.conj(evec[:, idx])) for idx in [0, 1]])
    prob = np.asarray([np.absolute((np.dot(np.conj(project), sol.y[:, i]))) for i in range(sol.t.size)])
    X = query.time
    return X, prob


def plot_timedepprob():
    offset = list(0.01 * (np.arange(11) - 5))[::-1]
    fig = plt.figure(figsize=p["figsize"])
    ax = plt.axes(p["aspect_ratio"])
    for os in offset:
        s, prob = gettimedependentprobability(os)
        ax.errorbar(x=s, y=prob, color=OFFSET_MAP[os], marker="None", ls="-")
    """labels
    """
    ax.set_xlabel("annealing time $s$ [$\mu s$]", p["textargs"])
    ax.set_ylabel("probability", p["textargs"])

    #plt.legend(title="offset")
    cax = fig.add_axes([0.7, 0.3, 0.21, 0.02])
    cbar = mpl.colorbar.ColorbarBase(
        cax,
        cmap=OFFSET_CMAP,
        norm=mpl.colors.Normalize(vmin=-0.05, vmax=0.05),
        orientation="horizontal",
        label="offset",
    )
    cax.set_xticklabels(colorbar_label)
    fig.savefig("../new_figures/time_dependent_probability.pdf", transparent=True)


def gettimedependentsz(offset=0.05):
    sim = Sim()
    query, sol, tdse = sim.get_data(offset)
    # H = tdse._constructIsingH(np.array(tdse.ising["Jij"]), np.array(tdse.ising["hi"])).todense().tolist()
    # eval, evec = eigh(H)
    # project = sum([np.kron(evec[:, idx], np.conj(evec[:, idx])) for idx in [0, 1]])
    from functools import reduce
    totalsz = reduce(lambda a, b: a + b, tdse.FockZ)
    fockn = tdse.Focksize  # totalsz.shape[0]
    sz = np.asarray([(np.trace(totalsz @ sol.y[:, i].reshape(fockn, fockn))).real for i in range(sol.t.size)])
    X = query.time
    return X, sz


def plot_timedepsz():
    offset = list(0.01 * (np.arange(11) - 5))[::-1]
    plt.figure(figsize=p["figsize"])
    ax = plt.axes(p["aspect_ratio"])
    for os in offset:
        if os == 0.05:
            color = red
        elif os == -0.05:
            color = "k"
        elif os > 0:
            color = yellow
        elif os < 0:
            color = green
        else:
            color = blue
        s, sz = gettimedependentsz(os)
        ax.errorbar(x=s, y=sz, color=color, alpha=os_alpha[os], marker="None", ls="-", label=os)
    """labels
    """
    ax.set_xlabel("annealing time $s$ [$\mu s$]", p["textargs"])
    ax.set_ylabel("sz", p["textargs"])

    plt.legend(title="offset")
    plt.savefig("../new_figures/time_dependent_sz.pdf", transparent=True)


"""
###########################
#####  hybridization  #####
###########################
"""


# how large is off-diagonal term in Hamiltonian
# large is delocalized
def gethybridization(offset=0.05):
    sim = Sim()
    query, sol, tdse = sim.get_data(offset)
    ngrid = 100
    timegrid = np.linspace(0, 1, ngrid)
    offdiag = np.zeros(ngrid)
    for i in range(ngrid):
        H = tdse.annealingH(timegrid[i]).todense()
        offdiag[i] = np.linalg.norm(H - np.diag(np.diag(H)))
    return timegrid, offdiag


def plot_hybridization():
    offset = list(0.01 * (np.arange(11) - 5))[::-1]
    plt.figure("many body hybridization", figsize=p["figsize"])
    ax = plt.axes(p["aspect_ratio"])
    for os in offset:
        if os > 0:
            color = yellow
        elif os < 0:
            color = green
        else:
            color = blue
        timegrid, offdiag = gethybridization(os)
        ax.errorbar(x=timegrid, y=offdiag, marker="None", color=color, alpha=os_alpha[os], ls="-", label=os)
    ax.legend()
    # plt.yscale("log")
    plt.savefig("../new_figures/hybridization.pdf", transparent=True)


"""
###########################
#####  level spacing  #####
###########################
"""


def getlevelspacing(offset=0.05):
    sim = Sim()
    query, sol, tdse = sim.get_data(offset)
    ngrid = 1000
    fockn = 3 #tdse.Focksize
    timegrid = np.linspace(0, 1, ngrid)
    r = np.zeros(ngrid)
    for i in range(ngrid):
        H = tdse.annealingH(timegrid[i]).todense()
        val, evec = eigh(H)
        spacing = val[1:fockn] - val[0:fockn - 1]
        temp = np.asarray(
            [min((spacing[n], spacing[n + 1])) / max((spacing[n], spacing[n + 1])) for n in range(fockn - 2)])
        r[i] = np.sum(temp) / (fockn - 2)
    return timegrid, r


def plot_levelspacing():
    offset = list(0.01 * (np.arange(11) - 5))[::-1]
    all_levels = []
    for os in offset:
        timegrid, r = getlevelspacing(os)
        all_levels.append(r)
    print(np.shape(all_levels))
    sns.heatmap(all_levels, yticklabels=offset)
    # ax.legend()
    # plt.yscale("log")
    plt.savefig("../new_figures/levelspacing.pdf", transparent=True)
    #plt.show()


"""
############################################
#####  time-dependent energy spectrum  #####
############################################
"""


def getspectrum(offset=0.05):
    sim = Sim()
    query, sol, tdse = sim.get_data(offset)
    ngrid = 100
    es = 7
    timegrid = np.linspace(0, 1, ngrid)
    r = {idx: [] for idx in range(es)}
    gap = {0: [], 1: []}
    gap_density = []
    for i in range(ngrid):
        H = tdse.annealingH(timegrid[i]).todense()
        val, evec = eigh(H)
        if i == 0:
            norm = abs(val[1] - val[0])
            shift = val[0] / norm - 1
        for idx in range(es):
            r[idx].append(val[idx]/norm - shift)
        gap[1].append(val[2] - val[1])
        gap[0].append(val[2] - val[0])
        gap_density.append((val[2] + val[1] - 2 * val[0]) / 2)
    return timegrid, r, gap, gap_density


def plot_spectrum():
    offset = [0.0] #[-0.05, -0.04, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05]
    plt.figure("spectrum", figsize=p["figsize"])
    ax = plt.axes(p["aspect_ratio"])
    color = ["k"] #[red, "k", blue]
    for idx, os in enumerate(offset):
        timegrid, r, gap, gap_density = getspectrum(os)
        for es in r:
            ax.errorbar(x=timegrid, y=r[es], marker="None", ls="-", color=color[idx])
        #for es in [0]:  # gap:
        #    ax.errorbar(x=timegrid, y=gap[es], marker="None", ls="-")
        # ax.errorbar(x=timegrid, y=gap_density, marker="None", ls="-")
        ax.set_xlabel("annealing time $s$ [$\mu s$]", **p["textargs"])
        ax.set_ylabel("dimensionless energy", **p["textargs"])
        plt.savefig(f"../new_figures/spectrum_{os}.pdf", transparent=True)


"""
################################
#####  mutual information  #####
################################
"""


def ent_entropy(rho, nA, indicesA, reg):
    """
    calculate the entanglement entropy
    input:
       rho: density matrix
       n: number of qubits
       nA: number of qubits in partition A
       indicesA: einsum string for partial trace
       reg: infinitesimal regularization
    """
    tensorrho = rho.reshape(
        tuple([2 for i in range(2 * 5)])
    )
    rhoA = np.einsum(indicesA, tensorrho)
    matrhoA = rhoA.reshape(2 ** nA, 2 ** nA) + reg * np.identity(2 ** nA)
    s = -np.trace(matrhoA @ logm(matrhoA)) / np.log(2)
    return s


def vonNeumann_entropy(rho, reg):
    totaln = 5
    matrho = rho.reshape(2 ** totaln, 2 ** totaln) + reg * np.identity(2 ** totaln)
    s = -np.trace(matrho @ logm(matrho)) / np.log(2)
    return s


def q_mutual_info(rho, nA, nB, indicesA, indicesB, reg):
    """
    calculate the quantum mutual information
    """
    sa = ent_entropy(rho, nA, indicesA, reg)
    sb = ent_entropy(rho, nB, indicesB, reg)
    sab = vonNeumann_entropy(rho, reg)
    s = sa + sb - sab
    return s


def getmi(offset=0.05, type="quantum"):
    sim = Sim()
    query, sol, tdse = sim.get_data(offset)
    nA = 4
    nB = 1
    indicesA = "abcdeafghi->bcdefghi"
    indicesB = "abcdefbcde->af"
    # indicesA = "abdfhcbegi->adfhcegi"
    # indicesB = "acdefabdef->cb"
    # indicesA = "abcdefgchi->abdefghi"
    # indicesB = "abcdeabfde->cf"
    # indicesA = "abcdefghdi->abcefghi"
    # indicesB = "abcdeabcfe->df"
    # indicesA = "abcdefghie->abcdfghi"
    # indicesB = "abcdeabcdf->ef"

    # nA = 2
    # indicesA = "abdfhabdgi->fhgi"
    # nB = 3
    # indicesB = "abcghdefgh->abcdef"

    reg = 1E-6
    s = query.time
    mi = []
    for idx, si in enumerate(s):
        rho = sol.y[:, idx]
        if type == "classical":
            rho = np.diag(np.diag(rho))
        mi.append(q_mutual_info(rho, nA, nB, indicesA, indicesB, reg).real)
    return s, mi


def plot_mi():
    offset = list(0.01 * (np.arange(11) - 5))[::-1]
    plt.figure("muutal information", figsize=p["figsize"])
    ax = plt.axes(p["aspect_ratio"])
    for os in offset:
        if os > 0:
            color = yellow
        elif os < 0:
            color = green
        else:
            color = blue
        s, mi = getmi(os)
        ax.errorbar(x=s, y=mi, marker="None", color=color, alpha=os_alpha[os], ls="-", label=os)
    ax.legend()
    plt.savefig("../new_figures/mutual_information.pdf", transparent=True)


"""
#####################
#####  DWave MI #####
#####################
"""


def calculate_dwave_mi(offset=-0.05):
    from scipy.stats import entropy
    # get partition
    n = 5
    nA = 4
    nB = 1
    indicesA = 'ijklm->jklm'
    indicesB = 'ijklm->i'
    # read dwave
    scount = getallspinconfig(offset=offset)
    scount = {key[0]: scount[key] for key in scount}
    pr = np.zeros(2 ** n)
    for I in range(2 ** n):
        dwstate = tuple(np.array([int(i) for i in '{0:05b}'.format(I)]))
        if dwstate in scount:
            pr[I] = scount[(tuple(dwstate))]
        else:
            pr[I] = 0
    prtensor = pr.reshape([2 for i in range(n)])
    prAtensor = np.einsum(indicesA, prtensor)
    prBtensor = np.einsum(indicesB, prtensor)
    prA = prAtensor.reshape(2 ** nA)
    prB = prBtensor.reshape(2 ** nB)
    mi = entropy(prA, base=2) + entropy(prB, base=2) - entropy(pr, base=2)
    print(mi)
    return mi


def plot_dwave_mi():
    offset = list(0.01 * (np.arange(11) - 5))[::-1]
    plt.figure(figsize=p["figsize"])
    ax = plt.axes(p["aspect_ratio"])
    for os in offset:
        mi = calculate_dwave_mi(os)
        ax.errorbar(x=os, y=mi, marker="o", color="k")
        # plot simulation final mi
        _, mi = getmi(os, type="classical")
        ax.errorbar(x=os, y=mi[-1], marker="o", color=red)
    plt.savefig("../new_figures/dwave_mutual_information.pdf", transparent=True)


"""
##################
#####  main  #####
##################
"""
if __name__ == "__main__":
    """
    For DWave only
    """
    # plot_anneal_time() # this is not current, maybe drop this
    plot_all()

    #plot_random_ratio()
    # plot_dwave_mi()
    """
    For TDSE simulation
    """
    plot_tdse()
    plot_tdse_extended()
    plot_distribution()
    plot_annealcurve()
    plot_annealcurve_extended()
    plot_timedepprob()


    # plot_hybridization()
    # plot_mi()
    # plot_timedepsz()
    #plot_levelspacing()
    #plot_spectrum()
