import matplotlib.pyplot as plt
import pickle

import numpy as np

from qlpdb.tdse.models import Tdse

from qlp.mds.mds_qlpdb import graph_summary
from qlp.mds import graph_tools as gt
from qlp.mds.qubo import get_mds_qubo
from qlp.tdse import convert_params, embed_qubo_example

from django.conf import settings
from ta_plots import get_tdse_data, plot_tdse, getallspinconfig


label_params = dict()
label_params["fontsize"] = 7
tick_params = dict()
tick_params["labelsize"] = 7
tick_params["width"] = 0.5
errorbar_params = dict()
errorbar_params["mew"] = 0.5
errorbar_params["markersize"] = 5
red = "#c82506"
green = "#70b741"
blue = "#51a7f9"


class Data:
    def __init__(self):
        self.params = self.parameters()

    def parameters(self):
        # offset params
        offset_params = dict()
        offset_params["annealing_time"] = 1
        offset_params["normalized_time"] = [0, 1]
        offset_params["offset"] = "binary"
        offset_params["offset_min"] = 0
        offset_params["offset_range"] = 0
        offset_params["fill_value"] = "extrapolate"
        offset_params["anneal_curve"] = "dwave"

        # wave params
        wave_params = dict()
        wave_params["type"] = "mixed"
        wave_params["temp"] = 0.001
        wave_params["temp_local"] = 0.1
        wave_params["gamma"] = 1 / 0.1
        wave_params["gamma_local"] = 1/8
        wave_params["initial_wavefunction"] = "transverse"

        # graph params
        nvertices = 2
        graph, tag = gt.generate_nn_graph(nvertices)
        directed = False
        penalty = 2
        embed = True  # nvertices = [2, 3] available

        if embed:
            qubo, embedding = embed_qubo_example(nvertices)
        else:
            qubo = get_mds_qubo(
                graph, directed=directed, penalty=penalty, triangularize=True, dtype="d"
            )
            qubo = qubo / 4
        graph_params = graph_summary(tag, graph, qubo)

        # solver params
        solver_params = dict()
        solver_params["method"] = "RK45"
        solver_params["rtol"] = 9e-8
        solver_params["atol"] = 9e-9

        params = {
            "offset": offset_params,
            "wave": wave_params,
            "graph": graph_params,
            "solver": solver_params,
        }
        return params

    def get_data(self):
        print(convert_params(self.params["offset"]))
        print(self.params["solver"])
        print(self.params["wave"])
        query = Tdse.objects.filter(
            graph__tag=self.params["graph"]["tag"],
            offset__contains=convert_params(self.params["offset"]),
            solver__contains=self.params["solver"],
            wave__contains=self.params["wave"],
        ).first()
        return query


def aggregate():
    adata = dict()
    data = Data()
    for offset in [
        0.05,
        0.04,
        0.03,
        0.02,
        0.01,
        0.0,
        -0.01,
        -0.02,
        -0.03,
        -0.04,
        -0.05,
    ]:
        # for offset in [-0.04, 0.04]:
        # for offset in [-0.04, 0.04]:
        data.params["offset"]["offset"] = "binary"
        data.params["offset"]["offset_min"] = offset
        data.params["offset"]["offset_range"] = abs(offset) * 2
        adata[offset] = data.get_data()
        with open(f"{settings.MEDIA_ROOT}/{adata[offset].solution}", "rb") as file:
            adata[offset].sol = pickle.load(file)
        with open(f"{settings.MEDIA_ROOT}/{adata[offset].instance}", "rb") as file:
            adata[offset].tdse = pickle.load(file)
    return adata


def aggregate_nodeco():
    adata = dict()
    data = Data()
    data.params["wave"]["gamma"] = 0.0
    data.params["offset"]["annealing_time"] = 0.001
    data.params["offset"]["offset"] = "binary"
    for offset in [
        0.05,
        0.04,
        0.03,
        0.02,
        0.01,
        0.0,
        -0.01,
        -0.02,
        -0.03,
        -0.04,
        -0.05,
    ]:
        # for offset in [0.04, -0.04]:
        data.params["offset"]["offset_min"] = offset
        data.params["offset"]["offset_range"] = abs(offset) * 2
        adata[offset] = data.get_data()
        with open(f"{settings.MEDIA_ROOT}/{adata[offset].solution}", "rb") as file:
            adata[offset].sol = pickle.load(file)
        with open(f"{settings.MEDIA_ROOT}/{adata[offset].instance}", "rb") as file:
            adata[offset].tdse = pickle.load(file)
    return adata


def aggregate_gamma():
    adata = dict()
    data = Data()
    data.params["offset"]["offset"] = "binary"
    data.params["wave"]["temp"] = 0.04
    for gamma in [
        1 / 10,
        1 / 20,
        1 / 30,
        1 / 40,
        1 / 50,
        1 / 60,
        1 / 70,
        1 / 80,
        1 / 90,
        1 / 100,
    ]:
        print(gamma)
        data.params["wave"]["gamma"] = gamma
        adata[gamma] = data.get_data()
        with open(f"{settings.MEDIA_ROOT}/{adata[gamma].solution}", "rb") as file:
            adata[gamma].sol = pickle.load(file)
        with open(f"{settings.MEDIA_ROOT}/{adata[gamma].instance}", "rb") as file:
            adata[gamma].tdse = pickle.load(file)
    return adata


def plot_mbl(adata):
    ngrid = 10
    timegrid = np.linspace(0, 1, ngrid)
    offdiag = np.zeros(ngrid)
    plt.figure("many body hybridization", figsize=(7, 4))
    ax = plt.axes([0.15, 0.15, 0.8, 0.8])
    for offset in adata:
        for i in range(ngrid):
            H = adata[offset].tdse.annealingH(timegrid[i]).todense()
            offdiag[i] = np.linalg.norm(H - np.diag(np.diag(H)))
        ax.errorbar(x=timegrid, y=offdiag, marker="None", ls="-", label=offset)
    ax.legend()
    plt.title(
        "many body hybridization (Frobenius norm of off diagonal matrix elements) "
    )
    plt.draw()
    plt.savefig("./hybridization.pdf", transparent=True)


def plot_centropy(adata):
    import scipy.stats as stats

    plt.figure("centropy", figsize=(7, 4))
    ax = plt.axes([0.15, 0.15, 0.8, 0.8])
    for offset in adata:
        centropy = np.asarray(
            [
                stats.entropy(
                    abs(np.diag(adata[offset].sol.y[:, i].reshape(32, 32))), base=2
                )
                for i in range(adata[offset].sol.t.size)
            ]
        ).real
        timegrid = np.linspace(0, 1, adata[offset].sol.t.size)
        ax.errorbar(x=timegrid, y=centropy, marker="None", ls="-", label=offset)
    ax.legend()
    plt.title("classical entropy")
    plt.draw()
    plt.show()


def plot_distribution(adata):
    prdict = getallspinconfig()

    plt.figure("distribution", figsize=(7, 11))
    rhodim = 2 ** 5
    height = 0.085
    top = 0.98
    ax0 = plt.axes([0.15, top - height, 0.8, height])
    ax0.bar(
        x=np.linspace(0, rhodim - 1, rhodim),
        height=np.diagonal(adata[-0.05].sol.y[:, -1].reshape((rhodim, rhodim))).real,
        width=1,
        color=red,
        label="simulation"
    )
    ax0.get_xaxis().set_ticks([])
    ax0.set_ylim([0, 1.1])
    ax0.text(
        top-0.03, top-0.13, "-10% offset", horizontalalignment="right", transform=ax0.transAxes
    )

    ax1 = plt.axes([0.15, top - height * 2, 0.8, height])
    ax1.bar(
        x=np.linspace(0, rhodim - 1, rhodim),
        height=np.diagonal(adata[-0.04].sol.y[:, -1].reshape((rhodim, rhodim))).real,
        width=1,
        color=red

    )
    ax1.set_ylim([0, 1.1])
    ax1.text(
        top-0.03, top-0.13, "-8% offset", horizontalalignment="right", transform=ax1.transAxes
    )
    ax1.get_xaxis().set_ticks([])

    ax2 = plt.axes([0.15, top - height * 3, 0.8, height])
    ax2.bar(
        x=np.linspace(0, rhodim - 1, rhodim),
        height=np.diagonal(adata[-0.03].sol.y[:, -1].reshape((rhodim, rhodim))).real,
        width=1,
        color=red
    )
    ax2.set_ylim([0, 1.1])
    ax2.text(
        top-0.03, top-0.13, "-6% offset", horizontalalignment="right", transform=ax2.transAxes
    )
    ax2.get_xaxis().set_ticks([])

    ax3 = plt.axes([0.15, top - height * 4, 0.8, height])
    ax3.bar(
        x=np.linspace(0, rhodim - 1, rhodim),
        height=np.diagonal(adata[-0.02].sol.y[:, -1].reshape((rhodim, rhodim))).real,
        width=1,
        color=red
    )
    ax3.set_ylim([0, 1.1])
    ax3.text(
        top-0.03, top-0.13, "-4% offset", horizontalalignment="right", transform=ax3.transAxes
    )
    ax3.get_xaxis().set_ticks([])

    ax4 = plt.axes([0.15, top - height * 5, 0.8, height])
    ax4.bar(
        x=np.linspace(0, rhodim - 1, rhodim),
        height=np.diagonal(adata[-0.01].sol.y[:, -1].reshape((rhodim, rhodim))).real,
        width=1,
        color=red
    )
    ax4.set_ylim([0, 1.1])
    ax4.text(
        top-0.03, top-0.13, "-2% offset", horizontalalignment="right", transform=ax4.transAxes
    )
    ax4.get_xaxis().set_ticks([])

    ax5 = plt.axes([0.15, top - height * 6, 0.8, height])
    ax5.bar(
        x=np.linspace(0, rhodim - 1, rhodim),
        height=np.diagonal(adata[0.0].sol.y[:, -1].reshape((rhodim, rhodim))).real,
        width=1,
        color=red
    )
    ax5.set_ylim([0, 1.1])
    ax5.text(
        top-0.03, top-0.13, "0% offset", horizontalalignment="right", transform=ax5.transAxes
    )
    ax5.get_xaxis().set_ticks([])

    ax6 = plt.axes([0.15, top - height * 7, 0.8, height])
    ax6.bar(
        x=np.linspace(0, rhodim - 1, rhodim),
        height=np.diagonal(adata[0.01].sol.y[:, -1].reshape((rhodim, rhodim))).real,
        width=1,
        color=red
    )
    ax6.set_ylim([0, 1.1])
    ax6.text(
        top-0.03, top-0.13, "2% offset", horizontalalignment="right", transform=ax6.transAxes
    )
    ax6.get_xaxis().set_ticks([])

    ax7 = plt.axes([0.15, top - height * 8, 0.8, height])
    ax7.bar(
        x=np.linspace(0, rhodim - 1, rhodim),
        height=np.diagonal(adata[0.02].sol.y[:, -1].reshape((rhodim, rhodim))).real,
        width=1,
        color=red
    )
    ax7.set_ylim([0, 1.1])
    ax7.text(
        top-0.03, top-0.13, "4% offset", horizontalalignment="right", transform=ax7.transAxes
    )
    ax7.get_xaxis().set_ticks([])

    ax8 = plt.axes([0.15, top - height * 9, 0.8, height])
    ax8.bar(
        x=np.linspace(0, rhodim - 1, rhodim),
        height=np.diagonal(adata[0.03].sol.y[:, -1].reshape((rhodim, rhodim))).real,
        width=1,
        color=red
    )
    ax8.set_ylim([0, 1.1])
    ax8.text(
        top-0.03, top-0.13, "6% offset", horizontalalignment="right", transform=ax8.transAxes
    )
    ax8.get_xaxis().set_ticks([])

    ax9 = plt.axes([0.15, top - height * 10, 0.8, height])
    ax9.bar(
        x=np.linspace(0, rhodim - 1, rhodim),
        height=np.diagonal(adata[0.04].sol.y[:, -1].reshape((rhodim, rhodim))).real,
        width=1,
        color=red
    )
    ax9.set_ylim([0, 1.1])
    ax9.text(
        top-0.03, top-0.13, "8% offset", horizontalalignment="right", transform=ax9.transAxes
    )
    ax9.get_xaxis().set_ticks([])

    ax10 = plt.axes([0.15, top - height * 11, 0.8, height])
    ax10.bar(
        x=np.linspace(0, rhodim - 1, rhodim),
        height=np.diagonal(adata[0.05].sol.y[:, -1].reshape((rhodim, rhodim))).real,
        width=1,
        color=red
    )
    ax10.set_ylim([0, 1.1])
    ax10.text(
        top-0.03, top-0.13, "10% offset", horizontalalignment="right", transform=ax10.transAxes
    )

    ax0.bar(x=range(len(prdict[1][-0.05])), height=prdict[1][-0.05], width=1, alpha=0.5, color=blue, label="D-Wave")
    ax1.bar(x=range(len(prdict[1][-0.04])), height=prdict[1][-0.04], width=1, alpha=0.5, color=blue)
    ax2.bar(x=range(len(prdict[1][-0.03])), height=prdict[1][-0.03], width=1, alpha=0.5, color=blue,)
    ax3.bar(x=range(len(prdict[1][-0.02])), height=prdict[1][-0.02], width=1, alpha=0.5, color=blue,)
    ax4.bar(x=range(len(prdict[1][-0.01])), height=prdict[1][-0.01], width=1, alpha=0.5, color=blue,)
    ax5.bar(x=range(len(prdict[1][0.0])), height=prdict[1][0.0], width=1, alpha=0.5, color=blue,)
    ax6.bar(x=range(len(prdict[1][0.01])), height=prdict[1][0.01], width=1, alpha=0.5, color=blue,)
    ax7.bar(x=range(len(prdict[1][0.02])), height=prdict[1][0.02], width=1, alpha=0.5, color=blue,)
    ax8.bar(x=range(len(prdict[1][0.03])), height=prdict[1][0.03], width=1, alpha=0.5, color=blue,)
    ax9.bar(x=range(len(prdict[1][0.04])), height=prdict[1][0.04], width=1, alpha=0.5, color=blue,)
    ax10.bar(x=range(len(prdict[1][0.05])), height=prdict[1][0.05], width=1, alpha=0.5, color=blue,)
    ax0.legend(loc=2)
    ax5.set_ylabel("probability")
    ax10.set_xlabel("eigenstates (00000 to 11111)")
    plt.draw()
    plt.savefig("./final_state_distribution.pdf", transparent=True)


def plot_aggregate(adata, tag):
    if False:
        plt.figure("full probability", figsize=(7, 4))
        ax = plt.axes([0.15, 0.15, 0.8, 0.8])
        for idx, key in enumerate(adata):
            if key > 0:
                color = red
            else:
                color = blue
            tdse = adata[key].tdse
            idx, en, evec = tdse.ground_state_degeneracy(tdse.IsingH, 2e-2, debug=False)
            ax.errorbar(
                x=adata[key].time,
                y=adata[key].prob,
                color=color,
                label=f"{int(key*2*100)}%",
            )
        ax.set_xlabel("normalized time")
        ax.set_ylabel("MDS probability")
        ax.legend()
        plt.draw()
        plt.savefig(f"full_prob_{tag}.pdf", transparent=True)

    dwave_data = get_tdse_data()
    plt.figure("prob", figsize=(7, 4))
    ax = plt.axes([0.15, 0.15, 0.8, 0.8])
    x = list(adata.keys())
    y = [adata[key].prob[-1] for key in x]
    for idx, xi in enumerate(x):
        if float(xi) < 0 and float(xi) > -0.01:
            color = red
        else:
            color = "k"
        ax.errorbar(x=2 * xi, y=y[idx], ls="none", marker="o", color=color)
    X = 2*np.array([-0.05, -0.04, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05])
    y = dwave_data["Binary"]
    #ax.errorbar(x=X, y=y, ls="none", marker='o', color=red, label="D-Wave")
    ax.set_xlabel("offset range (%)")
    ax.set_ylabel("MDS probability")
    if tag == "deco":
        ax.set_ylim([0.8985, 0.9017])
    else:
        ax.set_ylim([0.9938, 0.99485])
    plt.draw()
    plt.savefig(f"./sim_{tag}.pdf", transparent=True)

    if False:
        reg = 1e-9
        plt.figure(f"mutual information", figsize=(7, 4))
        ax = plt.axes([0.15, 0.15, 0.8, 0.8])
        mi = dict()
        for idx, key in enumerate(adata):
            if key > 0:
                color = red
            else:
                color = blue
            sol = adata[key].sol
            tdse = adata[key].tdse
            entropy_params = {"nA": 2, "indicesA": "abcdfabceg->dfeg", "reg": reg}
            entropyA = np.asarray(
                [tdse.ent_entropy(sol.y[:, i], **entropy_params) for i in range(sol.t.size)]
            ).real
            entropy_params = {"nA": 3, "indicesA": "aceghbdfgh->acebdf", "reg": reg}
            entropyB = np.asarray(
                [tdse.ent_entropy(sol.y[:, i], **entropy_params) for i in range(sol.t.size)]
            ).real
            entropyAB = np.asarray(
                [tdse.vonNeumann_entropy(sol.y[:, i], reg) for i in range(sol.t.size)]
            ).real
            mutual_information = entropyA + entropyB - entropyAB
            ax.errorbar(
                x=adata[key].time,
                y=mutual_information,
                color=color,
                label=f"{int(key*2*100)}%",
            )
            mi[key] = mutual_information
        ax.set_xlabel("normalized time")
        ax.set_ylabel("mutual information")
        ax.legend()
        plt.draw()
        plt.savefig(f"mutual_info_{tag}.pdf", transparent=True)

        plt.figure(f"final mutual information", figsize=(7, 4))
        ax = plt.axes([0.15, 0.15, 0.8, 0.8])
        x = list(mi.keys())
        y = [mi[key][-1] for key in x]
        ax.errorbar(x=x, y=y, ls="None", marker="o")
        ax.set_xlabel("offset")
        plt.show()


def plot_gamma(gdata):
    plt.figure("decoherence", figsize=(7, 4))
    ax = plt.axes([0.15, 0.15, 0.8, 0.8])
    x = list(gdata.keys())
    X = 1 / (np.array(x) * 1e9) * 1e9
    y = [gdata[key].prob[-1] for key in x]
    for idx, xi in enumerate(X):
        if idx == 7:
            color = blue
        else:
            color = "k"
        ax.errorbar(x=xi, y=y[idx], ls="none", color=color, marker="o")
    ax.set_xlabel("coherence time (ns)")
    ax.set_ylabel("MDS probability")
    plt.draw()
    plt.savefig("./coherence.pdf", transparent=True)
    plt.show()

    plt.figure("decoherence full", figsize=(7, 4))
    ax = plt.axes([0.15, 0.15, 0.8, 0.8])
    t = list(gdata.keys())
    T1 = 1 / (np.array(t) * 1e9) * 1e9
    for idx, key in enumerate(t):
        x = gdata[key].time
        y = gdata[key].prob
        ax.errorbar(x=x, y=y, ls="-", label=T1[idx])
    ax.set_xlabel("normalized time")
    ax.legend()
    plt.draw()
    plt.show()


def plot_schedule(adata):
    X, y1, y2 = adata[-0.05].tdse.AS.plot([0, 1])
    X, z1, z2 = adata[0.0].tdse.AS.plot([0, 1])

    plt.figure("anneal schedule", figsize=(7, 4))
    ax = plt.axes([0.15, 0.15, 0.8, 0.8])
    ax.errorbar(x=X, y=y1[:, 0], ls="-", color=blue, label="Positive offset A(s)")
    ax.errorbar(x=X, y=z1[:, 0], ls="-", color="k", label="Default A(s)")
    ax.errorbar(x=X, y=y1[:, -1], ls="-", color=red, label="Negative offset A(s)")
    ax.errorbar(x=X, y=y2[:, 0], ls="--", color=blue, label="Positive offset B(s)")
    ax.errorbar(x=X, y=z2[:, 0], ls="--", color="k", label="Default B(s)")
    ax.errorbar(x=X, y=y2[:, -1], ls="--", color=red, label="Negative offset B(s)")
    ax.set_xlabel("normalized time")  # , **label_params)
    ax.set_ylabel("GHz")  # , **label_params)
    ax.legend()
    plt.draw()
    plt.savefig("./anneal_schedule.pdf")
    plt.show()


def plot_spectrum(adata):
    from numpy.linalg import eigh

    # plot spectrum
    tdse = adata[0.0].tdse
    normalized_time = tdse.offset["normalized_time"]

    num_es = 5
    # unit conversion
    # sol.y = [GHz / h]
    energyscale = tdse.ising["energyscale"]
    make_dimensionless = 1 / energyscale * 1 / tdse.AS.B(normalized_time[1])[0]
    print("ENERGY SCALE:", energyscale)
    fig = plt.figure("spectrum", figsize=(7, 4))
    ax = plt.axes([0.15, 0.15, 0.8, 0.8])

    x = []
    y = {es: [] for es in range(num_es)}
    for s in np.linspace(normalized_time[0], normalized_time[1]):
        eigval, eigvec = eigh((tdse.annealingH(s)).toarray())
        seigval = (
            np.sort(eigval) * make_dimensionless
            + tdse.ising["c"]
            + tdse.ising["qubo_constant"]
        )
        x.append(s)
        for es in y.keys():
            y[es].append(seigval[es])
    for es in y.keys():
        ax.errorbar(x=x, y=y[es])
    gap = {es: np.array(y[es]) - np.array(y[0]) for es in y.keys()}
    ax.set_xlabel("normalized time")
    ax.set_ylabel("energy (dimensionless)")
    plt.draw()
    plt.savefig("./spectrum.pdf", transparent=True)
    plt.show()
    print("ground state energy:", y[0][-1])
    print("1st ex state energy:", y[1][-1])
    eigval, eigvec = eigh((tdse.annealingH(s)).toarray())
    print("ground state eigvec:", np.round(eigvec[0]))
    print("1st ex state eigvec:", np.round(eigvec[1]))
    kb = 8.617333262145e-5  # eV⋅K−1
    h = 4.135667696e-15  # eV⋅s
    h_kb = h / kb  # K*s
    print(
        "start gap energy (Kelvins):",
        energyscale
        * tdse.AS.B(normalized_time[1])[0]
        * (y[1][0] - y[0][0])
        * 1e9
        * h_kb,
    )
    mingap = min(np.array(y[2]) - np.array(y[0]))
    print("MINGAP:", mingap)
    print("min gap energy (Kelvins):", mingap / make_dimensionless * 1e9 * h_kb)

    # plot spectrum
    tdse = adata[-0.05].tdse
    normalized_time = tdse.offset["normalized_time"]

    num_es = 5
    # unit conversion
    # sol.y = [GHz / h]
    energyscale = tdse.ising["energyscale"]
    make_dimensionless = 1 / energyscale * 1 / tdse.AS.B(normalized_time[1])[0]

    fig = plt.figure("offset spectrum", figsize=(7, 4))
    ax = plt.axes([0.15, 0.15, 0.8, 0.8])

    x = []
    y = {es: [] for es in range(num_es)}
    for s in np.linspace(normalized_time[0], normalized_time[1]):
        eigval, eigvec = eigh((tdse.annealingH(s)).toarray())
        seigval = (
            np.sort(eigval) * make_dimensionless
            + tdse.ising["c"]
            + tdse.ising["qubo_constant"]
        )
        x.append(s)
        for es in y.keys():
            y[es].append(seigval[es])
    for es in y.keys():
        ax.errorbar(x=x, y=y[es])
    gap = {es: np.array(y[es]) - np.array(y[0]) for es in y.keys()}
    ax.set_xlabel("normalized time")
    ax.set_ylabel("energy (dimensionless)")
    plt.draw()
    plt.savefig("./spectrum_offset.pdf", transparent=True)
    plt.show()
    print("ground state energy:", y[0][-1])
    print("1st ex state energy:", y[1][-1])
    eigval, eigvec = eigh((tdse.annealingH(s)).toarray())
    print("ground state eigvec:", np.round(eigvec[0]))
    print("1st ex state eigvec:", np.round(eigvec[1]))
    kb = 8.617333262145e-5  # eV⋅K−1
    h = 4.135667696e-15  # eV⋅s
    h_kb = h / kb  # K*s
    print(
        "start gap energy (Kelvins):",
        energyscale
        * tdse.AS.B(normalized_time[1])[0]
        * (y[1][0] - y[0][0])
        * 1e9
        * h_kb,
    )
    mingap = min(np.array(y[2]) - np.array(y[0]))
    print(
        "min gap energy (Kelvins):",
        energyscale * tdse.AS.B(normalized_time[1])[0] * mingap * 1e9 * h_kb,
    )


if __name__ == "__main__":
    # vs offset
    adata = aggregate()
    # print(list(adata.keys()))
    plot_aggregate(adata, "deco")
    # plot_mbl(adata)
    # plot_centropy(adata)
    plot_distribution(adata)

    # plot AS
    # print(list(adata.keys()))
    # plot_schedule(adata)

    # plot spectrum
    # plot_spectrum(adata)

    # vs offset no decoherence
    # bdata = aggregate_nodeco()
    # plot_aggregate(bdata, "nodeco")

    # vs gamma
    #gdata = aggregate_gamma()
    #plot_gamma(gdata)
