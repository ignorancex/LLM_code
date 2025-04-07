import matplotlib.pyplot as plt

from qlpdb.tdse.models import Tdse

from qlp.mds.mds_qlpdb import graph_summary
from qlp.mds import graph_tools as gt
from qlp.mds.qubo import get_mds_qubo
from qlp.tdse import convert_params

class Data():
    def __init__(self):
        self.params = self.parameters()

    def parameters(self):
        # offset params
        offset_params = dict()
        offset_params["annealing_time"] = 0.0003
        offset_params["normalized_time"] = [-0.2, 1.2]
        offset_params["offset"] = "negbinary"
        offset_params["offset_min"] = 0
        offset_params["offset_range"] = 0
        offset_params["fill_value"] = "truncate"
        offset_params["anneal_curve"] = "linear"

        # wave params
        wave_params = dict()
        wave_params["type"] = "mixed"
        wave_params["temp"] = 15e-3
        wave_params["initial_wavefunction"] = "true"

        # graph params
        nvertices = 3
        graph, tag = gt.generate_nn_graph(nvertices)
        directed = False
        penalty = 2
        qubo = get_mds_qubo(
            graph, directed=directed, penalty=penalty, triangularize=True, dtype="d"
        )
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

    def get_data(self):
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
    for annealtime in [0.001, 0.0003]:
        data.params["offset"]["annealing_time"] = annealtime
        adata[annealtime] = data.get_data()
    return adata

def plot_aggregate(adata):
    plt.figure("probability")
    ax = plt.axes([0.15, 0.15, 0.8, 0.8])
    for key in adata:
        ax.errorbar(x=adata[key].time, y=adata[key].prob)
    plt.draw()

    plt.figure("entropy")
    ax = plt.axes([0.15, 0.15, 0.8, 0.8])
    for key in adata:
        ax.errorbar(x=adata[key].time, y=adata[key].entropy)
    plt.draw()

    plt.show()

if __name__ == "__main__":
    adata = aggregate()
    plot_aggregate(adata)
