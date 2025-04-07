import numpy as np
import hashlib

from minorminer import find_embedding
from dwave.system.composites import FixedEmbeddingComposite

import matplotlib.pyplot as plt
import yaml


class AnnealOffset:
    """https://docs.dwavesys.com/docs/latest/c_qpu_0.html#anneal-offsets
    """

    def __init__(self, tag, graph_params={}):
        self.tag = tag
        self.graph_params = graph_params

    def fcn(self, h, offset_min, offset_range):
        h = np.array(h)
        abshrange = max(abs(h)) - min(abs(h))
        fullrange = max(h) - min(h)

        if self.tag == "advproblem":
            offset_tag = f"FixEmbedding_AdvanceProblem_{offset_min}_{offset_range}"
            adv = offset_min + offset_range
            offset_fcn = [adv for q in range(self.graph_params["total_vertices"])]
            nconstraint = self.graph_params["total_qubits"] - self.graph_params["total_vertices"]
            offset_constraint = [offset_min for q in range(nconstraint)]
            offset_fcn.extend(offset_constraint)
            return offset_fcn, offset_tag
        if self.tag == "advconstraint":
            offset_tag = f"FixEmbedding_AdvanceConstraint_{offset_min}_{offset_range}"
            offset_fcn = [offset_min for q in range(self.graph_params["total_vertices"])]
            adv = offset_min + offset_range
            nconstraint = self.graph_params["total_qubits"] - self.graph_params["total_vertices"]
            offset_constraint = [adv for q in range(nconstraint)]
            offset_fcn.extend(offset_constraint)
            return offset_fcn, offset_tag
        if self.tag == "constant":
            return (
                np.zeros(len(h)),
                f"FixEmbedding_Constant_{offset_min}_{offset_range}_v3_1",
            )
        if self.tag == "single_sided_binary":
            offset_tag = f"FixEmbedding_Single_Sided_Binary_{offset_min}_z4"
            offset_fcn = []
            hmid = abshrange * 0.5 + min(abs(h))
            for hi in h:
                if abs(hi) <= hmid:
                    if offset_min < 0:
                        offset_fcn.append(offset_min)
                    else:
                        offset_fcn.append(0)
                else:
                    if offset_min < 0:
                        offset_fcn.append(0)
                    else:
                        offset_fcn.append(-1*offset_min)
            return offset_fcn, offset_tag
        if self.tag == "binary":
            offset_tag = f"FixEmbedding_Binary_{offset_min}_{offset_range}_z0"
            offset_fcn = []
            hmid = abshrange * 0.5 + min(abs(h))
            for hi in h:
                if abs(hi) <= hmid:
                    offset_fcn.append(offset_min)
                else:
                    if offset_min < 0:
                        offset_fcn.append(offset_min + offset_range)
                    else:
                        offset_fcn.append(offset_min - offset_range)
            return offset_fcn, offset_tag
        if self.tag == "test0":
            offset_tag = "test0"
            offset_fcn = [offset_min, 0]
            return offset_fcn, offset_tag
        if self.tag == "test1":
            offset_tag = "test1"
            offset_fcn = [0, offset_min]
            return offset_fcn, offset_tag
        if self.tag == "negbinary":
            offset_tag = f"FixEmbedding_NegBinary_{offset_min}_{offset_range}_v3_1"
            offset_fcn = []
            hmid = abshrange * 0.5 + min(abs(h))
            for hi in h:
                if abs(hi) >= hmid:
                    offset_fcn.append(offset_min)
                else:
                    if offset_min < 0:
                        offset_fcn.append(offset_min + offset_range)
                    else:
                        offset_fcn.append(offset_min - offset_range)
            return offset_fcn, offset_tag
        if self.tag == "shiftlinear":
            offset_tag = f"FixEmbedding_ShiftLinear_{offset_min}_{offset_range}"
            absshifth = abs(h) - min(abs(h))
            shiftnormh = absshifth / abshrange
            offset_fcn = shiftnormh * offset_range + offset_min
            return offset_fcn, offset_tag
        if self.tag == "negshiftlinear":
            offset_tag = f"FixEmbedding_NegShiftLinear_{offset_min}_{offset_range}"
            invhnorm = -1 * (abs(h) - max(abs(h))) / abshrange
            offset_fcn = invhnorm * offset_range + offset_min
            return offset_fcn, offset_tag
        if self.tag == "signedshiftlinear":
            offset_tag = f"FixEmbedding_SignedShiftLinear_{offset_min}_{offset_range}"
            shifth = h - min(h)
            normh = shifth / fullrange
            offset_fcn = normh * offset_range + offset_min
            return offset_fcn, offset_tag
        if self.tag == "signednegshiftlinear":
            offset_tag = (
                f"FixEmbedding_SignedNegShiftLinear_{offset_min}_{offset_range}"
            )
            shifth = -1 * (h - max(h)) / fullrange
            offset_fcn = shifth * offset_range + offset_min
            return offset_fcn, offset_tag
        if self.tag == "linear":
            offset_tag = f"Linear_{offset_min}_{offset_range}"
            hnorm = abs(h) / max(abs(h))
            offset_fcn = hnorm * offset_range * 0.9 + offset_min * 0.9
            return offset_fcn, offset_tag
        if self.tag == "neglinear":
            hnorm = abs(h) / max(abs(h))
            return (
                -1.0 * hnorm * offset_range * 0.9 + offset_range + offset_min * 0.9,
                f"Neglinear_{offset_min}_{offset_range}",
            )
        if self.tag == "signedlinear":
            hnorm = 0.5 * (1.0 + h / max(abs(h)))
            return (
                hnorm * offset_range * 0.9 + offset_min * 0.9,
                f"Signedlinear_{offset_min}_{offset_range}",
            )
        if self.tag == "negsignedlinear":
            hnorm = 0.5 * (1.0 - h / max(abs(h)))
            return (
                hnorm * offset_range * 0.9 + offset_min * 0.9,
                f"Negsignedlinear_{offset_min}_{offset_range}",
            )
        else:
            print(
                "Anneal offset not defined.\nDefine in AnnealOffset class inside qlp.mds.mds_qlpdb"
            )


def retry_embedding(
        sampler,
        qubo_dict,
        qpu_graph,
        graph_tag,
        target_min=-0.1,
        target_range=0.12,
        n_tries=100,
):
    def get_embed_min_max_offset(sampler, embedding):
        embed = FixedEmbeddingComposite(sampler, embedding)
        embedding_idx = [idx for embed_list in embedding.values() for idx in embed_list]
        anneal_offset_ranges = np.array(
            embed.properties["child_properties"]["anneal_offset_ranges"]
        )
        min_offset = max(
            [offsets[0] for offsets in anneal_offset_ranges[embedding_idx]]
        )
        max_offset = min(
            [offsets[1] for offsets in anneal_offset_ranges[embedding_idx]]
        )
        return embed, min_offset, max_offset

    try:
        with open(
                f"../qlp/mds/embeddings/{graph_tag}_{target_min}_{target_range}_v6.yaml", "r"
        ) as file:
            embedding = yaml.safe_load(file)
        embed, min_offset, max_offset = get_embed_min_max_offset(sampler, embedding)
        embedding_set = {k: set(embedding[k]) for k in embedding}
        return embed, embedding, min_offset, max_offset
    except Exception as e:
        print(e)
        pass

    for i in range(n_tries):
        try:
            embedding = find_embedding(qubo_dict, qpu_graph)
            embed, min_offset, max_offset = get_embed_min_max_offset(sampler, embedding)
            if (target_range > max_offset - target_min) or (min_offset > target_min):
                raise ValueError(
                    f"\n{target_range} > {max_offset - target_min}: Not enough offset range for inhomogeneous driving."
                    f"\n{min_offset} > {target_min}: min_offset needs to be lower."
                    "Try another embedding."
                )
            else:
                with open(
                        f"../qlp/mds/embeddings/{graph_tag}_{target_min}_{target_range}_v6.yaml",
                        "w",
                ) as file:
                    safe_embed = {int(k): list(embedding[k]) for k in embedding}
                    yaml.safe_dump(safe_embed, file)
                return embed, embedding, min_offset, max_offset
        except Exception as e:
            # print(e)
            continue


def plot_anneal_offset(sampler):
    offsets = np.array(sampler.properties["anneal_offset_ranges"])
    offset_min = offsets[:, 0]
    offset_max = offsets[:, 1]
    fig, axs = plt.subplots(2, 2, figsize=(15, 15), sharey=True, tight_layout=True)
    axs[0, 0].hist(offset_min, bins=30)
    axs[0, 0].set_title("offset min")
    axs[0, 1].hist(offset_max, bins=30)
    axs[0, 1].set_title("offset max")
    axs[1, 0].hist(offset_max - offset_min, bins=30)
    axs[1, 0].set_title("offset range")
    axs[1, 1].hist(0.5 * (offset_max + offset_min), bins=30)
    axs[1, 1].set_title("offset mean")
    plt.draw()
    plt.show()


def find_offset(h, fcn, embedding, offset_min, offset_range):
    anneal_offset = np.zeros(2048)  # expects full yield 2000Q
    hlist = []
    hkey = []
    for key in h:
        hlist.append(h[key])
        hkey.append(key)
    offset_value, tag = fcn(hlist, offset_min, offset_range)
    offset_value = {hkey[idx]: offset_value[idx] for idx in range(len(hkey))}
    offset_dict = dict()
    for logical_qubit, qubit in embedding.items():
        for idx in qubit:
            # sets same offset for all qubits in chain
            anneal_offset[idx] = offset_value[idx]
            offset_dict[idx] = offset_value[idx]
    offset_list = []
    for idx in range(len(embedding)):
        for qi in embedding[idx]:
            offset_list.append(offset_dict[qi])
    return list(anneal_offset), tag, offset_list


def insert_result(graph_params, experiment_params, data_params):
    from qlpdb.graph.models import Graph as graph_Graph
    from qlpdb.experiment.models import Experiment as experiment_Experiment
    from qlpdb.data.models import Data as data_Data

    # select or insert row in graph
    graph, created = graph_Graph.objects.get_or_create(
        tag=graph_params["tag"],  # Tag for graph type (e.g. Hamming(n,m) or K(n,m))
        total_vertices=graph_params[
            "total_vertices"
        ],  # Total number of vertices in graph
        total_edges=graph_params["total_edges"],  # Total number of edges in graph
        max_edges=graph_params["max_edges"],  # Maximum number of edges per vertex
        adjacency=graph_params[
            "adjacency"
        ],  # Sorted adjacency matrix of dimension [N, 2]
        adjacency_hash=graph_params[
            "adjacency_hash"
        ],  # md5 hash of adjacency list used for unique constraint
    )

    # select or insert row in experiment
    experiment, created = experiment_Experiment.objects.get_or_create(
        graph=graph,  # Foreign Key to `graph`
        machine=experiment_params["machine"],  # Hardware name (e.g. DW_2000Q_5)
        settings=experiment_params["settings"],  # Store DWave machine parameters
        settings_hash=experiment_params[
            "settings_hash"
        ],  # md5 hash of key sorted settings
        p=experiment_params["p"],  # Coefficient of penalty term, 0 to 9999.99
        chain_strength=experiment_params["chain_strength"],
        tag=experiment_params["tag"],
    )

    # select or insert row in data
    for idx in range(len(data_params["spin_config"])):
        measurement = data_Data.objects.filter(experiment=experiment).order_by(
            "-measurement"
        )
        if measurement.exists():
            measurement = measurement.first().measurement + 1
        else:
            measurement = 0
        data, created = data_Data.objects.get_or_create(
            experiment=experiment,  # Foreign Key to `experiment`
            measurement=measurement,  # Increasing integer field labeling measurement number
            spin_config=list(
                data_params["spin_config"][idx]
            ),  # Spin configuration of solution, limited to 0, 1
            chain_break_fraction=9999, #data_params["chain_break_fraction"][idx],
            energy=data_params["energy"][
                idx
            ],  # Energy corresponding to spin_config and QUBO
            constraint_satisfaction=data_params["constraint_satisfaction"][idx],
        )
    return data


def graph_summary(tag, graph, qubo):
    """
    Get summary statistics of input graph
    :param graph:
    :return:
    """
    vertices = np.unique(np.array([i for i in graph]).flatten())
    neighbors = {v: 0 for v in vertices}
    for i in graph:
        neighbors[i[0]] += 1
        neighbors[i[1]] += 1
    params = dict()
    params["tag"] = tag
    params["total_vertices"] = len(vertices)
    params["total_edges"] = len(graph)
    try:
        keylist = np.unique(np.array([key for key in qubo]).flatten())
        params["total_qubits"] = len(keylist)
    except:
        params["total_qubits"] = len(qubo.todense().tolist())
    params["max_edges"] = max(neighbors.values())
    params["adjacency"] = [list(i) for i in list(graph)]
    params["adjacency_hash"] = hashlib.md5(
        str(np.sort(list(graph))).replace(" ", "").encode("utf-8")
    ).hexdigest()
    return params


def experiment_summary(machine, settings, penalty, chain_strength, tag):
    params = dict()
    params["machine"] = machine
    params["settings"] = {
        key: settings[key] for key in settings if key not in ["anneal_offsets"]
    }
    params["p"] = penalty
    params["chain_strength"] = chain_strength
    params["tag"] = tag
    params["settings_hash"] = hashlib.md5(
        str([[key, params["settings"][key]] for key in sorted(params["settings"])])
            .replace(" ", "")
            .encode("utf-8")
    ).hexdigest()
    return params


def data_summary(raw, graph_params, experiment_params):
    params = dict()
    params["spin_config"] = raw.iloc[:, : graph_params["total_qubits"]].values
    params["energy"] = (
            raw["energy"].values + experiment_params["p"] * graph_params["total_vertices"]
    )
    #params["chain_break_fraction"] = raw["chain_break_fraction"].values
    params["constraint_satisfaction"] = np.equal(
        params["energy"],
        np.sum(params["spin_config"][:, : graph_params["total_vertices"]], axis=1),
    )
    return params


def QUBO_to_Ising(Q):
    q = np.diagonal(Q)
    QD = np.copy(Q)
    for i in range(len(QD)):
        QD[i, i] = 0.0
    QQ = np.copy(QD + np.transpose(QD))
    J = np.triu(QQ) / 4.0
    uno = np.ones(len(QQ))
    h = q / 2 + np.dot(QQ, uno) / 4
    g = np.dot(uno, np.dot(QD, uno)) / 4.0 + np.dot(q, uno) / 2.0
    # 0-1 basis transform
    h=(-1)*h
    return (J, h, g)

def Ising_to_QUBO(J, h):
    Q = None
    return Q
