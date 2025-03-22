import torch
from itertools import combinations_with_replacement
import numpy as np
from fairlearn.metrics import (
    demographic_parity_difference,
    equalized_odds_difference,
)


def fair_metrics(gt, y, group):
    metrics_dict = {
        "DPd": demographic_parity_difference(gt, y, sensitive_features=group),
        "EOd": equalized_odds_difference(gt, y, sensitive_features=group),
    }
    return metrics_dict

def get_link_labels(pos_edge_index, neg_edge_index):
    E = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(E, dtype=torch.float)
    link_labels[: pos_edge_index.size(1)] = 1.0
    return link_labels

def prediction_fairness(test_edge_idx, test_edge_labels, te_y, group):
    te_dyadic_src = group[test_edge_idx[0]]
    te_dyadic_dst = group[test_edge_idx[1]]

    # SUBGROUP DYADIC
    u = list(combinations_with_replacement(np.unique(group), r=2))

    te_sub_diatic = []
    for i, j in zip(te_dyadic_src, te_dyadic_dst):
        for k, v in enumerate(u):
            if (i, j) == v or (j, i) == v:
                te_sub_diatic.append(k)
                break
    te_sub_diatic = np.asarray(te_sub_diatic)
    # MIXED DYADIC 
    
    te_mixed_dyadic = te_dyadic_src != te_dyadic_dst
    # GROUP DYADIC
    
    te_gd_dict = fair_metrics(
        np.concatenate([test_edge_labels, test_edge_labels], axis=0),
        np.concatenate([te_y, te_y], axis=0),
        np.concatenate([te_dyadic_src, te_dyadic_dst], axis=0),
    )

    te_md_dict = fair_metrics(test_edge_labels, te_y, te_mixed_dyadic)

    te_sd_dict = fair_metrics(test_edge_labels, te_y, te_sub_diatic)

    fair_list = [
        te_md_dict["DPd"],
        te_md_dict["EOd"],
        te_gd_dict["DPd"],
        te_gd_dict["EOd"],
        te_sd_dict["DPd"],
        te_sd_dict["EOd"],
    ]

    return fair_list