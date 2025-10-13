import scanpy as sc
import anndata as ad
import pandas as pd
import numpy as np
import torch
import optuna

from gears import PertData, GEARS
from analysis.benchmarks.evaluation import Evaluation


class GEARSHParamsRange:
    """
    Hyperparameter ranges for GEARS.
    """

    @staticmethod
    def get_distributions():
        return {
            "hidden_size": optuna.distributions.IntDistribution(32, 512, step=32),
            "num_go_gnn_layers": optuna.distributions.IntDistribution(1, 3),
            "num_gene_gnn_layers": optuna.distributions.IntDistribution(1, 3),
            "decoder_hidden_size": optuna.distributions.IntDistribution(
                16, 48, step=16
            ),
            "num_similar_genes_go_graph": optuna.distributions.IntDistribution(
                10, 30, step=5
            ),
            "num_similar_genes_co_express_graph": optuna.distributions.IntDistribution(
                10, 30, step=5
            ),
            "coexpress_threshold": optuna.distributions.FloatDistribution(
                0.2, 0.5, step=0.01
            ),
            "lr": optuna.distributions.FloatDistribution(5e-6, 1e-3, log=True),
            "wd": optuna.distributions.FloatDistribution(1e-8, 1e-3, log=True),
        }


def run_gears(
    pert_data_path: str = "/weka/prime-shared/prime-data/gears/",
    dataset_name: str = "norman19",
    split_dict_path: str = "/weka/prime-shared/prime-data/gears/norman19_gears_split.pkl",
    eval_split: str = "val",
    batch_size: str = 32,
    epochs: int = 10,
    lr: float = 5e-4,
    wd: float = 1e-4,
    hidden_size: str = 128,
    num_go_gnn_layers: int = 1,
    num_gene_gnn_layers: int = 1,
    decoder_hidden_size: int = 16,
    num_similar_genes_go_graph: int = 20,
    num_similar_genes_co_express_graph: int = 20,
    coexpress_threshold: float = 0.4,
    device="cuda:0",
    seed=0,
):
    """
    Helper function to train and evaluate a GEARS model
    """
    pert_data = PertData(pert_data_path)  # specific saved folder
    pert_data.load(
        data_path=pert_data_path + dataset_name
    )  # load the processed data, the path is saved folder + dataset_name
    pert_data.prepare_split(split="custom", split_dict_path=split_dict_path)
    pert_data.get_dataloader(batch_size=batch_size, test_batch_size=batch_size)

    gears_model = GEARS(
        pert_data,
        device=device,
        weight_bias_track=False,
        proj_name="pertnet",
        exp_name="pertnet",
    )
    gears_model.model_initialize(
        hidden_size=hidden_size,
        num_go_gnn_layers=num_go_gnn_layers,
        num_gene_gnn_layers=num_gene_gnn_layers,
        decoder_hidden_size=decoder_hidden_size,
        num_similar_genes_go_graph=num_similar_genes_go_graph,
        num_similar_genes_co_express_graph=num_similar_genes_co_express_graph,
        coexpress_threshold=coexpress_threshold,
        seed=seed,
    )
    gears_model.train(epochs=epochs, lr=lr, weight_decay=wd)
    torch.cuda.empty_cache()

    val_perts = []
    for p in pert_data.set2conditions[eval_split]:
        newp_list = []
        for gene in p.split("+"):
            if gene in gears_model.pert_list:
                newp_list.append(gene)
        if len(newp_list) > 0:
            val_perts.append(newp_list)

    val_avg_pred = gears_model.predict(val_perts)
    pred_df = pd.DataFrame(val_avg_pred).T
    pred_df.columns = gears_model.adata.var_names.values
    torch.cuda.empty_cache()

    ctrl_adata = gears_model.adata[gears_model.adata.obs.condition == "ctrl"]
    val_conditions = ["+".join(p) for p in val_perts] + ["ctrl"]
    ref_adata = gears_model.adata[gears_model.adata.obs.condition.isin(val_conditions)]

    pred_adata = sc.AnnData(pred_df)
    pred_adata.obs["condition"] = [x.replace("_", "+") for x in pred_adata.obs_names]
    pred_adata.obs["condition"] = pred_adata.obs["condition"].astype("category")
    pred_adata = ad.concat([pred_adata, ctrl_adata])

    ev = Evaluation(
        model_adatas={
            "GEARS": pred_adata,
        },
        ref_adata=ref_adata,
        pert_col="condition",
        ctrl="ctrl",
    )

    aggr_metrics = [
        ("average", "rmse"),
        ("logfc", "cosine"),
    ]
    summary_metrics_dict = {}
    for aggr, metric in aggr_metrics:
        ev.evaluate(aggr_method=aggr, metric=metric)
        ev.evaluate_pairwise(aggr_method=aggr, metric=metric)
        ev.evaluate_rank(aggr_method=aggr, metric=metric)

        metric_df = ev.get_eval(aggr_method=aggr, metric=metric)
        rank_df = ev.get_rank_eval(aggr_method=aggr, metric=metric)
        summary_metrics_dict[f"{metric}_{aggr}"] = np.mean(metric_df["GEARS"])
        summary_metrics_dict[f"{metric}_rank_{aggr}"] = np.mean(rank_df["GEARS"])

    return pd.Series(summary_metrics_dict)
