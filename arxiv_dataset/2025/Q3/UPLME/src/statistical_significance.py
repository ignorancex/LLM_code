import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statannotations.Annotator import Annotator
from scipy.stats import wilcoxon
# plt.style.use(['science', 'tableau-colorblind10'])

def wilcoxon_test(x: np.ndarray, y: np.ndarray) -> float:
    d = x - y
    res = wilcoxon(d)
    print(f"WILCOXON TEST: statistic={res.statistic}, p-value={res.pvalue}")
    return res.pvalue

def _extract_test_stat(file: str, metric: str = "pcc") -> np.ndarray:
    df = pd.read_csv(file, index_col=0)
    df = df.drop(["mean", "std", "median"], axis=0)
    metric = df[f"test_{metric}"].to_numpy()
    return metric

def _plot_significance(file_x: str, file_y: str, save_suffix: str, metrics: list) -> None:
    expt_x = "UCVME"
    expt_y = "UPLME"

    # Prepare data for plotting
    data = []
    for metric in metrics:
        metric_x = _extract_test_stat(file_x, metric=metric)
        metric_y = _extract_test_stat(file_y, metric=metric)
        data.extend([
            {"Experiment": expt_x, "Metric": metric.upper(), "Value": val} for val in metric_x
        ])
        data.extend([
            {"Experiment": expt_y, "Metric": metric.upper(), "Value": val} for val in metric_y
        ])

    df = pd.DataFrame(data)

    # Create the plot
    plt.figure(figsize=(5, 4))
    ax = sns.barplot(data=df, x="Metric", y="Value", hue="Experiment", errorbar="sd")

    # Add significance annotations
    pairs = [
        (("PCC", expt_x), ("PCC", expt_y)),
        (("CCC", expt_x), ("CCC", expt_y)),
        (("SCC", expt_x), ("SCC", expt_y)),
        (("RMSE", expt_x), ("RMSE", expt_y)),
        (("CAL", expt_x), ("CAL", expt_y)),
        (("SHP", expt_x), ("SHP", expt_y)),
        (("NLPD", expt_x), ("NLPD", expt_y)),
    ]

    annotator = Annotator(ax, pairs, data=df, x="Metric", y="Value", hue="Experiment")
    annotator.configure(test="t-test_ind", text_format="star", loc="inside", verbose=2)
    annotator.apply_and_annotate()

    ax.set_ylabel("Metric Value")
    ax.set_xlabel("Metric")
    ax.set_xticklabels([r"PCC $\uparrow$", r"CCC $\uparrow$", r"SCC $\uparrow$", r"RMSE $\downarrow$", r"CAL $\downarrow$", r"SHP $\downarrow$", r"NLPD $\downarrow$"], rotation=45)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.show()
    plt.savefig(f"outputs/significance_plot_{save_suffix}.pdf", format="pdf", bbox_inches="tight")

if __name__ == "__main__":
    file_x = "outputs/2025-07-30/16-35-59_newsemp_cross-prob_lambdas-[1, None, None, None, None]_tune-False_4-passes-ucvme/results_val-True_test-True.csv"
    file_y = "outputs/2025-07-28/14-22-44_newsemp_cross-prob_lambdas-[1, 9.110462266012783, 5.5635098435909764]_tune-False_single-model_4-passes-best/results_val-True_test-True-3seeds.csv"
    metrics = ["pcc", "ccc", "scc", "rmse", "cal", "shp", "nlpd"]
    _plot_significance(file_x, file_y, save_suffix="newsemp", metrics=metrics)

    # empstories
    file_x = "outputs/2025-08-02/07-35-19_empstories_cross-prob_lambdas-[12.108875317622909, None, None, None, None]_tune-False_4-passes-ucvme/results_val-True_test-True.csv"
    file_y = "outputs/2025-07-31/00-28-34_empstories_cross-prob_lambdas-[1, 44.267427035816425, 21.34894435683186]_tune-False_single-model_4-passes-optuna-best-lambdas-updated-rescaling/results_val-True_test-True.csv"
    _plot_significance(file_x, file_y, save_suffix="empstories", metrics=metrics)