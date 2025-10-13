import pandas as pd

from results_scripts.figure1 import load_rows_spearman
from results_scripts.figure2 import load_df_human_agreement_scaling

df_results_human = load_df_human_agreement_scaling()

# 6500 annotations / 26 models => 250 instructions
rows_spearman = load_rows_spearman(n_instrs=[250])

df_spearman = pd.DataFrame(rows_spearman)
df_spearman = df_spearman[(df_spearman.model != "length")]
df_spearman.model = df_spearman.model.apply(lambda s: s.split("/")[1].split("-")[1])

model_sizes = ["0.5B", "1.5B", "3B", "7B", "32B", "72B"]

rows = []
metric_col = "corr"
df_spearman = (
    df_spearman.loc[:, ["model", "#instructions", metric_col]]
    .groupby(["model", "#instructions"])
    .agg(["mean", "std"])[metric_col]
)
df_human_agreement_sub = df_results_human[
    df_results_human["#instructions"] == 6500
].copy()
df_human_agreement_sub.model = df_human_agreement_sub.model.apply(
    lambda s: s.split("/")[1].split("-")[1]
)

metric_col = "Human agreement"
df_human_agreement_sub = (
    df_human_agreement_sub.loc[:, ["model", metric_col]]
    .groupby(["model"])
    .agg(["mean", "std"])[metric_col]
)

# Qwen/Qwen2.5-0.5B-Instruct -> 0.5B

for model_size in model_sizes:
    corr_mean, corr_std = df_spearman.loc[model_size].values[0]
    human_mean, human_std = df_human_agreement_sub.loc[model_size].values
    format_str = lambda mean, std: f"{mean:.2f} $\\pm$ {std:.3f}"
    rows.append(
        {
            "$\#_\\text{params}$": model_size,
            "Sp. corr.": format_str(corr_mean, corr_std),
            "CV Sp (\\%)": f"{corr_std / corr_mean * 100:.2f}",
            "Hum. agr.": format_str(human_mean, human_std),
            "CV HA (\\%)": f"{human_std / human_mean * 100:.2f}",
            #        "Cost (\\$)": f"{human_cost_mean:.1f}",
        }
    )
print(pd.DataFrame(rows).to_latex(index=False))
