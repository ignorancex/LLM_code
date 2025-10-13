import seaborn as sns
from matplotlib import pyplot as plt

from judgetuning.utils import figure_path, load_judge_evaluations


def rename(s: str):
    s = s.split("/")[1].lower()
    for token in ["-instruct", "-gptq-int8", "-it", "-fp8", "meta-"]:
        s = s.replace(token, "")
    return s


df_configs_per_eval, mat_cost_per_eval, mat_human_agreement_per_eval = (
    load_judge_evaluations()
)
df_configs = df_configs_per_eval[400]

prompt_cols = [
    "provide_answer",
    "provide_explanation",
    "provide_confidence",
    "provide_example",
    "score_type",
    "json_output",
]
df_configs_copy = df_configs.copy()
df_configs_copy["model"] = df_configs_copy.loc[:, "model"].apply(rename)
df_corr = (
    df_configs_copy.loc[:, prompt_cols + ["model", "human_agreement"]]
    .pivot_table(index="model", columns=prompt_cols, values="human_agreement")
    .T.corr()
)
cg = sns.clustermap(
    df_corr,
    vmin=None,
    vmax=1,
    cmap="Reds",
    annot=True,
    figsize=(6.5, 6.5),
)
cg.ax_row_dendrogram.set_visible(True)
cg.ax_col_dendrogram.set_visible(False)
cg.cax.set_visible(False)
ax = cg.ax_heatmap
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_title("Correlation between prompts")
plt.tight_layout()
plt.savefig(figure_path(None, "clustermap.pdf"))
plt.show()
