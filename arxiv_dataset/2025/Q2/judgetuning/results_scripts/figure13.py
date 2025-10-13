import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr

from judgetuning.utils import load_judge_evaluations, figure_path

df_configs_per_eval, mat_cost_per_eval, mat_human_agreement_per_eval = (
    load_judge_evaluations()
)

np.random.seed(0)
# split randomly maximum number of instruction into two buckets
# look at correlation of HA between both groups
fig, axes = plt.subplots(1, 3, sharey=True, sharex=True, figsize=(12, 3.5))

for i, n_instr in enumerate([400, 1200, 3548]):

    random_instr = np.random.permutation(n_instr)[:n_instr]
    instr1 = random_instr[: n_instr // 2]
    instr2 = random_instr[n_instr // 2 :]
    sns.scatterplot(
        pd.DataFrame(
            {
                "x": mat_human_agreement_per_eval[n_instr][:, instr1].mean(axis=1),
                "y": mat_human_agreement_per_eval[n_instr][:, instr2].mean(axis=1),
            }
        ),
        x="x",
        y="y",
        ax=axes[i],
    )
    sp = spearmanr(
        mat_human_agreement_per_eval[n_instr][:, instr1].mean(axis=1),
        mat_human_agreement_per_eval[n_instr][:, instr2].mean(axis=1),
    )[0]
    axes[i].set_xlim([0.2, 0.61])
    axes[i].set_ylim([0.2, 0.61])
    axes[i].grid()
    axes[i].set_xlabel(f"Human agr. on first {n_instr // 2} instr.")
    axes[i].set_ylabel(f"Human agr. on last {n_instr // 2} instr.")
    axes[i].set_title(f"{n_instr} instr - $\\rho$={sp:.2f}")
plt.tight_layout()

plt.savefig(figure_path(None, "rung-correlation.pdf"))
plt.show()
