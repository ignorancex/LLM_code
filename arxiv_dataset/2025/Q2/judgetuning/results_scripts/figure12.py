import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from judgetuning.utils import load_judge_evaluations, figure_path

def model_rename(s: str):
    if "/" in s:
        s = s.split("/")[1].lower()
        for token in ["-instruct", "-gptq-int8", "-it", "-fp8", "meta-"]:
            s = s.replace(token, "")
        return s
    else:
        return s


def model_family(s: str):
    # TODO we ran Llama 3 instead of 3.1 for 8B
    if "llama" in s:
        return "llama3"
    s = "".join(s.split("-")[:-1])
    return s.capitalize()


def model_size(s: str):
    return int(s.split("-")[-1].replace("b", ""))


df_configs_per_eval, mat_cost_per_eval, mat_human_agreement_per_eval = (
    load_judge_evaluations()
)

cols = [
    "n_sample_with_shift",
    "provide_answer",
    "provide_explanation",
    "provide_example",
    "json_output",
    "temperature",
    "score_type",
    "model",
]
fig, axes = plt.subplots(2, 4, figsize=(12, 6), sharex=True, sharey=True)
axes = np.ravel(axes)

for i, col in enumerate(cols):
    n_instr = 400
    dd = df_configs_per_eval[n_instr].copy()
    dd["cost"] = mat_cost_per_eval[n_instr].mean(axis=1)
    dd.model = dd.model.apply(model_rename)

    sns.scatterplot(
        dd,
        x="cost",
        y="human_agreement",
        hue=col,
        ax=axes[i],
        marker=".",
    )
    axes[i].set_ylabel("Human agr. (val set)")
    axes[i].set_title(col.replace("_", "-"))
    axes[i].legend(loc="lower right")
    axes[i].grid()
plt.tight_layout()
plt.savefig(figure_path(None, f"all-hyperparameters.pdf"), dpi=100)
plt.show()
