import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap

task_fullname_map = {
    "APR": "Automated Program Repair",
    "CG": "Code Generation",
    "CR": "Code Review",
    "CRF": "Code Refinement",
    "CRG": "Code Review Generation",
    "CSum": "Code Summarization",
    "CTrans": "Code Translation",
    "CMG": "Commit Message Generation",
    "CComp": "Code Completion",
    "HFP": "Header File Prediction",
    "JITCU": "Just-in-time Comment Update",
    "MNR": "Method Name Recommendation",
    "PBT": "Protocol Buffer Transformation",
    "UTG": "Unit Test Generation",
    "CT": "Cloze Test",
    "CCD": "Code Clone Detection",
    "CS": "Code Search",
    "DD": "Defect Detection",
    "MNCC": "Method Name Consistency Check"
}

ordered_methods = [
    "Base LoRA", "QLoRA", "AdaLoRA", "FF-LoRA", "FF-LoRA + AT",
    "Base Adapter", "L-Adapter", "T-Adapter", "L-Adapter + T-Adapter", "L-Adapter + NER-Adapter + AdapterFusion",
    "Prompt Tuning", "Prefix Tuning", "P-Tuning",
    "BitFit", "Telly-K", "MAM", "(IA)\u00B3", "Pass-Tuning"
]

peft_task_data = {
    "Base LoRA": {"APR": 3, "CG": 7, "CComp": 2, "CMG": 2, "CR": 1, "CRF": 1, "CRG": 1, "CSum": 2, "CTrans": 3, "HFP": 1, "JITCU": 1, "PBT": 1, "CCD": 2, "DD": 2},
    "QLoRA": {"CG": 1, "CSum": 2, "CTrans": 1, "CComp": 1, "UTG": 1},
    "AdaLoRA": {"APR": 1},
    "FF-LoRA": {"CG": 1, "CSum": 1, "CTrans": 1, "CCD": 1},
    "FF-LoRA + AT": {"CG": 1, "CSum": 1, "CTrans": 1, "CCD": 1},
    "Base Adapter": {"CG": 2, "CMG": 1, "CRF": 1, "CSum": 4, "CTrans": 3, "CCD": 3, "CS": 1, "DD": 3},
    "L-Adapter": {"CCD": 1, "CT": 2},
    "T-Adapter": {"CSum": 1, "CCD": 1},
    "L-Adapter + T-Adapter": {"CCD": 1},
    "L-Adapter + NER-Adapter + AdapterFusion": {"CRF": 1, "CSum": 1},
    "Prompt Tuning": {"CRF": 1, "CSum": 3, "CTrans": 3, "MNR": 1, "CCD": 1, "CS": 1, "DD": 3, "MNCC": 1},
    "Prefix Tuning": {"APR": 1, "CG": 1, "CMG": 1, "CRG": 1, "CSum": 3, "CTrans": 2, "JITCU": 1, "UTG": 1, "CCD": 1, "DD": 1},
    "P-Tuning": {"APR": 1, "CG": 1, "CRF": 1, "CSum": 1, "CTrans": 1, "CCD": 1, "DD": 1},
    "BitFit": {"CG": 1, "CRF": 1, "CSum": 1, "CTrans": 1, "CCD": 1, "DD": 1},
    "Telly-K": {"CComp": 1, "CG": 1, "CSum": 1, "CCD": 1, "CS": 1},
    "MAM": {"CSum": 1, "CTrans": 1, "CCD": 1, "DD": 1},
    "(IA)\u00B3": {"APR": 2},
    "Pass-Tuning": {"CG": 1, "CRF": 1, "CSum": 1, "CTrans": 1, "CCD": 1, "DD": 1}
}

all_tasks = list(task_fullname_map.keys())

heatmap_df = pd.DataFrame(index=all_tasks, columns=ordered_methods).fillna(0)
for method, tasks in peft_task_data.items():
    for task, count in tasks.items():
        heatmap_df.loc[task, method] = count

heatmap_df.index = [f"{task_fullname_map[abbr]} ({abbr})" for abbr in heatmap_df.index]

wrap_width = 30
heatmap_df.index = ['\n'.join(textwrap.wrap(task, width=wrap_width)) for task in heatmap_df.index]

plt.figure(figsize=(26, 16))
custom_cmap = sns.color_palette(["#F5F5F5", "#B0E0E6", "#87CEEB", "#4682B4", "#1E3A5F", "#0B1F33"], as_cmap=True)

ax = sns.heatmap(
    heatmap_df,
    annot=True,
    annot_kws={"size": 18},
    cmap=custom_cmap,
    linewidths=0.5,
    cbar=True,
    vmin=0.01,
    vmax=6
)

cbar = ax.collections[0].colorbar
cbar.set_ticks([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
cbar.set_ticklabels(["1", "2", "3", "4", "5", "6"])
cbar.ax.tick_params(labelsize=18)



plt.xticks(rotation=45, ha="right", fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel("PEFT Methods", fontsize=18, fontweight="bold", labelpad=20)
plt.ylabel("SE Tasks", fontsize=18, fontweight="bold", labelpad=20)

plt.tight_layout()
output_path = "../visualizations/heatmap_of_peft_methods_across_se_tasks.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()

output_path
