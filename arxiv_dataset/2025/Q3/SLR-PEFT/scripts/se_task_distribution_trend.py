import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patheffects as path_effects
from matplotlib.patches import Patch

file_path = "../data/peft_se_benchmark_metadata.csv"
df = pd.read_csv(file_path, encoding="iso-8859-1")

task_trends = df.groupby(["Year", "SE Task"]).size().unstack(fill_value=0)

generative_tasks = {
    "Automated Program Repair",
    "Code Generation",
    "Code Refinement",
    "Code Review Generation",
    "Code Summarization",
    "Code Translation",
    "Commit Message Generation",
    "Code Completion",
    "Just-in-time Comment Update",
    "Method Name Recommendation",
    "Unit Test Generation",
}

non_generative_tasks = {
    "Cloze Test",
    "Code Clone Detection",
    "Code Review",
    "Code Search",
    "Defect Detection",
    "Method Name Consistency Check",
}

colors = plt.cm.get_cmap("tab20", len(task_trends.columns)).colors

task_color_mapping = {}
for i, task in enumerate(task_trends.columns):
    if task == "Cloze Test":
        task_color_mapping[task] = "#FF5733"
    elif task == "Unit Test Generation":
        task_color_mapping[task] = "#33FF57"
    else:
        task_color_mapping[task] = colors[i]

plot_colors = [task_color_mapping[task] for task in task_trends.columns]

fig, ax = plt.subplots(figsize=(12, 8))
task_trends.plot(kind="bar", stacked=True, color=plot_colors, ax=ax, width=0.8, edgecolor="black")

if 2025 in task_trends.index:
    total_height_2025 = task_trends.loc[2025].sum()
    max_y_value = int(task_trends.sum(axis=1).max())
    placeholder_height = max_y_value - total_height_2025

    ax.bar(len(task_trends) - 1, placeholder_height, bottom=total_height_2025,
           color="#c5d5e4", edgecolor="#4a6486", alpha=0.4, hatch="..", linewidth=1.5)

    ax.text(len(task_trends) - 1, total_height_2025 + placeholder_height / 2,
            "Ongoing\n(2025)", ha='center', va='center',
            fontsize=13, color="#2e4053", fontweight="medium",
            fontstyle="italic", family="serif",
            path_effects=[path_effects.withStroke(linewidth=3, foreground='white')])


for i, bars in enumerate(ax.containers):
    if i >= len(task_trends.columns):
        continue
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            se_task = task_trends.columns[i]
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + height / 2,
                    int(height), ha='center', va='center',
                    fontsize=10, color="black", fontweight="bold")
            if se_task in generative_tasks:
                marker = "★"
            elif se_task in non_generative_tasks:
                marker = "◆"
            else:
                marker = None

            x_marker = bar.get_x() + 0.1
            y_marker = bar.get_y() + height - 0.5
            if marker:
                text_marker = ax.text(x_marker, y_marker, marker, ha='left', va='center',
                                      fontsize=15, color="white", fontweight="bold")
                text_marker.set_path_effects([path_effects.withStroke(linewidth=3, foreground="black")])


ax.set_xlabel("Year", fontsize=16, fontweight="bold")
ax.set_ylabel("Number of Tasks", fontsize=16, fontweight="bold")
ax.tick_params(axis="x", labelrotation=45, labelsize=14)
ax.tick_params(axis="y", labelsize=14)

max_y_value = int(task_trends.sum(axis=1).max())
ax.set_yticks(np.arange(0, max_y_value + 6, 5))
ax.set_ylim(0, max_y_value + 1)

handles, labels = ax.get_legend_handles_labels()
more_patch = Patch(facecolor="#c5d5e4", edgecolor="#4a6486", hatch="..", label="Ongoing (2025)", alpha=0.4)
handles.append(more_patch)
ax.legend(handles=handles, title="SE Task", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=15, title_fontsize=15)

plt.tight_layout()
plt.savefig("../visualizations/enhanced_se_task_trends.png", dpi=300, bbox_inches="tight")
plt.show()
