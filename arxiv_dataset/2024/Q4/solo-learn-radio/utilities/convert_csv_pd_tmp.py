import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ast import literal_eval
import seaborn as sns


exps = pd.read_csv("accuracies.csv")

exps.accuracies = exps.accuracies.apply(literal_eval)
#exps['accuracies'] = exps['accuracies'].astype(float)
result_df = exps.explode('accuracies').drop_duplicates().reset_index(drop=1)
result_df = result_df[result_df["K"].isin([1,3])]
result_df = result_df[result_df["eval_method"] == "finetune"]

print(result_df)


sns.set_theme(style="whitegrid")

result_df["aug"] = result_df["pretrain_aug"] + "-" + result_df["eval_aug"]
result_df["display_name"] = result_df["pretrain_method"] + "-" + result_df["pretrain_dataset"] + "-" + result_df["backbone"]
result_df = result_df.sort_values(by=["backbone", "pretrain_dataset", "pretrain_method"], ignore_index=True)
#result_df["class"] = result_df["eval_dataset"] + "-" + result_df["pretrain_method"] + "-" + result_df["backbone"] 

#result_df_vlass = result_df[result_df["eval_dataset"] == "vlass"]
#result_df_frg = result_df[result_df["eval_dataset"] == "frg"]
result_df_rgz = result_df[result_df["eval_dataset"] == "rgz"]
#result_df_robin = result_df[result_df["eval_dataset"] == "robin"]
result_df_mirabest = result_df[result_df["eval_dataset"] == "mirabest"]

# Draw a nested barplot by species and sex
g = sns.catplot(
    data=result_df_mirabest, kind="bar",
    y="display_name", x="accuracies", hue="aug",
    hue_order=['minmax-minmax', 'minmax-mixed', 'mixed-minmax', 'mixed-mixed'],
    errorbar="sd", palette="dark", alpha=.7, height=6, orient="h"
)
g.set(xticks=range(20, 100, 2))
g.set(xlim=(20, 100))
g.tight_layout()
g.despine(left=True)
g.set_axis_labels("", "Accuracy")
g.legend.set_title("")
g.tick_params(axis='y', labelsize=11)
g.tick_params(axis='x', labelsize=6)
sns.move_legend(
    g, "lower center",
    bbox_to_anchor=(.5, 1), ncol=4, title=None, frameon=False,
)

#plt.xticks(rotation=60)
plt.tight_layout()
plt.savefig("mirabest_finetune", bbox_inches='tight')


# Draw a nested barplot by species and sex
g = sns.catplot(
    data=result_df_rgz, kind="bar",
    y="display_name", x="accuracies", hue="aug",
    hue_order=['minmax-minmax', 'minmax-mixed', 'mixed-minmax', 'mixed-mixed'],
    errorbar="sd", palette="dark", alpha=.7, height=6, orient="h"
)
g.set(xticks=range(20, 100, 2))
g.set(xlim=(20, 100))
g.tight_layout()
g.despine(left=True)
g.set_axis_labels("", "Accuracy")
g.legend.set_title("")
g.tick_params(axis='y', labelsize=11)
g.tick_params(axis='x', labelsize=6)
sns.move_legend(
    g, "lower center",
    bbox_to_anchor=(.5, 1), ncol=4, title=None, frameon=False,
)

#plt.xticks(rotation=60)
plt.tight_layout()
plt.savefig("rgz_finetune", bbox_inches='tight')
"""
# Draw a nested barplot by species and sex
g = sns.catplot(
    data=result_df_vlass, kind="bar",
    y="display_name", x="accuracies", hue="aug",
    hue_order=['minmax-minmax', 'minmax-mixed', 'mixed-minmax', 'mixed-mixed'],
    errorbar="sd", palette="dark", alpha=.7, height=6, orient="h"
)
g.set(xticks=range(20, 100, 2))
g.set(xlim=(20, 100))
g.tight_layout()
g.despine(left=True)
g.set_axis_labels("", "Accuracy")
g.legend.set_title("")
g.tick_params(axis='y', labelsize=11)
g.tick_params(axis='x', labelsize=6)
sns.move_legend(
    g, "lower center",
    bbox_to_anchor=(.5, 1), ncol=4, title=None, frameon=False,
)

#plt.xticks(rotation=60)
plt.tight_layout()
plt.savefig("vlass", bbox_inches='tight')

# Draw a nested barplot by species and sex
g = sns.catplot(
    data=result_df_frg, kind="bar",
    y="display_name", x="accuracies", hue="aug",
    hue_order=['minmax-minmax', 'minmax-mixed', 'mixed-minmax', 'mixed-mixed'],
    errorbar="sd", palette="dark", alpha=.7, height=6, orient="h"
)
#g.set_xticklabels(labels=None)
g.set(xticks=range(20, 100, 2))
g.set(xlim=(20, 100))
g.tight_layout()
g.despine(left=True)
g.set_axis_labels("", "Accuracy")
g.legend.set_title("")
g.tick_params(axis='y', labelsize=11)
g.tick_params(axis='x', labelsize=6)
sns.move_legend(
    g, "lower center",
    bbox_to_anchor=(.5, 1), ncol=4, title=None, frameon=False,
)

#plt.xticks(rotation=60)
plt.tight_layout()
plt.savefig("frg", bbox_inches='tight')
#plt.show()

# Draw a nested barplot by species and sex
g = sns.catplot(
    data=result_df_robin, kind="bar",
    y="display_name", x="accuracies", hue="aug",
    hue_order=['minmax-minmax', 'minmax-mixed', 'mixed-minmax', 'mixed-mixed'],
    errorbar="sd", palette="dark", alpha=.7, height=6, orient="h"
)
#g.set_xticklabels(labels=None)
g.set(xticks=range(20, 100, 2))
g.set(xlim=(20, 100))
g.tight_layout()
g.despine(left=True)
g.set_axis_labels("", "Accuracy")
g.legend.set_title("")
g.tick_params(axis='y', labelsize=11)
g.tick_params(axis='x', labelsize=6)
sns.move_legend(
    g, "lower center",
    bbox_to_anchor=(.5, 1), ncol=4, title=None, frameon=False,
)

#plt.xticks(rotation=60)
plt.tight_layout()
plt.savefig("robin", bbox_inches='tight')
#plt.show()

# Draw a nested barplot by species and sex
g = sns.catplot(
    data=result_df_mirabest, kind="bar",
    y="display_name", x="accuracies", hue="aug",
    hue_order=['minmax-minmax', 'minmax-mixed', 'mixed-minmax', 'mixed-mixed'],
    errorbar="sd", palette="dark", alpha=.7, height=6, orient="h"
)
#g.set_xticklabels(labels=None)
g.set(xticks=range(0, 100, 2))
g.set(xlim=(0, 100))
g.tight_layout()
g.despine(left=True)
g.set_axis_labels("", "Accuracy")
g.legend.set_title("")
g.tick_params(axis='y', labelsize=11)
g.tick_params(axis='x', labelsize=6)
sns.move_legend(
    g, "lower center",
    bbox_to_anchor=(.5, 1), ncol=4, title=None, frameon=False,
)

#plt.xticks(rotation=60)
plt.tight_layout()
plt.savefig("mirabest", bbox_inches='tight')
#plt.show()

"""