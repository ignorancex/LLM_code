import seaborn as sns
sns.set_theme(style="whitegrid")

penguins = sns.load_dataset("penguins")

print(penguins)

# Draw a nested barplot by species and sex
g = sns.catplot(
    data=penguins, kind="bar",
    x="method", y="accuracy", hue="normalization",
    errorbar="sd", palette="dark", alpha=.6, height=6
)
g.despine(left=True)
g.set_axis_labels("", "Body mass (g)")
g.legend.set_title("")