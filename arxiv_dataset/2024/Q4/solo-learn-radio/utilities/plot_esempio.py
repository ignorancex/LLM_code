import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="whitegrid")

penguins = sns.load_dataset("penguins")
print(penguins)

# Draw a nested barplot by species and sex
g = sns.catplot(
    data=penguins, kind="bar",
    x="species", y="body_mass_g", hue="sex",
    errorbar="sd", palette="dark", alpha=.6, height=6
)
g.despine(left=True)
g.set_axis_labels("", "Body mass (g)")
g.legend.set_title("")

plt.tight_layout()
plt.show()
