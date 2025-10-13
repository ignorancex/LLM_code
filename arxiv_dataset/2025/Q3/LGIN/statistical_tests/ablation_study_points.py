import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = {
    'Curvature': ['Variable Curvature', 'Variable Curvature', 'Fixed Curvature', 'Fixed Curvature'],
    'Transport': ['Parallel Transport', 'No Parallel Transport', 'Parallel Transport', 'No Parallel Transport'],
    'AUC Score': [89.08, 84.23, 91.54, 86.92]
}
df = pd.DataFrame(data)

sns.set_theme(style="whitegrid", context="paper")

plt.figure(figsize=(6, 5))

ax = sns.pointplot(data=df, x='Curvature', y='AUC Score', hue='Transport',
                   markers=True, linestyles='--', palette='viridis', dodge=0.1)

for i in range(df.shape[0]):
    row = df.iloc[i]
    x_pos = list(df['Curvature'].unique()).index(row['Curvature'])
    offset = -0.05 if row['Transport'] == df['Transport'].unique()[0] else 0.05
    plt.text(x_pos + offset, row['AUC Score'] + 0.3, f'{row["AUC Score"]:.2f}',
             ha='center', va='bottom', fontsize=9)


plt.xlabel("Curvature Type", fontsize=11)
plt.ylabel("AUC Score (%)", fontsize=11)
plt.ylim(82, 94)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=labels,
          title="Transport Method", fontsize=9, title_fontsize=10)

plt.tight_layout()

plt.savefig("ablation_auc_pointplot.png", dpi=600, bbox_inches='tight')
plt.show()
