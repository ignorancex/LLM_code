import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_point():
    data = {
        'Curvature': ['Variable Curvature', 'Variable Curvature', 'Fixed Curvature', 'Fixed Curvature'],
        'Transport': ['Parallel Transport', 'No Parallel Transport', 'Parallel Transport', 'No Parallel Transport'],
        'AUC Score': [89.08, 84.23, 91.54, 86.92]
    }

    df = pd.DataFrame(data)

    sns.set_theme(style="whitegrid", context="paper")

    plt.figure(figsize=(6, 5), dpi=600)

    ax = sns.pointplot(data=df, x='Curvature', y='AUC Score', hue='Transport',
                       markers=True, linestyles='--', palette='viridis', dodge=0.1)

    for i in range(df.shape[0]):
        row = df.iloc[i]
        x_pos = list(df['Curvature'].unique()).index(row['Curvature'])
        offset = - \
            0.05 if row['Transport'] == df['Transport'].unique()[0] else 0.05
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

        plt.savefig("statistical_tests/plots/ablation_auc_pointplot.png",
                    dpi=600)


def plot_3d():
    x_labels = ['Variable', 'Fixed']
    y_labels = ['With PT', 'Without PT']
    z_values = np.array([[89.08, 84.23], [91.54, 86.92]])

    x = np.repeat(np.arange(len(x_labels)), len(y_labels))
    y = np.tile(np.arange(len(y_labels)), len(x_labels))
    z = z_values.flatten()

    fig = plt.figure(figsize=(8, 6), dpi=600)
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(x, y, z, c=z, cmap='cividis',
                         s=100, marker='x')

    ax.set_xlabel('Curvature Type', labelpad=10)
    ax.set_ylabel('Transport Mode', labelpad=10)
    ax.set_zlabel('AUC')
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(x_labels)
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels)

    ax.tick_params(
        axis='y',
        which='major',
        pad=5
    )

    cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label('AUC Score')

    plt.savefig('statistical_tests/plots/plot.png')


if __name__ == '__main__':
    inp = int(input("Enter 1 to for 2d line plot and 2 to plot 3d figure: "))
    if inp == 1:
        plot_point()
    else:
        plot_3d()
