import matplotlib.pyplot as plt
import numpy as np

def plot_mean_att_distance(mean_att_dist):
    'mean_att_dist shape: (num_layers, num_heads)'
    num_layers = mean_att_dist.shape[0]
    num_heads = mean_att_dist.shape[1]
    # Create the plot
    plt.figure(figsize=(10, 6))

    for head in range(num_heads):
        values = mean_att_dist[:, head]
        plt.scatter(range(num_layers), values, label=f'Head {head}', s=20)

    plt.xlabel('Network depth (layer)')
    plt.ylabel('Mean attention distance (pixels)')
    plt.xlim(0, num_layers - 1)
    plt.ylim(0, 128)

    # Customize legend
    plt.legend(loc='lower right', ncol=2, fontsize='small')

    # Add ellipsis to legend
    handles, labels = plt.gca().get_legend_handles_labels()
    handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=5))
    labels.append('...')
    plt.legend(handles, labels, loc='lower right', ncol=2, fontsize='small')
    plt.tight_layout()
    
    return plt