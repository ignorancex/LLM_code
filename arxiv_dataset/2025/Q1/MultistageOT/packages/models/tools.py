import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import patches as patches


def plot_couplings(ax, coupling_matrix, X, Y,  scale=1, alpha=1, cmap= None,color = 'k', scatter_size=50):
    """ Plot couplings between supports X and Y, with coupling strength
        given by the coupling_matrix with elements m_ij, describing the
        coupling strength between data point i in support X and data point
        j in support Y """

    size = scatter_size

    # Create scatter plots for both point clouds
    ax.scatter(X[:, 0], X[:, 1], s=size, color=[0.9,0.9,0.9], edgecolor='k',zorder=0)
    ax.scatter(Y[:, 0], Y[:, 1], s=size, color=[0.9,0.9,0.9], edgecolor='k',zorder=0)

    # Flatten the coupling matrix to get a 1D array
    coupling_strength = coupling_matrix.flatten()


    # Iterate through each pair of points
    if cmap is not None:
        cmap_dyn = plt.get_cmap('viridis')
        colors_cmap = cmap_dyn(coupling_strength)

    for i in range(len(X)):
        for j in range(len(Y)):
            # Get the coupling strength for this pair of points
            strength = coupling_strength[i*len(Y) + j]

            # Calculate the thickness based on the coupling strength
            coupling_thickness = strength * scale 

            # Create an line connecting the two points
            if cmap is not None:
                coupling = FancyArrowPatch((X[i, 0], X[i, 1]), (Y[j, 0], Y[j, 1]),
                                        arrowstyle='-', mutation_scale=coupling_thickness,
                                        lw=coupling_thickness, color=colors_cmap[i * len(Y) + j], alpha=alpha, zorder=1)
            else:
                coupling = FancyArrowPatch((X[i, 0], X[i, 1]), (Y[j, 0], Y[j, 1]),
                                        arrowstyle='-', mutation_scale=coupling_thickness,
                                        lw=coupling_thickness, color=color, alpha=alpha, zorder=1)
            ax.add_patch(coupling)

def plot_fate_probabilities_with_pie_charts(ax,data_df, probability_df, entropy_fade=True, plot_order=None, frac=1, indices=None, s=1, outline=1.5, background = None, colors = None):
    """ Plot fate probabilities as pie charts. """  
    

    # Compute entropy for each probability distribution
    if entropy_fade:
        entropies = -np.sum(probability_df * np.log2(probability_df), axis=1)
    
    data_df = (data_df - data_df.min()) / (data_df.max() - data_df.min())

    num_data = data_df.shape[0]

    
    if indices is not None:
        data_df_sampled = data_df.loc[indices]
    else:
        data_df_sampled = data_df

    data_df_sampled = data_df_sampled.sample(frac=frac)
    
    # Define colors for each class
    if plot_order is not None:
        probability_df = probability_df[plot_order]

    if colors is None:
        colors = plt.cm.tab10(range(len(probability_df.columns)))
    
    if background is not None:
        ax.scatter(data_df_sampled.iloc[:,0],data_df_sampled.iloc[:,1], facecolor='none',edgecolor='k', linewidth=outline, s=background)

    if entropy_fade:
        ax.scatter(data_df_sampled.iloc[:,0],data_df_sampled.iloc[:,1], color=[0.9,0.9,0.9], s=2)


    
    # Iterate over each data point
    it = 0
    for i, row in data_df_sampled.iterrows():
        it += 1
        
        x, y = row[0], row[1]
        probs = probability_df.loc[i]
        probs = probs/np.sum(probs)
        
        # Compute alpha value based on entropy
        if entropy_fade:
            entropy = entropies[i]
            alpha = (1 - entropy / np.log2(len(probs)))
        else:
            alpha = 1

        

        print("\r", "Plotting cell nr {j}/{all}".format(j=it,all=num_data), end="")


        radius = np.sqrt(s)/(2*3*72)

        wedges = [patches.Wedge([x,y], radius, 360*np.sum(probs[:k]), 
                                360*np.sum(probs[:k])+360*probs[k], 
                                linewidth=0,color=colors[k], alpha = alpha) for k in range(len(probs))]
        
        [ax.add_patch(wedges[k]) for k in range(len(probs))]

        labels = probability_df.columns
        # Create legend patches
        legend_patches = [mpatches.Patch(color=color, label=label) for color, label in zip(colors, labels)]
        # Add the legend
        ax.legend(handles=legend_patches, loc=[1,0.5])
        

