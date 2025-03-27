# import libraries
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
tqdm.pandas()
import datetime
import imageio
from matplotlib.colors import LinearSegmentedColormap
from utils.models import encoder

def plot_review(model, rev_features, label, int2word, depth, dir_path = None, options = None):
    """
    Plots the tokens in a single review, when the encoding dimensions is emb = 2.
    The evolution of tokens is plotted for K = depth number of layers.
    The arguments rev_features and label contain the indeces of words in a single review and its label, respectively.
    The function int2word establishes the correspondence between indices and words.
    The results are saved in the directory dir_path.
    As options, it can be specified:
        - movie: if True, saves an animation of the evolution as a .gif
        - save_plots: if True, saves initial (K = 0) and final (K = depth) token representations
        - trail: if True, plots also the trajectories of tokens as they evolve through the layers
        - start_white: if True, colors all non-leaders white and progressively colors all tokens (and their trails) as they evolve
    """
    if options is None:
        options = {
            'movie': True,
            'save_plots': False,
            'mean': True,
            'levels': True,
            'trail': False,
            'start_white': False
        }
    E = model.state_dict()['encoder.weight'] 
    ## if d \neq 2, return an error
    if np.shape(E)[1] != 2:
        raise ValueError("Plot only possible if embedding dimension d = 2.")    
    v = model.state_dict()['decoder.weight'].numpy()
    b = model.state_dict()['decoder.bias'].numpy()
    alpha = model.state_dict()['attention.alpha'].numpy() 
    z0 = encoder(E, rev_features).T #initial configuration embedding
    d, n = np.shape(z0)
    ## Run the HARDMAX dynamics ##
    s1 = 1 / (1 + alpha)
    s2 = alpha * s1
    W = np.zeros((d, n))
    z = np.zeros((d, n, depth+1)) #we want to do K = depth number of iterations, +1 to save the initial configuration
    z[:, :, 0] = z0
    f = z0.copy()
    for iter in range(1,depth+1):
        # Run dynamics
        for i in range(n):
            IP = np.dot(f[:, i].T,f)
            Pij = np.zeros(n)
            ind = IP == np.max(IP)
            Pij[ind] = 1. / np.sum(ind)
            W[:, i] =  s2 * np.sum(Pij * f, axis=1)
        f = s1 * f + W
        z[:, :, iter] = f
    # Get all leaders (in the layer = depth, i.e. last layer)
    leaders = []
    for i in range(n):
        y = np.dot(z[:, i, depth], z[:, :, depth])
        ind = y == np.max(y)
        if i == np.where(ind)[0][0]:
            leader_particle = np.where(ind)[0][0]
            leaders = np.append(leaders, leader_particle)
    # Identify tokens that are not leaders but also followed (attractors)
    attractors = np.zeros((n,depth+1))
    for iter in range(depth+1):
        for p in range(n):
            y = np.dot(z[:, p, iter], z[:, :, iter])
            attractors[p,iter] = np.argmax(y)
    unique_attractors = np.unique(attractors)
    color_index_mapping = {value: index for index, value in enumerate(unique_attractors)}

    colors = ["#FFAE00", "#56B4E9", "#009E73", "#D55E00", "#CC79A7", "#F0E442", '#8c564b', '#e377c2', '#7f7f7f', '#9467bd', '#17becf', '#1f77b4']
    dt_string = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = dt_string + ".gif"
    base_filename = dt_string
    if label == 1:
        sentiment = 'Positive'
    else:
        sentiment = 'Negative'
    m = 0.5 #margin for the plots
    x_min = z[0,:,:].min() - m
    x_max = z[0,:,:].max() + m
    y_min = z[1,:,:].min() - m
    y_max = z[1,:,:].max() + m

    # Plots of the decoder output, hyperplane and half-spaces
    x = np.linspace(x_min, x_max, 50)
    y = np.linspace(y_min, y_max, 50)
    y2 = -v[:,0]/v[:,1] * x - b/v[:,1]
    X, Y = np.meshgrid(x, y)
    Z = v[:,0] * X + v[:,1] * Y + b
    ct = 0.1 #constant to make the transition of colors less sharp
    Z = 1 / (1 + np.exp(-ct * Z))
    lev = np.linspace(0, 1, 2 + 1)
    start_color = '#FFFFFF'  # White
    for iter in range(depth+1):
        plt.subplots(figsize=(4, 3))#plt.subplots(figsize=(4, 3))
        # Set the x-axis and y-axis limits
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max) 
        if options['levels']:
            plt.contourf(X,Y,Z, levels = lev, cmap = create_custom_colormap(), alpha = 0.5, antialiased=True)
        plt.plot(x, y2, color='gray',linewidth = 1, linestyle = '--')
        if options['mean']: #plot the mean word
            z_mean = np.mean(z[:,:,iter], axis = 1)
            plt.scatter(z_mean[0],z_mean[1],
                            color = 'black',
                            alpha=1,
                            marker='^',
                            zorder=4,
                            s=75)
        for p in range(n):
            if options['trail']: #plot trail of tokens
                if iter > 0:
                    for t in range(0,iter):
                        x_val = z[0, p, t:t+2]
                        y_val = z[1, p, t:t+2]
                        if options['start_white']:
                            end_color = colors[color_index_mapping[attractors[p,t]]]
                            plot_color = interpolate_color(start_color, end_color, depth + 1, t)
                        else:
                            plot_color = colors[color_index_mapping[attractors[p,t]]]
                        plt.plot(x_val, y_val, 
                            color = plot_color, 
                            alpha = 1, 
                            linewidth = 1, 
                            zorder = 1) 
            x_val1 = z[0, p, iter]
            y_val2 = z[1, p, iter]
            if p not in leaders:
                if p in attractors[:,iter]:
                    if options['start_white']:
                        end_color = colors[color_index_mapping[attractors[p,iter]]]
                        plot_color = interpolate_color(start_color, end_color, depth + 1, iter)
                        edge_end_color = colors[color_index_mapping[p]]
                        edge_plot_color = interpolate_color(start_color, edge_end_color, depth + 1, iter)
                    else:
                        plot_color = colors[color_index_mapping[attractors[p,iter]]]
                        edge_plot_color = colors[color_index_mapping[p]]
                    plt.scatter(x_val1, y_val2,
                                    color = plot_color,
                                    alpha=1,
                                    marker='o',
                                    linewidth=2,
                                    edgecolors=edge_plot_color,
                                    zorder=2,
                                    s=40)
                else:
                    if options['start_white']:
                        end_color = colors[color_index_mapping[attractors[p,iter]]]
                        plot_color = interpolate_color(start_color, end_color, depth + 1, iter)
                    else:
                        plot_color = colors[color_index_mapping[attractors[p,iter]]]
                    plt.scatter(x_val1, y_val2,
                                        color = plot_color,
                                        alpha= 1,
                                        marker='o',
                                        linewidth=2,
                                        edgecolors=plot_color,
                                        zorder=2,
                                        s=40)
            # Tag leaders with the word they encode
            if p in leaders:
                plot_color = colors[color_index_mapping[attractors[p,iter]]]
                plt.scatter(x_val1, y_val2,
                                    c = plot_color,
                                    alpha=1,
                                    marker='*',
                                    zorder=3,
                                    s=150)
                plt.annotate(int2word[int(rev_features[p])], (x_val1, y_val2), 
                        textcoords="offset points", 
                        xytext=(7,-17), 
                        ha='center', 
                        fontsize=9, 
                        zorder = 5,
                        bbox=dict(boxstyle="round", fc="1.0"),
                        )   
        plt.title(str(sentiment) +' review ($K =$' + str(iter) + ')', fontsize=12)
        if options['save_plots']:
            if iter == 0:
                plt.savefig(dir_path + '/' + base_filename + 'initial' + "{}.pdf".format(iter),
                            format='pdf', dpi=250, bbox_inches='tight')
            if iter == depth:
                plt.savefig(dir_path + '/' + base_filename + 'final' + "{}.pdf".format(iter),
                            format='pdf', dpi=250, bbox_inches='tight')
        if options['movie']:
            plt.savefig(base_filename + "{}.png".format(iter),
                            format='png', dpi=250)
    if options['movie']:
            imgs = []
            for t in range(depth+1):
                img_file = base_filename + "{}.png".format(t)
                imgs.append(imageio.imread(img_file))
                os.remove(img_file)
            imageio.mimwrite(os.path.join(dir_path, filename), imgs, fps = 1, loop = 0)

def create_custom_colormap():
    colors = [(0, '#FFB5B5'), (0.5, 'white'), (1, '#B4F28D')]  # Red to White to Green
    cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)
    return cmap

def interpolate_color(start_color, end_color, depth, k_steps):
    start_rgb = hex_to_rgb(start_color)
    end_rgb = hex_to_rgb(end_color)

    interpolated_colors = []
    for i in range(depth):
        # Perform linear interpolation between RGB values
        interpolated_rgb = tuple(int(start_rgb[j] + (float(i)/(depth-1)) * (end_rgb[j] - start_rgb[j])) for j in range(3))
        # Convert interpolated RGB to HEX
        interpolated_hex = rgb_to_hex(interpolated_rgb)
        interpolated_colors.append(interpolated_hex)
    return interpolated_colors[k_steps]

## Allow particles to progressively transition from white to the color of the particle they follow
def hex_to_rgb(hex_color):
    # Remove '#' if present
    hex_color = hex_color.lstrip('#')
    # Convert HEX to RGB
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_hex(rgb_color):
    # Convert RGB to HEX
    return '#{:02x}{:02x}{:02x}'.format(*rgb_color)