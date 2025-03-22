# Plot the quantiles of a dataset against each other
# https://en.wikipedia.org/wiki/Q%E2%80%93Q_plot

import matplotlib.pyplot as plt
from torch_utils import gen_utils
import torch

# Set a predefined style
plt.style.use('ggplot')

# Define color and marker mappings
layer_colors = {8: 'red', 14: 'blue', 16: 'green', 18: 'purple'}
resolution_markers = {256: 'o', 512: 's', 1024: '^'}


# Set device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def calculate_quantiles_torch(data, num_quantiles=100):
    quantiles = torch.linspace(0, 1, num_quantiles)
    return torch.quantile(data, quantiles).cpu().numpy()


# TODO: use 'stylegan3-t', 'stylegan3-r'
cfg = 'stylegan2'
dlatents = []

# Set seed for reproducibility
seed = 0
torch.manual_seed(seed)  # TODO: do we need to sample different latents for each?
latents = torch.randn(10000, 512, device=device)

# TODO: separate by number of mapping layers?
# TODO: separate by image resolution

# networks = ['ffhq1024', 'ffhqu1024', 'ffhq512', 'ffhq256', 'ffhqu256', 'ffhq256', 'celebahq256', 'anime256',
#             'lsundog256', 'lsuncat256', 'lsunchurch256', 'lsunhorse256', 'afhqcat256', 'cub256', 'sdhorses256', 'sdbicycles256']

# Networks @ 1024x1024
networks = ['ffhq1024', 'ffhqu1024', 'metfaces1024', 'metfacesu1024', 'minecraft1024', 'maps1024', 'lhq1024', 'sddogs1024']

# Add networks @ 512x512
# networks.extend(['ffhq512', 'afhqcat512', 'afhqdog512', 'afhqwild512', 'afhq512', 'brecahad512', 'lsuncar512',
#             'fursona512', 'mlpony512', 'sdelephant512', 'sdlions512', 'sdgiraffes512', 'sdparrots512'])

# Add Networks @ 256x256
# networks.extend(['ffhq256', 'ffhqu256', 'celebahq256', 'lsundog256', 'lsuncat256', 'lsunchurch256', 'lsunhorse256', 'afhqcat256',
#             'anime256', 'cub256', 'sdhorses256', 'sdbicycles256'])

networks_features = {}

for network_pkl in networks:
    try:
        G = gen_utils.load_network('G_ema', network_pkl, cfg, device)
        dlatent = G.mapping(z=latents, c=None, truncation_psi=1.0).detach().cpu()
        dlatents.append(dlatent[:, 0])  # [1, num_ws, w_dim] -> [1, w_dim]

        # Save the features of the Generator
        networks_features[network_pkl] = {'num_ws': G.mapping.num_ws, 'w_dim': G.mapping.w_dim,
                                          'num_layers': G.mapping.num_layers, 'img_resolution': G.img_resolution}

        del G, dlatent

    except Exception as e:
        print(f'Failed to load network {network_pkl}: {e}. Removing it and continuing...')
        networks.remove(network_pkl)
        continue

print(networks_features)

N = len(networks)

quantiles = [calculate_quantiles_torch(data) for data in dlatents]

# Plotting
fig, axs = plt.subplots(N, N, figsize=(4*N, 4*N), constrained_layout=True)
# Set title
fig.suptitle('Quantile-Quantile Plots: $p(\mathcal{W})$', fontsize=40)

# for i in range(N):
#     for j in range(N):
#         if i == j:
#             # Plot histogram on the diagonal
#             data = dlatents[i].squeeze().cpu().numpy()
#             means = data.mean(axis=0)  # Calculate mean across samples for each dimension
#             stds = data.std(axis=0)  # Calculate std across samples for each dimension
#             axs[i, j].hist(means, bins=50, density=True, alpha=0.7, color="skyblue", label='Means')
#             axs[i, j].hist(stds, bins=50, density=True, alpha=0.7, color="lightgreen", label='Std Devs')
#             axs[i, j].set_title(f'{networks[i]}\nDistribution of Means and Std Devs')
#             axs[i, j].set_xlabel('Value')
#             axs[i, j].set_ylabel('Density')
#             axs[i, j].legend()
#         elif i<j:
#             axs[i, j].scatter(quantiles[j], quantiles[i], color="darkblue")
#             axs[i, j].plot(quantiles[j], quantiles[j], 'r--', alpha=0.6)  # y=x reference line
#             axs[i, j].set_title(f'{networks[i]} vs {networks[j]}')
#             axs[i, j].set_xlabel(f'Quantiles for {networks[j]}')
#             axs[i, j].set_ylabel(f'Quantiles for {networks[i]}')
#         else:
#             # Remove plots below the diagonal
#             axs[i, j].axis('off')

for i in range(N):
    for j in range(N):
        if i == j:
            # Plot histogram on the diagonal
            data = dlatents[i].squeeze().cpu().numpy()
            flattened_data = data.flatten()
            mean, std = flattened_data.mean(), flattened_data.std()
            axs[i, j].hist(flattened_data, bins=100, density=True, alpha=0.7, color="skyblue")
            axs[i, j].axvline(mean, color='darkred', linestyle='dashed', linewidth=2)

            model_info = networks_features[networks[i]]
            title = f"{networks[i]}\n"
            title += f"Layers: {model_info['num_layers']}, "
            title += f"W_dim: {model_info['w_dim']}, "
            title += f"Num_ws: {model_info['num_ws']}, "
            title += f"Res: {model_info['img_resolution']}\n"
            title += f"Mean: {mean:.2f}, Std Dev: {std:.2f}"

            axs[i, j].set_title(title, fontsize=10)
            axs[i, j].set_xlabel('W values')
            axs[i, j].set_ylabel('Density')
        elif i < j:
            # Plot Q-Q plot above the diagonal
            model_i = networks_features[networks[i]]
            model_j = networks_features[networks[j]]

            color = layer_colors.get(model_i['num_ws'], 'gray')
            marker = resolution_markers.get(model_i['img_resolution'], 'x')

            axs[i, j].scatter(quantiles[j], quantiles[i], color=color, marker=marker, s=10, alpha=0.7)
            axs[i, j].plot(quantiles[j], quantiles[j], 'k--', alpha=0.6)  # y=x reference line

            title = f"{networks[i]} vs {networks[j]}"
            axs[i, j].set_title(title, fontsize=10)
            axs[i, j].set_xlabel(f'Quantiles for {networks[j]}')
            axs[i, j].set_ylabel(f'Quantiles for {networks[i]}')
        else:
            # Remove plots below the diagonal
            axs[i, j].axis('off')

# Add a legend for colors (layers) and shapes (resolutions)
handles = []
labels = []
for layers, color in layer_colors.items():
    handles.append(plt.Line2D([0], [0], marker='o', color=color, label=f'{layers} layers', markersize=8, linestyle=''))
    labels.append(f'{layers} layers')
for res, marker in resolution_markers.items():
    handles.append(plt.Line2D([0], [0], marker=marker, color='black', label=f'{res}x{res}', markersize=8, linestyle=''))
    labels.append(f'{res}x{res}')

fig.legend(handles, labels, loc='center right', bbox_to_anchor=(0.98, 0.5), title='Model Properties')


plt.show()