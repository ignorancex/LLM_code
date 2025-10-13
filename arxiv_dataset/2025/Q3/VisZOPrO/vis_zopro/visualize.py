import argparse
from copy import deepcopy
import os

import torch
import numpy as np
from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
from scipy.spatial import procrustes
from scipy.stats import pearsonr
from safetensors import safe_open
from openTSNE import TSNE
from umap import UMAP
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import plotly.graph_objects as go


def load_lora_weights(directory, layer=None):
    """
    Loads LoRA weights from a directory.

    Args:
        directory: The directory containing the weight file.

    Returns:
        A dictionary of LoRA weights, or None if no file is found.
    """
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if filename.endswith(".pt") or filename.endswith(".bin"):
            try:
                # Attempt to load as a PyTorch state_dict
                return torch.load(filepath, map_location='cpu')
            except (torch.TorchError, RuntimeError, IOError) as e:
                print(f"Could not load weights from {filepath} as a PyTorch state_dict. Skipping. Error: {e}")
                return None
        elif filename.endswith(".safetensors"):
            tensors = {}
            with safe_open(filepath, framework="pt", device="cpu") as f:
                for k in f.keys():
                    if layer is None or ("layer" in k and layer == int(k.split(".")[4])):
                        tensors[k] = f.get_tensor(k)
                return tensors

    print(f"No .pt, .bin or .safetensors file found in directory: {directory}")
    return None


def extract_lora_weights(state_dict):
    """
    Extracts and concatenates LoRA weights from a state dictionary.

    Args:
        state_dict: A dictionary containing the model's state_dict.

    Returns:
        A concatenated numpy array of LoRA weights, or None if no LoRA weights are found.
    """
    lora_weights = []
    for key, value in state_dict.items():
        if 'lora_A' in key or 'lora_B' in key:
            # Flatten and convert to numpy array
            lora_weights.append(value.cpu().detach().to(torch.float32).numpy().flatten())

    if not lora_weights:
        print("No LoRA weights (lora_A or lora_B) found in the state_dict.")
        return None

    return np.concatenate(lora_weights)


def normalize_vector(v):
    """
    Normalizes a vector.

    Args:
        v: The vector to normalize.

    Returns:
        The normalized vector.
    """
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def distance_correlation(v1, v2):
    """
    Calculates the distance correlation between two vectors.

    Args:
        v1: The first vector (numpy array).
        v2: The second vector (numpy array).

    Returns:
        The distance correlation between v1 and v2.
    """

    def _d(X, Y):
        """Compute Euclidean distance matrix."""
        # Reshape X and Y to be column vectors if they are 1D arrays
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        return np.sqrt(np.sum((X - Y[:, np.newaxis])**2, axis=-1))

    n = len(v1)
    a = _d(v1, v1)
    b = _d(v2, v2)

    # Calculate means along the appropriate axis, handling both 1D and 2D cases
    A = a - a.mean(axis=0, keepdims=True) - a.mean(axis=1, keepdims=True) + a.mean()
    B = b - b.mean(axis=0, keepdims=True) - b.mean(axis=1, keepdims=True) + b.mean()

    dcov_XY = np.sqrt(np.sum(A * B)) / n
    dcov_XX = np.sqrt(np.sum(A * A)) / n
    dcov_YY = np.sqrt(np.sum(B * B)) / n

    if dcov_XX * dcov_YY == 0:
        return 0
    else:
        dcor = dcov_XY / np.sqrt(dcov_XX * dcov_YY)
        return dcor


# Function to calculate the angle between two vectors
def calculate_angle(v1, v2):
    """
    Calculates the angle between two vectors.

    Args:
        v1: The first vector.
        v2: The second vector.

    Returns:
        The angle in degrees between v1 and v2.
    """
    v1_u = normalize_vector(v1)
    v2_u = normalize_vector(v2)
    return np.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))


def plot_trajectories_plotly(model_weights, save_dir, method='tsne', layer=None):
    """
    Plots the trajectories of model weights in 3D using Plotly.

    Args:
        model_weights: A dictionary where keys are model names and values are lists of weight vectors.
        save_dir: The directory to save the plot.
        method: The dimensionality reduction method ('umap' or 'tsne').
        layer: The layer to analyze. If None, all layers are analyzed.
    """
    fig = go.Figure()
    for i, (model_name, model_w) in enumerate(model_weights.items()):

        fig.add_trace(go.Scatter3d(
            x=model_w[:, 0],
            y=model_w[:, 1],
            z=model_w[:, 2],
            mode='lines+markers',
            marker=dict(size=5, color=i, colorscale='Viridis'),  # Color by model index
            line=dict(color=i, colorscale='Viridis', width=2),
            name=model_name
        ))

        # Start and end markers
        fig.add_trace(go.Scatter3d(
            x=[model_w[0, 0]],
            y=[model_w[0, 1]],
            z=[model_w[0, 2]],
            mode='markers',
            marker=dict(size=10, color=i, symbol='circle', colorscale='Viridis'),
            name=f"{model_name} Start"
        ))
        fig.add_trace(go.Scatter3d(
            x=[model_w[-1, 0]],
            y=[model_w[-1, 1]],
            z=[model_w[-1, 2]],
            mode='markers',
            marker=dict(size=10, color=i, symbol='x', colorscale='Viridis'),
            name=f"{model_name} End"
        ))

    fig.update_layout(
        title=f"Weight Trajectories in 3D ({method.upper()})",
        scene=dict(
            xaxis_title="Reduced Dimension 1",
            yaxis_title="Reduced Dimension 2",
            zaxis_title="Reduced Dimension 3"
        ),
        legend=dict(
            title="Models",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )

    # fig.show()
    # this will save an interactive html file
    fig.write_html(os.path.join(save_dir, f"weight_trajectories_3d_{method}_layer{layer if layer is not None else '_all'}.html"))


def plot_trajectories(model_weights, save_dir, method='tsne', layer=None, predict=10):
    """
    Plots the trajectories of model weights in 2 or 3d depending on data.

    Args:
        model_weights: A dictionary where keys are model names and values are lists of weight vectors. (assumes weights' dimension has already been reduced)
        save_dir: The directory to save the plot.
        method: The dimensionality reduction method.
        layer: The layer to analyze. If None, all layers are analyzed.
        predict: The number of steps to predict into the future. If 0, no prediction is made.
    """
    # Infer dimensionality from the data
    num_dimensions = next(iter(model_weights.values())).shape[1]

    if num_dimensions not in (2, 3):
        print("Plotting is only supported for 2D and 3D data.")
        return

    # Standardized parameters
    fig_size = (12, 10)
    marker_size = 150
    font_size = 14

    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111, projection='3d' if num_dimensions == 3 else None)

    cmap = cm.get_cmap('viridis')
    prediction_cmap = cm.get_cmap('plasma')  # Different colormap for predictions

    for i, (model_name, weights) in enumerate(model_weights.items()):
        color = cmap(i / len(model_weights))

        # Prediction (if requested)
        if predict > 0:
            n_samples, dim = weights.shape
            predicted_weights = np.zeros((predict, dim))

            n_last_points = 10
            if n_samples < n_last_points:
                n_last_points = n_samples

            # Fit a linear model to the last n_last_points
            for i in range(dim):
                x = np.arange(n_samples - n_last_points, n_samples)
                y = weights[-n_last_points:, i]
                slope, intercept = np.polyfit(x, y, 1)
                predicted_weights[:, i] = np.array([slope * (n_samples + j) + intercept for j in range(predict)])
            all_weights = np.concatenate([weights, predicted_weights], axis=0)
        else:
            all_weights = weights

        # Plotting (common to 2D and 3D)
        plot_args = {'marker': 'o', 'color': color, 'label': model_name, 'alpha': 0.7}
        pred_plot_args = {'marker': 'o', 'color': prediction_cmap(0.8), 'linestyle': '--', 'alpha': 0.7}  # Highlight predictions

        if num_dimensions == 3:
            ax.plot(all_weights[:, 0], all_weights[:, 1], all_weights[:, 2], **plot_args, linewidth=2, markersize=5)
            if predict > 0:
                ax.plot(predicted_weights[:, 0], predicted_weights[:, 1], predicted_weights[:, 2], **pred_plot_args, linewidth=2, markersize=5)
        else:
            ax.plot(all_weights[:, 0], all_weights[:, 1], **plot_args)
            if predict > 0:
                ax.plot(predicted_weights[:, 0], predicted_weights[:, 1], **pred_plot_args, linewidth=2, markersize=5)

        # Start and end markers (common to 2D and 3D)
        ax.scatter(*weights[0, :], marker='*', s=marker_size, color=color)
        ax.scatter(*weights[-1, :], marker='X', s=marker_size, color=color)

        # Predicted end marker
        if predict > 0:
            ax.scatter(*predicted_weights[-1, :], marker='X', s=marker_size, color=prediction_cmap(0.8))

    # Axes and title (common to 2D and 3D)
    ax.set_xlabel("Reduced Dimension 1", fontsize=font_size)
    ax.set_ylabel("Reduced Dimension 2", fontsize=font_size)
    if num_dimensions == 3:
        ax.set_zlabel("Reduced Dimension 3", fontsize=font_size)

    ax.set_title(f"Weight Trajectories in {num_dimensions}D ({method.upper()})", fontsize=font_size)

    # Customize legend and appearance
    legend = ax.legend(loc='upper right', fontsize=font_size)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('black')

    ax.grid(True, linestyle='--', alpha=0.5, color='gray')
    ax.set_facecolor('whitesmoke')
    if num_dimensions == 3:
        ax.view_init(elev=30, azim=120)

    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(save_dir, f"weight_trajectories_{num_dimensions}d_{method}_layer{layer if layer is not None else '_all'}_predict{predict}.png"))
    plt.close()


def reduce_dimensions(weights, model_names, method='tsne', n_components=2, seed=None):
    """
    Reduces the dimensions of a set of weights using a specified method.

    Args:
        weights: The weights to reduce.
        model_names: A list of model names to include in the plot.
        method: The dimensionality reduction method ('umap' or 'tsne').
        n_components: The number of components to reduce to.
        seed: Random seed for reproducibility.

    Returns:
        A dictionary of <model_name: reduced weights>.
    """

    all_weights = []
    model_labels = []
    for model_name, weights in weights.items():
        if model_name in model_names:
            all_weights.extend(weights)
            model_labels.extend([model_name] * len(weights))

    if method == 'tsne':
        # in case TSNE crashes, try reducing the number of components with one of these first
        all_weights = UMAP(n_components=100, random_state=seed).fit_transform(np.array(all_weights))
        # all_weights = PCA(n_components=min(100, len(all_weights))).fit_transform(np.array(all_weights))
        reducer = TSNE(n_components=n_components, perplexity=min(30, len(all_weights)-1), n_jobs=-1, verbose=True)  # can use , neighbors="annoy" if segmentation fault during neighbors search
        reduced_weights = reducer.fit(np.array(all_weights))
    elif method == 'umap':
        all_weights = UMAP(n_components=n_components, random_state=seed).fit_transform(np.array(all_weights))
        reduced_weights = all_weights
    else:
        raise ValueError("Invalid dimensionality reduction method. Choose 'umap' or 'tsne'.")
    start_index = 0
    reduced_weights_dict = {}
    for model_name in model_names:
        end_index = start_index
        while end_index < len(model_labels) and model_labels[end_index] == model_name:
            end_index += 1
        reduced_weights_dict[model_name] = reduced_weights[start_index:end_index]
        start_index = end_index
    return reduced_weights_dict


# Main function to process directories and perform analysis
def analyze_lora_weights(base_dir, model_names, save_dir, layer=None, seed=None, obi_wan=False, low_memory=False, keep_rm=None):
    """
    Analyzes LoRA weights from multiple models.

    Args:
        base_dir: Folder containing the checkpoints
        model_names: A list of model names to analyze.
        save_dir: Folder to save the plots.
        layer: The layer to analyze. If None, all layers are analyzed.
        seed: Random seed for reproducibility.
        obi_wan: Use this flag to analyze the weights off by one.
        low_memory: Use this flag to reduce memory usage.
        keep_rm: Keep the previous reward model weight with this ratio.
    """
    os.makedirs(save_dir, exist_ok=True)

    # collect all model weights
    model_weights = {}
    all_dirs = os.listdir(base_dir)
    for model_name in model_names:
        weight_vectors = []
        # remove all folders that do not have lora in the name
        dirs = [d for d in all_dirs if f"{model_name}_model" in d]
        sorted_dirs = sorted(dirs, key=lambda x: int(x.split('_')[3]))
        # if very big, take at most 50, equally spaced
        if low_memory and len(sorted_dirs) > 50:
            sorted_dirs = sorted_dirs[::len(sorted_dirs)//50]
        for subdir in sorted_dirs:
            subdir_path = os.path.join(base_dir, subdir)
            # needed for new checkpoints
            adapter_name = "pl" if model_name == "policy" else "reward"
            subdir_path = os.path.join(subdir_path, adapter_name)
            if os.path.isdir(subdir_path):
                print("Loading weights from:", subdir_path)
                state_dict = load_lora_weights(subdir_path, layer)
                if state_dict:
                    weights = extract_lora_weights(state_dict)
                    if weights is not None:
                        weight_vectors.append(weights)
        model_weights[model_name] = weight_vectors
    print("Loaded weights from the following models:", model_weights.keys())

    if keep_rm:
        for i in range(1, len(model_weights["reward"])):
            model_weights["reward"][i] = model_weights["reward"][i] * (1 - keep_rm) + model_weights["reward"][i-1] * keep_rm

    # 1. Plot trajectories (2D and then 3D)
    reduced_weights = reduce_dimensions(model_weights, model_names, method='tsne', n_components=2, seed=seed)
    plot_trajectories(reduced_weights, save_dir, method='tsne', layer=layer)
    reduced_weights = reduce_dimensions(model_weights, model_names, method='umap', n_components=2, seed=seed)
    plot_trajectories(reduced_weights, save_dir, method='umap', layer=layer)

    # 3D plots
    reduced_weights = reduce_dimensions(model_weights, model_names, method='tsne', n_components=3, seed=seed)
    plot_trajectories(reduced_weights, save_dir, method='tsne', layer=layer)
    plot_trajectories_plotly(reduced_weights, model_names, method='tsne', layer=layer)
    reduced_weights = reduce_dimensions(model_weights, model_names, method='umap', n_components=3, seed=seed)
    plot_trajectories(reduced_weights, save_dir, method='umap', layer=layer)
    plot_trajectories_plotly(reduced_weights, save_dir, method='umap', layer=layer)

    # Now do the analysis, use weight updates instead of whole weights
    # i.e. instead of using weights[i], use weights[i] - weights[i-1] (the difference with the previous weight)
    # this helps visualize the correlation between the *updates* rather than the weights themselves

    # load initial weights before training (so that we can get the difference for the first point)
    # ppo initial model is under final_sft_model
    if not obi_wan:
        state_dict = load_lora_weights(os.path.join(base_dir, "final_sft_model"), layer)
        weights = extract_lora_weights(state_dict)
        model_weights["policy"].insert(0, weights)

    # reduce dimensions and *then* calculate the difference
    # Doing the opposite creates instabilities due to very low values        
    model_weights = reduce_dimensions(model_weights, model_names, method='umap', n_components=100, seed=seed)
    for model_name in model_names:
        for i in reversed(range(1, len(model_weights[model_name]))):
            model_weights[model_name][i] = model_weights[model_name][i] - model_weights[model_name][i-1]

    # remove the initial weights (i.e. reward_model 0 and policy_model 0 - likely sft_model)
    for model_name in model_names:
        model_weights[model_name] = model_weights[model_name][1:]

    # make sure the lengths are the same
    min_len = min(len(model_weights[model_names[0]]), len(model_weights[model_names[1]]))
    model_weights[model_names[0]] = model_weights[model_names[0]][:min_len]
    model_weights[model_names[1]] = model_weights[model_names[1]][:min_len]

    # 2. Procrustes analysis
    # Pad the shorter trajectory with its last vector to make them the same length
    max_len = max(len(model_weights[model_names[0]]), len(model_weights[model_names[1]]))
    proc_weights = deepcopy(model_weights)
    proc_weights[model_names[0]] = np.vstack([proc_weights[model_names[0]], np.tile(model_weights[model_names[0]][-1], (max_len - len(model_weights[model_names[0]]), 1))])
    proc_weights[model_names[1]] = np.vstack([proc_weights[model_names[1]], np.tile(model_weights[model_names[1]][-1], (max_len - len(model_weights[model_names[1]]), 1))])
    _, _, disparity = procrustes(np.array(proc_weights[model_names[0]]), np.array(proc_weights[model_names[1]]))
    print(f"Procrustes Disparity between {model_names[0]} and {model_names[1]}: {disparity}")

    # 3. Distance correlation
    for i in range(len(model_weights[model_names[0]])):
        dcor = distance_correlation(model_weights[model_names[0]][i], model_weights[model_names[1]][i])
        print(f"Distance Correlation at time step {i} between {model_names[0]} and {model_names[1]}: {dcor}")

    # 4. Angle between weight vectors
    for i in range(len(model_weights[model_names[0]])):
        angle = calculate_angle(model_weights[model_names[0]][i], model_weights[model_names[1]][i])
        print(f"Angle between weight vectors at time step {i} for {model_names[0]} and {model_names[1]}: {angle:.2f} degrees")

    # 5. Angle between velocity vectors
    for i in range(len(model_weights[model_names[0]]) - 1):
        velocity1 = model_weights[model_names[0]][i+1] - model_weights[model_names[0]][i]
        velocity2 = model_weights[model_names[1]][i+1] - model_weights[model_names[1]][i]
        angle = calculate_angle(velocity1, velocity2)
        print(f"Angle between velocity vectors at time step {i} for {model_names[0]} and {model_names[1]}: {angle:.2f} degrees")

    # 6. Pearson correlation of weight changes
    # Pad the shorter trajectory with its last vector to make them the same length
    max_len = max(len(model_weights[model_names[0]]), len(model_weights[model_names[1]]))
    corr_weights = deepcopy(model_weights)
    corr_weights[model_names[0]] = np.vstack([corr_weights[model_names[0]], np.tile(model_weights[model_names[0]][-1], (max_len - len(model_weights[model_names[0]]), 1))])
    corr_weights[model_names[1]] = np.vstack([corr_weights[model_names[1]], np.tile(model_weights[model_names[1]][-1], (max_len - len(model_weights[model_names[1]]), 1))])

    correlations = []
    for i in range(len(model_weights[model_names[0]]) - 1):
        change1 = model_weights[model_names[0]][i+1] - model_weights[model_names[0]][i]
        change2 = model_weights[model_names[1]][i+1] - model_weights[model_names[1]][i]
        correlation, _ = pearsonr(change1, change2)
        correlations.append(correlation)
        print(f"Pearson correlation of weight changes at time step {i} for {model_names[0]} and {model_names[1]}: {correlation}")

    # Plotting the correlations over time
    plt.figure(figsize=(10, 6))
    plt.plot(correlations, marker='o')
    plt.title('Pearson Correlation of Weight Changes Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Correlation')
    plt.grid(True)
    plt.show()
    plt.savefig(os.path.join(save_dir, f"weight_change_correlation_layer{layer if layer is not None else '_all'}.png"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-dir', type=str, required=True, help='Base directory containing model checkpoints.')
    parser.add_argument('--layer', type=int, default=None, help='Layer to analyze. Leave empty to analyze all layers.')
    parser.add_argument('--random-seed', type=int, default=None, help='Random seed for reproducibility. Giving one may slow some computations.')
    parser.add_argument('--obi-wan', action='store_true', help='Use this flag to analyze the weights off by one. If given, will compare reward_model iter_1 with policy_model iter_2 (and so on). If not given, will compare reward_model iter_1 with policy_model iter_1. With this option, you see the effect of the reward model on the policy model.')
    parser.add_argument('--low-memory', action='store_true', help='Use this flag to reduce memory usage. This will reduce the number of points used for the analysis.')
    parser.add_argument('--save-dir', type=str, default='./plots', help='Directory to save the plots.')
    parser.add_argument('--keep-rm', type=float, default=None, help='Keep the previous weight with this ratio. Use this option if the training was done with a moving average of the weights.')
    args = parser.parse_args()
    model_names = ['reward', 'policy']

    analyze_lora_weights(
        base_dir=args.base_dir,
        model_names=model_names,
        save_dir=args.save_dir,
        layer=args.layer,
        seed=args.random_seed,
        obi_wan=args.obi_wan,
        low_memory=args.low_memory,
        keep_rm=args.keep_rm
    )
