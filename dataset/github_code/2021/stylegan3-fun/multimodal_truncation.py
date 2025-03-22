import os
from typing import List, Optional, Union, Tuple
import click

import dnnlib
from torch_utils import gen_utils

import scipy
import numpy as np
import PIL.Image
import torch

import legacy

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# ----------------------------------------------------------------------------

# TODO/hax: Use this for generation: https://huggingface.co/spaces/SIGGRAPH2022/Self-Distilled-StyleGAN/blob/main/model.py
# SGANXL uses it for generation w/L2 norm: https://github.com/autonomousvision/stylegan_xl/blob/4241ff9cfeb69d617427107a75d69e9d1c2d92f2/torch_utils/gen_utils.py#L428
@click.group()
def main():
    pass


# ----------------------------------------------------------------------------


@main.command(name='get-centroids')
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename: can be URL, local file, or the name of the model in torch_utils.gen_utils.resume_specs', required=True)
@click.option('--cfg', type=click.Choice(['stylegan2', 'stylegan3-t', 'stylegan3-r']), help='Config of the network, used only if you want to use the pretrained models in torch_utils.gen_utils.resume_specs')
@click.option('--device', help='Device to use for image generation; using the CPU is slower than the GPU', type=click.Choice(['cpu', 'cuda']), default='cuda', show_default=True)
# Centroids options
@click.option('--seed', type=int, help='Random seed to use', default=0, show_default=True)
@click.option('--num-latents', type=int, help='Number of latents to use for clustering; not recommended to change', default=60000, show_default=True)
@click.option('--num-clusters', type=click.Choice(['32', '64', '128']), help='Number of cluster centroids to find', default='64', show_default=True)
# Extra parameters
@click.option('--anchor-latent-space', '-anchor', is_flag=True, help='Anchor the latent space to w_avg to stabilize the video')
@click.option('--plot-pca', '-pca', is_flag=True, help='Plot and save the PCA of the disentangled latent space W')
@click.option('--dim-pca', '-dim', type=click.IntRange(min=2, max=3), help='Number of dimensions to use for the PCA', default=3, show_default=True)
@click.option('--verbose', type=bool, help='Verbose mode for KMeans (during centroids calculation)', show_default=True, default=False)
@click.option('--outdir', type=click.Path(file_okay=False), help='Directory path to save the results', default=os.path.join(os.getcwd(), 'out', 'clusters'), show_default=True, metavar='DIR')
@click.option('--description', '-desc', type=str, help='Description name for the directory path to save results', default='pure_centroids', show_default=True)
def get_centroids(
        ctx: click.Context,
        network_pkl: str,
        cfg: Optional[str],
        device: Optional[str],
        seed: Optional[int],
        num_latents: Optional[int],
        num_clusters: Optional[str],
        anchor_latent_space: Optional[bool],
        plot_pca: Optional[bool],
        dim_pca: Optional[int],
        verbose: Optional[bool],
        outdir: Union[str, os.PathLike],
        description: Optional[str]
):
    """Find the cluster centers in the latent space of the selected model"""
    device = torch.device('cuda') if torch.cuda.is_available() and device == 'cuda' else torch.device('cpu')

    # Load the network
    G = gen_utils.load_network('G_ema', network_pkl, cfg, device)

    # Setup for using CPU
    if device.type == 'cpu':
        gen_utils.use_cpu(G)

    # Stabilize/anchor the latent space
    if anchor_latent_space:
        gen_utils.anchor_latent_space(G)

    desc = f'multimodal-truncation-{num_clusters}clusters'
    desc = f'{desc}-{description}' if len(description) != 0 else desc
    # Create the run dir with the given name description
    run_dir = gen_utils.make_run_dir(outdir, desc)

    print('Generating all the latents...')
    z = torch.from_numpy(np.random.RandomState(seed).randn(num_latents, G.z_dim)).to(device)
    w = G.mapping(z, None)[:, 0, :]

    # Get the centroids
    print('Finding the cluster centroids. Patience...')
    scaler = StandardScaler()
    scaler.fit(w.cpu())

    # Scale the dlatents and perform KMeans with the selected number of clusters
    w_scaled = scaler.transform(w.cpu())
    kmeans = KMeans(n_clusters=int(num_clusters), random_state=0, init='random', verbose=int(verbose)).fit(w_scaled)

    # Get the centroids and inverse transform them to the original space
    w_avg_multi = torch.from_numpy(scaler.inverse_transform(kmeans.cluster_centers_)).to(device)

    print('Success! Saving the centroids...')
    for idx, w_avg in enumerate(w_avg_multi):
        w_avg = torch.tile(w_avg, (1, G.mapping.num_ws, 1))
        img = gen_utils.w_to_img(G, w_avg)[0]
        # Save image and dlatent/new centroid
        PIL.Image.fromarray(img, 'RGB').save(os.path.join(run_dir, f'pure_centroid_no{idx+1:03d}-{num_clusters}clusters.jpg'))
        np.save(os.path.join(run_dir, f'centroid_{idx+1:03d}-{num_clusters}clusters.npy'), w_avg.unsqueeze(0).cpu().numpy())

    # Save the configuration used
    ctx.obj = {
        'model_options': {
            'network_pkl': network_pkl,
            'model_configuration': cfg},
        'centroids_options': {
            'seed': seed,
            'num_latents': num_latents,
            'num_clusters': num_clusters},
        'extra_parameters': {
            'anchor_latent_space': anchor_latent_space,
            'outdir': run_dir,
            'description': description}
    }
    # Save the run configuration
    gen_utils.save_config(ctx=ctx, run_dir=run_dir)

    if plot_pca:
        print('Plotting the PCA of the disentangled latent space...')
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA

        pca = PCA(n_components=dim_pca)
        fit_pca = pca.fit(w_scaled)
        fit_pca = pca.fit_transform(w_scaled)
        kmeans_pca = KMeans(n_clusters=int(num_clusters), random_state=0, verbose=0, init='random').fit_predict(fit_pca)

        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(111, projection='3d' if dim_pca == 3 else None)
        axes = fit_pca[:, 0], fit_pca[:, 1], fit_pca[:, 2] if dim_pca == 3 else fit_pca[:, 0], fit_pca[:, 1]
        ax.scatter(*axes, c=kmeans_pca, cmap='inferno', edgecolor='k', s=40, alpha=0.5)
        ax.set_title(r"$| \mathcal{W} | \rightarrow $" + f'{dim_pca}')
        ax.axis('off')
        plt.savefig(os.path.join(run_dir, f'pca_{dim_pca}dim_{num_clusters}clusters.png'))

    print('Done!')


# ----------------------------------------------------------------------------


if __name__ == '__main__':
    main()


# ----------------------------------------------------------------------------
