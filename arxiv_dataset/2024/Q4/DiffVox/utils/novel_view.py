import time
from pathlib import Path

import astra
import click
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import submitit
import torch
from monai.metrics import MSEMetric, PSNRMetric, SSIMMetric
from torchio import ScalarImage
from tqdm import tqdm

from diffdrr.data import read
from diffdrr.drr import DRR


def trafo(image):
    """
    A transformation to apply to each image. Converts an image from the
    raw scanner output to the form described by the projection geometry.
    """
    return np.transpose(np.flipud(image))


def load(
    datapath,
    proj_rows,
    proj_cols,
    subsample,
    orbits_to_recon=[1, 2, 3],
    geometry_filename="scan_geom_corrected.geom",
    dark_filename="di000000.tif",
    flat_filenames=["io000000.tif", "io000001.tif"],
):
    """Load and preprocess raw projection data."""

    # Create a numpy array to geometry projection data
    projs = np.zeros((proj_rows, 0, proj_cols), dtype=np.float32)

    # And create a numpy array to projection geometry
    vecs = np.zeros((0, 12), dtype=np.float32)
    orbit = range(0, 1200, subsample)
    n_projs_orbit = len(orbit)

    # Projection file indices, reversed due to portrait mode acquisition
    projs_idx = range(1200, 0, -subsample)

    # Read the images and geometry from each acquisition
    for orbit_id in orbits_to_recon:

        # Load the scan geometry
        orbit_datapath = datapath / f"tubeV{orbit_id}"
        vecs_orbit = np.loadtxt(orbit_datapath / f"{geometry_filename}")
        vecs = np.concatenate((vecs, vecs_orbit[orbit]), axis=0)

        # Load flat-field and dark-fields
        dark = trafo(imageio.imread(orbit_datapath / dark_filename))
        flat = np.zeros((2, proj_rows, proj_cols), dtype=np.float32)
        for idx, fn in enumerate(flat_filenames):
            flat[idx] = trafo(imageio.imread(orbit_datapath / fn))
        flat = np.mean(flat, axis=0)

        # Load projection data directly on the big projection array
        projs_orbit = np.zeros((n_projs_orbit, proj_rows, proj_cols), dtype=np.float32)
        for idx, fn in enumerate(
            tqdm(projs_idx, desc=f"Loading images (tube {orbit_id})")
        ):
            projs_orbit[idx] = trafo(
                imageio.imread(orbit_datapath / f"scan_{fn:06}.tif")
            )

        # Preprocess the projection data
        projs_orbit -= dark
        projs_orbit /= flat - dark
        projs_orbit = np.clip(projs_orbit, 0, 1)
        np.log(projs_orbit, out=projs_orbit)
        np.negative(projs_orbit, out=projs_orbit)

        # Permute data to ASTRA convention
        projs_orbit = np.transpose(projs_orbit, (1, 0, 2))
        projs = np.concatenate((projs, projs_orbit), axis=1)
        del projs_orbit

    projs = np.ascontiguousarray(projs)
    return projs, vecs


def get_source_target_vec(vecs: np.ndarray):
    projs_rows = 972  # Image height
    projs_cols = 768  # Image width

    sources = []
    targets = []
    for idx in range(len(vecs)):
        src = vecs[idx, :3]  # X-ray source
        det = vecs[idx, 3:6]  # Center of the detector plane
        u = vecs[idx, 6:9]  # Basis vector one of the detector plane
        v = vecs[idx, 9:12]  # Basis vector two of the detector plane

        src = torch.from_numpy(src).to(torch.float32)
        det = torch.from_numpy(det).to(torch.float32)
        u = torch.from_numpy(u).to(torch.float32)
        v = torch.from_numpy(v).to(torch.float32)

        # Create a canonical basis for the detector plane
        rows = (
            torch.arange(-projs_rows // 2, projs_rows // 2) + 0.5
            if projs_rows % 2 == 0
            else 1.0
        )
        cols = (
            torch.arange(-projs_cols // 2, projs_cols // 2) + 0.5
            if projs_cols % 2 == 0
            else 1.0
        )

        # Change of basis to u and v from the dataset
        i, j = torch.meshgrid(rows, cols, indexing="ij")
        x = torch.einsum("ij, n -> ijn", j, -u)
        y = torch.einsum("ij, n -> ijn", i, v)

        # Move the center of the detector plane to `det`
        source = src
        target = det + x + y
        source = source.expand(target.shape)
        sources.append(source.flip([1, 2]))
        targets.append(target.flip([1, 2]))

    return sources, targets


def render_drr(renderer, volume, vecs):
    subject = read("../data/Walnut8/gt.nii.gz")
    drr = DRR(
        subject,
        sdd=199.006188,
        height=972,
        width=768,
        delx=0.074800,
        renderer=renderer,
    ).cuda()

    s, t = get_source_target_vec(vecs[3:4])
    s = torch.stack(s)
    t = torch.stack(t)
    s = s.view(1, -1, 3).cuda()
    t = t.view(1, -1, 3).cuda()

    splits = 16
    cutoff = 972 * 768 // splits

    x = []
    with torch.no_grad():
        for jdx in range(splits):
            tmp = drr.render(
                torch.load(volume, weights_only=False)["tensors"]["est"]
                .squeeze()
                .cuda(),
                s[:, jdx * cutoff : (jdx + 1) * cutoff],
                t[:, jdx * cutoff : (jdx + 1) * cutoff],
            )
            x.append(tmp)

    x = torch.cat(x, dim=-1)
    x = drr.reshape_transform(x, 1)

    return x.cpu()


def render_astra(volume, vecs):
    subject = read(volume)

    vol_geom = astra.create_vol_geom(
        *subject.density.spatial_shape,
        subject.density.get_bounds()[0][0],
        subject.density.get_bounds()[0][1],
        subject.density.get_bounds()[1][0],
        subject.density.get_bounds()[1][1],
        subject.density.get_bounds()[2][0],
        subject.density.get_bounds()[2][1],
    )

    # Create projection data from this
    proj_geom = astra.create_proj_geom("cone_vec", 972, 768, vecs[3:4])
    proj_id, proj_data = astra.create_sino3d_gpu(
        subject.volume.data.squeeze().numpy(),
        proj_geom,
        vol_geom,
    )

    # Display a single projection image
    proj = proj_data[:, 0]
    return torch.from_numpy(proj)[None, None]

def render_nerf(volume, vecs):
    subject = read("../data/Walnut8/gt.nii.gz")
    drr = DRR(
        subject,
        sdd=199.006188,
        height=972,
        width=768,
        delx=0.074800,
    ).cuda()

    s, t = get_source_target_vec(vecs[3:4])
    s = torch.stack(s)
    t = torch.stack(t)
    s = s.view(1, -1, 3).cuda()
    t = t.view(1, -1, 3).cuda()

    splits = 16
    cutoff = 972 * 768 // splits

    x = []
    with torch.no_grad():
        for jdx in range(splits):
            tmp = drr.render(
                torch.load(volume, weights_only=False).detach().squeeze().cuda() / 1000,
                s[:, jdx * cutoff : (jdx + 1) * cutoff],
                t[:, jdx * cutoff : (jdx + 1) * cutoff],
            )
            x.append(tmp)

    x = torch.cat(x, dim=-1)
    x = drr.reshape_transform(x, 1)

    return x.cpu()

def evaluate(true, pred):
    return (
        PSNRMetric(true.max())(true, pred).item(),
        SSIMMetric(2, true.max())(true, pred).item(),
        MSEMetric()(true, pred).item(),
    )


def main(walnut_id, n_view):
    projs, vecs = load(
        Path(f"../data/Walnut{walnut_id}/Projections"),
        972,
        768,
        subsample=30,
        orbits_to_recon=[3],
    )
    true = torch.from_numpy(projs[:, 3])[None, None]
    results = []
    for algorithm in ["trilinear", "siddon", "nesterov", "sirt", "cgls", "fdk"]:

        # Render the novel view
        if algorithm in ["trilinear", "siddon"]:
            volume = list(
                Path("../results/final_v2").glob(
                    f"walnut{walnut_id}_{n_view}_{algorithm}*.pt"
                )
            )[0]
            pred = render_drr(algorithm, volume, vecs)


        else:
            volume = f"../baselines/Walnut{walnut_id}/{n_view}/{algorithm}.nii.gz"
            pred = render_astra(volume, vecs)

        # Calculate metrics
        metrics = evaluate(true, pred)
        for metric, value in zip(["psnr", "ssim", "mse"], metrics):
            results.append([walnut_id, n_view, algorithm, metric, value])

    df = pd.DataFrame(
        results, columns=["walnut_id", "n_view", "algorithm", "metric", "value"]
    )
    df.to_csv(f"../csvs/novel_views/{walnut_id}_{n_view}.csv", index=False)


if __name__ == "__main__":

    walnut_ids = list(range(3, 43))
    n_views = [5, 10, 15, 20, 30, 60]

    walnut_id = []
    n_view = []
    for w in walnut_ids:
        for n in n_views:
            walnut_id.append(w)
            n_view.append(n)

    executor = submitit.AutoExecutor(folder="logs")
    executor.update_parameters(
        name="walnut",
        gpus_per_node=1,
        mem_gb=10,
        slurm_array_parallelism=len(walnut_id),
        slurm_partition="2080ti",
        timeout_min=10_000,
    )
    jobs = executor.map_array(main, walnut_id, n_view)
