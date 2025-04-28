from pathlib import Path

import pandas as pd
import submitit
import torch
from monai.metrics import MSEMetric, PSNRMetric, SSIMMetric
from torchio import ScalarImage


def pcc(x, y):
    return torch.dot(x.flatten(), y.flatten()) / (x.norm() * y.norm())


def evaluate(walnut_id, n_view):
    # Load the true walnut
    true = ScalarImage(f"../data/Walnut{walnut_id}/gt.nii.gz").data[None].cuda()

    # Load the predicted volumes
    preds = {}
    results = Path("../results/final_v2")
    nn_dir = Path('../SAX-NeRF/logs')
    sax_dir = nn_dir / 'Lineformer_walnut'
    naf_dir = nn_dir / 'naf_walnut'
    for algorithm in ["fdk", "cgls", "sirt", "nesterov", "siddon", "trilinear", "sax", "naf"]:
        if algorithm == "siddon":
            try:
                pred = list(results.glob(f"walnut{walnut_id}_{n_view}_siddon*"))[0]
                pred = torch.load(pred, weights_only=False)["tensors"]["est"].detach()[None, None]
            except:
                continue
        elif algorithm == "trilinear":
            try:
                pred = list(results.glob(f"walnut{walnut_id}_{n_view}_trilinear*"))[0]
                pred = torch.load(pred, weights_only=False)["tensors"]["est"].detach()[None, None]
            except:
                continue
        elif algorithm == "sax":
            try:
                pred = list(sax_dir.glob(f'walnut_{walnut_id}/w_{walnut_id}_{n_view}/*/volume.pt'))[0]
                pred = torch.load(pred, weights_only=False).cpu().detach()[None, None] / 1000
            except:
                continue
        elif algorithm == "naf":
            try:
                pred = list(naf_dir.glob(f'walnut_{walnut_id}/w_{walnut_id}_{n_view}/*/volume.pt'))[0]
                pred = torch.load(pred, weights_only=False).cpu().detach()[None, None] / 1000
            except:
                continue
        else:
            pred = ScalarImage(
                f"../baselines/Walnut{walnut_id}/{n_view}/{algorithm}.nii.gz"
            ).data[None]
        preds[algorithm] = pred.cuda()

    # Make the metrics
    psnr = PSNRMetric(true.max())
    ssim = SSIMMetric(3, true.max())
    mse = MSEMetric()

    # Calculate the metrics
    evaluation_metrics = []
    for metric, fn in {"psnr": psnr, "ssim": ssim, "mse": mse, "pcc": pcc}.items():
        for algorithm, pred in preds.items():
            value = fn(pred, true).item()
            evaluation_metrics.append([walnut_id, n_view, algorithm, metric, value])

    # Save the results
    df = pd.DataFrame(evaluation_metrics, columns=["walnut_id", "n_view", "algorithm", "metric", "value"])
    df.to_csv(f"../csvs/baseline_recons/{walnut_id}_{n_view}.csv", index=False)


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
        mem_gb=8,
        slurm_array_parallelism=len(walnut_id),
        slurm_partition="2080ti",
        timeout_min=10_000,
    )
    jobs = executor.map_array(evaluate, walnut_id, n_view)