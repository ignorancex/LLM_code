import torch
import numpy as np
import matplotlib.pyplot as plt

EPS = 1e-12

def safe_sqrt(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.clamp(x, min=EPS))

def fft_density(density: torch.Tensor, norm='backward') -> torch.Tensor:
    return torch.fft.fftn(density, norm=norm)

def check_nan_inf(tensor: torch.Tensor, name: str = "tensor") -> torch.Tensor:
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        print(f"Warning: NaN/Inf detected in {name}, replacing with zeros")
        tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1e6, neginf=-1e6)
    return tensor

def _to_serialisable(obj):
    if isinstance(obj, torch.Tensor):
        return obj.cpu().tolist()
    return obj

def _strip_unpicklable(state: dict) -> dict:
    safe = {}
    for k, v in state.items():
        if k in {"backbone_network", "multiply_R", "multiply_R_inverse", "multiply_covariance"}:
            continue
        if isinstance(v, torch.Tensor):
            safe[k] = v.cpu()
        else:
            safe[k] = v
    return safe

def plot_results_extended(metrics, measurements, save_path=None, enable_noise_estimation=True, algorithm="Adam-PnP"):
    epochs = metrics["epoch"]
    best_r = [min(r) for r in metrics["rmsd"]]
    mean_r = [float(np.mean(r)) for r in metrics["rmsd"]]
    best_rmsd_curve = metrics.get("best_rmsd", best_r)

    n = len(measurements)
    n_rows = 3 if enable_noise_estimation and n > 0 else 2
    fig, axes = plt.subplots(n_rows, max(2, n), figsize=(6 * max(2, n), 5 * n_rows), squeeze=False)
    
    ax = axes[0,0]
    ax.plot(epochs, best_r, "b-", label="Best in Population")
    ax.plot(epochs, mean_r,"b--", label="Mean in Population")
    ax.plot(epochs, best_rmsd_curve, "r-", linewidth=2, label="Best Overall")
    ax.set(title="Backbone Ca-RMSD", xlabel="Epoch", ylabel="RMSD (Å)")
    ax.legend(); ax.grid(True)

    ax = axes[0,1]
    for k,v in metrics.items():
        if k.startswith("loss_"):
            ax.semilogy(epochs, v, label=k.replace("loss_",""))
    ax.set(title="Loss Evolution", xlabel="Epoch", ylabel="Log Loss")
    if len(measurements) > 0: ax.legend()
    ax.grid(True)

    if enable_noise_estimation and n > 0:
        for i, name in enumerate(measurements.keys()):
            ax = axes[1,i]
            ax.plot(epochs, metrics[f"sigma_meas_{name}"], "k-", label="Estimated sigma")
            true_noise = measurements[name].config.get("true_noise_level",0.0)
            ax.axhline(true_noise, color="g", linestyle="--", label="True sigma")
            ax.set(title=f"'{name}' Noise Level", xlabel="Epoch", ylabel="Sigma")
            ax.legend(); ax.grid(True)

        for i, name in enumerate(measurements.keys()):
            ax = axes[2,i]
            ax.plot(epochs, metrics[f"lipschitz_{name}"], "m-")
            ax.set(title=f"'{name}' Lipschitz Constant", xlabel="Epoch", ylabel="L")
            ax.grid(True)
    else:
        for i, name in enumerate(measurements.keys()):
            ax = axes[1,i]
            fixed_noise = measurements[name].config.get("noise_level", 1.0)
            true_noise = measurements[name].config.get("true_noise_level", 0.0)
            ax.axhline(fixed_noise, color="k", linestyle="-", label=f"Fixed sigma={fixed_noise:.2f}")
            ax.axhline(true_noise, color="g", linestyle="--", label=f"True sigma={true_noise:.2f}")
            ax.set(title=f"'{name}' (Fixed Noise)", xlabel="Epoch", ylabel="sigma")
            ax.legend(); ax.grid(True)
            ax.set_ylim([0, max(fixed_noise, true_noise) * 1.5])

    plt.suptitle(f"{algorithm} Reconstruction Results")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()