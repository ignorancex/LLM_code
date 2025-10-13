import os
import json
import argparse
import pickle
import torch
from chroma import Protein
from adam_pnp.src.reconstruction import AdaptiveMultiModalReconstruction
from adam_pnp.src.utils import plot_results_extended, _to_serialisable, _strip_unpicklable

def parse_args():
    parser = argparse.ArgumentParser(description='Adam-PnP: Adaptive Multi-Modal Protein Reconstruction')
    parser.add_argument('--config', type=str, default='configs/base_config.json', help='Path to config JSON file')
    parser.add_argument('--protein_path', type=str, required=True, help='Path to protein CIF file')
    parser.add_argument('--partial', action='store_true', help='Enable partial structure modality')
    parser.add_argument('--distances', action='store_true', help='Enable distances modality')
    parser.add_argument('--density', action='store_true', help='Enable density modality')
    parser.add_argument('--n_distances', type=int, help='Number of distance observations')
    parser.add_argument('--n_partial', type=int, help='Number of partial structure observations')
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--output_dir', type=str, default='./results', help='Output directory')
    parser.add_argument('--exp_name', type=str, help='Experiment name for output folder')
    return parser.parse_args()

def deep_merge(base, custom):
    for key, value in custom.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            deep_merge(base[key], value)
        else:
            base[key] = value

def main():
    args = parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    # Override with command-line arguments
    if args.partial: config["measurements"]["partial_structure"]["enabled"] = True
    if args.distances: config["measurements"]["distances"]["enabled"] = True
    if args.density: config["measurements"]["density"]["enabled"] = True
    if args.n_distances is not None: config["measurements"]["distances"]["n_distances"] = args.n_distances
    if args.n_partial is not None: config["measurements"]["partial_structure"]["n_observations"] = args.n_partial
    if args.seed is not None: config["seed"] = args.seed
    
    if args.exp_name:
        exp_name = args.exp_name
    else:
        modalities = []
        if config["measurements"]["partial_structure"]["enabled"]:
            n_obs = config["measurements"]["partial_structure"].get("n_observations", "all")
            modalities.append(f"P{n_obs}")
        if config["measurements"]["distances"]["enabled"]:
            n_dist = config["measurements"]["distances"]["n_distances"]
            modalities.append(f"D{n_dist}")
        if config["measurements"]["density"]["enabled"]:
            modalities.append("E")
        modalities_str = "_".join(modalities) if modalities else "unconditioned"
        exp_name = f"{modalities_str}_seed{config['seed']}"
    
    output_dir = os.path.join(args.output_dir, exp_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Running experiment: {exp_name}")
    print(f"Output will be saved to: {output_dir}")

    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("Loading protein...")
    protein = Protein.from_CIF(args.protein_path, device=config["device"])
    X_gt, C_gt, S_gt = protein.to_XCS(all_atom=False)
    mask = (torch.abs(C_gt.reshape(-1)) == 1)
    X_gt, C_gt, S_gt = X_gt[:, mask], C_gt[:, mask], S_gt[:, mask]
    X_gt = X_gt - X_gt.mean(dim=(0,1,2), keepdim=True)
    print(f"Protein '{os.path.basename(args.protein_path)}' with {X_gt.shape[1]} residues loaded to {config['device']}.")

    framework = AdaptiveMultiModalReconstruction(config)
    X_best, X_final, metrics, measurements = framework.run(X_gt, C_gt, S_gt)

    best_rmsd = metrics["best_rmsd_overall"]
    best_epoch = metrics["best_rmsd_epoch"]
    
    print(f"\nExperiment Finished. Best RMSD = {best_rmsd:.3f} Å (at epoch {best_epoch})")
    
    plot_path = os.path.join(output_dir, "reconstruction_summary.png")
    plot_results_extended(metrics, measurements, save_path=plot_path, 
                         enable_noise_estimation=config["enable_noise_estimation"],
                         algorithm="Adam-PnP")
    print(f"Saved summary plot to {plot_path}")

    try:
        Protein.from_XCS(X_best, C_gt, S_gt).to_PDB(os.path.join(output_dir, "reconstructed_best.pdb"))
        Protein.from_XCS(X_final, C_gt, S_gt).to_PDB(os.path.join(output_dir, "reconstructed_final.pdb"))
        Protein.from_XCS(X_gt, C_gt, S_gt).to_PDB(os.path.join(output_dir, "ground_truth.pdb"))
        print("Saved PDB files for best, final, and ground truth structures.")
    except Exception as e:
        print(f"Error saving PDB files: {e}")

    try:
        with open(os.path.join(output_dir, "metrics.pkl"), "wb") as f:
            pickle.dump(metrics, f)
        with open(os.path.join(output_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2, default=_to_serialisable)
        print("Saved metrics and configuration files.")
    except Exception as e:
        print(f"Error saving metrics/config: {e}")

    try:
        safe_measurements = {
            name: {
                "config": m.config,
                "data": _strip_unpicklable(m.data),
                "lipschitz_constant": m.noise_scheduler.L,
                "final_sigma_meas": m.get_learned_sigma_meas()
            }
            for name, m in measurements.items()
        }
        with open(os.path.join(output_dir, "measurements.pkl"), "wb") as f:
            pickle.dump(safe_measurements, f)
        print("Saved measurements data.")
    except Exception as e:
        print(f"Error saving measurements: {e}")

if __name__ == "__main__":
    main()