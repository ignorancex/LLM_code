import functools
import itertools
import os
import sys
from typing import Callable, Optional, Dict

import pandas as pd

from gravann.training import training as trainer
from gravann.util import get_asteroid_bounding_box, enableCUDA


def patch_legacy_config(cfg: Dict) -> Dict:
    """This function patches the legacy config files to the new format"""
    if not isinstance(cfg["training"]["iterations"], list):
        cfg["training"]["iterations"] = [cfg["training"]["iterations"]]

    return cfg


def run(cfg: Dict, stop_running: Optional[Callable[[], bool]] = None, cuda_device: str = "0") -> None:
    """This function runs all the permutations of above settings
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
    results_df = pd.DataFrame()

    enableCUDA()
    print("Using the following samples:", cfg["samples"])
    print("Will use device ", os.environ["TORCH_DEVICE"])

    # Legacy support for old config files
    cfg = patch_legacy_config(cfg)

    iterable_parameters = [
        cfg["seed"],
        cfg.get("pretrained_model", [""]),
        cfg["ground_truth"],
        cfg["noise"],
        cfg["training"]["iterations"],
        cfg["training"]["loss"],
        cfg["training"]["batch_size"],
        cfg["training"]["learning_rate"],
        cfg["model"]["encoding"],
        cfg["model"]["activation"],
        cfg["model"]["hidden_layers"],
        cfg["model"]["n_neurons"],
        cfg["siren"]["omega"],
        cfg["target_sampling"]["method"],
        cfg["target_sampling"]["domain"]
    ]

    combi = functools.reduce(lambda x, y: x * y, map(lambda x: len(x), iterable_parameters)) * len(cfg["samples"])
    print(f"#### - TOTAL AMOUNT OF ITERATIONS {combi}")

    for sample in cfg["samples"]:
        print(f"###### - SAMPLE START {sample}")
        if cfg["integration"]["limit_domain"]:
            cfg["integration"]["domain"] = get_asteroid_bounding_box(asteroid_pk_path=f"3dmeshes/{sample}.pk")
        for it, (
                seed, pretrained_model, ground_truth, noise,
                iterations, loss, batch_size, learning_rate,
                encoding, activation, hidden_layers, n_neurons,
                omega,
                sample_method, sample_domain,
        ) in enumerate(itertools.product(*iterable_parameters)):
            print("######## - SINGLE RUN START")
            csv_checkpoint_path = os.path.join("results", cfg['name'], f"results_checkpoint_{it:04d}.csv")
            if os.path.exists(csv_checkpoint_path):
                results_df = pd.read_csv(csv_checkpoint_path)
                print("######## - LOADED RUN FROM CHECKPOINT")
                continue
            run_results = trainer.run_training_configuration({
                ########################################################################################################
                # Name of the sample and other administrative stuff like the chosen seed
                ########################################################################################################
                "sample": sample,
                "pretrained_model": pretrained_model,
                "output_folder": f"{os.path.join('results', cfg['name'])}",
                "run_id": it,
                "plotting_points": cfg["plotting_points"],
                "seed": seed,
                ########################################################################################################
                # Chosen Ground Truth
                ########################################################################################################
                "ground_truth": ground_truth,
                "resolution": cfg.get("resolution", "100%"),
                ########################################################################################################
                # Training configuration & Validation Configuration
                ########################################################################################################
                "loss": loss,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "iterations": iterations,
                "validation_points": cfg["training"]["validation_points"],
                "validation_ground_truth": cfg["training"]["validation_ground_truth"],
                "use_acceleration": cfg["model"]["use_acceleration"],
                "validation_sampling_altitudes": cfg["training"]["validation_sampling_altitudes"],
                "validation_batch_size": cfg["training"].get("validation_batch_size", 100),
                ########################################################################################################
                # Model Configuration
                ########################################################################################################
                "encoding": encoding,
                "activation": activation,
                "hidden_layers": hidden_layers,
                "n_neurons": n_neurons,
                "model_type": cfg["model"]["type"],
                ########################################################################################################
                # Sirene Configuration
                ########################################################################################################
                "omega": omega,
                ########################################################################################################
                # Target Point Sampling Configuration
                ########################################################################################################
                "sample_method": sample_method,
                "sample_domain": sample_domain,
                ########################################################################################################
                # Noise Configuration
                ########################################################################################################
                "noise": noise,
                ########################################################################################################
                # Integration Configuration
                ########################################################################################################
                "integrator": "trapezoid",
                "integration_points": cfg["integration"]["points"],
                "integration_domain": cfg["integration"]["domain"]
            })
            results_df = results_df.append(run_results, ignore_index=True)
            results_df.to_csv(csv_checkpoint_path, index=False)
            print("######## - SINGLE RUN DONE")
            if stop_running is not None and stop_running():
                print("Requested to stop earlier")
                sys.exit(0)
        print(f"###### - SAMPLE {sample} DONE")
    print("#### - ALL ITERATIONS DONE")
    print(f"Writing results csv to {cfg['name']}")
    if os.path.isfile(os.path.join("results", cfg['name'], "results.csv")):
        previous_results = pd.read_csv(os.path.join("results", cfg['name'], "results.csv"))
        results_df = pd.concat([previous_results, results_df])
    results_df.to_csv(os.path.join("results", cfg['name'], "results.csv"), index=False)
    print("#### - EVERYTHING DONE")
