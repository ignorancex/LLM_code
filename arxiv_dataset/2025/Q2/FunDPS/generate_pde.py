import json
import os
from datetime import datetime

import numpy as np
import torch
import wandb

from utils.yaml_config import Config, process_arguments
from generation.dps import PDESolverDPS
from generation.daps import PDESolverDAPS
from generation.dps_multires import PDESolverDPS_MultiRes


def get_solver(config):
    """Get the appropriate solver based on config."""
    solver_type = config["guidance"]["type"].lower()
    if solver_type == "dps":
        return PDESolverDPS(config)
    elif solver_type == "dps_multires":
        return PDESolverDPS_MultiRes(config)
    elif solver_type == "daps":
        return PDESolverDAPS(config)
    else:
        raise ValueError(f"Unknown solver type: {solver_type}")


if __name__ == "__main__":
    # Parse arguments and load config
    args = process_arguments()
    config = Config(args)

    # Set the seed
    seed = config["seed"]
    if seed is None:
        seed = torch.randint(1 << 31, size=[]).item()
        config.update("seed", seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Create output directory
    formatted_time = datetime.now().strftime("%m%d_%H%M%S")
    desc = [formatted_time, config["dataset"], str(config["resolution"]), config["guidance"]["type"]]
    if config["name"]:
        desc.append(config["name"])
    desc = "G" + "-".join(desc)
    save_dir = os.path.join(config["outdir"], desc)
    os.makedirs(save_dir)
    config.update("outdir", save_dir)

    # Initialize wandb if enabled
    if config["wandb"]:
        wandb.init(
            config=config.to_dict(),
            name=desc,
            mode=config["wandb"],
        )
        wandb.run.log_code(root=".")

    # Save the config
    with open(os.path.join(save_dir, "config.json"), "wt") as f:
        json.dump(config.to_dict(), f, indent=2)

    # Initialize and run solver
    solver = get_solver(config)
    solver.generate()

    if config["wandb"] == "offline":
        wandb.finish()
