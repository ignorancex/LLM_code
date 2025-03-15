# File: main.py
"""
Top-level orchestrator or entry-point script.

This script:
1. Creates backup folders and moves existing results/logs there.
2. Reads the config.json file.
3. Depending on the CLI argument:
   - If an environment name is passed, runs the experiments only for that environment.
   - Otherwise, runs the experiments for each environment in config:
       * baseline experiment (train + PCMCI)
       * Gaussian-noise experiment (observations)
       * Gaussian-noise experiment (actions)
       * AR-noise experiment (observations)
       * AR-noise experiment (actions)
       * (Optionally dimension-dropping experiments if needed)
"""

import argparse
import json
import os
import pathlib
import time
from datetime import datetime

# Import your experiment scripts
from markovianess.experiments.baseline import run as run_baseline
from markovianess.experiments.noised_gaussian import run_noised_gaussian
from markovianess.experiments.noised_gaussian_action import run_noised_gaussian_action
from markovianess.experiments.noised_auto_regressive import run_noised_auto_regressive
from markovianess.experiments.noised_action_auto_regressive import run_noised_action_auto_regressive
from markovianess.experiments.dropped import run_dropped


def setup(root_path: pathlib.Path, config_path: str):
    """
    Moves old results/logs to a backup directory, then creates fresh
    directories for the new run. This ensures you start with empty
    'results' folders for each environment.
    """
    # Create top-level directories
    os.makedirs(root_path / "backup", exist_ok=True)
    os.makedirs(root_path / "results", exist_ok=True)

    results_path = root_path / "results"
    backup_path = root_path / "backup"

    # Create a unique backup dir (timestamp-based)
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    backup_dir = backup_path / timestamp
    os.makedirs(backup_dir, exist_ok=True)

    # Move existing files/folders from results/ into the backup_dir
    for item in os.listdir(results_path):
        old_path = results_path / item
        new_path = backup_dir / item
        os.rename(old_path, new_path)

    # If logs.txt exists at project root, move it too
    logs_path = root_path / "logs.txt"
    if logs_path.exists():
        os.rename(logs_path, backup_dir / "logs.txt")

    # Create subfolders for each environment as stated in config
    with open(config_path, "r") as f:
        config = json.load(f)
    environments = [env["name"] for env in config["environments"]]

    for environment in environments:
        (results_path / environment / "csv").mkdir(parents=True, exist_ok=True)
        (results_path / environment / "plots" / "baseline").mkdir(parents=True, exist_ok=True)
        (results_path / environment / "models").mkdir(parents=True, exist_ok=True)
        (results_path / environment / "pcmci").mkdir(parents=True, exist_ok=True)


def run_all_experiments_for_env(env_name, config_path, start_time):
    """
    Run the full pipeline (baseline, noised, etc.) for a single environment.
    """

    print(f"\n===== RUNNING BASELINE FOR ENV={env_name} =====")
    used_seed = run_baseline(config_path=config_path, env_name=env_name)
    baseline_end_time = time.perf_counter()
    print(f"Baseline time for {env_name}: {baseline_end_time - start_time:.2f} s")

    print(f"===== RUNNING NOISED GAUSSIAN ACTION FOR ENV={env_name} =====")
    run_noised_gaussian_action(config_path=config_path, env_name=env_name, baseline_seed=used_seed)
    gauss_action_end_time = time.perf_counter()
    print(f"Noised (Gaussian) action time for {env_name}: {gauss_action_end_time - baseline_end_time:.2f} s")

    print(f"===== RUNNING NOISED GAUSSIAN FOR ENV={env_name} =====")
    run_noised_gaussian(config_path=config_path, env_name=env_name, baseline_seed=used_seed)
    gauss_end_time = time.perf_counter()
    print(f"Noised (Gaussian) time for {env_name}: {gauss_end_time - gauss_action_end_time:.2f} s")

    print(f"===== RUNNING AR NOISED FOR ENV={env_name} =====")
    run_noised_auto_regressive(config_path, env_name=env_name, baseline_seed=used_seed)
    ar_end_time = time.perf_counter()
    print(f"AR Noised time for {env_name}: {ar_end_time - gauss_end_time:.2f} s")

    print(f"===== RUNNING NOISED ACTION AR FOR ENV={env_name} =====")
    run_noised_action_auto_regressive(config_path, env_name=env_name, baseline_seed=used_seed)
    ar_action_end_time = time.perf_counter()
    print(f"AR Noised Action time for {env_name}: {ar_action_end_time - ar_end_time:.2f} s")

    # If you also want dimension-dropping experiments, keep the line below:
    print(f"===== RUNNING DROPPED FOR ENV={env_name} =====")
    run_dropped(config_path, env_name, baseline_seed=used_seed)
    dropped_end_time = time.perf_counter()
    print(f"Dropped time for {env_name}: {dropped_end_time - ar_action_end_time:.2f} s")


def main(root_path: pathlib.Path, env_name=None):
    """
    Reads config.json, sets up fresh results directories, and runs
    the baseline + noised experiments for either:
      1) The single environment specified by env_name, or
      2) All environments in config (if env_name is None).
    """

    config_path = str(root_path / "config.json")

    # 1) Setup backup & fresh directories
    setup(root_path, config_path)

    # 2) Read config for environment(s)
    with open(config_path, "r") as f:
        config = json.load(f)

    start_time = time.perf_counter()

    if env_name is not None:
        # Run only for the requested environment
        # Check if it exists in config
        all_env_names = [e["name"] for e in config["environments"]]
        if env_name not in all_env_names:
            raise ValueError(
                f"Environment '{env_name}' not found in config. "
                f"Available: {all_env_names}"
            )
        run_all_experiments_for_env(env_name, config_path, start_time)

    else:
        # No environment specified => run for all in config
        for env_item in config["environments"]:
            run_all_experiments_for_env(env_item["name"], config_path, start_time)

    end_time = time.perf_counter()
    total_hours = ((end_time - start_time) / 60.0)/60.0
    print(f"All experiments finished in {total_hours:.2f} hours.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Top-level orchestrator for Markovianess experiments."
    )
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        help="Name of the environment to run. If not provided, runs all."
    )
    args = parser.parse_args()

    main(pathlib.Path().absolute(), env_name=args.env)