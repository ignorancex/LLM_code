import os
import subprocess
import argparse


def submit_slurm_job(
    notebook_name: str,
    number_of_cores: int = 1,
) -> None:
    # template for the Slurm batch script
    slurm_template = f"""#!/bin/bash
#SBATCH --job-name={notebook_name}
#SBATCH --output=notebook_out/%x.out
#SBATCH --error=notebook_out/%x.err
#SBATCH --time=03:00:00
#SBATCH --partition=regular
#SBATCH --nodes=1
#SBATCH --cpus-per-task={str(number_of_cores)}
#SBATCH --mem=32GB

# load virtual env
source ~/.bashrc
mamba deactivate
mamba activate plato

# Run the script
python run_notebook.py {notebook_name}
    """

    # Write the job script to a file
    script_filename = f"slurm_job_{notebook_name}.sh"
    with open(script_filename, "w") as file:
        file.write(slurm_template)

    # Submit the job using sbatch
    result = subprocess.run(["sbatch", script_filename], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"Successfully submitted job for notebook {notebook_name}")
    else:
        print(f"Failed to submit job for notebook {notebook_name}: {result.stderr}")

    # remove the script file
    os.remove(script_filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Submit Slurm jobs for Jupyter notebooks."
    )
    parser.add_argument(
        "notebooks",
        nargs="*",
        default=[
            "03_plato_fields",
            "04_stellar_sample",
            "05_detection_efficiency",
            "06_planet_populations",
            "07_planet_population_metrics",
            "08_heatmaps",
            "A_survey_comparison",
            "B_stellar_variability",
            "C_halo_special_target_list",
        ],
        help="List of notebook names to submit as Slurm jobs.",
    )
    parser.add_argument(
        "--number_of_cores",
        type=int,
        default=1,
        help="Number of cores to use.",
    )

    args = parser.parse_args()
    notebooks = args.notebooks

    if isinstance(notebooks, str):
        notebooks = [notebooks]

    for notebook in notebooks:
        submit_slurm_job(notebook)
