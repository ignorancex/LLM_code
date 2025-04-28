import os
import subprocess
from typing import Optional


def submit_slurm_job(
    num_embryos: int,
    metallicity_limit: Optional[float] = None,
) -> None:
    # determine suffix based on metallicity presence
    if metallicity_limit is None:
        suffix = "no_limit"
        metallicity_arg = ""
    else:
        suffix = f"limit_{metallicity_limit}"
        metallicity_arg = f"--metallicity_limit {metallicity_limit}"

    # template for the Slurm batch script
    slurm_template = f"""#!/bin/bash
#SBATCH --job-name=mock_obs_{num_embryos}_{suffix}
#SBATCH --output=mock_obs_out/%x.out
#SBATCH --error=mock_obs_out/%x.err
#SBATCH --time=03:00:00
#SBATCH --partition=regular
#SBATCH --nodes=1
#SBATCH --mem=8GB

# load virtual env
source ~/.bashrc
mamba deactivate
mamba activate plato

# Run the script
python create_mock_observations.py {num_embryos} {metallicity_arg}
    """

    # write the job script to a file
    script_filename = f"slurm_job_{num_embryos}_{suffix}.sh"
    with open(script_filename, "w") as file:
        file.write(slurm_template)

    # submit the job using sbatch
    result = subprocess.run(["sbatch", script_filename], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"Successfully submitted job for num_embryos={num_embryos} with {suffix}")
    else:
        print(
            "Failed to submit job for num_embryos="
            f"{num_embryos} with {suffix}: {result.stderr}"
        )

    # remove the script file
    os.remove(script_filename)


# calls for different numbers of embryos and both conditions
num_embryos_values = [10, 20, 50, 100]
for num_embryos in num_embryos_values:
    submit_slurm_job(num_embryos)
    submit_slurm_job(num_embryos, -0.6)
