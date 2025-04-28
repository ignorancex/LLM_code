import os

pairs = [
    [0, 0],  # N0
    [0, 1],  # kappa
    [1, 0],  # kappa
    [0, 2],  # N1
    [1, 1],  # N1
    [2, 0],  # N1
    [0, 3],  # should vanish
    [1, 2],  # should vanish
    [2, 1],  # should vanish
    [3, 0],  # should vanish
    [0, 4],  # N2
    [1, 3],  # N2
    [2, 2],  # N2
    [3, 1],  # N2
    [4, 0],  # N2
    [-1, -1],  # QE
    [-2, -2],  # unlensed
]

# Create a directory to store the Slurm logs
if not os.path.exists("logs"):
    os.makedirs("logs")

for pair in pairs:
    # Extract the pair values
    pair_name = f"{pair[0]}-{pair[1]}"
    pair_args = f"{pair[0]} {pair[1]}"

    # Create the Slurm script content
    script_content = f"""#!/bin/bash
#SBATCH --job-name=gen-estimator-{pair_name}
#SBATCH --output=logs/2023-04-17-gen-estimator-{pair_name}.out
#SBATCH --error=logs/2023-04-17-gen-estimator-{pair_name}.err
#SBATCH --time=60:00
#SBATCH -p kipac
#SBATCH --nodes=1
#SBATCH --mem=8192
#SBATCH --cpus-per-task=1

conda init
conda activate nblensing

python gen_estimators_unmasked.py {pair_args}
"""

    # Write the Slurm script to a file
    script_filename = f"gen-estimator-{pair_name}.sh"
    with open(script_filename, "w") as f:
        f.write(script_content)

    # Submit the Slurm job
    os.system(f"sbatch {script_filename}")

    print(f"Submitted Slurm job for pair {pair_name}")
