import os

# Create a directory to store the Slurm logs
if not os.path.exists("logs"):
    os.makedirs("logs")

for d_idx in range(100):
    d_name = f"{d_idx}"

    # Create the Slurm script content
    script_content = f"""#!/bin/bash
#SBATCH --job-name=gen-RDN0-{d_name}
#SBATCH --output=logs/2023-04-17-gen-RDN0-{d_name}.out
#SBATCH --error=logs/2023-04-17-gen-RDN0-{d_name}.err
#SBATCH --time=600:00
#SBATCH -p kipac
#SBATCH --nodes=1
#SBATCH --mem=16384
#SBATCH --cpus-per-task=1

conda init
conda activate nblensing

python gen_RDN0.py {d_idx}
"""

    # Write the Slurm script to a file
    script_filename = f"gen-RDN0-{d_name}.sh"
    with open(script_filename, "w") as f:
        f.write(script_content)

    # Submit the Slurm job
    os.system(f"sbatch {script_filename}")

    print(f"Submitted Slurm job for index {d_idx}")
