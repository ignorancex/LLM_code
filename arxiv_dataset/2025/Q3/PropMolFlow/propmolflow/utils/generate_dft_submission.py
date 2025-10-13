import os
from rdkit import Chem

# Settings
prop = 'gap'
input_sdf = f"./sdf-files/{prop}_in.sdf" # your generated sdf file for DFT calculation
base_dir = f"{prop}/gaussian_jobs"
nprocs = 8
mem = "16GB"
route_line = "# B3LYP/6-31G(2df,p) freq" # single point calculation without relaxation
# route_line = "# B3LYP/6-31G(2df,p) Opt Freq" # relaxation job
account = "genai-mingjieliu"

# Make base output directory
os.makedirs(base_dir, exist_ok=True)

suppl = Chem.SDMolSupplier(input_sdf, removeHs=False)
atomic_numbers = {
    'H': 1, 'He': 2,
    'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
    'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18
}

from rdkit.Chem import rdmolops
def get_total_electrons(mol):
    total_atomic_number = sum(atomic_numbers.get(atom.GetSymbol(), 0) for atom in mol.GetAtoms())
    formal_charge = rdmolops.GetFormalCharge(mol)
    return total_atomic_number - formal_charge

for i, mol in enumerate(suppl):
    if mol is None:
        continue

    total_electrons = get_total_electrons(mol)
    if total_electrons % 2 != 0:
        print(f"Skipping molecule {i}: odd number of electrons ({total_electrons})")
        continue

    mol_name = "mol"
    job_name = f"{mol_name}_{i}"
    job_dir = os.path.join(base_dir, job_name)
    os.makedirs(job_dir, exist_ok=True)

    com_path = os.path.join(job_dir, f"{job_name}.com")
    slurm_path = os.path.join(job_dir, f"{job_name}.slurm")

    conf = mol.GetConformer()

    # Write Gaussian input (.com) file
    with open(com_path, 'w') as f:
        # f.write(f"%NProcShared={nprocs}\n")
        # f.write(f"%Mem={mem}\n")
        f.write(f"%Chk={job_name}.chk\n")
        f.write(f"{route_line}\n\n")
        f.write(f"{job_name}\n\n")
        f.write("0 1\n")
        for atom in mol.GetAtoms():
            pos = conf.GetAtomPosition(atom.GetIdx())
            symbol = atom.GetSymbol()
            f.write(f"{symbol} {pos.x:.6f} {pos.y:.6f} {pos.z:.6f}\n")
        f.write("\n")

    # Write SLURM job script
    with open(slurm_path, 'w') as f:
        f.write(f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={job_name}.out
#SBATCH --error={job_name}.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={nprocs}
#SBATCH --mem={mem}
#SBATCH --time=4:00:00
#SBATCH --distribution=cyclic:cyclic
#SBATCH --output=random%j.log 
#SBATCH --account={account}
#SBATCH --qos={account}

module load gaussian/16  # Change to your cluster's module

g16 < {job_name}.com > output.log
""")

print(f"Prepared: {job_name} in {job_dir}")
