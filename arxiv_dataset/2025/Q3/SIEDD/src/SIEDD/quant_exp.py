from pathlib import Path
import subprocess
import argparse
import os

PROJECT_ROOT = Path(__file__).absolute().parent.parent.parent
EXPERIMENT_SCRATCH = PROJECT_ROOT / "experiment_scratch"
EXPERIMENT_SCRATCH.mkdir(exist_ok=True)


def start_run(data_path: Path, save_path: Path, name: str):
    slurm_path = EXPERIMENT_SCRATCH / f"slurm_exp_cfg_{name}.sh"
    slurm_out_file = f"exp_cfg_{name}.out"
    args = f"--data_path {data_path} --save_path {save_path} --name {name}"
    cmd = f"uv run --env-file .env quant_test {args}"
    cmd = f"export WANDB_GROUP=ptq_sweep\n{cmd}"

    slurm_template = (PROJECT_ROOT / "slurm_template.sh").read_text()
    slurm = slurm_template.replace("{{PROJECT_ROOT}}", str(PROJECT_ROOT))
    slurm = slurm.replace("{{CMD}}", cmd)
    slurm_path.write_text(slurm)
    os.chdir(EXPERIMENT_SCRATCH)
    subprocess.run(["sbatch", str(slurm_path), "-o", slurm_out_file])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=Path)
    parser.add_argument("--save_path", type=Path)
    parser.add_argument("--name", type=str)
    args = parser.parse_args()
    data_path: Path = args.data_path
    save_path: Path = args.save_path
    name: str = args.name
    start_run(data_path, save_path, name)


if __name__ == "__main__":
    main()
