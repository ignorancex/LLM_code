"""Run for multiple hyperparameter values on slurm cluster."""

import json
import os
from copy import deepcopy
from itertools import product
from typing import Text, Sequence, Dict, Any

from absl import app
from absl import flags

flags.DEFINE_string("experiment_name", None,
                    "The name of the experiment (used for output folder).")
flags.DEFINE_string("result_dir", "/home/haicu/elisabath.ailer/Projects/UnderspecifiedIV/Output/",
                    "Base directory for all results.")
flags.DEFINE_bool("gpu", True, "Whether to use GPUs.")
flags.mark_flag_as_required("experiment_name")
FLAGS = flags.FLAGS

# Will create a *cartesian grid* of all combinations of these
sweep = {
  "p": [50, 150],
  "d": [30],
  "d_id": [20],
  "d_max": [4],
  "n_rounds": [5, 6],
  "seed": [1911]
}

# Other flags that should be fixed for all runs (won'y be in result folder names)
fixed_flags = {
  "n": 1000,
  "n_runs": 250
}



# Some values and paths to be set
user = "elisabath.ailer"  # CHANGE THIS
project = "UnderspecifiedIV"
executable = f"/home/haicu/{user}/miniconda3/envs/insufficient_iv/bin/python"  # IF YOU'RE USING MINICONDA
run_file = f"/home/haicu/{user}/Projects/{project}/Code/src/run_optimization_slurm.py"  # MAY NEED TO UPDATE

# Specify the resource requirements *per run*
num_cpus = 4
num_gpus = 1
mem_mb = 16000
max_runtime = "00-16:00:00"


def get_output_name(value_dict: Dict[Text, Any]) -> Text:
  """Get the name of the output directory."""
  name = ""
  for k, v in value_dict.items():
    name += f"_{k}{v}"
  return name[1:]


def get_flag(key: Text, value: Any) -> Text:
  if isinstance(value, bool):
    return f' --{key}' if value else f' --no{key}'
  else:
    return f' --{key} {value}'


def submit_all_jobs(args: Sequence[Dict[Text, Any]]) -> None:
  """Genereate submit scripts and launch them."""
  # Base of the submit file
  base = list()
  base.append(f"#!/bin/bash")
  base.append("")
  base.append(f"#SBATCH -J {project}{'_gpu' if FLAGS.gpu else ''}")
  base.append(f"#SBATCH -c {num_cpus}")
  base.append(f"#SBATCH --mem={mem_mb}")
  base.append(f"#SBATCH -t {max_runtime}")
  base.append(f"#SBATCH --nice=10000")
  if FLAGS.gpu:
    base.append(f"#SBATCH -p gpu_p")
    base.append(f"#SBATCH --qos gpu")
    base.append(f"#SBATCH --gres=gpu:{num_gpus}")
    base.append(f"#SBATCH --exclude=icb-gpusrv0[1-2]")  # keep for interactive
  else:
    base.append(f"#SBATCH -p cpu_p")

  for i, arg in enumerate(args):
    lines = deepcopy(base)
    output_name = get_output_name(arg)

    # Directory for slurm logs
    result_dir = os.path.join(FLAGS.result_dir, FLAGS.experiment_name)
    logs_dir = os.path.join(result_dir, output_name)

    # Create directories if non-existent (may be created by the program itself)
    if not os.path.exists(logs_dir):
      os.makedirs(logs_dir)

    # The output, logs, and errors from running the scripts
    logs_name = os.path.join(logs_dir, "slurm")
    lines.append(f"#SBATCH -o {logs_name}.out")
    lines.append(f"#SBATCH -e {logs_name}.err")

    # Queue job
    lines.append("")
    runcmd = executable
    runcmd += " "
    runcmd += run_file
    # ASSUMING RUNFILE TAKES THESE THREE ARGUMENTS
    runcmd += f' --fig_path {result_dir}'
    # Sweep arguments
    for k, v in arg.items():
      runcmd += get_flag(k, v)
    # Adaptive arguments (depending on sweep value)
    #for adaptive_k, func in adaptive_flags.items():
    #  runcmd += get_flag(adaptive_k, func(arg))
    # Fixed arguments
    for k, v in fixed_flags.items():
      runcmd += get_flag(k, v)

    lines.append(runcmd)
    lines.append("")

    # Now dump the string into the `run_all.sub` file.
    with open("run_job.cmd", "w") as file:
      file.write("\n".join(lines))

    print(f"Submitting {i}...")
    os.system("sbatch run_job.cmd")


def main(_):
  """Initiate multiple runs."""
  values = list(sweep.values())
  args = list(product(*values))
  keys = list(sweep.keys())
  args = [{keys[i]: arg[i] for i in range(len(keys))} for arg in args]
  n_jobs = len(args)
  sweep_dir = os.path.join(FLAGS.result_dir, FLAGS.experiment_name)

  # Create directories if non-existent
  if not os.path.exists(sweep_dir):
    os.makedirs(sweep_dir)
  print(f"Store sweep dictionary to {sweep_dir}...")
  with open(os.path.join(sweep_dir, "sweep.json"), 'w') as fp:
    json.dump(sweep, fp, indent=2)

  print(f"Generate all {n_jobs} submit script and launch them...")
  submit_all_jobs(args)

  print(f"DONE")


if __name__ == "__main__":
  app.run(main)
