# imports
import os
import subprocess
import sys
sys.path.append(os.path.abspath(os.path.join("..")))
sys.path.append(os.path.abspath(os.path.join("../src")))
from dataset import DOMAINS

# Define constants for default values

# domain name
DOMAIN = "Science"
DOWNSAMPLED = False

# models
MODEL = "gpt2"
# MODEL = "pythia-6.9b"
# MODEL = "Llama-3.2-1B"
# MODEL = "Llama-3.1-8B"

# model folder
MODEL_FOLDER = f"{MODEL}_full"

# experiment name
# EXPERIMENT = "copyVSfact"
EXPERIMENT = "copyVSfactQnA"
# EXPERIMENT = "copyVSfactDomain"

# if EXPERIMENT == "copyVSfactDomain":
#     MODEL_FOLDER += f"_{DOMAIN}"

scripts = [
    "plot_logit_lens_fig_2.py",
    "plot_logit_attribution_fig_3_4a.py",
    "plot_head_pattern_fig_4b.py",
    # "plot_ablation_fig_5.py"
]

def plot_results():
    for script in scripts:
        # for domain in DOMAINS:
        command = [
            "python",
            f"../plotting_scripts/{script}",
            "--model", MODEL,
            "--experiment", EXPERIMENT,
            "--model_folder", MODEL_FOLDER,
            # "--domain", domain,
        ]
        if DOWNSAMPLED:
            command.append("--downsampled")
        print(f"Running command: {' '.join(command)}")
        subprocess.run(command)
        print()


if __name__ == "__main__":
    plot_results()
