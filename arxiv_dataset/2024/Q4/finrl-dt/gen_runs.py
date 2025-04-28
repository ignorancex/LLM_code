import os
import glob

# Create the 'runs' directory if it doesn't exist
if not os.path.exists('runs'):
    os.makedirs('runs')

# Define the list of algorithms
algorithms = ['a2c', 'ddpg', 'td3', 'ppo', 'sac', 'ensemble']

# Base script template
base_script = '''#!/bin/bash
# Auto-generated run script for {algo_upper}

conda activate finrl-dt

export TRANSFORMERS_OFFLINE=0
export TOKENIZERS_PARALLELISM=0

seeds=(20742 55230 85125 96921 67851)
gpu=0 # GPU identification number

drl_alg={algo} # The deep reinforcement learning algorithm we are targeting to clone the behavior
model_type=dt

# Training parameters
lr=1e-3
weight_decay=1e-5 # For AdamW optimizer
dropout=0.1
warmup_steps=2500
num_steps=1000 # Total number of training steps; i.e., how many times we call env.step() for training

# Environment parameters
env=stock_trading
sample_ratio=1
K=20  # Context length for the decision transformer model
dataset_path={train_trajectory_file} # Path to the trajectory data
test_trajectory_file={test_trajectory_file}

# Device
device='cuda' # or 'cpu'

# Pretrained language model
pretrained_lm="/home/gridsan/syun/gpt2_model" # Path to the downloaded GPT-2 model

use_pretrained_lm=true
lora=true

for seed in "${{seeds[@]}}"; do
    exp_name="${{drl_alg}}_${{model_type}}_lora_gpt2_${{seed}}"
    outdir="/home/gridsan/syun/finrl-dt/checkpoints/${{exp_name}}"

    # Run the experiment
    CUDA_VISIBLE_DEVICES=${{gpu}} python ~/finrl-dt/experiment.py \\
        --device ${{device}} \\
        --env ${{env}} \\
        --dataset_path ${{dataset_path}} \\
        --seed ${{seed}} \\
        --K ${{K}} \\
        --learning_rate ${{lr}} \\
        --num_steps ${{num_steps}} \\
        --weight_decay ${{weight_decay}} \\
        --sample_ratio ${{sample_ratio}} \\
        --warmup_steps ${{warmup_steps}} \\
        --pretrained_lm ${{pretrained_lm}} \\
        --outdir ${{outdir}} \\
        --dropout ${{dropout}} \\
        --mlp_embedding \\
        --adapt_mode \\
        --adapt_embed \\
        --lora \\
        --exp_name ${{exp_name}} \\
        --drl_alg ${{drl_alg}} \\
        --model_type ${{model_type}} \\
        --test_trajectory_file ${{test_trajectory_file}}
done
'''

# Function to find the latest trajectory file for a given algorithm and dataset type
def find_latest_trajectory_file(algo, dataset_type):
    if algo == 'ensemble':
        search_pattern = f'data/{dataset_type}_{algo}_trajectories_*_*.pkl'
    else:
        search_pattern = f'data/{dataset_type}_{algo}_trajectory_*.pkl'
    files = glob.glob(search_pattern)
    if not files:
        raise FileNotFoundError(f"No {dataset_type} trajectory files found for algorithm '{algo}'.")
    # Get the latest file based on the timestamp in the filename
    latest_file = max(files, key=os.path.getctime)
    return latest_file

# Generate run scripts for each algorithm
for algo in algorithms:
    try:
        train_trajectory_file = find_latest_trajectory_file(algo, 'train')
        test_trajectory_file = find_latest_trajectory_file(algo, 'test')

        script_content = base_script.format(
            algo=algo,
            algo_upper=algo.upper(),
            train_trajectory_file=train_trajectory_file,
            test_trajectory_file=test_trajectory_file
        )

        script_filename = f'run_{algo}_dt_gpu.sh'
        script_path = os.path.join('runs', script_filename)

        with open(script_path, 'w') as f:
            f.write(script_content)
        os.chmod(script_path, 0o755)  # Make the script executable

        print(f"Generated run script for {algo.upper()} at '{script_path}'")
    except FileNotFoundError as e:
        print(e)

print("Run scripts generation completed.")

# New code to generate submit scripts
submit_script_template = '''#!/bin/bash
#SBATCH -c 20                # Request 20 CPU cores
#SBATCH --gres=gpu:volta:1   # Request 1 Volta GPU
#SBATCH -o run_{algo}_dt_gpu.log-%j  # Output file
#SBATCH -J run_{algo}_dt_gpu_job # Job name

# Dynamically find the path to conda.sh based on the conda executable
CONDA_PATH=$(dirname "$(dirname "$(which conda)")")/etc/profile.d/conda.sh

# Source the conda.sh script
source "$CONDA_PATH"

# Initialize conda for use in the script
eval "$(conda shell.bash hook)"

# Activate your conda environment
conda activate finrl-dt

# Run your main script
bash runs/run_{algo}_dt_gpu.sh
'''

# Generate individual submit scripts
for algo in algorithms:
    submit_script_content = submit_script_template.format(algo=algo)
    submit_script_filename = f'submit_run_{algo}_dt_gpu.sh'
    submit_script_path = os.path.join('runs', submit_script_filename)

    with open(submit_script_path, 'w') as f:
        f.write(submit_script_content)
    os.chmod(submit_script_path, 0o755)  # Make the script executable

    print(f"Generated submit script for {algo.upper()} at '{submit_script_path}'")

# Generate a combined submit script
combined_submit_script = '''#!/bin/bash

# Submit all run scripts
'''

for algo in algorithms:
    combined_submit_script += f"sbatch runs/submit_run_{algo}_dt_gpu.sh\n"

combined_submit_script_path = os.path.join('runs', 'submit_all_runs.sh')
with open(combined_submit_script_path, 'w') as f:
    f.write(combined_submit_script)
os.chmod(combined_submit_script_path, 0o755)  # Make the script executable

print(f"Generated combined submit script at '{combined_submit_script_path}'")
print("All scripts generation completed.")
