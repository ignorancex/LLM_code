import os
import re
import matplotlib.pyplot as plt
from tqdm import tqdm

# Path to the logs directory
logs_dir = 'logs'

# Get the list of experiments (directories)
# experiments = sorted(os.listdir(logs_dir))

experiments = [f'cnn_64_DB_encoder_loss_{i}_decoder_loss_{j}' for i in (0.0875, 0.175, 0.35, 0.7) for j in (1, 2, 4, 8)]

# Keep only the directories that include 'cnn_64_DB' in their name
experiments = [experiment for experiment in experiments if 'cnn_64_DB' in experiment]

# Initialize lists to store the extracted data
encoder_mse_values = []
bitwise_acc_values = []
experiment_names = []

# Regular expression patterns to extract encoder_mse and bitwise-acc
encoder_mse_pattern = re.compile(r"^encoder_mse\s+([\d.]+)")
bitwise_acc_pattern = re.compile(r"^bitwise-acc\s+([\d.]+)")


def process_log_file(log_file_path):
    with open(log_file_path, 'r') as f:
        lines = f.readlines()

    # Reverse the lines to start from the end
    lines = lines[::-1]
    # in_validation = False
    in_validation = True
    current_encoder_mse = None
    current_bitwise_acc = None

    for line in lines:
        line = line.strip()

        # Check for validation start
        # if line.startswith("Running validation for epoch"):
        #    in_validation = True
        #    continue

        if in_validation:
            # Check for encoder_mse
            encoder_mse_match = encoder_mse_pattern.match(line)
            if encoder_mse_match and current_encoder_mse is None:
                current_encoder_mse = float(encoder_mse_match.group(1))

            # Check for bitwise-acc
            bitwise_acc_match = bitwise_acc_pattern.match(line)
            if bitwise_acc_match and current_bitwise_acc is None:
                current_bitwise_acc = float(bitwise_acc_match.group(1))

            # If we have both values, return them
            if current_encoder_mse is not None and current_bitwise_acc is not None:
                return current_encoder_mse, current_bitwise_acc

    # Return None if metrics are not found
    return None, None


# Process each experiment
for experiment_name in tqdm(experiments):
    log_file_path = os.path.join(logs_dir, experiment_name, 'output.log')
    if os.path.exists(log_file_path):
        encoder_mse, bitwise_acc = process_log_file(log_file_path)
        if encoder_mse is not None and bitwise_acc is not None:
            encoder_mse_values.append(encoder_mse)
            bitwise_acc_values.append(bitwise_acc)
            experiment_names.append(experiment_name)
        else:
            print(f"Metrics not found in {experiment_name}")
    else:
        print(f"Log file not found for {experiment_name}")

# Plot the data
plt.figure(figsize=(10, 7))

# Scatter plot of bitwise-acc vs encoder_mse
plt.scatter(encoder_mse_values, bitwise_acc_values, color='blue')

# Annotate each point with the experiment name
for i, name in enumerate(experiment_names):
    plt.annotate(name[len('cnn_64_DB_encoder_loss_'):], (encoder_mse_values[i], bitwise_acc_values[i]),
                 textcoords="offset points",
                 xytext=(0, 10),
                 ha='right')

plt.xlabel('Encoder MSE')
plt.ylabel('Bitwise Accuracy')
plt.title('Bitwise Accuracy vs Encoder MSE Across Experiments')
plt.grid(True)

# Show the plot
plt.tight_layout()
plt.show()
# plt.savefig('cnn_64_visualization.png')
