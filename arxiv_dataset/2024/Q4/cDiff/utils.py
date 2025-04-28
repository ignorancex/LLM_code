import torch
import os
import pandas as pd
import matplotlib.pyplot as plt
from evaluation.SBC import compare_distributions
import numpy as np
import random

def SET_SEED(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def save_model(model, save_path, model_name, seed, model_type):
    """
    Save a PyTorch model to the specified path.

    Parameters:
    model (torch.nn.Module): The PyTorch model to save.
    save_path (str): The directory where the model will be saved.
    model_name (str): The name for the saved model file.
    seed (int): The seed value for the experiment (optional for filename).
    model_type (str): Type or name of the model for filename purposes.

    Returns:
    None
    """
    # Construct the file name with seed and model type for easy identification
    file_name = f"{save_path}/{model_name}_seed={seed}_{model_type}.pth"

    # Save the model
    torch.save(model.state_dict(), file_name)

    print(f"Model saved at {file_name}")


def load_torch_model(model, load_path, model_name, seed, model_type):
    """
    Load a PyTorch model from the specified path.

    Parameters:
    model_class (torch.nn.Module): The class definition of the model architecture.
    load_path (str): The directory where the model is saved.
    model_name (str): The name of the saved model file.
    seed (int): The seed value used in the filename.
    model_type (str): Type or name of the model for filename purposes.

    Returns:
    model (torch.nn.Module): The loaded PyTorch model with the saved weights.
    """
    # Construct the file name with seed and model type for easy identification
    file_name = f"{load_path}/{model_name}_seed={seed}_{model_type}.pth"

    # Load the state dictionary into the model
    model.load_state_dict(torch.load(file_name))

    # Set the model to evaluation mode (optional, depends on use case)
    model.eval()

    print(f"Model loaded from {file_name}")

    return model




def safe_update(df, save_path, axis=0):
    """
    Safely updates or creates a CSV file with the given DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame to be saved or updated.
    save_path (str): The full path to the CSV file where the DataFrame should be saved.
    """
    # Check if the file already exists
    if os.path.exists(save_path):
        existing_df = pd.read_csv(save_path)
        if axis == 1:
            df.index = existing_df.index
            updated_df = pd.concat([existing_df, df], axis=axis)
        else:
            updated_df = pd.concat([existing_df, df], ignore_index=True)
    else:
        updated_df = df

    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save the updated DataFrame to the specified path
    # updated_df.to_csv(save_path, index=False)
    with open(save_path, 'w', buffering=1) as file:
        updated_df.to_csv(file, index=False)
        file.flush()



def plot_hist(sbc_calstats,save_path,seed,model_type):
    n_bins = 20
    num_columns = sbc_calstats.shape[1]
    distance = {}

    for i in range(num_columns):
        # Calculate histogram for the i-th column
        hist = torch.histc(sbc_calstats[:, i], min=0, max=1, bins=n_bins)

        # Compare distributions and store in the distance dictionary
        dist_key = f'theta{i + 1}'
        distance[dist_key] = compare_distributions(sbc_calstats[:, i])

        # Plot and save the histogram
        plt.bar(range(n_bins), hist.numpy(), align='center')
        plt.title(f'{distance[dist_key]}')
        plt.savefig(f"{save_path}/{dist_key}_seed={seed}_{model_type}.png", bbox_inches='tight', pad_inches=0)
        plt.close()



def plot_scatter(y, theta, model, save_path, seed, model_type,device):
    """
    Function to plot scatter plots comparing true values and estimated values of theta.

    Parameters:
    - y: Input data.
    - theta: Ground truth values.
    - model: The model to estimate theta.
    - save_path: Path to save the plot.
    - seed: Random seed used for plotting.
    - model_type: A string representing the type of model being used.
    """
    # Get the estimated values using the model
    y = y.to(device)
    theta = theta.detach().cpu().numpy()
    theta_est = model.sample(y)
    theta_est = theta_est.detach().cpu().numpy()

    # Number of dimensions in theta
    n = theta.shape[-1]

    # Create a figure with subplots
    plt.figure(figsize=(n * 5, 5))  # Adjust the figure size based on the number of dimensions

    # Loop through each dimension of theta
    for i in range(n):
        # Create subplot for each dimension
        plt.subplot(1, n, i + 1)

        # Plot scatter for true vs estimated theta
        plt.scatter(theta[:, i], theta_est[:, i])

        # Add labels and title
        plt.xlabel(f'True Theta Dimension {i + 1}')
        plt.ylabel(f'Estimated Theta Dimension {i + 1}')
        plt.title(f'Scatter Plot for Dimension {i + 1}')

    # Save the plot as a file
    plt.savefig(f"{save_path}/scatter_seed={seed}_{model_type}.png", bbox_inches='tight', pad_inches=0)

    # Close the plot to free up memory
    plt.close()
