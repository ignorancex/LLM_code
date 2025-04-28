import os
import pickle
import numpy as np
import pandas as pd

def clear_folder(folder_path):
    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"The folder '{folder_path}' does not exist.")
        return

    # Iterate through files and delete them
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

    # Iterate through subfolders and delete them (optional)
    for subfolder in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, subfolder)
        if os.path.isdir(subfolder_path):
            clear_folder(subfolder_path)


def save_txt(file_path, filename, data):
    os.makedirs(file_path, exist_ok=True)
    path = os.path.join(file_path, filename)
    
    with open(path, "w") as f:
        f.write(str(data))


def save_pickle(file_path, filename, data):
    os.makedirs(file_path, exist_ok=True)
    path = os.path.join(file_path, filename)

    with open(path, "wb") as f:
        pickle.dump(data, f)

def load_all_max_snr_files(folder_path):
    # Function to load all pickle files in a directory
    max_snr_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.pkl')]

    # List to hold all the max_snr dictionaries
    all_max_snr_dicts = []

    # Load all max_snr dictionaries
    for file_path in max_snr_files:
        with open(file_path, 'rb') as f:
            max_snr_dict = pickle.load(f)
            all_max_snr_dicts.append(max_snr_dict)

    return all_max_snr_dicts

def calculate_averages(folder_path):
    all_max_snr_dicts = load_all_max_snr_files(folder_path)

    # Compute the averages
    avg_max_snr = np.mean([d['snr'] for d in all_max_snr_dicts])
    avg_total_iters = np.mean([d['total_iters'] for d in all_max_snr_dicts])
    avg_peak_iters = np.mean([d['iter'] for d in all_max_snr_dicts])

    return avg_max_snr, avg_total_iters, avg_peak_iters

def pickle_csv(directory):

    # List to hold all DataFrames
    data_list = []

    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        # If the file is a pickle file
        if filename.endswith(".pkl"):
            filepath = os.path.join(directory, filename)

            # Load the pickle file
            with open(filepath, 'rb') as f:
                data = pickle.load(f)

            # Convert the scalar values in the dictionary to lists and convert to a DataFrame
            df = pd.DataFrame({k: [v] for k, v in data.items()})
            data_list.append(df)

    # Concatenate all DataFrames in the list into a single DataFrame
    all_data = pd.concat(data_list, ignore_index=True)

    # Convert the DataFrame to a CSV file
    all_data.to_csv('../data/combined_data.csv', index=False)

