import os
import pickle


def create_path_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def pickle_save_to_file(data, file_path):
    """Save data to a file using pickle."""
    if file_path is not None:
        path_only = os.path.split(file_path)[0]
        if path_only and not os.path.isdir(path_only):
            os.makedirs(path_only, exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)

def pickle_load_from_file(file_path):
    """Load data from a pickle file."""
    with open(file_path, 'rb') as f:
        return pickle.load(f)