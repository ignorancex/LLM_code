import argparse
import json
import zipfile
from tqdm import tqdm
import numpy as np
import h5py


def zip_to_hdf5_with_index(zip_filename, h5_filename, json_filename):
    with zipfile.ZipFile(zip_filename, 'r') as zip_ref, h5py.File(h5_filename, 'w') as group:
        file_list = []
        for file_path in tqdm(zip_ref.namelist()):
            # Skip directories
            if file_path.endswith('/'):
                continue

            # Remove the outermost folder name if present
            if '/' in file_path:
                new_file_path = file_path.split('/', 1)[1]
            else:
                new_file_path = file_path

            # Ensure we're only processing .npy files
            if not new_file_path.endswith('.npz'):
                continue

            with zip_ref.open(file_path, 'r') as f:
                npy_data = f.read()
            group.create_dataset(new_file_path, data=np.array(npy_data, dtype='S'))
            file_list.append(new_file_path)
        
        with open(json_filename, 'w') as f:
            # Save the json, so we know all the filenames in the h5
            json.dump(file_list, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    args = parser.parse_args()

    assert args.input.endswith('.zip')
    h5_filename = args.input.replace('.zip', '.h5')
    json_filename = args.input.replace('.zip', '_h5.json')
    zip_to_hdf5_with_index(args.input, h5_filename, json_filename)