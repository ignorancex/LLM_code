import os
from glob import glob

from keras.layers import ELU

from train_binary import train as train_binary_model
from utils.dataset import MRIDataset
from utils.models import binary_model
from utils.preprocessing import create_dataset_from_patients_directory, create_binary_dataset_from_dataset, \
    create_cropped_dataset_from_dataset


def create_new_dataset(input_dataset_path: str, output_dataset_path: str) -> MRIDataset:
    non_cropped_dataset_path = os.path.join(output_dataset_path, "data")
    binary_dataset_path = os.path.join(output_dataset_path, "binary")
    binary_weights_path = os.path.join(binary_dataset_path, "BinaryWeights.hdf5")
    cropped_dataset_path = os.path.join(output_dataset_path, "cropped")

    print("Creating NonCropped Dataset From Patient Directory")
    create_dataset_from_patients_directory(input_dataset_path, non_cropped_dataset_path)

    print("\nCreating Binary Dataset From NonCropped Dataset")
    create_binary_dataset_from_dataset(non_cropped_dataset_path, binary_dataset_path)

    print("\nTraining the Binary Model using the binary dataset")
    train_binary_model(binary_dataset_path, binary_weights_path)

    n_channels = 20
    model = binary_model(128, 128, 128, 4, 1, n_channels, activation=ELU())
    model.load_weights(binary_weights_path)

    print("\nCreating Cropped Dataset From NonCropped Dataset using the Binary Model")
    create_cropped_dataset_from_dataset(non_cropped_dataset_path, model, cropped_dataset_path)

    # Validate the shape and content of the MRI dataset
    return MRIDataset(non_cropped_dataset_path=non_cropped_dataset_path,
                      binary_dataset_path=binary_dataset_path,
                      cropped_dataset_path=cropped_dataset_path)


def generate_train_test_split_lists(dataset_name: str, dataset_path: str) -> None:
    for subdirectory in ["train/images", "train/masks", "val/images", "val/masks"]:
        subdirectory_path = os.path.join(dataset_path, subdirectory)
        if not os.path.isdir(subdirectory_path):
            raise FileNotFoundError(f"Unable to find the subdirectory {subdirectory}")
        elif len(glob(os.path.join(subdirectory_path, "*.npy"))) == 0:
            raise FileNotFoundError(f"The {subdirectory} subdirectory is data (when looking for .npy files)")

    split_paths = os.path.join("splits", dataset_name)
    if not os.path.isdir(split_paths):
        os.makedirs(split_paths)

    with open(os.path.join(split_paths, "train_files.txt"), "w") as train_file:
        for img_name in os.listdir(os.path.join(os.path.join(dataset_path, "train/images"))):
            patient_index = img_name[6:-4]
            filled_index = str(patient_index).zfill(5)
            train_file.write(f"{dataset_name}_{filled_index}\n")

    with open(os.path.join(split_paths, "val_files.txt"), "w") as val_file:
        for img_name in os.listdir(os.path.join(os.path.join(dataset_path, "val/images"))):
            patient_index = img_name[6:-4]
            filled_index = str(patient_index).zfill(5)
            val_file.write(f"{dataset_name}_{filled_index}\n")
