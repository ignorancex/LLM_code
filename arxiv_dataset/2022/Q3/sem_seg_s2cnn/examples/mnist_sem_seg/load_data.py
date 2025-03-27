import os
import subprocess

import numpy as np
import torch
import torch.utils.data as data_utils

import get_paths


def get_data(train_set_rotated, test_set_rotated, len_train_data, len_test_data, bandwidth):
    rot_string_train = ("non-" if not train_set_rotated else "") + "rotated training samples"
    rot_string_test = ("non-" if not test_set_rotated else "") + "rotated test samples"
    print(
        "Loading MNIST semantic segmentation on S2 dataset with",
        len_train_data,
        rot_string_train,
        "and",
        len_test_data,
        rot_string_test,
    )

    datasets_dir = get_paths.get_datasets_path()

    suffix = "_rot_train" if train_set_rotated else ""
    suffix += "_rot_test" if test_set_rotated else ""
    suffix += "_" + str(len_test_data) + "_test_samples_" + str(len_train_data) + "_train_samples"
    suffix += "_bandwidth_" + str(bandwidth)
    filename = "s2_mnist_sem_seg" + suffix + ".npz"
    file_path = os.path.join(datasets_dir, filename)

    if not os.path.isfile(file_path):
        print("Could not find file", filename, "in", datasets_dir)
        base_dir = get_paths.get_base_path()
        command = (
            "python3 -u "
            + os.path.join(base_dir, "gendata.py")
            + (" --no_rotate_train" if not train_set_rotated else "")
            + (" --no_rotate_test" if not test_set_rotated else "")
            + " --bandwidth="
            + str(bandwidth)
            + " --len_train_data="
            + str(len_train_data)
            + " --len_test_data="
            + str(len_test_data)
        )
        print("Running", command)
        subprocess.run(command, shell=True)
        print("Loading generated dataset...")

    data = np.load(file_path)

    dataset = {"train": {}, "test": {}}
    dataset["train"]["images"] = data["arr_0"]
    dataset["train"]["labels"] = data["arr_1"]
    dataset["train"]["seg_masks"] = data["arr_2"]
    dataset["test"]["images"] = data["arr_3"]
    dataset["test"]["labels"] = data["arr_4"]
    dataset["test"]["seg_masks"] = data["arr_5"]

    print("Done.")

    return dataset


def get_dataloader(
    batch_size, train_set_rotated, test_set_rotated, len_train_data, len_test_data, bandwidth
):
    data = get_data(
        bandwidth=bandwidth,
        train_set_rotated=train_set_rotated,
        test_set_rotated=test_set_rotated,
        len_train_data=len_train_data,
        len_test_data=len_test_data,
    )

    train_data = torch.from_numpy(data["train"]["images"][:, None, :, :].astype(np.float32))
    train_labels = torch.from_numpy(data["train"]["seg_masks"].astype(np.uint8))
    train_dataset = data_utils.TensorDataset(train_data, train_labels)
    train_loader = data_utils.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_data = torch.from_numpy(data["test"]["images"][:, None, :, :].astype(np.float32))
    test_labels = torch.from_numpy(data["test"]["seg_masks"].astype(np.uint8))
    test_dataset = data_utils.TensorDataset(test_data, test_labels)
    test_loader = data_utils.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader
