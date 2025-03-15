from typing import Tuple
import numpy as np
import pandas as pd
import os


def model_name(config: dict) -> str:
    """
    Generate a model name based on the config.

    Parameters:
        config (dict): config dictionary

    Returns:
        str: model name
    """
    dataset = config["DATA"]["type"]
    model = config["MODEL"]["model_type"]
    layer = config["MODEL"]["layer_type"]
    latent_dim = config["MODEL"]["latent_dim"]
    if layer == "KAE":
        order = config["MODEL"]["order"]
        model_name = f"{dataset}_{model}_{layer}_{latent_dim}_{order}"
    else:
        model_name = f"{dataset}_{model}_{layer}_{latent_dim}"

    return model_name


def calc_mean_list(list: list) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate mean and std of a list of arrays.

    Parameters:
        list (list): a list of arrays

    Returns:
        Tuple[np.ndarray, np.ndarray]: a tuple of mean and std arrays
    """
    arr = np.vstack(list)
    return np.mean(arr, axis=0), np.std(arr, axis=0)


def calc_mean_arr(*arrays: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate mean and std of given arrays.

    Parameters:
        *arrays (np.ndarray): arrays to calculate mean and std

    Returns:
        Tuple[np.ndarray, np.ndarray]: a tuple of mean and std arrays
    """
    return [(np.mean(arr), np.std(arr)) for arr in arrays]


def save_data_to_excel(
    excel_path: str, sheet_name: str, columns: list, data: list
) -> None:
    """
    Save given data to an excel file.

    Parameters:
        excel_path (str): the path of the excel file
        sheet_name (str): the name of the sheet to save the data
        columns (list): a list of column names
        data (list): a list of data to save

    Returns:
        None
    """
    new_df = pd.DataFrame(data, columns=columns)

    if not os.path.exists(excel_path):
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            new_df.to_excel(writer, sheet_name=sheet_name, index=False)
    else:
        with pd.ExcelWriter(excel_path, engine="openpyxl", mode="a") as writer:
            if sheet_name in writer.book.sheetnames:
                existing_df = pd.read_excel(excel_path, sheet_name=sheet_name)
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                writer.book.remove(writer.book[sheet_name])
                combined_df.to_excel(writer, sheet_name=sheet_name, index=False)
            else:
                new_df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"Data saved to sheet '{sheet_name}' in {excel_path}")


def save_results_acc(
    excel_path: str,
    model_name: str,
    test_acc: list,
    train_acc: list,
    test_std: list,
    train_std: list,
) -> None:
    """
    Save classification results to an excel file.

    Parameters:
        excel_path (str): the path of the excel file
        model_name (str): the name of the model
        test_acc (list): a list of test accuracy
        train_acc (list): a list of train accuracy
        test_std (list): a list of standard deviation of test accuracy
        train_std (list): a list of standard deviation of train accuracy

    Returns:
        None
    """
    sheet_name = "classify"
    columns_classify = (
        ["Model"]
        + [f"Acc_test_{m}" for m in ["knn1", "knn10", "svm", "mlp"]]
        + [f"Acc_train_{m}" for m in ["knn1", "knn10", "svm", "mlp"]]
    )
    data_classify = [
        [model_name] + test_acc + train_acc,
        ["std"] + test_std + train_std,
    ]
    save_data_to_excel(excel_path, sheet_name, columns_classify, data_classify)


def save_results_loss(
    excel_path: str,
    model_name: str,
    loss_test_epoch: list,
    loss_train_epoch: list,
    time_train_epoch: list,
    loss_test_std: list,
    loss_train_std: list,
    time_train_std: list,
) -> None:
    """
    Save the loss and time results to an excel file.

    Parameters:
        excel_path (str): the path of the excel file
        model_name (str): the name of the model
        loss_test_epoch (list): a list of test loss per epoch
        loss_train_epoch (list): a list of train loss per epoch
        time_train_epoch (list): a list of train time per epoch
        loss_test_std (list): a list of standard deviation of test loss per epoch
        loss_train_std (list): a list of standard deviation of train loss per epoch
        time_train_std (list): a list of standard deviation of train time per epoch

    Returns:
        None
    """
    sheet_name = "loss"
    columns_loss = (
        ["Model", "Test_loss", "Train_loss", "Train_time"]
        + [f"Loss_Test_{i+1}" for i in range(len(loss_test_epoch))]
        + [f"Loss_Train_{i+1}" for i in range(len(loss_train_epoch))]
        + [f"Time_Train_{i+1}" for i in range(len(time_train_epoch))]
    )

    data_loss = [
        [model_name, loss_test_epoch[-1], loss_train_epoch[-1], time_train_epoch[-1]]
        + loss_test_epoch.tolist()
        + loss_train_epoch.tolist()
        + time_train_epoch.tolist(),
        ["std", loss_test_std[-1], loss_train_std[-1], time_train_std[-1]]
        + loss_test_std.tolist()
        + loss_train_std.tolist()
        + time_train_std.tolist(),
    ]
    save_data_to_excel(excel_path, sheet_name, columns_loss, data_loss)
