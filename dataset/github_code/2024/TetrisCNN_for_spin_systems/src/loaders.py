import pickle
import re

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset


def extract_data_from_folder(
    path_to_folder,
    verbose=False,
    parse_cl=True,
    multiple=None,
    phys_model="TFIM",
    phase="TFIM_ferro",
):
    p = path_to_folder
    print(f"Folder : ./{p}")
    files = [el.name for el in p.rglob("*") if (re.search(".csv", el.name) is not None)]
    if "TFIM" in phys_model:
        parameters = {"N": [], "J": [], "g": [], "data": [], "class": []}
    elif "FH" in phys_model:
        parameters = {
            "N": [],
            "J": [],
            "v1": [],
            "v2": [],
            "data": [],
            "class": [],
        }
    elif "BLBQ" in phys_model:
        parameters = {
            "N": [],
            "theta": [],
            "data": [],
            "class": [],
        }
    else:
        raise ValueError("This model is not supported yet!")
    files.sort()
    for f in files:
        if multiple is None:
            N = int(
                re.search("\d+\.*\d*", re.search(f"N=\d+\.*\d*", f).group()).group()
            )
            if parse_cl:
                cl = int(
                    re.search(
                        "\d+\.*\d*", re.search(f"class=\d+\.*\d*", f).group()
                    ).group()
                )
            df = pd.read_csv(p.joinpath(f"{f}"))
            parameters["N"].append(N)
            if parse_cl:
                parameters["class"].append(cl)
            parameters["data"].append(df)
            if "TFIM" in phys_model:
                J = float(
                    re.search(
                        "[-]*\d+\.*\d*",
                        re.search(f"J=[-]*\d+\.*\d*", f).group(),
                    ).group()
                )
                g = float(
                    re.search(
                        "[-]*\d+\.\d+", re.search(f"g=[-]*\d+\.\d+", f).group()
                    ).group()
                )
                parameters["J"].append(J)
                parameters["g"].append(g)
                if verbose:
                    print(f"N = {N} | J = {J} | g = {g:.2f} | Filename: {f}")
                del N, J, f, df, g
            elif "FH" in phys_model:
                J = float(
                    re.search(
                        "[-]*\d+\.*\d*",
                        re.search(f"J=[-]*\d+\.*\d*", f).group(),
                    ).group()
                )
                v1 = float(
                    re.search(
                        "[-]*\d+\.\d+",
                        re.search(f"V1=[-]*\d+\.\d+", f).group(),
                    ).group()
                )
                v2 = float(
                    re.search(
                        "[-]*\d+\.\d+",
                        re.search(f"V2=[-]*\d+\.\d+", f).group(),
                    ).group()
                )
                parameters["J"].append(J)
                parameters["v1"].append(v1)
                parameters["v2"].append(v2)
                if verbose:
                    print(
                        f"N = {N} | J = {J} | V1 = {v1:.2f} | V2 = {v2:.2f} | Filename: {f}"
                    )
                del N, J, f, df, v1, v2
            elif "BLBQ" in phys_model:
                theta = float(
                    re.search(
                        "[-]*\d+\.*\d*",
                        re.search(f"theta=[-]*\d+\.*\d*", f).group(),
                    ).group()
                )
                parameters["theta"].append(theta)
                if verbose:
                    print(f"N = {N} | θ/π = {(theta/np.pi):.4f} | Filename: {f}")
                del N, f, df, theta
            else:
                raise ValueError("This model is not supported yet!")
        else:
            N = int(
                re.search("\d+\.*\d*", re.search(f"N=\d+\.*\d*", f).group()).group()
            )
            if N == multiple:
                if parse_cl:
                    cl = int(
                        re.search(
                            "\d+\.*\d*",
                            re.search(f"class=\d+\.*\d*", f).group(),
                        ).group()
                    )
                df = pd.read_csv(p.joinpath(f"{f}"))
                parameters["N"].append(N)
                if parse_cl:
                    parameters["class"].append(cl)
                parameters["data"].append(df)
                if "TFIM" in phys_model:
                    J = float(
                        re.search(
                            "[-]*\d+\.*\d*",
                            re.search(f"J=[-]*\d+\.*\d*", f).group(),
                        ).group()
                    )
                    g = float(
                        re.search(
                            "[-]*\d+\.\d+",
                            re.search(f"g=[-]*\d+\.\d+", f).group(),
                        ).group()
                    )
                    parameters["J"].append(J)
                    parameters["g"].append(g)
                    if verbose:
                        print(f"N = {N} | J = {J} | g = {g:.2f} | Filename: {f}")
                    del N, J, f, df, g
                elif "FH" in phys_model:
                    J = float(
                        re.search(
                            "[-]*\d+\.*\d*",
                            re.search(f"J=[-]*\d+\.*\d*", f).group(),
                        ).group()
                    )
                    v1 = float(
                        re.search(
                            "[-]*\d+\.\d+",
                            re.search(f"V1=[-]*\d+\.\d+", f).group(),
                        ).group()
                    )
                    v2 = float(
                        re.search(
                            "[-]*\d+\.\d+",
                            re.search(f"V2=[-]*\d+\.\d+", f).group(),
                        ).group()
                    )
                    parameters["J"].append(J)
                    parameters["v1"].append(v1)
                    parameters["v2"].append(v2)
                    if verbose:
                        print(
                            f"N = {N} | J = {J} | V1 = {v1:.2f} | V2 = {v2:.2f} | Filename: {f}"
                        )
                    del N, J, f, df, v1, v2
                elif "BLBQ" in phys_model:
                    theta = float(
                        re.search(
                            "[-]*\d+\.*\d*",
                            re.search(f"theta=[-]*\d+\.*\d*", f).group(),
                        ).group()
                    )
                    parameters["theta"].append(theta)
                    if verbose:
                        print(f"N = {N} | θ/π = {(theta/np.pi):.4f} | Filename: {f}")
                    del N, f, df, theta
                else:
                    raise ValueError("This model is not supported yet!")
            else:
                continue
    if "TFIM" in phys_model:
        parameters["J"] = np.array(parameters["J"])
        parameters["g"] = np.array(parameters["g"])
        order = np.argsort(parameters["g"])
        parameters["g"] = parameters["g"][order]
        parameters["J"] = parameters["J"][order]
    elif "FH" in phys_model:
        parameters["J"] = np.array(parameters["J"])
        parameters["v1"] = np.array(parameters["v1"])
        parameters["v2"] = np.array(parameters["v2"])
        if "LL-CDW1" in phase:
            order = np.argsort(parameters["v1"])
        else:
            order = np.argsort(parameters["v2"])
        parameters["v1"] = parameters["v1"][order]
        parameters["v2"] = parameters["v2"][order]
        parameters["J"] = parameters["J"][order]
    elif "BLBQ" in phys_model:
        parameters["theta"] = np.array(parameters["theta"])
        order = np.argsort(parameters["theta"])
        parameters["theta"] = parameters["theta"][order]
    else:
        raise ValueError("This model is not supported yet!")
    parameters["N"] = np.array(parameters["N"])[order]
    if parse_cl:
        parameters["class"] = np.array(parameters["class"])[order]
    parameters["data"] = [parameters["data"][i] for i in order]
    return parameters


def preprocess_meas_list(meas_list, observables, how="snaps", calc_params=False):
    """
    How:
    'snaps' - calculate magnetization for snapshots
    'expvals' - calculate magnetization from <S_j>
    'corrs' - calculate <S_i S_{i+1}>
    """
    mags = {}
    snaps = meas_list
    if calc_params:
        N = observables[snaps[0]]["N"][0]
        J = observables[snaps[0]]["J"][0]
        sample_num = observables[snaps[0]]["data"][0].shape[1]
    for num, meas_type in enumerate(snaps):
        if how == "expvals":
            meas_save = str(meas_type)
            meas_type = how
        g_range = observables[meas_type]["g"]
        mag_tab = []
        for g in g_range:
            ind = np.where(observables[meas_type]["g"] == g)[0][0]
            df = observables[meas_type]["data"][ind]
            for col in list(df.columns):
                if str(df[col].dtype) == "object":
                    df[col] = df[col].str.replace(" ", "")
                    df[col] = df[col].str.replace("im", "j").apply(lambda x: complex(x))
            if how == "snaps":
                mag = df.mean(axis=0).mean()
                mag_tab.append(mag)
            elif how == "expvals":
                mag = df.mean(axis=0)
                mag_tab.append(mag[num])
            elif how == "corrs":
                mat = np.real(df.to_numpy())
                sh = mat.shape[0]
                corr = 0
                for i in range(sh - 1):
                    corr += mat[i, i + 1]
                corr /= sh - 1
                mag_tab.append(corr)
        if how == "expvals":
            meas_type = meas_save
        mags[meas_type] = mag_tab
    if calc_params:
        return mags, g_range, N, J, sample_num
    else:
        return mags, g_range


class NumpyToPyTorch_Tensor(Dataset):
    def __init__(self, X, Y, transform=None, reg=False):
        self.X = torch.from_numpy(X).float()  # image
        if reg is None:
            self.Y = torch.from_numpy(Y).long()  # label for classification
        else:
            self.Y = torch.from_numpy(Y).float()  # label for classification
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        label = self.Y[index]
        img = self.X[index]

        if self.transform:
            img = self.transform(img)

        return img, label


class Importer(object):
    """
    Parameters
    ----------
    dataset_folder_path : pathlib.PosixPath
        A pathlib Path object pointing to the directory in which you have
        stored the pickled train/validation/test datasets.
    batch : int
        Specifies the default batch size.
    device : torch.device
        Allows for enabling of memory pin for CUDA-enabled computers.
        If is not None, then if enables pin_memory option for CUDA devices.
        Default is None.

    Returns
    -------
    Nothing, just states the internal parameters state.

    """

    def __init__(
        self,
        dataset_folder_path,
        batch,
        snaps=["x"],
        device=None,
        dim=1,
        n_workers=0,
        rescale_y=False,
        phys_model="TFIM",
        phase="TFIM_ferro",
        reg=False,
    ):
        """
        Parameters
        ----------
        dataset_folder_path : pathlib.PosixPath
            A pathlib Path object pointing to the directory in which you have
            stored the pickled train/validation/test datasets.
        batch : int
            Specifies the default batch size.
        snaps : list

        device : torch.device
            Allows for enabling of memory pin for CUDA-enabled computers.
            If is not None, then if enables pin_memory option for CUDA devices.
            Default is None.

        Returns
        -------
        Nothing, just states the internal parameters state.

        """
        self._is_supported(phys_model, phase)
        self.batch = batch
        self.dim = dim
        self.snaps = snaps
        self.rescale_y = rescale_y
        self.y_scaler = None
        self.n_workers = n_workers
        self.datasets_path = dataset_folder_path
        self._train_path = dataset_folder_path.joinpath("train_set")
        self._val_path = dataset_folder_path.joinpath("val_set")
        self._test_path = dataset_folder_path.joinpath("test_set")
        self.phys_model = phys_model
        self.trn_val_test_sizes = {}
        self.phase = phase
        if device is not None:
            self.mem_pin_bool = True if device.type == "cuda" else False
        else:
            self.mem_pin_bool = False
        self.reg = reg
        if reg:
            self.rescale_y = False

    def _is_supported(self, model, phase):
        supported_models = ["FH", "TFIM", "BLBQ"]
        supported_transitions = {
            "TFIM": ["TFIM_ferro", "TFIM_antiferro", "TFIM_all"],
            "FH": ["LL_CDW1", "LL_BO_CDW2", "LL_CDW2"],
            "BLBQ": ["BLBQ_Full", "BLBQ_Full_ED"],
        }
        if model in supported_models:
            if phase in supported_transitions[model]:
                return
            else:
                raise ValueError(
                    f"Phase transition {phase} is not supported for model {model}! \nSupported phase transitions are: {supported_transitions[model]}"
                )
        else:
            raise ValueError(
                f"Model {model} is not supported! \nSupported models are: {supported_models}"
            )

    def csv_to_numpy_tensors(
        self,
        p,
        shuffle=False,
        save_mask=False,
        seed=2137,
        set_designation=None,
    ):
        measurements = [el.name for el in p.rglob("*") if (el.is_dir())]
        measurements.sort()
        n_reals = 0
        if type(self.snaps) is list:
            chosen_meas = [
                m
                for m in measurements
                for snap in self.snaps
                if (re.search(f"_{snap}", m) is not None)
            ]
            chosen_meas.sort()

            # Extracting data shape
            files = [
                el.name
                for el in (p.joinpath(f"{chosen_meas[0]}")).rglob("*")
                if (re.search(".csv", el.name) is not None)
            ]
            files.sort()
            n_set = set()
            counts = {}
            for f in files:
                extracted_N = int(
                    re.search("\d+\.*\d*", re.search(f"N=\d+\.*\d*", f).group()).group()
                )
                if extracted_N in n_set:
                    counts[extracted_N] += 1
                else:
                    n_set.add(extracted_N)
                    counts[extracted_N] = 1
            n_tab = list(n_set)
            n_tab.sort()
            if len(n_tab) == 1:
                kind = "normal"
            else:
                kind = "multiple"

            transition = self.phase
            if "FH" in self.phys_model:
                trans_set = set()
                for f in files:
                    extracted_trans = str(
                        re.search(
                            "\w+-[a-zA-Z]+\d?",
                            re.search(f"transition=\w+-[a-zA-Z]+\d?", f).group(),
                        ).group()
                    )
                    if extracted_trans in trans_set:
                        continue
                    else:
                        trans_set.add(extracted_trans)
                transition = trans_set.pop()

            if kind == "normal":
                N, n_reals = (
                    pd.read_csv(p.joinpath(f"{chosen_meas[0]}/{files[0]}"))
                    .to_numpy()
                    .shape
                )
                self.shape = (len(files) * n_reals, 1)
                data_tensor = np.zeros(
                    shape=(len(files) * n_reals, len(self.snaps), N, self.dim)
                )
                if "TFIM" in self.phys_model:
                    g_tensor = np.zeros(shape=(len(files) * n_reals, 1))
                    j_tensor = np.zeros(shape=(len(files) * n_reals, 1))

                elif "FH" in self.phys_model:
                    v1_tensor = np.zeros(shape=(len(files) * n_reals, 1))
                    v2_tensor = np.zeros(shape=(len(files) * n_reals, 1))
                    j_tensor = np.zeros(shape=(len(files) * n_reals, 1))
                elif "BLBQ" in self.phys_model:
                    theta_tensor = np.zeros(shape=(len(files) * n_reals, 1))
                else:
                    raise ValueError("This model is not supported yet!")
                label_tensor = np.zeros(shape=(len(files) * n_reals, 1))

                for ch_num, meas in enumerate(chosen_meas):
                    params = extract_data_from_folder(
                        p.joinpath(f"{meas}"),
                        phys_model=self.phys_model,
                        phase=transition,
                    )
                    classes_no = np.unique(params["class"]).shape[0]

                    if "TFIM" in self.phys_model:
                        g_range = params["g"]
                        var_range = g_range
                    elif "FH" in self.phys_model:
                        if transition == "LL-CDW1":
                            v1_range = params["v1"]
                            var_range = v1_range
                        else:
                            v2_range = params["v2"]
                            var_range = v2_range
                    elif "BLBQ" in self.phys_model:
                        theta_range = params["theta"]
                        var_range = theta_range
                    else:
                        raise ValueError("This model is not supported yet!")

                    for pt_num, var_param in enumerate(var_range):
                        beg_ind = pt_num * n_reals
                        deg_ind = pt_num % (classes_no - 1)
                        if "TFIM" in self.phys_model:
                            g = var_param
                            ind = np.where(params["g"] == g)[0][deg_ind]
                            j = params["J"][ind]
                            g_tensor[beg_ind : (beg_ind + n_reals)] = (
                                np.ones(shape=(n_reals, 1)) * g
                            )
                            j_tensor[beg_ind : (beg_ind + n_reals)] = (
                                np.ones(shape=(n_reals, 1)) * j
                            )

                        elif "FH" in self.phys_model:
                            if self.phase == "LL_CDW1":
                                v1 = var_param
                                ind = np.where(params["v1"] == v1)[0][0]
                                v2 = params["v2"][ind]
                            elif self.phase == "LL_CDW2":
                                v2 = var_param
                                ind = np.where(params["v2"] == v2)[0][0]
                                v1 = params["v1"][ind]
                            elif self.phase == "LL_BO_CDW2":
                                v2 = var_param
                                ind = np.where(params["v2"] == v2)[0][0]
                                v1 = params["v1"][ind]
                            else:
                                raise ValueError("This model is not supported yet!")
                            j = params["J"][ind]
                            v1_tensor[beg_ind : (beg_ind + n_reals)] = (
                                np.ones(shape=(n_reals, 1)) * v1
                            )
                            v2_tensor[beg_ind : (beg_ind + n_reals)] = (
                                np.ones(shape=(n_reals, 1)) * v2
                            )
                            j_tensor[beg_ind : (beg_ind + n_reals)] = (
                                np.ones(shape=(n_reals, 1)) * j
                            )
                        elif "BLBQ" in self.phys_model:
                            if "BLBQ_Full" in self.phase:
                                theta = var_param
                                ind = np.where(params["theta"] == theta)[0][0]
                            else:
                                raise ValueError("This model is not supported yet!")
                            theta_tensor[beg_ind : (beg_ind + n_reals)] = (
                                np.ones(shape=(n_reals, 1)) * theta
                            )
                        else:
                            raise ValueError("This model is not supported yet!")
                        df = params["data"][ind]
                        label = params["class"][ind]
                        label_tensor[beg_ind : (beg_ind + n_reals)] = (
                            np.ones(shape=(n_reals, 1)) * label
                        )
                        multiple_meas = df.to_numpy()
                        data_tensor[
                            beg_ind : (beg_ind + n_reals), ch_num, :, -1
                        ] = multiple_meas.T
            elif kind == "multiple":
                _, n_reals = (
                    pd.read_csv(p.joinpath(f"{chosen_meas[0]}/{files[0]}"))
                    .to_numpy()
                    .shape
                )
                if "TFIM" in self.phys_model:
                    g_tensors_vec = {}
                    j_tensors_vec = {}
                elif "FH" in self.phys_model:
                    v1_tensors_vec = {}
                    v2_tensors_vec = {}
                    j_tensors_vec = {}
                elif "BLBQ" in self.phys_model:
                    theta_tensors_vec = {}
                else:
                    raise ValueError("This model is not supported yet!")
                label_tensors_vec = {}
                data_tensors_vec = {}
                for (
                    N
                ) in (
                    n_tab
                ):  # <-- Here you could possibly change it to choosing which sizes you would like
                    n_points = counts[N]
                    data_tensor = np.zeros(
                        shape=(
                            n_points * n_reals,
                            len(self.snaps),
                            N,
                            self.dim,
                        )
                    )
                    if "TFIM" in self.phys_model:
                        g_tensor = np.zeros(shape=(n_points * n_reals, 1))
                        j_tensor = np.zeros(shape=(n_points * n_reals, 1))

                    elif "FH" in self.phys_model:
                        v1_tensor = np.zeros(shape=(n_points * n_reals, 1))
                        v2_tensor = np.zeros(shape=(n_points * n_reals, 1))
                        j_tensor = np.zeros(shape=(n_points * n_reals, 1))
                    elif "BLBQ" in self.phys_model:
                        theta_tensor = np.zeros(shape=(n_points * n_reals, 1))
                    else:
                        raise ValueError("This model is not supported yet!")
                    label_tensor = np.zeros(shape=(n_points * n_reals, 1))
                    for ch_num, meas in enumerate(chosen_meas):
                        params = extract_data_from_folder(
                            p.joinpath(f"{meas}"),
                            multiple=N,
                            phys_model=self.phys_model,
                            phase=transition,
                        )

                        classes_no = np.unique(params["class"]).shape[0]

                        if "TFIM" in self.phys_model:
                            g_range = params["g"]
                            var_range = g_range
                        elif "FH" in self.phys_model:
                            if transition == "LL-CDW1":
                                v1_range = params["v1"]
                                var_range = v1_range
                            else:
                                v2_range = params["v2"]
                                var_range = v2_range
                        elif "BLBQ" in self.phys_model:
                            theta_range = params["theta"]
                            var_range = theta_range
                        else:
                            raise ValueError("This model is not supported yet!")
                        for pt_num, var_param in enumerate(var_range):
                            beg_ind = pt_num * n_reals
                            deg_ind = pt_num % (classes_no - 1)
                            if "TFIM" in self.phys_model:
                                g = var_param
                                ind = np.where(params["g"] == g)[0][deg_ind]
                                j = params["J"][ind]
                                g_tensor[beg_ind : (beg_ind + n_reals)] = (
                                    np.ones(shape=(n_reals, 1)) * g
                                )
                                j_tensor[beg_ind : (beg_ind + n_reals)] = (
                                    np.ones(shape=(n_reals, 1)) * j
                                )

                            elif "FH" in self.phys_model:
                                if self.phase == "LL_CDW1":
                                    v1 = var_param
                                    ind = np.where(params["v1"] == v1)[0][0]
                                    v2 = params["v2"][ind]
                                elif self.phase == "LL_CDW2":
                                    v2 = var_param
                                    ind = np.where(params["v2"] == v2)[0][0]
                                    v1 = params["v1"][ind]
                                elif self.phase == "LL_BO_CDW2":
                                    v2 = var_param
                                    ind = np.where(params["v2"] == v2)[0][0]
                                    v1 = params["v1"][ind]
                                else:
                                    raise ValueError("This model is not supported yet!")
                                j = params["J"][ind]
                                v1_tensor[beg_ind : (beg_ind + n_reals)] = (
                                    np.ones(shape=(n_reals, 1)) * v1
                                )
                                v2_tensor[beg_ind : (beg_ind + n_reals)] = (
                                    np.ones(shape=(n_reals, 1)) * v2
                                )
                                j_tensor[beg_ind : (beg_ind + n_reals)] = (
                                    np.ones(shape=(n_reals, 1)) * j
                                )
                            elif "BLBQ" in self.phys_model:
                                if "BLBQ_Full" in self.phase:
                                    theta = var_param
                                    ind = np.where(params["theta"] == theta)[0][0]
                                else:
                                    raise ValueError("This model is not supported yet!")
                                theta_tensor[beg_ind : (beg_ind + n_reals)] = (
                                    np.ones(shape=(n_reals, 1)) * theta
                                )
                            else:
                                raise ValueError("This model is not supported yet!")
                            df = params["data"][ind]
                            label = params["class"][ind]
                            label_tensor[beg_ind : (beg_ind + n_reals)] = (
                                np.ones(shape=(n_reals, 1)) * label
                            )
                            multiple_meas = df.to_numpy()
                            data_tensor[
                                beg_ind : (beg_ind + n_reals), ch_num, :, -1
                            ] = multiple_meas.T
                    if "TFIM" in self.phys_model:
                        g_tensors_vec[N] = g_tensor
                        j_tensors_vec[N] = j_tensor
                    elif "FH" in self.phys_model:
                        v1_tensors_vec[N] = v1_tensor
                        v2_tensors_vec[N] = v2_tensor
                        j_tensors_vec[N] = j_tensor
                    elif "BLBQ" in self.phys_model:
                        theta_tensors_vec[N] = theta_tensor
                    else:
                        raise ValueError("This model is not supported yet!")
                    label_tensors_vec[N] = label_tensor
                    data_tensors_vec[N] = data_tensor

        # Baseline classificator only implemented for TFIM model
        elif type(self.snaps) is dict:
            kind = "blank"
            N = self.snaps["N"]
            labels = self.snaps["labels"]
            nsamples = self.snaps["nsamples"]
            channels = self.snaps["channels"]
            j = self.snaps["J"]
            percentages = self.snaps["class_shares"]
            data_tensor = np.zeros(shape=(nsamples, channels, N, self.dim))
            g_tensor = np.zeros(shape=(nsamples, 1))
            j_tensor = np.ones(shape=(nsamples, 1)) * j
            label_tensor = np.concatenate(
                [
                    np.ones(shape=(round(nsamples * perc), 1)) * lab
                    for (perc, lab) in zip(percentages, labels)
                ]
            )
        else:
            raise TypeError("Invalid value!")

        if kind != "multiple":
            if not self.reg:
                label_tensor = label_tensor.reshape(self.shape)
                if "TFIM" in self.phys_model:
                    g_tensor = g_tensor.reshape(self.shape)
                    j_tensor = j_tensor.reshape(self.shape)

                elif "FH" in self.phys_model:
                    v1_tensor = v1_tensor.reshape(self.shape)
                    v2_tensor = v2_tensor.reshape(self.shape)
                    j_tensor = j_tensor.reshape(self.shape)

                elif "BLBQ" in self.phys_model:
                    theta_tensor = theta_tensor.reshape(self.shape)

                else:
                    raise ValueError("This model is not supported yet!")
            else:
                if "TFIM" in self.phys_model:
                    g_tensor = g_tensor.reshape(self.shape)
                    j_tensor = j_tensor.reshape(self.shape)
                    label_tensor = g_tensor.copy()

                elif "FH" in self.phys_model:
                    if self.phase == "LL_CDW1":
                        v1_tensor = v1_tensor.reshape(self.shape)
                        v2_tensor = v2_tensor.reshape(self.shape)
                        j_tensor = j_tensor.reshape(self.shape)
                        label_tensor = v1_tensor.copy()
                    else:
                        v1_tensor = v1_tensor.reshape(self.shape)
                        v2_tensor = v2_tensor.reshape(self.shape)
                        j_tensor = j_tensor.reshape(self.shape)
                        label_tensor = v2_tensor.copy()

                elif "BLBQ" in self.phys_model:
                    theta_tensor = theta_tensor.reshape(self.shape)
                    label_tensor = theta_tensor.copy()

                else:
                    raise ValueError("This model is not supported yet!")

        else:
            for N in n_tab:
                if not self.reg:
                    label_tensors_vec[N] = label_tensors_vec[N].reshape(
                        shape=self.shape
                    )
                    if "TFIM" in self.phys_model:
                        g_tensors_vec[N] = g_tensors_vec[N].reshape(shape=self.shape)
                        j_tensors_vec[N] = j_tensors_vec[N].reshape(shape=self.shape)
                    elif "FH" in self.phys_model:
                        v1_tensors_vec[N] = v1_tensors_vec[N].reshape(shape=self.shape)
                        v2_tensors_vec[N] = v2_tensors_vec[N].reshape(shape=self.shape)
                        j_tensors_vec[N] = j_tensors_vec[N].reshape(shape=self.shape)
                    elif "BLBQ" in self.phys_model:
                        theta_tensors_vec[N] = theta_tensors_vec[N].reshape(
                            shape=self.shape
                        )
                    else:
                        raise ValueError("This model is not supported yet!")
                else:
                    if "TFIM" in self.phys_model:
                        g_tensors_vec[N] = g_tensors_vec[N].reshape(shape=self.shape)
                        j_tensors_vec[N] = j_tensors_vec[N].reshape(shape=self.shape)
                        label_tensors_vec[N] = g_tensors_vec[N]
                    elif "FH" in self.phys_model:
                        if self.phase == "LL_CDW1":
                            v1_tensors_vec[N] = v1_tensors_vec[N].reshape(
                                shape=self.shape
                            )
                            v2_tensors_vec[N] = v2_tensors_vec[N].reshape(
                                shape=self.shape
                            )
                            j_tensors_vec[N] = j_tensors_vec[N].reshape(
                                shape=self.shape
                            )
                            label_tensors_vec[N] = v1_tensors_vec[N]
                        else:
                            v1_tensors_vec[N] = v1_tensors_vec[N].reshape(
                                shape=self.shape
                            )
                            v2_tensors_vec[N] = v2_tensors_vec[N].reshape(
                                shape=self.shape
                            )
                            j_tensors_vec[N] = j_tensors_vec[N].reshape(
                                shape=self.shape
                            )
                            label_tensors_vec[N] = v2_tensors_vec[N]
                    elif "BLBQ" in self.phys_model:
                        theta_tensors_vec[N] = theta_tensors_vec[N].reshape(
                            shape=self.shape
                        )
                        label_tensors_vec[N] = theta_tensors_vec[N]
                    else:
                        raise ValueError("This model is not supported yet!")

        if (
            shuffle and kind != "multiple"
        ):  # <-- For now shuffling if hard-coded to off for multiple system sizes, because we don't need it for testing
            # Shuffling ordered data, but preserving the mask (since we want to remember for which 'v' the datapoint was calculated)
            if kind == "normal":
                mask = np.arange(len(files) * n_reals)
            else:
                mask = np.arange(nsamples)
            np.random.seed(seed)
            np.random.shuffle(mask)

            # Saving the mask in the object's variables to retrieve original indices afterwards
            self.train_mask = mask
            if save_mask:
                with open(
                    self.datasets_path.joinpath("train_set_mask.pickle"), "wb"
                ) as f:
                    pickle.dump(mask, f)

            masked_data = data_tensor[mask]
            masked_labels = label_tensor[mask]
            if "TFIM" in self.phys_model:
                masked_g = g_tensor[mask]
                masked_j = j_tensor[mask]
                masked_params = (masked_g, masked_j)
            elif "FH" in self.phys_model:
                masked_v1 = v1_tensor[mask]
                masked_v2 = v2_tensor[mask]
                masked_j = j_tensor[mask]
                masked_params = (masked_v1, masked_v2, masked_j)
            elif "BLBQ" in self.phys_model:
                masked_theta = theta_tensor[mask]
                masked_params = masked_theta
            else:
                raise ValueError("This model is not supported yet!")

        elif not shuffle and kind != "multiple":
            masked_data = data_tensor
            masked_labels = label_tensor
            if "TFIM" in self.phys_model:
                masked_g = g_tensor
                masked_j = j_tensor
                masked_params = (masked_g, masked_j)
            elif "FH" in self.phys_model:
                masked_v1 = v1_tensor
                masked_v2 = v2_tensor
                masked_j = j_tensor
                masked_params = (masked_v1, masked_v2, masked_j)
            elif "BLBQ" in self.phys_model:
                masked_theta = theta_tensor
                masked_params = masked_theta
            else:
                raise ValueError("This model is not supported yet!")
        else:
            masked_data = data_tensors_vec
            masked_labels = label_tensors_vec
            if "TFIM" in self.phys_model:
                masked_g = g_tensors_vec
                masked_j = j_tensors_vec
                masked_params = (masked_g, masked_j)
            elif "FH" in self.phys_model:
                masked_v1 = v1_tensors_vec
                masked_v2 = v2_tensors_vec
                masked_j = j_tensors_vec
                masked_params = (masked_v1, masked_v2, masked_j)
            elif "BLBQ" in self.phys_model:
                masked_theta = theta_tensors_vec
                masked_params = masked_theta
            else:
                raise ValueError("This model is not supported yet!")
        if set_designation is not None:
            self.trn_val_test_sizes[set_designation] = [n_reals] * pt_num

        return masked_data, masked_labels, masked_params

    def get_train_loader(
        self, batch_size=None, shuffle=True, save_mask=False, seed=2137
    ):
        """
        A function for preparation of PyTorch DataLoader class instance with
        training dataset loaded.

        Parameters
        ----------
        batch_size : int, optional
            Custom batch size for training dataset loader.
            If None, Importer instance's default value is used.
            The default is None.
        shuffle : bool, optional
            Should the training data be shuffled on import. The default is True.
        save_mask : bool, optional
            Should the shuffle mask be saved to hard drive apart from it being
            saved as variable in Importer instance. The default is False.
        seed : int, optional
            What seed should the random number generator use for shuffling.
            The default is 2137.

        Returns
        -------
        train_loader : torch.utils.data.dataloader.DataLoader
            DataLoader instance fed with training dataset.

        """

        if batch_size is None:
            batch_size = self.batch

        p = self._train_path
        if "TFIM" in self.phys_model:
            (
                data_tensor,
                labels_tensor,
                (g_tensor, j_tensor),
            ) = self.csv_to_numpy_tensors(
                p,
                shuffle=shuffle,
                save_mask=save_mask,
                seed=seed,
                set_designation="train",
            )
        elif "FH" in self.phys_model:
            (
                data_tensor,
                labels_tensor,
                (v1_tensor, v2_tensor, j_tensor),
            ) = self.csv_to_numpy_tensors(
                p,
                shuffle=shuffle,
                save_mask=save_mask,
                seed=seed,
                set_designation="train",
            )
        elif "BLBQ" in self.phys_model:
            (
                data_tensor,
                labels_tensor,
                (theta_tensor),
            ) = self.csv_to_numpy_tensors(
                p,
                shuffle=shuffle,
                save_mask=save_mask,
                seed=seed,
                set_designation="train",
            )
        else:
            raise ValueError("This model is not supported yet!")

        if self.rescale_y:
            labels_tensor = labels_tensor / labels_tensor.max()

        data_set = NumpyToPyTorch_Tensor(data_tensor, labels_tensor, reg=self.reg)

        if "TFIM" in self.phys_model:
            self.train_g_tab = g_tensor
            self.train_j_tab = j_tensor

        elif "FH" in self.phys_model:
            self.train_v1_tab = v1_tensor
            self.train_v2_tab = v2_tensor
            self.train_j_tab = j_tensor

        elif "BLBQ" in self.phys_model:
            self.train_theta_tab = theta_tensor

        else:
            raise ValueError("This model is not supported yet!")

        train_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.n_workers,
            pin_memory=self.mem_pin_bool,  # CUDA only, this lets your DataLoader allocate the samples in page-locked memory, which speeds-up the transfer from CPU to GPU during training
        )
        return train_loader

    def get_val_loader(self, batch_size=None):
        """
        A function for preparation of PyTorch DataLoader class instance with
        validation dataset loaded.

        Parameters
        ----------
        batch_size : int, optional
            Custom batch size for validation dataset loader.
            If None, Importer instance's default value is used.
            The default is None.

        Returns
        -------
        val_loader : torch.utils.data.dataloader.DataLoader
            DataLoader instance fed with validation dataset.

        """

        if batch_size is None:
            batch_size = self.batch

        p = self._val_path
        if "TFIM" in self.phys_model:
            (
                data_tensor,
                labels_tensor,
                (g_tensor, j_tensor),
            ) = self.csv_to_numpy_tensors(
                p,
                set_designation="validation",
            )
        elif "FH" in self.phys_model:
            (
                data_tensor,
                labels_tensor,
                (v1_tensor, v2_tensor, j_tensor),
            ) = self.csv_to_numpy_tensors(
                p,
                set_designation="validation",
            )
        elif "BLBQ" in self.phys_model:
            (
                data_tensor,
                labels_tensor,
                (theta_tensor),
            ) = self.csv_to_numpy_tensors(
                p,
                set_designation="validation",
            )
        else:
            raise ValueError("This model is not supported yet!")

        if self.rescale_y:
            labels_tensor = labels_tensor / labels_tensor.max()

        data_set = NumpyToPyTorch_Tensor(data_tensor, labels_tensor, reg=self.reg)

        if "TFIM" in self.phys_model:
            self.val_g_tab = g_tensor
            self.val_j_tab = j_tensor

        elif "FH" in self.phys_model:
            self.val_v1_tab = v1_tensor
            self.val_v2_tab = v2_tensor
            self.val_j_tab = j_tensor

        elif "BLBQ" in self.phys_model:
            self.val_theta_tab = theta_tensor

        else:
            raise ValueError("This model is not supported yet!")

        val_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.n_workers,
            pin_memory=self.mem_pin_bool,  # CUDA only, this lets your DataLoader allocate the samples in page-locked memory, which speeds-up the transfer from CPU to GPU during training
        )
        return val_loader

    def get_test_loader(self, batch_size=None):
        """
        A function for preparation of PyTorch DataLoader class instance with
        test dataset loaded.

        Parameters
        ----------
        batch_size : int, optional
            Custom batch size for test dataset loader.
            If None, Importer instance's default value is used.
            The default is None.

        Returns
        -------
        test_loader : torch.utils.data.dataloader.DataLoader
            DataLoader instance fed with test dataset.

        """

        if batch_size is None:
            batch_size = self.batch

        p = self._test_path
        if "TFIM" in self.phys_model:
            (
                data_tensor,
                labels_tensor,
                (g_tensor, j_tensor),
            ) = self.csv_to_numpy_tensors(
                p,
                set_designation="test",
            )
        elif "FH" in self.phys_model:
            (
                data_tensor,
                labels_tensor,
                (v1_tensor, v2_tensor, j_tensor),
            ) = self.csv_to_numpy_tensors(
                p,
                set_designation="test",
            )
        elif "BLBQ" in self.phys_model:
            (
                data_tensor,
                labels_tensor,
                (theta_tensor),
            ) = self.csv_to_numpy_tensors(
                p,
                set_designation="test",
            )
        else:
            raise ValueError("This model is not supported yet!")

        if "TFIM" in self.phys_model:
            self.test_g_tab = g_tensor
            self.test_j_tab = j_tensor

        elif "FH" in self.phys_model:
            self.test_v1_tab = v1_tensor
            self.test_v2_tab = v2_tensor
            self.test_j_tab = j_tensor

        elif "BLBQ" in self.phys_model:
            self.test_theta_tab = theta_tensor

        else:
            raise ValueError("This model is not supported yet!")

        self.test_data = data_tensor

        if type(data_tensor) is not dict:
            if batch_size == -1:
                batch_size = labels_tensor.shape[0]
            if self.rescale_y:
                labels_tensor = labels_tensor / labels_tensor.max()

            self.test_labels = labels_tensor
            data_set = NumpyToPyTorch_Tensor(data_tensor, labels_tensor, reg=self.reg)

            test_loader = DataLoader(
                data_set,
                batch_size=batch_size,
                shuffle=False,
                num_workers=self.n_workers,
                pin_memory=self.mem_pin_bool,  # CUDA only, this lets your DataLoader allocate the samples in page-locked memory, which speeds-up the transfer from CPU to GPU during training
            )
            return test_loader
        else:
            if batch_size == -1:
                batch_size = labels_tensor[list(labels_tensor.keys())[-1]].shape[0]
            loaders = {}
            if self.rescale_y:
                for key in labels_tensor.keys():
                    labels_tensor[key] = labels_tensor[key] / labels_tensor[key].max()
            self.test_labels = labels_tensor

            for key in labels_tensor.keys():
                data_set = NumpyToPyTorch_Tensor(
                    data_tensor[key], labels_tensor[key], reg=self.reg
                )

                test_loader = DataLoader(
                    data_set,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=self.n_workers,
                    pin_memory=self.mem_pin_bool,  # CUDA only, this lets your DataLoader allocate the samples in page-locked memory, which speeds-up the transfer from CPU to GPU during training
                )
                loaders[key] = test_loader
            return loaders
