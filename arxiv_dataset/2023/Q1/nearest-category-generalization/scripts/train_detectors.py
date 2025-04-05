import os
import gc

from skimage.color import rgb2gray
from skimage.transform import resize
import torch
import numpy as np
from tqdm.notebook import tqdm
import joblib
from mkdir_p import mkdir_p

from lolip.models.detectors import get_detector
from lolip.variables import auto_var
from lolip.models.torch_utils import archs

RESULT_DIR = "./results/notebooks_detection/"

def predict_real(model, X, device):
    if len(X.shape) == 4:
        X = X.transpose(0, 3, 1, 2)
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(X).float())
    loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False, num_workers=2)
    ret = []
    for [x] in loader:
        ret.append(model(x.to(device)).detach().cpu().numpy())
    return np.concatenate(ret, axis=0)

def get_repr(model, X, device):
    if len(X.shape) == 4:
        X = X.transpose(0, 3, 1, 2)
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(X).float())
    loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False, num_workers=2)
    ret = []
    for [x] in loader:
        ret.append(model.get_repr(x.to(device)).detach().cpu().numpy())
    return np.concatenate(ret, axis=0)

class ModelWrapper():
    def __init__(self, model):
        self.model = model
    def predict_real(self, X, device="cpu"):
        return predict_real(self.model, X, device=device)
    def predict(self, X, device="cpu"):
        return predict_real(self.model, X, device=device).argmax(axis=1)

def cifar_pixel(in_dataset):
    out_datasets = ["mnist", "svhn", "tinyimgnet"]
    detector_names = ["IsolationForest", "IsolationForestv2", "ExtIsoForest", "LocalOutlierFactor", "linearOneClassSVM"]

    X, _, tstX, _, rest = auto_var.get_var_with_argument("dataset", in_dataset)
    oodX = np.concatenate((rest[0], rest[1]), axis=0)
    X = X.reshape(len(X), -1)
    tstX = tstX.reshape(len(tstX), -1)
    oodX = oodX.reshape(len(oodX), -1)

    for out_dataset in out_datasets:
        print(out_dataset)
        ood2X, _, ood3X, _ = auto_var.get_var_with_argument("dataset", out_dataset)
        if out_dataset == "mnist":
            ood2X = np.concatenate([ood2X] * 3, axis=3)
            ood3X = np.concatenate([ood3X] * 3, axis=3)
        ood2X = np.concatenate([resize(x, (32, 32)).reshape(1, 32, 32, 3) for x in ood2X], axis=0)
        ood3X = np.concatenate([resize(x, (32, 32)).reshape(1, 32, 32, 3) for x in ood3X], axis=0)
        ood2X, ood3X = ood2X.reshape(len(ood2X), -1), ood3X.reshape(len(ood3X), -1)

        for detector_name in detector_names:
            print(detector_name)
            filepath = f"results/notebooks_detection/{detector_name}_{in_dataset}_{out_dataset}.pkl"
            if os.path.exists(filepath):
                pass
            else:
                clf = get_detector(detector_name).fit(X.reshape(len(X), -1))
                ret = (clf.predict(X), clf.predict(tstX), clf.predict(oodX), clf.predict(ood2X), clf.predict(ood3X))
                joblib.dump(ret, filepath)

def cifar_feature(in_dataset, base_model_path, n_classes):
    architecture = "WRN_40_10"
    out_datasets = ['mnist', "svhn", "tinyimgnet"]
    detector_names = ["IsolationForest", "IsolationForestv2", "ExtIsoForest", "LocalOutlierFactor", "linearOneClassSVM"]
    #detector_names = ["IsolationForest", "IsolationForestv2", "ExtIsoForest", #]
    device = "cuda"

    X, _, tstX, _, rest = auto_var.get_var_with_argument("dataset", in_dataset)
    oodX = np.concatenate((rest[0], rest[1]), axis=0)

    model = getattr(archs, architecture)(n_classes=n_classes)
    model.load_state_dict(torch.load(base_model_path)['model_state_dict'])
    model.to(device)
    model.eval()

    X = get_repr(model, X, device=device).reshape(len(X), -1)
    tstX = get_repr(model, tstX, device=device).reshape(len(tstX), -1)
    oodX = get_repr(model, oodX, device=device).reshape(len(oodX), -1)
    print(X.shape)

    for out_dataset in out_datasets:
        print(out_dataset)
        ood2X, _, ood3X, _ = auto_var.get_var_with_argument("dataset", out_dataset)
        if out_dataset == "mnist":
            ood2X = np.concatenate([ood2X] * 3, axis=3)
            ood3X = np.concatenate([ood3X] * 3, axis=3)
        ood2X = np.concatenate([resize(x, (32, 32)).reshape(1, 32, 32, 3) for x in ood2X], axis=0)
        ood3X = np.concatenate([resize(x, (32, 32)).reshape(1, 32, 32, 3) for x in ood3X], axis=0)
        ood2X, ood3X = get_repr(model, ood2X, device=device).reshape(len(ood2X), -1), get_repr(model, ood3X, device=device).reshape(len(ood3X), -1)
        gc.collect()

        for detector_name in detector_names:
            print(detector_name)
            filepath = f"results/notebooks_detection/feature_{detector_name}_{in_dataset}_{out_dataset}.pkl"
            if os.path.exists(filepath):
                pass
            else:
                clf = get_detector(detector_name, n_jobs=4).fit(X.reshape(len(X), -1))
                ret = (clf.predict(X), clf.predict(tstX), clf.predict(oodX), clf.predict(ood2X), clf.predict(ood3X))
                joblib.dump(ret, filepath)
                del clf
                gc.collect()

        del ood2X, ood3X
        gc.collect()


def mnist_pixel(in_dataset="mnistwo9"):
    out_datasets = ["cifar10", "svhn", "tinyimgnet"]
    detector_names = ["IsolationForest", "IsolationForestv2", "ExtIsoForest", "LocalOutlierFactor", "linearOneClassSVM"]
    mkdir_p(RESULT_DIR)

    X, _, tstX, _, rest = auto_var.get_var_with_argument("dataset", in_dataset)
    oodX = np.concatenate((rest[0], rest[1]), axis=0)
    X = X.reshape(len(X), -1)
    tstX = tstX.reshape(len(tstX), -1)
    oodX = oodX.reshape(len(oodX), -1)

    for out_dataset in out_datasets:
        print(out_dataset)
        ood2X, _, ood3X, _ = auto_var.get_var_with_argument("dataset", out_dataset)
        ood2X, ood3X = rgb2gray(ood2X), rgb2gray(ood3X)
        ood2X = np.concatenate([resize(x, (28, 28)).reshape(1, 28, 28, 1) for x in ood2X], axis=0)
        ood3X = np.concatenate([resize(x, (28, 28)).reshape(1, 28, 28, 1) for x in ood3X], axis=0)
        ood2X, ood3X = ood2X.reshape(len(ood2X), -1), ood3X.reshape(len(ood3X), -1)

        for detector_name in detector_names:
            print(detector_name)
            filepath = os.path.join(RESULT_DIR, f"{detector_name}_{in_dataset}_{out_dataset}.pkl")
            if os.path.exists(filepath):
                print(f"{filepath} exists")
            else:
                clf = get_detector(detector_name).fit(X.reshape(len(X), -1))
                ret = (clf.predict(X), clf.predict(tstX), clf.predict(oodX), clf.predict(ood2X), clf.predict(ood3X))
                joblib.dump(ret, filepath)

def mnist_feature():
    in_dataset = "mnistwo9"
    base_model_path = "./models/out_of_sample/pgd-128-mnistwo0-70-1.0-0.01-ce-vtor2-CNN002-0.9-2-sgd-0-0.0-ep0070.pt"
    architecture = "CNN002"
    n_classes = 9
    out_datasets = ['cifar10', "svhn", "tinyimgnet"]
    device = "cuda"
    detector_names = ["IsolationForest", "IsolationForestv2", "ExtIsoForest", "LocalOutlierFactor", "linearOneClassSVM"]

    X, _, tstX, _, rest = auto_var.get_var_with_argument("dataset", in_dataset)
    oodX = np.concatenate((rest[0], rest[1]), axis=0)

    model = getattr(archs, architecture)(n_classes=n_classes, n_features=None)
    model.load_state_dict(torch.load(base_model_path)['model_state_dict'])
    model = model.feature_extractor
    model.to(device)
    model.eval()
    extractor = ModelWrapper(model)

    X = extractor.predict_real(X, device=device).reshape(len(X), -1)
    tstX = extractor.predict_real(tstX, device=device).reshape(len(tstX), -1)
    oodX = extractor.predict_real(oodX, device=device).reshape(len(oodX), -1)

    for out_dataset in out_datasets:
        print(out_dataset)
        ood2X, _, ood3X, _ = auto_var.get_var_with_argument("dataset", out_dataset)
        ood2X, ood3X = rgb2gray(ood2X), rgb2gray(ood3X)
        ood2X = np.concatenate([resize(x, (28, 28)).reshape(1, 28, 28, 1) for x in ood2X], axis=0)
        ood3X = np.concatenate([resize(x, (28, 28)).reshape(1, 28, 28, 1) for x in ood3X], axis=0)
        ood2X, ood3X = extractor.predict_real(ood2X, device=device).reshape(len(ood2X), -1), extractor.predict_real(ood3X, device=device).reshape(len(ood3X), -1)

        for detector_name in detector_names:
            print(detector_name)
            filepath = os.path.join(RESULT_DIR, f"feature_{detector_name}_{in_dataset}_{out_dataset}.pkl")
            if os.path.exists(filepath):
                pass
            else:
                clf = get_detector(detector_name).fit(X.reshape(len(X), -1))
                ret = (clf.predict(X), clf.predict(tstX), clf.predict(oodX), clf.predict(ood2X), clf.predict(ood3X))
                joblib.dump(ret, filepath)

def main():
    #mnist_pixel("mnistwo9")
    #mnist_pixel("mnistwo0")
    #mnist_feature()
    #cifar_pixel("cifar10wo0")
    cifar_pixel("cifar10wo4")
    #cifar_pixel("cifar100coarsewo0")
    cifar_pixel("cifar100coarsewo4")

    #in_dataset = "cifar10wo0"
    #base_model_path = f"./models/out_of_sample/pgd-64-{in_dataset}-70-1.0-0.01-ce-vtor2-WRN_40_10-0.0-2-adam-0-0.0-ep0070.pt"
    #n_classes = 9
    #cifar_feature(in_dataset, base_model_path, n_classes)

    #in_dataset = "cifar100coarsewo0"
    #base_model_path = f"./models/out_of_sample/pgd-64-{in_dataset}-70-1.0-0.01-ce-vtor2-WRN_40_10-0.0-2-adam-0-0.0-ep0070.pt"
    #n_classes = 19
    #cifar_feature(in_dataset, base_model_path, n_classes)

if __name__ == "__main__":
    main()
