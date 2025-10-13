from collections.abc import Mapping
import torch
import matplotlib.pyplot as plt
import numpy as np
import random


def rounds_mask(n_rounds):
    mask = torch.zeros((len(n_rounds), 10), dtype=torch.bool)
    for i, max_j in enumerate(n_rounds):
        mask[i, :max_j] = True
    return mask


def move_to(obj, to_device):
    if torch.is_tensor(obj):
        return obj.to(to_device)
    elif isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            res[k] = move_to(v, to_device)
        return res
    elif isinstance(obj, list):
        res = []
        for v in obj:
            res.append(move_to(v, to_device))
        return res
    if isinstance(obj, Mapping):
        return obj.to(to_device)
    else:
        raise TypeError("Invalid type for move_to")


def get_model_name(model_params, model_name=None):
    model_params = [(key, value) for key, value in model_params.items()] if len(model_params) else []
    model_params = sorted(model_params)
    path = str(model_name) if model_name is not None else ""
    for param, value in model_params:
        path += "_"*len(path) + f"{param}={value}"
    return path


from scipy.optimize import curve_fit


def learn_sigmoid_weighting_by_reaction_time(data, bin_window=10, show_graph=False, normalized=True):

    bins = np.linspace(0, 101, 101).reshape(-1, 1)
    bin_centers = np.zeros_like(bins)
    bin_size = 100 / (len(bins) - 1)
    print(bin_size)
    for i in range(len(bins)):
        bins[i] = np.percentile(data["reaction_time"], i * bin_size)
    for i in range(len(bins) - 1):
        bin_centers[i] = np.percentile(data["reaction_time"], (i + 0.5) * bin_size)
    y = np.zeros_like(bins)
    c = np.zeros_like(bins)

    y[0] = data[(data["reaction_time"] <= bins[1].item())].didWin.mean()
    c[0] = data[(data["reaction_time"] <= bins[1].item())].didWin.count()

    for i in range(1, len(bins) - 1):
        min_bin = np.clip(i - bin_window, a_min=0, a_max=len(bins) - 1)
        max_bin = np.clip(i + bin_window + 1, a_min=0, a_max=len(bins) - 1)
        d = data[
            (bins[min_bin].item() < data["reaction_time"]) & (data["reaction_time"] <= bins[max_bin].item())].didWin
        y[i] = d.mean()
        c[i] = d.count()

    reaction_time = bin_centers[:-1].reshape(-1)
    log_reaction_time = np.log(reaction_time)
    winning_percentage = y[:-1].reshape(-1)

    def sigmoid(x, L, x0, k, b):
        y = L / (1 + np.exp(-k * (x - x0))) + b
        return (y)

    p0 = [max(log_reaction_time), np.median(log_reaction_time), 1,
          min(log_reaction_time)]  # this is an mandatory initial guess

    popt, pcov = curve_fit(sigmoid, log_reaction_time, winning_percentage, p0, method='dogbox')

    lower_bound = min(log_reaction_time)
    upper_bound = max(log_reaction_time)

    myline = np.linspace(lower_bound, upper_bound, 100)
    myline_y = sigmoid(myline, *popt)

    if show_graph:
        plt.plot(log_reaction_time, winning_percentage, 'o', label='data')
        plt.plot(myline, myline_y, label='fit')
        plt.ylim(0.5, 0.8)
        plt.xlabel("log(reaction time)")
        plt.ylabel("winning chance")
        plt.legend(loc='best')

    def weighting_function(x, sigmoid, popt):
        x = np.log(x)
        return sigmoid(x, *popt)

    best_sigmoid_value = weighting_function(8000, sigmoid, popt)

    def weighting_function_normalized(x, sigmoid, popt, best_sigmoid_value):
        x = np.log(x)
        sig = sigmoid(x, *popt)
        weight = (sig - 0.5) / (best_sigmoid_value-0.5)
        weight = np.clip(weight, 0, 1)
        return weight

    if normalized:
        return lambda x: weighting_function_normalized(x, sigmoid, popt, best_sigmoid_value)
    else:
        return lambda x: weighting_function(x, sigmoid, popt)


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

