import argparse
import contextlib
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import humanize
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib.ticker import AutoMinorLocator
from prettytable.colortable import ColorTable, Theme
from sklearn.metrics import classification_report, confusion_matrix


sys.stdout.flush()
from termcolor import cprint
from tqdm.auto import tqdm

from analysis_pipeline_1D import sr_routine
from src.loaders import Importer
from src.architectures import ShapeAdaptiveConvNet, SmallModel
from src.metrics import calculate_r2_and_mse, reshape_by_time
from src.auxiliary_functions import (
    CheckpointBest,
    EarlyStopper,
    count_parameters,
    prediction_based_method,
)

parser = argparse.ArgumentParser(
    prog="TetrisCNN training routine",
    description="""
    Program for TetrisCNN adaptive kernel network training
    """,
    epilog="For analysis of a NN generated from this routine please refer to TeTriss_load.ipynb or Full_stack*.*",
)
parser.add_argument(
    "--model",
    help="Overrides chosen physical model. Default is TFIM",
    default="TFIM",
)
parser.add_argument(
    "--phase",
    help="Overrides chosen initial phase in physical model. Default is TFIM_ferro",
    default="TFIM_ferro",
)
parser.add_argument(
    "--phys_data_path",
    help="Overrides path to physical data. Default is './TFIM_datasets/'.",
    default=None,
)
parser.add_argument(
    "--classes_no",
    help="Overrides Number of classes in classification training. Default is 2",
    default=2,
)
parser.add_argument(
    "-v",
    "--verbose",
    help="Toggles printing of extended training statistics",
    action="count",
    default=0,
)
parser.add_argument(
    "-ep",
    "--epochs",
    help="Overrides training epoch count. Default is 500",
    default=500,
)
parser.add_argument(
    "-bs",
    "--batch_size",
    help="Overrides batch size. Default is 32",
    default=32,
)
parser.add_argument(
    "-lr",
    "--learning_rate",
    help="Overrides learning rate. Default is 5e-4",
    default=5e-4,
)
parser.add_argument(
    "-wd",
    "--weight_decay",
    help="Overrides weight decay. Default is 1e-2",
    default=1e-1,
)
parser.add_argument(
    "-opt",
    "--optimizer",
    help="Overrides choice of optimizer. Default is AdamW, can be changed to SGD or Adagrad.",
    default="AdamW",
)
parser.add_argument(
    "-d",
    "--bases",
    nargs="*",
    help="Bases to use for network training",
    default=["y"],
)
parser.add_argument(
    "-es",
    "--early_stop",
    nargs="+",
    help="Early stopping parameters override. The order needed is: [patience, tolerance, warmup_proc]. Set to -1 to turn off. Defaults are [5, 1e-4, 0.35].",
    default=["20", "1e-4", "0.35"],
)
parser.add_argument(
    "-lb",
    "--lambdas",
    nargs="+",
    help="Kernel penalties parameters override. The order needed is: [base, penmin, penmax, n_pen]. From this, a penalties logspace in constructed, with"
    + " base `base`, and exponents ranging from `penmin` to `penmax`. Also, for each kernel of the same size, the penalty will be multiplied by `n_pen`."
    + " Default is [10, -4, -1, 1].",
    default=["10", "-4", "-1", "1"],
)
parser.add_argument(
    "--back_depth",
    help="Overrides the depth of the backportion. The default is 1",
    default=1,
)
parser.add_argument(
    "--model_save_root",
    help="Defines the root directory for saving the model. Default is 'Models/'",
    default="Models/",
)
parser.add_argument(
    "--cutoff",
    help="At what val_accuracy should the model be consider as succesfully trained",
    default=0.65,
)
parser.add_argument(
    "--simult",
    help="Allows for simultaneous training of networks with the same parameters, saving them to 'Duplicated' folder instead of 'Generated'.",
    action="count",
    default=0,
)
parser.add_argument("-gpu", help="When set to 0, disables the use of GPU device. Default is 1.", default=1)
parser.add_argument(
    "-reg",
    help="Changes the task from regression on prediction of variable parameter of the model to classification. The possible targets for regression are: 'params', 'time' or 'none' (classification). The default is 'params'.",
    default="params",
)
parser.add_argument(
    "--skip_sr",
    help="Skips symbolic regression routine",
    action="count",
    default=0,
)
args = parser.parse_args()

# =========================
#   INDEPENDENT VARIABLES
# =========================
BS = int(args.batch_size)
EP = int(args.epochs)
LR = float(args.learning_rate)
WD = float(args.weight_decay)
conv_cutoff = args.cutoff

# Parameters of physical model

phys_model = args.model

if phys_model == "TFIM":
    out_labels = {
        "params": [r"$g$"],
    }
else:
    raise ValueError("This model is not implemented yet!")

snaps = args.bases
n_channels = len(snaps)

if args.phys_data_path is not None:
    root = Path(args.phys_data_path)
else:
    root = Path(f"./TFIM_datasets/")

# Model save location
save_root = Path(args.model_save_root)
if bool(args.simult):
    save_root = Path(
        f"Models/Duplicated/{time.strftime('%d-%m-%y_%H.%M.%S', time.localtime())}"
    )
save_root.mkdir(exist_ok=True, parents=True)

ws = 40  # Width of the table

# NN task parameters
try:
    reg = None if args.reg.lower() == "none" else args.reg
except AttributeError:
    reg = None
normalize_reg = True
classes_no = int(args.classes_no)  # Ignored in regression task
model_name = "TetrisCNN"
checkpoint_best = True
es_parser = args.early_stop  # Early stopping parameters read from argparse
print_all = bool(args.verbose)  # Print the technical information

# NN optimization parameters
optimizer = args.optimizer  # Optimizer read from argparse

# Parameter 'n_pen' regulates the penalization for each additional kernel of the same size. Models trained with this are marker as `VarLambdas`.
# To load model without varying penalty, set it to 1
exp_set = {
    "base": float(args.lambdas[0]),
    "penmin": float(args.lambdas[1]),
    "penmax": float(args.lambdas[2]),
    "n_pen": float(args.lambdas[3]),
}

# Computation device for Pytorch
if args.gpu > 0:
    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
else:
    device = torch.device("cpu")

# Data loading
importer_class = Importer
exp = importer_class(
    root,
    BS,
    reg=bool(reg is not None),
    snaps=snaps,
    device=device,
    phys_model=phys_model,
    phase=args.phase,
    rescale_y=normalize_reg,
)

trn_loader_straight = exp.get_train_loader(batch_size=BS, shuffle=False)
trn_sizes = exp.trn_val_test_sizes["train"]
trn_loader = exp.get_train_loader(batch_size=BS, shuffle=True)
val_loader = exp.get_val_loader()
test_loader = exp.get_test_loader()

# List of tested kernels
filt_num = 1
tested_kernels_list = [
    # Format is [(kernel height, kernel width), filter number, dilation]
    [(1, 1), filt_num, 1],
    [(2, 1), filt_num, 1],
    [(2, 1), filt_num, 2],
    [(2, 1), filt_num, 3],
    [(3, 1), filt_num, 1],
    [(3, 1), filt_num, 2],
    [(3, 1), filt_num, 3],
]
n_tested_kernels = len(tested_kernels_list)
max_area = max([k[0][0] * k[0][1] for k in tested_kernels_list])
max_dil = max([k[2] for k in tested_kernels_list])
max_filt_num = max([k[1] for k in tested_kernels_list])
out_feat_num = np.array([f_num for _, f_num, _ in tested_kernels_list]).sum()

# =========================
#    DEPENDENT VARIABLES
# =========================

# Determination of the penalty values
if type(exp_set) is dict:
    exponents = np.logspace(
        base=exp_set["base"],
        start=exp_set["penmin"],
        stop=exp_set["penmax"],
        num=max_area,
    ).round(5)
elif type(exp_set) is np.ndarray:
    exponents = exp_set

lambdas = []
for branch in tested_kernels_list:
    k_shape, k_num, dilation = branch
    k_area = k_shape[0] * k_shape[1]
    lb_tmp = []
    if k_area > 1:
        l = exponents[k_area - 1]
        for fnum in range(0, k_num):
            if exp_set["n_pen"] == 1:
                lb_tmp.append(l * dilation)
            else:
                lb_tmp.append(l * np.power(exp_set["n_pen"], fnum) * dilation)
    else:
        for fnum in range(0, k_num):
            l = exponents[k_area - 1]
            if exp_set["n_pen"] == 1:
                lb_tmp.append(l)
            else:
                lb_tmp.append(l * np.power(exp_set["n_pen"], fnum))
    lambdas += lb_tmp.copy()

lambdas = np.array(lambdas).flatten()

# Determination of model parameters
out_dims = 1 if classes_no < 3 else classes_no
if reg is not None:
    out_dims = len(out_labels[reg])

# Model definition
backportion_depth = int(args.back_depth)
backportion_dims_dict = {
    0: [out_feat_num, 32, out_dims],
    1: [out_feat_num, 32, 16, out_dims],
    2: [out_feat_num, 32, 64, 32, 8, 4, out_dims],
    3: [out_feat_num, 32, 64, 32, 16, 8, 4, out_dims],
}
backportion_dims = backportion_dims_dict[backportion_depth]

net = ShapeAdaptiveConvNet(
    input_dim=n_channels, shape_list=tested_kernels_list, device=device
).to(device)
classifier = SmallModel(backportion_dims, device=device).to(device)


# Model checkpointing
if checkpoint_best:
    checkpointer = CheckpointBest()

# Determination of activation and loss functions
if out_dims == 1:
    activation = torch.sigmoid
else:
    activation = torch.softmax

if reg is not None:
    criterion = nn.MSELoss()

# Determination of optimizer
if "Adagrad".lower() in optimizer.lower():
    opt = optim.Adagrad(
        list(net.parameters()) + list(classifier.parameters()),
        lr=LR,
        weight_decay=WD,
    )
    chosen_opt = "adagrad"
elif "SGD".lower() in optimizer.lower():
    opt = optim.SGD(
        list(net.parameters()) + list(classifier.parameters()),
        lr=LR,
        weight_decay=WD,
        momentum=0.5,
    )
    chosen_opt = "sgd"
elif "AdamW".lower() in optimizer.lower():
    opt = optim.AdamW(
        list(net.parameters()) + list(classifier.parameters()),
        lr=LR,
        weight_decay=WD,
    )
    chosen_opt = "adamw"
else:
    raise ValueError("Incorrect optimizer chosen!")

# Define early stopping
if int(es_parser[0]) != -1:
    early_stopping = {
        "patience": int(es_parser[0]),
        "tolerance": float(es_parser[1]),
        "warmup": EP * float(es_parser[2]),
    }
    early_stopper = EarlyStopper(
        patience=early_stopping["patience"],
        min_delta=early_stopping["tolerance"],
        warmup=early_stopping["warmup"],
    )
else:
    early_stopping = None

# Define training metrics tensors
if reg is None:
    all_trn_accs = np.zeros(EP)
    all_val_accs = np.zeros(EP)
    all_trn_correct = np.zeros((EP, len(trn_loader)))
    all_val_correct = np.zeros(EP)
    all_trn_errs = np.zeros((EP, len(trn_loader)))
else:
    all_trn_r2 = np.zeros(EP)
    all_val_r2 = np.zeros(EP)

all_losses = np.zeros((EP, len(trn_loader), 4))
all_trn_losses = np.zeros(EP)
all_trn_MSE = np.zeros(EP)
all_val_losses = np.zeros(EP)
all_val_activations = np.zeros((EP, out_feat_num))

# Define the theme for the table
th = Theme(
    default_color="36",
    vertical_color="32",
    horizontal_color="32",
    junction_color="36",
)

startTime = time.time()
if print_all:
    cprint("\n[INFO]", "magenta", end=" ")
    print("Training the network...")
    cprint("\n" + "".join(["=" for i in range(ws)]), "green")
    cprint(f"{'Neural Network parameters':^{ws}}\n", "green")
    count_parameters(net, device)
    count_parameters(classifier, device)
    cprint("\n" + "".join(["=" for i in range(ws)]), "green")

trn_len = trn_loader_straight.dataset.Y.shape[0]
val_len = val_loader.dataset.Y.shape[0]

trnY = trn_loader_straight.dataset.Y.numpy()
valY = val_loader.dataset.Y.numpy()

trn_tensor = trn_loader_straight.dataset.X.float().to(device)
val_tensor = val_loader.dataset.X.float().to(device)

val_time_tab = exp.val_g_tab
time_unique = np.unique(val_time_tab)
trn_time_tab = np.repeat(time_unique, 100)

del trn_loader_straight

# =========================
#      TRAINING LOOP
# =========================

train_pbar = tqdm(
    range(0, EP),
    colour="green",
    unit="epoch",
    position=1,
    leave=True,
    dynamic_ncols=True,
)

if not print_all:
    trn_bar = tqdm(
        total=1,
        colour="yellow",
        leave=True,
        position=2,
        desc="Trn stats || None",
        bar_format="{desc}",
        dynamic_ncols=True,
    )

    val_bar = tqdm(
        total=1,
        colour="yellow",
        leave=True,
        position=3,
        desc="Val stats || None",
        bar_format="{desc}",
        dynamic_ncols=True,
    )

for k in range(0, EP):
    dtype = torch.float32
    # ---TRAINING PHASE---
    for j, (x_batch, y_batch) in enumerate(trn_loader):
        net.train(True)
        classifier.train(True)
        opt.zero_grad()
        if out_dims != 1 and reg is None:
            y_batch = F.one_hot(y_batch, num_classes=classes_no)
        (x_batch, y_batch) = (x_batch.to(device), y_batch.to(device))
        logits_list = net(x_batch)
        # This is to avoid nan values in the output in case of using a dataset with defects
        logits_list = torch.nan_to_num(logits_list)
        if reg is None:
            if out_dims == 1:
                answer = classifier(logits_list)[
                    ..., 0
                ]  # <---- This worked with 2 classes and no one-hot encoding
            else:
                answer = classifier(logits_list)
        else:
            answer = classifier(logits_list)
        # This is to avoid nan values in the output in case of using a dataset with defects
        answer = torch.nan_to_num(answer)

        l1_reg = torch.zeros(1).to(device)
        for f, lambda1 in enumerate(lambdas):
            l1_reg += lambda1 * torch.norm(logits_list[:, f], p=1)

        # Here we will need an additional L1 on just the `net` parameters
        y_long = y_batch.type(dtype).to(device)
        if reg is None:
            if out_dims == 1:
                probs = activation(answer)
                the_loss = nn.BCEWithLogitsLoss()(answer.squeeze(), y_long.squeeze())
            else:
                probs = activation(answer, dim=1)
                the_loss = nn.CrossEntropyLoss()(answer, y_long)
        else:
            the_loss = criterion(
                torch.squeeze(answer.float()), torch.squeeze(y_long.float())
            )

        if reg is None:
            if out_dims == 1:
                all_trn_correct[k, j] = (y_long == (probs > 0.5)).sum().item()
                all_trn_errs[k, j] = np.power(
                    y_long.cpu().detach().numpy() - probs.cpu().detach().numpy(),
                    2,
                ).sum()
            else:
                all_trn_correct[k, j] = (
                    (probs.argmax(1) == y_batch.argmax(1))
                    .type(torch.float)
                    .sum()
                    .item()
                )
                all_trn_errs[k, j] = np.power(
                    y_batch.cpu().detach().numpy() - probs.cpu().detach().numpy(),
                    2,
                ).sum()

        loss = the_loss + l1_reg
        loss.backward()
        all_losses[k, j, 0] = loss.item()
        all_losses[k, j, 1] = the_loss.item()
        all_losses[k, j, 2] = l1_reg.item()

        opt.step()

    # ---VALIDATION PHASE---
    net.eval()
    classifier.eval()
    # training accuracy
    with torch.no_grad():
        logits_list = net(trn_tensor)
        # This is to avoid nan values in the output in case of using a dataset with defects
        logits_list = torch.nan_to_num(logits_list)
        if reg is None:
            if out_dims == 1:
                logits = classifier(logits_list)[
                    ..., 0
                ]  # <----- this worked without onehot encoding
                probs = activation(logits).cpu().detach().numpy()
            else:
                logits = classifier(logits_list)
            all_trn_accs[k] = (all_trn_correct.sum(axis=1) / trn_len)[k]
            all_trn_MSE[k] = all_trn_errs[k, :].sum() / trn_len
        else:
            answer = classifier(logits_list)
            y_long = torch.from_numpy(trnY).type(dtype).to(device)
            all_trn_MSE[k] = criterion(
                torch.squeeze(answer.float()), torch.squeeze(y_long.float())
            )
            answer = np.nan_to_num(answer.detach().cpu().numpy())
            y_long = y_long.detach().cpu().numpy()
            r2, _ = calculate_r2_and_mse(
                y_long, answer, time_tab=trn_time_tab, exp=exp, old_way=False
            )
            all_trn_r2[k] = r2
        all_trn_losses[k] = np.mean(all_losses[k, :, 0])

        # validation accuracy
        logits_list = net(val_tensor)
        # This is to avoid nan values in the output in case of using a dataset with defects
        logits_list = torch.nan_to_num(logits_list)
        if reg is None:
            if out_dims == 1:
                logits = classifier(logits_list)[
                    ..., 0
                ]  # <---- this worked without onehot encoding
                probs = activation(logits)
                y_val = (val_loader.dataset.Y).type(dtype).to(device)
                all_val_correct[k] = (y_val == (probs > 0.5)).sum().item()
            else:
                logits = classifier(logits_list)
                probs = activation(logits, dim=1)
                y_val = (
                    F.one_hot(val_loader.dataset.Y, num_classes=classes_no)
                    .type(dtype)
                    .to(device)
                )
                all_val_correct[k] = (
                    (probs.argmax(1) == y_val.argmax(1)).type(torch.float).sum().item()
                )
            all_val_accs[k] = all_val_correct[k] / val_len
            probs = probs.cpu().detach().numpy()
            all_val_losses[k] = ((y_val.cpu() - probs) ** 2).mean().numpy()
        else:
            answer = classifier(logits_list)
            y_long = torch.from_numpy(valY).type(dtype).to(device)
            if out_dims == 1:
                answer, y_long = torch.squeeze(answer.float()), torch.squeeze(
                    y_long.float()
                )
            else:
                answer, y_long = answer.float(), y_long.float()
            all_val_losses[k] = criterion(answer, y_long)
            # This is to avoid nan values in the output in case of using a dataset with defects
            answer = np.nan_to_num(answer.detach().cpu().numpy())
            y_long = y_long.detach().cpu().numpy()
            y_long = np.nan_to_num(y_long)
            r2, _ = calculate_r2_and_mse(
                y_long, answer, time_tab=val_time_tab, exp=exp, old_way=False
            )
            all_val_r2[k] = r2

        # Calculate activations
        activations_tmp = np.zeros(logits_list.shape[1])
        for ac_num, act_tmp in enumerate(logits_list.T):
            act_tmp, _ = reshape_by_time(act_tmp.cpu().detach().numpy(), val_time_tab)
            activation_max = act_tmp.max(axis=0)
            activation_min = act_tmp.min(axis=0)
            if np.abs(activation_max).max() > np.abs(activation_min).max():
                activations_tmp[ac_num] = activation_max
            else:
                activations_tmp[ac_num] = activation_min

        all_val_activations[k] = activations_tmp

    if checkpoint_best:
        saved_best = checkpointer(
            all_val_losses[k],
            k,
            [net, classifier],
        )

    if early_stopping is not None:
        if early_stopper.early_stop(all_trn_losses[k], k):
            break

    logits_numpy = logits_list.detach().cpu().numpy()
    logits_avg, logits_std = reshape_by_time(logits_numpy, val_time_tab)

    alive_thresh = 0.1 * np.abs(logits_avg).max()

    if print_all:
        print("Mean kernel activations:")
        for sh, (shape, num, dil) in enumerate(tested_kernels_list):
            table_tmp = ColorTable(
                [
                    f"{'Kernels'} {shape}",
                    f"{'Activation threshold':^24} = {alive_thresh:+4.2e}",
                ],
                theme=th,
            )
            table_tmp.add_row(
                [
                    f"{'Lambdas':^15}",
                    " ".join(
                        [f"{lb:+8.2e}" for lb in lambdas.reshape([-1, filt_num])[sh]]
                    ),
                ]
            )
            table_tmp.add_row(
                [
                    f"{'Mean':^15}\n{'Activations':^15} ",
                    "\n"
                    + " ".join(
                        [
                            f"{lb:+8.2e}"
                            for lb in logits_list.abs()
                            .max(axis=0)
                            .values.reshape([-1, filt_num])[sh]
                        ]
                    ),
                ]
            )
            table_tmp.add_row(
                [
                    f"{'Dominant':^15}\n{'Kernels':^15} ",
                    "\n"
                    + f"{int((logits_list.abs().max(axis=0).values.reshape([-1, filt_num])[sh] > alive_thresh).sum())}",
                ]
            )
            table_tmp.add_row(
                [
                    f"{'Convolution':^15}\n{'Dilation':^15} ",
                    "\n" + f"{dil}",
                ]
            )
            print(table_tmp)

        cprint("\n[INFO]", "magenta", end=" ")
        cprint(
            f"Current time : {time.strftime('%H:%M:%S', time.localtime())}",
            end=" ",
        )
        cprint("EPOCH:", "white", end=" ")
        print("{}/{}".format(k + 1, EP))
        if reg is None:
            print(
                "Train loss: {:.6f}, Train  MSE: {:.6f}, Train accuracy: {:.4f}".format(
                    all_trn_losses[k], all_trn_MSE[k], all_trn_accs[k]
                )
            )
            print(
                "                       Val   MSE: {:.6f},  Val  accuracy: {:.4f}\n".format(
                    all_val_losses[k], all_val_accs[k]
                )
            )
        else:
            print(
                "Train loss: {:.6f}, Train  MSE: {:.6f}, Train R^2: {:.4f}".format(
                    all_trn_losses[k], all_trn_MSE[k], all_trn_r2[k]
                )
            )
            print(
                "                       Val   MSE: {:.6f},  Val  R^2: {:.4f}\n".format(
                    all_val_losses[k], all_val_r2[k]
                )
            )

    else:
        if not print_all:
            if reg is None:
                trn_bar.set_description(
                    "Previous epoch stats || "
                    + f"Train loss: {all_trn_losses[k]:^12.6f} | Train MSE: {all_trn_MSE[k]:.6f} | Train accuracy: {all_trn_accs[k]:.6f}"
                ),
                val_bar.set_description(
                    "Previous epoch stats || "
                    + f" Val  loss: {all_val_losses[k]:^12.6f} |  Val  MSE: {all_val_losses[k]:.6f} | Val accuracy: {all_val_accs[k]:.6f}"
                ),
            else:
                trn_bar.set_description(
                    "Trn stats || "
                    + f"Train loss: {all_trn_losses[k]:^12.6f} | Train MSE: {all_trn_MSE[k]:.6f} | Train R^2: {all_trn_r2[k]:.6f}"
                ),
                val_bar.set_description(
                    "Val stats || "
                    + f" Val  loss: {all_val_losses[k]:^12.6f} |  Val  MSE: {all_val_losses[k]:.6f} |  Val  R^2: {all_val_r2[k]:.6f}"
                ),

    train_pbar.update(1)
endTime = time.time()
cprint("\n[INFO]", "magenta", end=" ")
print(
    "Total time taken to train the model: {}".format(
        humanize.precisedelta(endTime - startTime)
    )
)

#  =========================
#     CALCULATING STATS
#  =========================

if reg is None:
    converged = int(
        all_trn_accs[k] > float(conv_cutoff) and all_val_accs[k] > float(conv_cutoff)
    )
else:
    converged = int(
        all_trn_r2[k] > float(conv_cutoff) and all_val_r2[k] > float(conv_cutoff)
    )

kern_gen = {
    "fn": max_filt_num,
    "fs": max_area,
    "fd": max_dil,
    "kn": n_tested_kernels,
}

if exp_set["n_pen"] == 1:
    if reg is None:
        stats = [
            "model={}".format(phys_model),
            # "phase=[{}]".format(phase),
            "converged={}".format(
                int(
                    all_trn_accs[k] > float(conv_cutoff)
                    and all_val_accs[k] > float(conv_cutoff)
                )
            ),
            "kernels={}".format(
                str(tested_kernels_list if n_tested_kernels < 4 else kern_gen)
            ),
            "lambdas={}".format(str(list(exponents))),
            "backportion_depth={}".format(backportion_depth),
        ]
    else:
        stats = [
            "model={}".format(phys_model),
            # "phase=[{}]".format(phase),
            "converged={}".format(
                int(
                    all_trn_r2[k] > float(conv_cutoff)
                    and all_val_r2[k] > float(conv_cutoff)
                )
            ),
            "kernels={}".format(
                str(tested_kernels_list if n_tested_kernels < 4 else kern_gen)
            ),
            "lambdas={}".format(str(list(exponents))),
            "backportion_depth={}".format(backportion_depth),
        ]
else:
    if reg is None:
        stats = [
            "model={}".format(phys_model),
            # "phase=[{}]".format(phase),
            "converged={}".format(
                int(
                    all_trn_accs[k] > float(conv_cutoff)
                    and all_val_accs[k] > float(conv_cutoff)
                )
            ),
            "kernels={}".format(
                str(tested_kernels_list if n_tested_kernels < 4 else kern_gen)
            ),
            "lambdas={}".format(str(list(exponents))),
            "backportion_depth={}".format(backportion_depth),
            "VarLambdas",
        ]
    else:
        stats = [
            "model={}".format(phys_model),
            # "phase=[{}]".format(phase),
            "converged={}".format(
                int(
                    all_trn_r2[k] > float(conv_cutoff)
                    and all_val_r2[k] > float(conv_cutoff)
                )
            ),
            "kernels={}".format(
                str(tested_kernels_list if n_tested_kernels < 4 else kern_gen)
            ),
            "lambdas={}".format(str(list(exponents))),
            "backportion_depth={}".format(backportion_depth),
            "VarLambdas",
        ]
if max_dil > 1:
    stats.append(f"Dilated")
if chosen_opt != "adagrad":
    stats.append(f"opt={chosen_opt}")
if BS != 32:
    stats.append(f"bs={BS}")
stats.append(f"epochs={k}")
if type(snaps) is not dict:
    stats.append(
        "channels={}".format(str(snaps)),
    )
else:
    stats.append(
        "channels={}".format(str(["None"])),
    )
if reg is not None:
    stats.append(f"Regression={reg}")

if phys_model == "XY" or (phys_model == "Ising" and args.phys_phase != "ferro"):
    stats.append(f"Phase={args.phys_phase}")

stats = "_".join(stats).replace(":", "")

if reg is None:
    outs = {
        "losses": {"train": all_trn_losses, "val": all_val_losses},
        "accs": {"train": all_trn_accs, "val": all_val_accs},
        "raw": all_losses,
        "activations": all_val_activations,
    }
else:
    outs = {
        "losses": {"train": all_trn_losses, "val": all_val_losses},
        "R^2": {"train": all_trn_r2, "val": all_val_r2},
        "raw": all_losses,
        "activations": all_val_activations,
    }

# =========================
#      SAVING THE MODEL
# =========================

save_path = save_root.joinpath(stats)
save_path.mkdir(exist_ok=True)

save_path.joinpath("PNG").mkdir(parents=True, exist_ok=True)

torch.save(net.state_dict(), str(save_path.joinpath(f"{model_name}_net.dict")))
torch.save(
    classifier.state_dict(),
    str(save_path.joinpath(f"{model_name}_classifier.dict")),
)

net_checkpoint, classifier_checkpoint = checkpointer.get_checkpointed_model()
best_ep = checkpointer.get_checkpointed_epoch()
torch.save(
    net_checkpoint.state_dict(),
    str(save_path.joinpath(f"{model_name}_net_best_epoch={best_ep}.dict")),
)
torch.save(
    classifier_checkpoint.state_dict(),
    str(save_path.joinpath(f"{model_name}_classifier_best_epoch={best_ep}.dict")),
)

torch.save(outs, str(save_path.joinpath(f"{model_name}_history.pickle")))

# =========================
#      SAVING STATS
# =========================

if type(snaps) is not dict:
    s_params = [
        (
            "Training started",
            datetime.fromtimestamp(startTime).strftime("%H:%M:%S, %d/%m/%Y"),
        ),
        (
            "Training finished",
            time.strftime("%H:%M, %d/%m/%y", time.localtime()),
        ),
        ("Training time", humanize.precisedelta(endTime - startTime)),
        ("Physical model", phys_model),
        # ("Phase in model", phase),
        ("Used bases", ", ".join(snaps)),
        ("Used device", str(device)),
        ("Batch size", BS),
        ("Number of epochs", f"{k}/{EP}"),
        ("Learning rate", LR),
        ("Optimizer", opt.__class__),
        ("Weight decay", WD),
        ("Early stopping", str(early_stopping)),
        ("Used kernels", ", ".join([str(s) for s in tested_kernels_list])),
        ("Kernel penalties", ", ".join(exponents.astype(str))),
        ("Penalties params", ", ".join(args.lambdas)),
        ("Number of classes", classes_no),
        ("One-hot encoded", f"{'Yes' if classes_no > 2 else 'No'}"),
        ("Activation function", activation.__name__),
        ("Regression", reg),
        ("Backportion depth", backportion_depth),
        ("Backportion dims", backportion_dims),
    ]
else:
    s_params = [
        (
            "Training started",
            datetime.fromtimestamp(startTime).strftime("%H:%M:%S, %d/%m/%Y"),
        ),
        (
            "Training finished",
            time.strftime("%H:%M, %d/%m/%y", time.localtime()),
        ),
        ("Training time", humanize.precisedelta(endTime - startTime)),
        ("Physical model", phys_model),
        # ("Phase in model", phase),
        ("Used bases", str(snaps)),
        ("Used device", str(device)),
        ("Batch size", BS),
        ("Number of epochs", f"{k}/{EP}"),
        ("Learning rate", LR),
        ("Optimizer", opt.__class__),
        ("Weight decay", WD),
        ("Early stopping", str(early_stopping)),
        ("Used kernels", ", ".join([str(s) for s in tested_kernels_list])),
        ("Kernel penalties", ", ".join(exponents.astype(str))),
        ("Penalties params", ", ".join(args.lambdas)),
        ("Number of classes", classes_no),
        ("One-hot encoded", f"{'Yes' if classes_no > 2 else 'No'}"),
        ("Activation function", activation.__name__),
    ]

if phys_model == "XY" or (phys_model == "Ising" and args.phys_phase != "ferro"):
    s_params.append(("Phase", args.phys_phase))

if reg is None:
    res_tab = [
        (
            "Converged",
            f"{'Yes' if all_val_accs[k] > float(conv_cutoff) else 'No'}",
        ),
        ("Loss", f"{all_trn_losses[k]:.6f}"),
        ("Train  MSE", f"{ all_trn_MSE[k]:.6f}"),
        ("Val  MSE", f"{ all_val_losses[k]:.6f}"),
        ("Train accuracy", f"{ all_trn_accs[k]:.6f}"),
        ("Val accuracy", f"{ all_val_accs[k]:.6f}"),
    ]
else:
    res_tab = [
        (
            "Converged",
            f"{'Yes' if all_val_r2[k] > float(conv_cutoff) else 'No'}",
        ),
        ("Loss", f"{all_trn_losses[k]:.6f}"),
        ("Train  MSE", f"{ all_trn_MSE[k]:.6f}"),
        ("Val  MSE", f"{ all_val_losses[k]:.6f}"),
        ("Train R^2", f"{ all_trn_r2[k]:.6f}"),
        ("Val R^2", f"{ all_val_r2[k]:.6f}"),
    ]

param_file = save_path.joinpath("params.txt")
tmp_file = save_path.joinpath("tmp.txt")

with open(tmp_file, "w", encoding="utf-8") as f:
    with contextlib.redirect_stdout(f):
        cprint("\n" + "".join(["=" for i in range(ws)]), "green")
        print(f"{'Mean kernel activations':^{ws}}\n")
        for sh, (shape, num, dil) in enumerate(tested_kernels_list):
            table_tmp = ColorTable(
                [
                    f"{'Kernels'} {shape}",
                    f"{'Activation threshold':^24} = {alive_thresh:+4.2e}",
                ],
                theme=th,
            )
            table_tmp.add_row(
                [
                    f"{'Lambdas':^15}",
                    " ".join(
                        [f"{lb:+8.2e}" for lb in lambdas.reshape([-1, filt_num])[sh]]
                    ),
                ]
            )
            table_tmp.add_row(
                [
                    f"{'Mean':^15}\n{'Activations':^15} ",
                    "\n"
                    + " ".join(
                        [
                            f"{lb:+8.2e}"
                            for lb in logits_list.mean(axis=0).reshape([-1, filt_num])[
                                sh
                            ]
                        ]
                    ),
                ]
            )
            table_tmp.add_row(
                [
                    f"{'Dominant':^15}\n{'Kernels':^15} ",
                    "\n"
                    + f"{int(((logits_list.mean(axis=0).reshape([-1, filt_num])[sh]).abs() > alive_thresh).sum())}",
                ]
            )
            table_tmp.add_row(
                [
                    f"{'Convolution':^15}\n{'Dilation':^15} ",
                    "\n" + f"{dil}",
                ]
            )
            print(table_tmp)

with open(tmp_file, "r", encoding="utf-8") as f:
    kernels = f.read()
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    kernels = ansi_escape.sub("", kernels)

with open(tmp_file, "w", encoding="utf-8") as f:
    with contextlib.redirect_stdout(f):
        cprint("".join(["=" for i in range(ws)]), "green")
        cprint(f"{'Neural Network parameters':^{ws}}\n", "green")
        count_parameters(net, device)
        count_parameters(classifier, device)
        cprint("\n" + "".join(["=" for i in range(ws)]), "green")

with open(tmp_file, "r", encoding="utf-8") as f:
    nerd_stats = f.read()
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    nerd_stats = ansi_escape.sub("", nerd_stats)

tmp_file.unlink()

with open(param_file, "w", encoding="utf-8") as f:
    f.write(f"{'Training parameters':^20}\n\n")

    for lab, val in s_params:
        f.write(f"{lab:^20} : {val}\n")

    f.write(f"\n\n{'Results':^20}\n\n")
    for lab, val in res_tab:
        f.write(f"{lab:^20} : {val}\n")

    f.write("\n")
    f.write(kernels)
    f.write("\n")
    f.write(nerd_stats)

table2 = ColorTable(["Results", ""], theme=th)
for ln in res_tab:
    table2.add_row(list(ln))
print(table2)

for sh, (shape, num, dil) in enumerate(tested_kernels_list):
    table_tmp = ColorTable(
        [
            f"{'Kernels'} {shape}",
            f"{'Activation threshold':^24} = {alive_thresh:+4.2e}",
        ],
        theme=th,
    )
    table_tmp.add_row(
        [
            f"{'Lambdas':^15}",
            " ".join([f"{lb:+8.2e}" for lb in lambdas.reshape([-1, filt_num])[sh]]),
        ]
    )
    table_tmp.add_row(
        [
            f"{'Mean':^15}\n{'Activations':^15} ",
            "\n"
            + " ".join(
                [
                    f"{lb:+8.2e}"
                    for lb in logits_list.mean(axis=0).reshape([-1, filt_num])[sh]
                ]
            ),
        ]
    )
    table_tmp.add_row(
        [
            f"{'Dominant':^15}\n{'Kernels':^15} ",
            "\n"
            + f"{int(((logits_list.mean(axis=0).reshape([-1, filt_num])[sh]).abs() > alive_thresh).sum())}",
        ]
    )
    table_tmp.add_row(
        [
            f"{'Convolution':^15}\n{'Dilation':^15} ",
            "\n" + f"{dil}",
        ]
    )
    print(table_tmp)

# =========================
#     PLOTTING HISTORY
# =========================

history = outs

try:
    last_ep = np.where(history["losses"]["train"] == 0)[0][0]
except IndexError:
    last_ep = EP - 1

# Maybe move these parameters to the top or out of the function later
fs = 30
lw = 3
ms = 12
grid = True

fig = plt.figure(layout="constrained", figsize=(20, 20))
fig.suptitle("Training history", size=1.5 * fs)

subfigs = fig.subfigures(2, 1, wspace=0.07, width_ratios=[1])

axs = subfigs.flatten()[0].subplot_mosaic([["A", "B"], ["C", "B"]])

ax = axs["A"]

ax.plot(
    history["losses"]["train"][:last_ep],
    label="Full training set loss",
    linewidth=lw,
)
if reg is None:
    ax.plot(
        history["raw"][:last_ep, :, 1].mean(axis=1),
        "--",
        label="Training set BCE Loss",
        linewidth=lw,
    )
else:
    ax.plot(
        history["raw"][:last_ep, :, 1].mean(axis=1),
        "--",
        label="Training set MSE Loss",
        linewidth=lw,
    )
ax.set_ylabel("Loss", size=fs)
ax.plot(history["losses"]["val"][:last_ep], label="Validation set loss")
ax.legend(fontsize=0.8 * fs)
ax.set_title("Mean losses", size=fs)

if (
    history["losses"]["train"][:last_ep].max()
    > 2 * history["losses"]["train"][:last_ep].min()
):
    ax.set_yscale("symlog", linthresh=1e-4)

ax = axs["C"]

ax.plot(
    history["raw"][:last_ep, :, 2].mean(axis=1),
    "--",
    label="Training set L1\nkernel regularization",
    linewidth=lw,
)
ax.legend(fontsize=0.8 * fs)
ax.set_title("Mean L1 kernel regularization", size=fs)
ax.set_ylabel("L1 regularization", size=fs)
ax.set_xlabel("Epoch", size=fs)

ax = axs["B"]

if reg is None:
    ax.plot(
        history["accs"]["train"][:last_ep],
        ":",
        label="Training set",
        color="C1",
        linewidth=lw,
    )
    ax.plot(
        history["accs"]["val"][:last_ep],
        ":",
        label="Validation set",
        color="C2",
        linewidth=lw,
    )
    ax.plot(
        history["accs"]["train"][:last_ep],
        "o",
        alpha=0.5,
        color="C1",
        markersize=ms,
    )
    ax.plot(
        history["accs"]["val"][:last_ep],
        "o",
        alpha=0.5,
        color="C2",
        markersize=ms,
    )
else:
    ax.plot(
        history["R^2"]["train"][:last_ep],
        ":",
        label="Training set",
        color="C1",
        linewidth=lw,
    )
    ax.plot(
        history["R^2"]["val"][:last_ep],
        ":",
        label="Validation set",
        color="C2",
        linewidth=lw,
    )
    ax.plot(
        history["R^2"]["train"][:last_ep],
        "o",
        alpha=0.5,
        color="C1",
        markersize=ms,
    )
    ax.plot(
        history["R^2"]["val"][:last_ep],
        "o",
        alpha=0.5,
        color="C2",
        markersize=ms,
    )
ax.legend(fontsize=0.8 * fs)
if reg is None:
    ax.set_title("Accuracies", size=fs)
    ax.set_ylabel("Accuracy", size=fs)
else:
    ax.set_title(r"$R^2$", size=fs)
    ax.set_ylabel("Values", size=fs)
ax.set_xlabel("Epoch", size=fs)
ax.set_ylim(-0.1, 1.1)

for num, ax in enumerate(axs.values()):
    ax.tick_params(axis="both", which="major", labelsize=fs)
    if num > 0:
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        if grid:
            ax.grid(which="minor", color="gray", linestyle=":", linewidth=0.5)
            ax.grid(which="major", color="gray", linestyle="dashdot", linewidth=0.7)
    else:
        ax.grid(which="major", axis="both", linestyle="dashdot", linewidth=0.7)

axs = subfigs.flatten()[1].subplots(1, 1)

ax = axs

cmap = matplotlib.colormaps["tab10"]

for k_num, k_act in enumerate(history["activations"][:last_ep].T):
    ax.plot(
        k_act,
        "o",
        color=cmap(k_num),
        label=f"Kernel {k_num}\nSize {tested_kernels_list[k_num//filt_num][0]}, dilation {tested_kernels_list[k_num//filt_num][2]}",
        alpha=0.7,
        markersize=ms,
    )
    ax.plot(k_act, ":", color=cmap(k_num), alpha=0.4, linewidth=lw)
ax.set_title("Mean kernel activations on validation set", size=fs)
ax.set_ylabel(r"$a_k$", size=fs)
ax.set_xlabel("Epoch", size=fs)
ax.hlines(0, 0, last_ep, "k", "--", alpha=0.5, linewidth=lw)

if history["activations"][:last_ep].max() > 10 * history["activations"][:last_ep].min():
    linthresh = history["activations"][:last_ep].max() / 10
    ax.set_yscale("symlog", linthresh=linthresh)
    ax.hlines(
        linthresh,
        0,
        last_ep,
        "k",
        "--",
        alpha=0.3,
        linewidth=lw,
        label="Linear threshold",
    )
    ax.hlines(-linthresh, 0, last_ep, "k", "--", alpha=0.3, linewidth=lw)

ax.legend(
    fontsize=0.6 * fs,
    ncols=1,
    fancybox=True,
    shadow=True,
    loc="center left",
    bbox_to_anchor=(1, 0.5),
)

ax.tick_params(axis="both", which="major", labelsize=fs)
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
if grid:
    ax.grid(which="minor", color="gray", linestyle=":", linewidth=0.5)
    ax.grid(which="major", color="gray", linestyle="dashdot", linewidth=0.7)

plt.savefig(f"{str(save_path)}/training_history.pdf")
plt.savefig(f"{str(save_path.joinpath('PNG'))}/training_history.png")

# =========================
#     MODEL TESTING
# =========================

loader = test_loader
time_tab = exp.test_g_tab
with torch.no_grad():
    test_tensor = loader.dataset.X.clone().float().to(device)
    logits_list = net(test_tensor)
    # This is to avoid nan values in the output in case of using a dataset with defects
    logits_list = torch.nan_to_num(logits_list)
    if reg is None:
        if out_dims == 1:
            logits = classifier(logits_list)[
                ..., 0
            ]  # <---- this worked without onehot encoding
            probs = activation(logits)
            preds = (probs > 0.5).detach().cpu().numpy().astype(np.float32)
        else:
            logits = classifier(logits_list)
            probs = activation(logits, dim=1)
            preds = probs.argmax(1).type(torch.float).detach().cpu().numpy()
        probs = probs.detach().cpu().numpy()
        truth = loader.dataset.Y.numpy()
        print(classification_report(truth, preds))
        print(confusion_matrix(truth, preds))
        param_file = load_path.joinpath("test_accs.txt")
        with open(param_file, "w", encoding="utf-8") as f:
            with contextlib.redirect_stdout(f):
                print("Accuracies and confusion matrix on test set\n\n")
                print(classification_report(truth, preds))
                print(confusion_matrix(truth, preds))
    else:
        answer = classifier(logits_list)
        truth = loader.dataset.Y.float().to(device)
        if reg == "time":
            answer = answer.squeeze(-1)
            truth = truth.squeeze(-1)
        test_MSE = torch.nn.MSELoss()(answer, truth)
        # This is to avoid nan values in the output in case of using a dataset with defects
        answer = np.nan_to_num(answer.detach().cpu().numpy())
        truth = truth.detach().cpu().numpy()
        r2, _ = calculate_r2_and_mse(
            truth, answer, time_tab=time_tab, exp=exp, old_way=False
        )
        probs = answer
        param_file = save_path.joinpath("test_stats.txt")
        with open(param_file, "w", encoding="utf-8") as f:
            with contextlib.redirect_stdout(f):
                print("Statistics on test set\n\n")
                print(f"Test MSE : {test_MSE:.4f} | Test R^2 : {r2:.3f}")
        print(f"Test MSE : {test_MSE:.4f} | Test R^2 : {r2:.3f}")
logits_numpy = logits_list.detach().cpu().numpy()

#  =========================
#   PREDICTION BASED METHOD
#  =========================

prediction_path = save_path.joinpath("Prediction_based_method")
prediction_path.mkdir(exist_ok=True, parents=True)

exp.time = np.unique(exp.test_g_tab)

prediction_based_method(
    exp,
    truth,
    answer,
    reg,
    phys_model,
    (150, 1),  # Hardcoded TFIM
    logits_numpy,
    time_tab,
    pred_labels=out_labels[reg],
    kernel_names=tested_kernels_list,
    save=prediction_path,
    merge_params=True,
    spl_k=4,
    spl_s=0.0053,
)

# # ========================= # #
# #    ANALYSIS FILE CALL     # #
# # ========================= # #

# ======================================
#    Symbolic regression parameters
# ======================================

model_sr_folder = save_path.joinpath("SR")
model_sr_folder.mkdir(exist_ok=True, parents=True)

sr_tmp_path = model_sr_folder.joinpath("Halls_of_fame")
sr_tmp_path.mkdir(exist_ok=True, parents=True)

just_new = False

symbolic_reg_params = {
    "niterations": 400,  # < Increase me for better results
    "binary_operators": ["+", "*"],
    "unary_operators": [
        "neg",
        # "abs",
    ],
    "constraints": {"*": (1, 1)},
    "model_selection": "best",
    "maxdepth": 10,
    "tempdir": sr_tmp_path,
}

predictor_symbolic_reg_params = {
    "niterations": 400,  # < Increase me for better results
    "binary_operators": ["+", "*"],
    "unary_operators": [
        "neg",
        "square",
        "cube",
        "abs",
        "inv(x) = 1/x",
    ],
    "constraints": {"*": (1, 1)},
    "extra_sympy_mappings": {"inv": lambda x: 1 / x},
    "model_selection": "best",
    "maxdepth": 12,
    "tempdir": sr_tmp_path,
}

sr_routine(
    save_path,
    root,
    model_name=model_name,
    phys_model=phys_model,
    phys_phase=args.phase,
    snaps=snaps,
    filt_num=filt_num,
    normalize_reg=normalize_reg,
    approx=True,
    out_labels=out_labels,
    corrs_sr_params_override=symbolic_reg_params,
    predictor_sr_params_override=predictor_symbolic_reg_params,
    skip_sr=bool(args.skip_sr),
    fs=fs,
    lw=lw,
    ms=ms,
    figsize_base=(20, 10),
)
