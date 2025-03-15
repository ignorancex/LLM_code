# ==============================================================================
# deprecated functions
import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from dunedn.configdn import PACKAGE
from dunedn.utils.utils import confusion_matrix

# instantiate logger
logger = logging.getLogger(PACKAGE)


def plot_crops(out_dir, imgs, name, sample):
    """
    Plots ADC colormap of channel vs time of 5x5 samples.

    Parameters
    ----------
        - d: string, directory path of output img
        - imgs: torch.Tensor of shape (N,C,H,W)
        - name: string, additional string to output name
        - sample: torch.Tensor selected image indices to be printed
        - wire: torch.Tensor, selected wires indices to be printed
    """
    imgs = imgs.permute(0, 2, 3, 1).squeeze(-1)
    samples = imgs[sample]

    fname = os.path.join(out_dir, "_".join([name, "crops.png"]))
    fig, axs = plt.subplots(5, 5, figsize=(25, 25))
    for i in range(5):
        for j in range(5):
            ax = axs[i, j]
            z = ax.imshow(samples[i * 5 + j])
            fig.colorbar(z, ax=ax)
    plt.savefig(fname)
    plt.close()
    logger.info("Saved image at %s" % fname)


def plot_wires(out_dir, imgs, name, sample, wire):
    """
    Plots ADC vs time of 5x5 channels.
    Parameters
    ----------
        - out_dir: string, directory path of output img
        - imgs: torch.Tensor of shape (N,C,H,W)
        - name: string, additional string to output name
        - sample: torch.Tensor selected image indices to be printed
        - wire: torch.Tensor, selected wires indices to be printed
    """
    imgs = imgs.permute(0, 2, 3, 1).squeeze(-1)
    samples = imgs[sample]

    fname = os.path.join(out_dir, "_".join([name, "wires.png"]))
    fig = plt.figure(figsize=(25, 25))
    for i in range(5):
        for j in range(5):
            ax = fig.add_subplot(5, 5, i * 5 + j + 1)
            ax.plot(samples[i * 5 + j, wire[i * 5 + j]], linewidth=0.3)
    plt.savefig(fname)
    plt.close()
    logger.info("Saved image at %s" % fname)


def print_cm(a, f, epoch):
    """
    Print confusion matrix at a given epoch a for binary classification to file named f

    Parameters
    ----------
        - a: np.array, confusion matrix of shape=(2,2)
        - fname: str, output file name
        - epoch: int, epoch number
    """
    tot = a.sum()
    logger.info(f"Epoch: {epoch}", file=f)
    logger.info("Over a total of %d pixels:\n" % tot, file=f)
    logger.info("------------------------------------------------", file=f)
    logger.info("|{:>20}|{:>12}|{:>12}|".format("", "Hit", "No hit"), file=f)
    logger.info("------------------------------------------------", file=f)
    logger.info(
        "|{:>20}|{:>12.4e}|{:>12.4e}|".format(
            "Predicted hit", a[1, 1] / tot, a[0, 1] / tot
        ),
        file=f,
    )
    logger.info("------------------------------------------------", file=f)
    logger.info(
        "|{:>20}|{:>12.4e}|{:>12.4e}|".format(
            "Predicted no hit", a[1, 0] / tot, a[0, 0] / tot
        ),
        file=f,
    )
    logger.info("------------------------------------------------", file=f)
    logger.info(
        "{:>21}|{:>12}|{:>12}|".format("", "Sensitivity", "Specificity"), file=f
    )
    logger.info("                     ---------------------------", file=f)
    logger.info(
        "{:>21}|{:>12.4e}|{:>12.4e}|".format(
            "", a[1, 1] / (a[1, 1] + a[1, 0]), a[0, 0] / (a[0, 1] + a[0, 0])
        ),
        file=f,
    )
    logger.info("                     ---------------------------\n\n", file=f)


def save_ROI_stats(args, epoch, clear, dn, t, ana=False):
    """
    Plot stats of the ROI: confusion matrix and histogram of the classifier's
    scores.

    Parameters
    ----------
        - dn: torch.Tensor, NN output of shape=(N,C,H,W)
        - clear: torch.Tensor, targets of shape=(N,C,H,W)
        - t: float, threshold in [0,1] range
    """
    # mpl.rcParams.update(mpl.rcParamsDefault)
    y_true = clear.detach().cpu().numpy().flatten().astype(bool)
    y_pred = dn.detach().cpu().numpy().flatten()
    hit = y_pred[y_true]
    no_hit = y_pred[~y_true]
    tp, fp, fn, tn = confusion_matrix(hit, no_hit, t)
    cm = np.array([[tn, fp], [fn, tp]])
    fname = os.path.join(args.dir_testing, "cm.txt")
    with open(fname, "a+") as f:
        print_cm(cm, f, epoch)
        f.close()
    logger.info(f"Updated confusion matrix file at {fname}")


def weight_scan(module):
    """
    Computes weights' histogram and norm.

    Parameters
    ----------
        - module: torch.nn.Module

    Returns
    -------
        - float, norm
        - np.array, bins center points
        - np.array, histogram
    """
    p = []
    for i in list(module.parameters()):
        p.append(list(i.detach().cpu().numpy().flatten()))

    p = np.concatenate(p, 0)
    norm = np.sqrt((p * p).sum()) / len(p)

    hist, edges = np.histogram(p, 100)

    return norm, (edges[:-1] + edges[1:]) / 2, hist


def freeze_weights(model, task):
    """
    Freezes weights of either ROI finder or denoiser.

    Parameters
    ----------
        - model: torch.nn.Module
        - task: str, available options dn | roi
    """
    for child in model.children():
        c = "ROI" == child._get_name()
        cond = not c if task == "roi" else c
        if cond:
            for param in child.parameters():
                param.requires_grad = False
    # model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    # params = sum([np.prod(p.size()) for p in model_parameters])
    # net = 'roi' if ROI==0 else 'dn'
    # print('Trainable parameters in %s: %d'% (net, params))
    return model
