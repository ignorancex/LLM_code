import warnings

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

import os
import pathlib
import pickle

# Pytorch imports
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib import cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from PIL import Image
from scipy.ndimage.filters import gaussian_filter
from torch.autograd import Variable
from torchvision import transforms
from torchvision.transforms import ToTensor
from tqdm.auto import tqdm, trange

from src.thermometer_encoding import data_to_therm
from src.utils import get_preds_from_model


def returnCAM(feature_conv, weight_softmax, class_idx):
    size_upsample = (50, 50)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h * w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        cam_img = Image.fromarray(cam_img)
        cam_img = cam_img.resize(size_upsample)
        output_cam.append(cam_img)
    return output_cam


def get_plain_CAM(
    model,
    ds,
    device,
    cam_img_num=0,
    last_conv_name="cnn_layers_2",
    therm=False,
    therm_levels=100,
):
    features_blobs = []

    def hook_feature(module, input, output):
        features_blobs.append(output.data.cpu().numpy())

    model._modules.get(last_conv_name).register_forward_hook(hook_feature)

    params = list(model.parameters())
    weight_softmax = np.squeeze(params[-2].cpu().data.numpy())

    transform = transforms.Compose([ToTensor()])
    if therm:
        img_tensor = transform(ds.test_data[cam_img_num]).float().reshape(-1, 1, 50, 50)
        img_tensor = data_to_therm(img_tensor, therm_levels)
        img_tensor = img_tensor.to(device)
    else:
        img_tensor = (
            transform(ds.test_data[cam_img_num])
            .float()
            .reshape(-1, 1, 50, 50)
            .to(device)
        )
    img_variable = Variable(img_tensor)

    logit = model(img_variable)

    h_x = F.softmax(logit, dim=1).data.squeeze()

    probs, idx = h_x.cpu().sort(0, True)
    probs = probs.numpy()
    idx = idx.numpy()

    CAMs_topo = returnCAM(features_blobs[0], weight_softmax, [1])
    CAMs_trivial = returnCAM(features_blobs[0], weight_softmax, [0])
    return CAMs_topo, CAMs_trivial


def upsample_CAM(cam):
    size_upsample = (50, 50)
    cam_img = cam / np.max(cam)
    cam_img = np.uint8(255 * cam_img)
    cam_img = Image.fromarray(cam_img)
    cam_img = cam_img.resize(size_upsample)
    return cam_img


def get_grad_CAM(
    model,
    ds,
    device,
    cam_img_num=0,
    last_conv_name="cnn_layers_2",
    therm=False,
    therm_levels=100,
):
    features_blobs = []
    gradients = []

    def hook_feature(module, input, output):
        features_blobs.append(output)

    def hook_grad(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    model._modules.get(last_conv_name).register_forward_hook(hook_feature)
    model._modules.get(last_conv_name).register_backward_hook(hook_grad)

    transform = transforms.Compose([ToTensor()])
    if therm:
        img_tensor = transform(ds.test_data[cam_img_num]).float().reshape(1, 1, 50, 50)
        img_tensor = data_to_therm(img_tensor, therm_levels)
        img_tensor = img_tensor.to(device)
    else:
        img_tensor = (
            transform(ds.test_data[cam_img_num])
            .float()
            .reshape(1, 1, 50, 50)
            .to(device)
        )
    img_variable = Variable(img_tensor, requires_grad=True)

    model.eval()  # Set model to evaluation mode
    logit = model(img_variable)
    model.zero_grad()

    # Perform backpropagation
    one_hot = torch.zeros_like(logit, requires_grad=False)
    one_hot[0, torch.argmax(logit)] = 1
    logit.backward(gradient=one_hot)

    # Calculate the gradients and features
    gradients = gradients[0]
    features = features_blobs[0][0]

    # Calculate the weights using global average pooling
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

    # Generate the class activation map
    for i in range(features.size()[0]):
        features[i, :, :] *= pooled_gradients[i]

    CAMs_topo = torch.mean(features, dim=0).detach().numpy()
    CAMs_topo = np.maximum(CAMs_topo, 0)  # ReLU activation
    CAMs_topo = CAMs_topo / np.max(CAMs_topo)  # Normalize

    CAMs_topo = upsample_CAM(CAMs_topo)


def get_CAM(
    model,
    ds,
    device,
    cam_img_num=0,
    last_conv_name="cnn_layers_2",
    therm=False,
    therm_levels=100,
    grad_cam=False,
):
    if grad_cam:
        return get_grad_CAM(
            model,
            ds,
            device,
            cam_img_num=cam_img_num,
            last_conv_name="cnn_layers_2",
            therm=therm,
            therm_levels=therm_levels,
        )
    else:
        return get_plain_CAM(
            model,
            ds,
            device,
            cam_img_num=cam_img_num,
            last_conv_name="cnn_layers_2",
            therm=therm,
            therm_levels=therm_levels,
        )


def generate_CAMs(
    model,
    test_loader,
    class_names,
    device,
    ds,
    therm=False,
    therm_levels=100,
    grad_cam=False,
):

    CAMs_tab = []

    sample_num = test_loader.dataset.Y.shape[0]

    preds = get_preds_from_model(
        model,
        test_loader,
        class_names,
        device,
        therm=therm,
        therm_levels=therm_levels,
    )

    cam_pbar = tqdm(
        total=test_loader.dataset.Y.shape[0],
        desc="CAM generation",
        position=1,
        leave=False,
        unit="sample",
        colour="blue",
    )
    for img_idx in range(max(sample_num, 1)):

        CAMs_topo, CAMs_trivial = get_CAM(
            model,
            ds,
            device,
            cam_img_num=img_idx,
            therm=therm,
            therm_levels=therm_levels,
            grad_cam=grad_cam,
        )

        CAMs_tab.append([CAMs_topo, CAMs_trivial])
        cam_pbar.update(1)
    cam_pbar.close()

    return CAMs_tab, preds


def CAMs_plot_gen_final(
    fig,
    axs,
    CAMs_topo,
    CAMs_trivial,
    ds,
    ds_path,
    preds,
    labels,
    class_names,
    model_name,
    cam_img_num=0,
    M=50,
):

    map_img = cm.cividis
    # map_img = cm.viridis

    map_hmap = cm.viridis

    img = ds.test_data[cam_img_num]
    img_cmap = np.uint8(map_img(img) * 255)

    extent = 0, M, 0, M

    fig.suptitle(
        f"Model name : {model_name}\nDataset : {ds_path} | Img_num = {cam_img_num}\n Predicted label = {preds} | True "
        f"label = {labels[cam_img_num]}\n {class_names[preds]} | "
        f"{class_names[labels[cam_img_num]]}",
        size=25,
    )
    imdata = axs[0, 0].imshow(img_cmap, alpha=1, extent=extent, cmap=map_hmap)
    camdata = axs[0, 1].imshow(CAMs_topo, alpha=1, extent=extent, cmap=map_img)
    axs[0, 2].imshow(CAMs_trivial, alpha=1, extent=extent, cmap=map_img)
    axs[0, 0].set_title("Image", size=20)
    axs[0, 1].set_title("CAM - Topological", size=20)
    axs[0, 2].set_title("CAM - Trivial", size=20)

    axs[1, 1].set_title("Img + CAM - Topological", size=20)
    axs[1, 2].set_title("Img + CAM - Trivial", size=20)
    axs[1, 1].imshow(img_cmap, alpha=0.65, extent=extent, cmap=map_img)
    axs[1, 1].imshow(CAMs_topo, alpha=0.35, extent=extent, cmap=map_img)
    axs[1, 2].imshow(img_cmap, alpha=0.65, extent=extent, cmap=map_img)
    axs[1, 2].imshow(CAMs_trivial, alpha=0.35, extent=extent, cmap=map_img)

    fontsize = 20
    ticksize = 15

    ax = axs[0, 0]
    axins = inset_axes(
        ax,
        width="5%",
        height="100%",
        loc="lower left",
        bbox_to_anchor=(1.05, 0.0, 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
    cbar = plt.colorbar(imdata, cax=axins, orientation="vertical")
    cbar.set_label(
        "Tunneling amplitude [arb.u.]", rotation=270, labelpad=30, size=fontsize
    )
    cbar.set_ticks(np.linspace(0, 255, 6))
    cbar.set_ticklabels([f"{t:.1f}" for t in np.linspace(0, 1, 6)], size=ticksize)

    ax = axs[0, 2]
    axins_2 = inset_axes(
        ax,
        width="5%",
        height="100%",
        loc="lower left",
        bbox_to_anchor=(1.05, 0.0, 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
    cbar_2 = plt.colorbar(camdata, cax=axins_2, orientation="vertical")
    cbar_2.set_label(
        "CAM importance [arb.u.]", rotation=270, labelpad=30, size=fontsize
    )
    cbar_2.set_ticks(np.linspace(0, 255, 6))
    cbar_2.set_ticklabels([f"{t:.1f}" for t in np.linspace(0, 1, 6)], size=ticksize)

    for num, ax in enumerate(axs.flat):
        ax.grid(visible=False)
        if num != 3:
            ax.set_xticks([0, 9, 19, 29, 39, 49], size=ticksize)
            ax.set_xticklabels(
                [0, 10, 20, 30, 40, 50], size=ticksize, rotation=45, ha="right"
            )

            ax.set_yticks([0, 9, 19, 29, 39, 49][::-1], size=ticksize)
            ax.set_yticklabels(
                [0, 10, 20, 30, 40, 50],
                size=ticksize,
                rotation=0,
                ha="right",
                va="center",
            )

            ax.set_xlabel("Eigenstate index $i$", size=fontsize)
            ax.set_ylabel("Physical site index $j$", size=fontsize)

    del img, img_cmap, CAMs_topo, CAMs_trivial


def viz_final(
    CAMs_tab, pred_tab, img_ind, model_sig, ds, ds_path, test_loader, save=None
):

    class_names = ["Trivial", "Topological"]

    fig_cam, ax_cam = plt.subplots(2, 3, figsize=(20, 10), layout="constrained")
    ax_stat = ax_cam[1, 0]

    ax_stat.plot(
        ds.v_tab[img_ind],
        test_loader.dataset.Y[img_ind],
        marker="$|$",
        ls="",
        color="red",
        markersize=30,
    )

    preds = pred_tab
    ax_stat.set_title(r"$W/w = {}$".format(np.unique(ds.W_tab)[0]), fontsize=20)
    ax_stat.plot(ds.v_tab, ds.test_labels, lw=0, marker=6, label="True label")
    ax_stat.plot(ds.v_tab, preds, lw=0, marker=7, label="Predicted label")
    ax_stat.set_ylabel(r"Winding Number $\vartheta$", fontsize=18)
    ax_stat.set_xlabel("$v/w$", fontsize=18)
    ax_stat.set_yticks([0, 1])
    ax_stat.axvline(x=1.0, ls="dotted")
    ax_stat.legend(loc="lower left", fontsize=25)

    CAMs_plot_gen_final(
        fig_cam,
        ax_cam,
        CAMs_tab[img_ind][0],
        CAMs_tab[img_ind][1],
        ds,
        ds_path,
        pred_tab[img_ind],
        test_loader.dataset.Y,
        class_names,
        cam_img_num=img_ind,
        model_name=model_sig,
        M=50,
    )

    if save is not None:
        fig_cam.savefig(save, bbox_inches="tight")
    else:
        plt.show()

    plt.close(fig_cam)
    plt.close("all")
    for ax in ax_cam.flat:
        ax.cla()
    ax_stat.cla()

    del preds

    return


def recreate_fig_2(load_path, save_path):
    smart_1 = load_path.joinpath("well_generalizing_good_CAM")
    smart_2 = load_path.joinpath("well_generalizing_misleading_CAM")
    dumb_1 = load_path.joinpath("poorly_generalizing_misleading_CAM")
    dumb_2 = load_path.joinpath("poorly_generalizing_good_CAM")
    targets_path = load_path.joinpath("targets_25_reals.pkl")
    toy_path = load_path.joinpath("cams_all_toy_model.pkl")
    cam_ind = 26

    fig = plt.figure(figsize=(20, 10), constrained_layout=True)

    subplot_pairs = {
        "(a)": dumb_1,
        "(b)": smart_1,
        "(c)": dumb_2,
        "(d)": smart_2,
        "(e)": dumb_1,
        "(f)": smart_1,
        "(g)": dumb_2,
        "(h)": smart_2,
        "tm": toy_path,
        "tg": targets_path,
    }

    axs = fig.subplot_mosaic(
        [
            ["tm", "(a)", "(b)", "(c)", "(d)"],
            ["tg", "(e)", "(f)", "(g)", "(h)"],
        ]
    )

    # Toggle use of latex serif font
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")
    fontsize = 24
    ticksize = 20
    lw = 2.5
    sigma = 0

    cam_cmap = "cividis"
    gen_cmap = "coolwarm"

    gen_ticks_num = 11

    cmap = plt.cm.get_cmap("tab20")

    prd_pt = [102, 88, 105, 95]
    pt_extent = [50, 175, 50, 175]

    for s_num, (subplot_label, data_path) in enumerate(
        list(subplot_pairs.items())[:-2]
    ):
        plot_type = "CAM" if s_num < 4 else "Generalization"

        ax = axs[subplot_label]
        ax.set_title("{}".format(subplot_label), size=fontsize, pad=10)

        if plot_type == "CAM":
            cam_path = data_path.joinpath("CAM/viz_tuple.pkl")
            with open(cam_path, "rb") as f:
                cam_data = pickle.load(f)
            cam_topo, cam_trivial = cam_data[0][cam_ind]

            cam = ax.imshow(cam_topo, cmap=cam_cmap, alpha=0.95, interpolation="none")

            ax.set_xticks([0, 9, 19, 29, 39, 49], size=ticksize)
            ax.set_xticklabels(
                [0, 10, 20, 30, 40, 50], size=ticksize, rotation=45, ha="right"
            )

            ax.set_yticks([0, 9, 19, 29, 39, 49][::-1], size=ticksize)
            ax.set_yticklabels(
                [0, 10, 20, 30, 40, 50],
                size=ticksize,
                rotation=0,
                ha="right",
                va="center",
            )

            ax.set_xlabel("Eigenstate index $i$", size=fontsize)
            ax.set_ylabel("Physical site index $j$", size=fontsize)

            if s_num == 3:
                axins = inset_axes(
                    ax,
                    width="5%",  # szerokość = 5% osi rodzica
                    height="100%",  # wysokość = 100%
                    loc="lower left",
                    bbox_to_anchor=(1.05, 0.0, 1, 1),
                    bbox_transform=ax.transAxes,
                    borderpad=0,
                )
                cbar = plt.colorbar(cam, cax=axins, orientation="vertical")
                cbar.set_label(
                    "CAM importance [arb.u.]", rotation=270, labelpad=30, size=fontsize
                )
                cbar.set_ticks(np.linspace(0, 255, 6))
                cbar.set_ticklabels(
                    [f"{t:.1f}" for t in np.linspace(0, 1, 6)], size=ticksize
                )

        elif plot_type == "Generalization":
            data = pd.read_pickle(data_path.joinpath("generalization_data_raw.pkl"))

            image = gaussian_filter(data[:, ::-1].T, sigma=sigma)
            im = ax.imshow(
                image,
                cmap=gen_cmap,
                interpolation="none",
                aspect=5 / 2,
                alpha=0.95,
                zorder=1,
            )

            ax.set_xticks(np.linspace(0, 500, gen_ticks_num), size=ticksize)
            ax.set_xticklabels(
                [f"{t:.1f}" for t in np.linspace(0, 5, gen_ticks_num)],
                size=ticksize,
                rotation=45,
                ha="right",
            )

            ax.set_yticks(np.linspace(0, 199, gen_ticks_num), size=ticksize)
            ax.set_yticklabels(
                [f"{t:.1f}" for t in np.linspace(2, 0, gen_ticks_num)],
                size=ticksize,
                rotation=0,
                ha="right",
                va="center",
            )

            ax.set_xlabel(r"$W \,/\,\textit{w} $", size=1.2 * fontsize)
            ax.set_ylabel(
                r"$\textit{v}\,/\,\textit{w}$",
                size=1.5 * fontsize,
                rotation=0,
                labelpad=15,
                ha="center",
                va="center",
                position=(0, 0.5),
            )

            if s_num == 7:
                axins = inset_axes(
                    ax,
                    width="5%",  # szerokość = 5% osi rodzica
                    height="100%",  # wysokość = 100%
                    loc="lower left",
                    bbox_to_anchor=(1.05, 0.0, 1, 1),
                    bbox_transform=ax.transAxes,
                    borderpad=0,
                )
                cbar = plt.colorbar(im, cax=axins, orientation="vertical")
                cbar.set_label(
                    r"Winding number $\vartheta$ "
                    + "averaged\nover disorder realizations",
                    rotation=270,
                    labelpad=55,
                    size=fontsize,
                )
                cbar.set_ticks(np.linspace(0, 1, 6))
                cbar.set_ticklabels(
                    [f"{t:.1f}" for t in np.linspace(0, 1, 6)], size=ticksize
                )

        ax.tick_params(axis="both", which="both", direction="in")
        # Removing the internal labels and tick labels
        ax.set_yticklabels([])
        ax.set_ylabel("")

        if s_num in [0, 2, 4, 6]:
            ax.set_ylabel(".\n.", color="white", size=1.2 * fontsize)

    ax = axs["tg"]
    data = pd.read_pickle(subplot_pairs["tg"])

    image = gaussian_filter(data[:, ::-1].T, sigma=sigma)
    im = ax.imshow(image, cmap=gen_cmap, interpolation="none", aspect=5 / 2, alpha=0.95)

    ax.set_xticks(np.linspace(0, 500, gen_ticks_num), size=ticksize)
    ax.set_xticklabels(
        [f"{t:.1f}" for t in np.linspace(0, 5, gen_ticks_num)],
        size=ticksize,
        rotation=45,
        ha="right",
    )

    ax.set_yticks(np.linspace(0, 199, gen_ticks_num), size=ticksize)
    ax.set_yticklabels(
        [f"{t:.1f}" for t in np.linspace(2, 0, gen_ticks_num)],
        size=ticksize,
        rotation=0,
        ha="right",
        va="center",
    )

    ax.set_xlabel(r"$W\,/\,\textit{w}$", size=1 * fontsize)
    ax.set_ylabel(
        r"$\textit{v}\,/\,\textit{w}$",
        size=1 * fontsize,
        rotation=90,
        labelpad=25,
        ha="center",
        va="center",
        position=(0, 0.5),
    )

    ax.set_title("targets", size=fontsize, pad=10)

    ax.tick_params(axis="both", which="both", direction="in")

    ax = axs["tm"]
    ax.set_title("toy model", size=fontsize, pad=10)

    cam_path = subplot_pairs["tm"]
    with open(cam_path, "rb") as f:
        cam_data = pickle.load(f)
    cam_topo, cam_trivial = cam_data[0][cam_ind]

    cam = ax.imshow(cam_topo, cmap=cam_cmap, alpha=0.95, interpolation="none")

    ax.set_xticks([0, 9, 19, 29, 39, 49], size=ticksize)
    ax.set_xticklabels([0, 10, 20, 30, 40, 50], size=ticksize, rotation=45, ha="right")

    ax.set_yticks([0, 9, 19, 29, 39, 49][::-1], size=ticksize)
    ax.set_yticklabels(
        [0, 10, 20, 30, 40, 50], size=ticksize, rotation=0, ha="right", va="center"
    )

    ax.set_xlabel("Eigenstate index $i$", size=fontsize)
    ax.set_ylabel("Physical site index $j$", size=fontsize)

    if s_num == 3:
        axins = inset_axes(
            ax,
            width="5%",
            height="100%",
            loc="lower left",
            bbox_to_anchor=(1.05, 0.0, 1, 1),
            bbox_transform=ax.transAxes,
            borderpad=0,
        )
        cbar = plt.colorbar(cam, cax=axins, orientation="vertical")
        cbar.set_label(
            "CAM importance [arb.u.]", rotation=270, labelpad=30, size=fontsize
        )
        cbar.set_ticks(np.linspace(0, 255, 6))
        cbar.set_ticklabels([f"{t:.1f}" for t in np.linspace(0, 1, 6)], size=ticksize)

    ax.tick_params(axis="both", which="both", direction="in")

    fig.text(0.425, 1.00, "CAM analysis", size=1.2 * fontsize, ha="center", va="center")
    fig.text(
        0.825, 1.00, "counterexamples", size=1.2 * fontsize, ha="center", va="center"
    )

    fig.savefig(save_path.joinpath("Fig_2.pdf"), bbox_inches="tight")
    save_path.joinpath("PNG").mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path.joinpath("PNG").joinpath("Fig_2.png"), bbox_inches="tight")
