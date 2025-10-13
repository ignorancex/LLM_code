import os
import math
from torchvision.transforms import ToPILImage, Resize
from torchvision.io import read_image
from torchvision.utils import make_grid


def residual_grid(basedir, modelname, in_layer, mid_layer, out_layer, neurons):
    exc_imgs = []
    for n in neurons:
        img = read_image(os.path.join(basedir, modelname, in_layer, f"unit{n}", "0_distill_center.png"))
        exc_imgs.append(img)

        img = read_image(os.path.join(basedir, modelname, mid_layer, f"unit{n}", "0_distill_center.png"))
        exc_imgs.append(img)

        img = read_image(os.path.join(basedir, modelname, out_layer, f"unit{n}", "0_distill_center.png"))
        exc_imgs.append(img)

        # img = read_image(os.path.join(basedir, modelname, in_layer, f"unit{n}", "0_distill_channel.png"))
        # exc_imgs.append(img)

        # img = read_image(os.path.join(basedir, modelname, mid_layer, f"unit{n}", "0_distill_channel.png"))
        # exc_imgs.append(img)

        # img = read_image(os.path.join(basedir, modelname, out_layer, f"unit{n}", "0_distill_channel.png"))
        # exc_imgs.append(img)

    grid = make_grid(exc_imgs, nrow=6)
    grid = ToPILImage()(grid)

    return grid


def exemplar_grid(fzdir, modelname, in_layer, mid_layer, out_layer, neuron, savedir, val_dir=None):
    imgs = []

    img = read_image(os.path.join(fzdir, modelname, in_layer, f"unit{neuron}", "0_distill_center.png"))
    imgs.append(img)
    img = read_image(os.path.join(fzdir, modelname, mid_layer, f"unit{neuron}", "0_distill_center.png"))
    imgs.append(img)
    img = read_image(os.path.join(fzdir, modelname, out_layer, f"unit{neuron}", "0_distill_center.png"))
    imgs.append(img)

    img = read_image(os.path.join(fzdir, modelname, in_layer, f"unit{neuron}", "0_distill_channel.png"))
    imgs.append(img)
    img = read_image(os.path.join(fzdir, modelname, mid_layer, f"unit{neuron}", "0_distill_channel.png"))
    imgs.append(img)
    img = read_image(os.path.join(fzdir, modelname, out_layer, f"unit{neuron}", "0_distill_channel.png"))
    imgs.append(img)

    if val_dir is not None:
        img_path = os.path.join(val_dir, modelname, in_layer, 'all_grids', f"{in_layer}_neuron{neuron}.png")
        val_imgs = read_image(img_path)
        val_imgs = Resize(224)(val_imgs)
        imgs.append(val_imgs)

        img_path = os.path.join(val_dir, modelname, mid_layer, 'all_grids', f"{mid_layer}_neuron{neuron}.png")
        val_imgs = read_image(img_path)
        val_imgs = Resize(224)(val_imgs)
        imgs.append(val_imgs)

        img_path = os.path.join(val_dir, modelname, out_layer, 'all_grids', f"{out_layer}_neuron{neuron}.png")
        val_imgs = read_image(img_path)
        val_imgs = Resize(224)(val_imgs)
        imgs.append(val_imgs)

    grid = make_grid(imgs, nrow=3, padding=8)
    grid = ToPILImage()(grid)
    grid.save(f"{savedir}/{modelname}/{out_layer}_neuron{neuron}.png")