import os
import random
import numpy as np
import cv2
from matplotlib import pyplot as plt

# import albumentations as A
import imgaug as ia
import imgaug.augmenters as iaa
import imgaug.parameters as iap


def visualize_cell(log_input_img, y_logtrue, y_logpred, cmap="jet", **kwargs):
    input_img = np.exp(log_input_img)
    y_true = np.exp(y_logtrue)
    # Prediction and uncertainty outputed by the neural network
    y_pred_mu = y_logpred[..., 0]
    y_pred_sig2 = y_logpred[..., 1] ** 2
    # Aleatoric parameters
    y_pred_forces = np.exp(y_pred_mu + 0.5 * y_pred_sig2)
    y_pred_mean = y_pred_forces.mean(axis=0)
    y_pred_epis = y_pred_forces.var(axis=0)
    y_pred_alea = (
        np.exp(2 * y_pred_mu + y_pred_sig2) * (np.exp(y_pred_sig2) - 1)
    ).mean(axis=0)
    y_pred_var = y_pred_alea + y_pred_epis
    # Full uncertainties
    y_pred_cv = np.sqrt(np.exp(y_pred_sig2 + 0.5) - 1).mean(axis=0)
    y_pred_entropy = np.log2(
        np.sqrt(y_pred_sig2) * np.exp(y_pred_mu + 0.5) * np.sqrt(2 * np.pi)
    ).mean(axis=0)

    fig, ax = plt.subplots(2, 4, sharex="col", sharey="row", **kwargs)
    ax = ax.ravel()

    ax[0].set_title("Raw data")
    im = ax[0].imshow(input_img[..., 0], vmin=0, cmap=cmap)
    plt.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04)

    vmax = max(np.percentile(y_true[..., 0], 99.5), np.percentile(y_pred_mean, 99.5))
    ax[1].set_title("Ground-Truth")
    im = ax[1].imshow(y_true[..., 0], vmin=0, vmax=vmax, cmap=cmap)
    plt.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)

    ax[2].set_title("Prediction")
    im = ax[2].imshow(y_pred_mean, vmin=0, vmax=vmax, cmap=cmap)
    plt.colorbar(im, ax=ax[2], fraction=0.046, pad=0.04)

    ax[3].set_title("Std.dev. of the force")
    im = ax[3].imshow(np.sqrt(y_pred_var), cmap=cmap)
    plt.colorbar(im, ax=ax[3], fraction=0.046, pad=0.04)

    ax[4].set_title("Model uncertainty")
    im = ax[4].imshow(np.sqrt(y_pred_epis), cmap=cmap)
    plt.colorbar(im, ax=ax[4], fraction=0.046, pad=0.04)

    ax[5].set_title("Data uncertainty")
    im = ax[5].imshow(np.sqrt(y_pred_alea), cmap=cmap)
    plt.colorbar(im, ax=ax[5], fraction=0.046, pad=0.04)

    ax[6].set_title("Coeff. of variation (σ/μ)")
    im = ax[6].imshow(y_pred_cv, cmap=cmap)
    plt.colorbar(im, ax=ax[6], fraction=0.046, pad=0.04)

    ax[7].set_title("Entropy (bits)")
    im = ax[7].imshow(y_pred_entropy, cmap=cmap)
    plt.colorbar(im, ax=ax[7], fraction=0.046, pad=0.04)

    plt.subplots_adjust(wspace=0.1)
    plt.tight_layout()
    return fig, ax


def load_bboxes(path):
    bbox = np.loadtxt(path + "_bbox.csv", dtype=np.str, delimiter=",", skiprows=1)
    return bbox[:, 1:].astype(np.int32)


def get_cell_list(dataset_path):
    cell_list = []
    for directory in sorted(os.listdir(dataset_path)):
        dir_path = os.path.join(dataset_path, directory)
        # Is the path a directory?
        if not os.path.isdir(dir_path):
            continue
        # What did we find? RAW/FRET/FORCE
        dir_type = None
        if directory.endswith("RAW"):
            dir_type = "RAW"
        elif directory.endswith("FRET"):
            dir_type = "FRET"
        elif directory.endswith("FORCE"):
            dir_type = "FORCE"
        else:
            continue

        # Adding trimmed name to list if needed
        short_name = directory[: -len(dir_type)]
        if short_name not in cell_list:
            cell_list.append(short_name)

    return cell_list


def crop_to_size(im, size):
    up, mod = divmod(im.shape[0] - size, 2)
    up = max(0, up)
    down = up + mod
    left, mod = divmod(im.shape[1] - size, 2)
    left = max(0, left)
    right = left + mod
    if down == 0 and right == 0:
        return im[up:, left:]
    elif down == 0:
        return im[up:, left:-right]
    elif right == 0:
        return im[up:down, left:]
    else:
        return im[up:-down, left:-right]


def tukey2d(size, p):
    assert p <= 0.5, "p must be <=0.5"
    w = np.ones(size)
    t = int(len(w) * p)  # Length of the tapering window
    tapering = 0.5 - 0.5 * np.cos(2 * np.pi * np.linspace(0, 1, 2 * t))
    w[:t] = tapering[:t]
    w[-t:] = tapering[-t:]
    return np.outer(w, w)


def tukey2d(sizew, sizeh, p):
    assert p <= 0.5, "p must be <=0.5"
    wh = np.ones(sizeh)
    t = int(sizeh * p)  # Length of the tapering window
    tapering = 0.5 - 0.5 * np.cos(2 * np.pi * np.linspace(0, 1, 2 * t))
    wh[:t] = tapering[:t]
    wh[-t:] = tapering[-t:]
    ww = np.ones(sizew)
    t = int(sizew * p)  # Length of the tapering window
    tapering = 0.5 - 0.5 * np.cos(2 * np.pi * np.linspace(0, 1, 2 * t))
    ww[:t] = tapering[:t]
    ww[-t:] = tapering[-t:]
    return np.outer(wh, ww)


def load_crop_image(filename, bbox=None, margin_size=None, tukey_p=None):
    im = cv2.imread(filename, cv2.IMREAD_UNCHANGED)

    if bbox is not None and margin_size is not None:
        x, y, w, h = bbox
        x, y = x - margin_size, y - margin_size
        w, h = w + 2 * margin_size, h + 2 * margin_size
        sizew, sizeh = im.shape
        # Crop of the image, clamped not to go out of bounds
        im = im[max(0, y) : min(sizeh, y + h), max(0, x) : min(sizew, x + w)]

    if tukey_p:
        mask = tukey2d(im.shape[1], im.shape[0], tukey_p)
        im = im * mask

    im = im.astype(np.float32)
    if im.ndim == 2:
        return np.expand_dims(im, axis=-1)

    return im


def import_cells(
    dataset_path,
    crop=False,
    crop_size=None,
    margin_size=0,
    tukey_p=None,
    image_types=["RAW", "FRET", "FORCE"],
    allowed_formats=("tif", "jpg", "jpeg", "bmp", "png"),
):
    # 1st step is identifiying the different cell experiments
    cell_list = get_cell_list(dataset_path)
    # Create the input dataset
    X = []
    y = []
    for cell in cell_list:
        print("*", cell)
        cell_path = os.path.join(dataset_path, cell)
        # Loading BBOXes
        if crop:
            bbox = load_bboxes(cell_path)
        # Listing files in each folder type
        image_paths = {
            im_type: [
                os.path.join(cell_path + im_type, file)
                for file in sorted(os.listdir(cell_path + im_type))
                if file.endswith(allowed_formats)
            ]
            for im_type in image_types
        }

        # Sanity check that we have the same number of files
        image_lengths = np.array([len(image_paths[key]) for key in image_paths.keys()])
        image_len = image_lengths[0]
        # image_len contains the number of image for each folder type, should have three equal values
        assert (
            image_lengths == image_len
        ).all(), (
            f"Each dataset should have as many images of each type ({image_lengths})"
        )

        # Loading the images
        for i in range(image_len):
            current_bbox = bbox[i] if crop else None
            input_image = []
            # Loading the input images
            if "RAW" in image_types:
                file_path = image_paths["RAW"][i]
                image = load_crop_image(file_path, current_bbox, margin_size)
                input_image.append(image)
            if "FRET" in image_types:
                file_path = image_paths["FRET"][i]
                image = load_crop_image(file_path, current_bbox, margin_size)
                input_image.append(image)
            input_image = np.block(input_image)
            if input_image.ndim == 2:
                input_image = np.expand_dims(input_image, axis=-1)

            # Loading the ground truth images
            file_path = image_paths["FORCE"][i]
            output_image = load_crop_image(
                file_path, current_bbox, margin_size, tukey_p
            )
            if output_image.ndim == 2:
                output_image = np.expand_dims(output_image, axis=-1)

            # Checking if the images are the right size
            if crop_size is not None:
                if input_image.shape[0] < crop_size or input_image.shape[1] < crop_size:
                    continue
                if (
                    output_image.shape[0] < crop_size
                    or output_image.shape[1] < crop_size
                ):
                    continue

            X.append(input_image)
            y.append(output_image)
    return X, y


def generate_batch(X, y, batch_size=32):
    while True:
        if isinstance(X, list):
            idx = np.random.choice(np.arange(len(X)), batch_size)
        else:
            idx = np.random.choice(np.arange(X.shape[0]), batch_size)
        yield X[idx], y[idx]


def generate_batch_augmented(X, y, batch_size=32, crop_size=130):
    while True:
        batch_X, batch_y = next(generate_batch(X, y, batch_size))
        seq = iaa.Sequential(
            [
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5),
                iaa.Affine(
                    rotate=(-180, 180),  # rotate by -45 to +45 degrees
                    order=[3],  # use nearest neighbour or bilinear interpolation (fast)
                ),
                iaa.CropToFixedSize(crop_size, crop_size),
            ]
        )
        seq_X = iaa.Sometimes(0.5, [iaa.ReplaceElementwise(0.01, iap.Uniform(0, 2000))])

        if batch_X[0].ndim == 2:
            channels = 1
        else:
            channels = batch_X[0].shape[-1]
        batch_aug_X = np.empty((batch_size, crop_size, crop_size, channels))
        batch_aug_y = np.empty((batch_size, crop_size, crop_size, 1))
        for i in range(batch_size):
            seq_det = (
                seq.to_deterministic()
            )  # call this for each batch again, NOT only once at the start
            images_aug = seq_det.augment_image(batch_X[i])
            forces_aug = seq_det.augment_image(batch_y[i])
            images_aug = seq_X.augment_image(images_aug)
            if images_aug.ndim == 2:
                batch_aug_X[i] = np.expand_dims(images_aug, axis=-1)
            else:
                batch_aug_X[i] = images_aug

            if forces_aug.ndim == 2:
                batch_aug_y[i] = np.expand_dims(forces_aug, axis=-1)
            else:
                batch_aug_y[i] = forces_aug

        yield batch_aug_X, batch_aug_y


def generate_log_batch_augmented(X, y, batch_size, crop_size):
    while True:
        batch_X, batch_y = next(generate_batch_augmented(X, y, batch_size, crop_size))
        batch_X = np.log(batch_X, where=batch_X >= 1.0, out=np.zeros_like(batch_X))
        batch_y = np.log(batch_y, where=batch_y >= 1.0, out=np.zeros_like(batch_y))
        yield batch_X, batch_y
