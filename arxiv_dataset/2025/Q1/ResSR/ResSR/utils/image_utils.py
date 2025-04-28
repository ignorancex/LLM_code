"""
@author: hdsullivan
"""

"""
Various functions used for manipulating images
"""

from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
try:
    from osgeo import gdal
except:
    print("GDAL must be installed using `conda install -c conda-forge gdal'. Reference ReadMe for correct conda environment set-up instructions.")
    quit()

EPS = np.finfo(float).eps

def load_S2_images(img_dir_path, 
                   band_name_list_10m = ['B02_10m.jp2', 'B03_10m.jp2', 'B04_10m.jp2', 'B08_10m.jp2'], 
                   band_name_list_20m = ['B05_20m.jp2', 'B06_20m.jp2', 'B07_20m.jp2', 'B8A_20m.jp2', 'B11_20m.jp2', 'B12_20m.jp2'], 
                   band_name_list_60m = ['B01_60m.jp2', 'B09_60m.jp2']):
    """
    Load Sentinel-2 MSI bands contained in img_dir_path as specificed by the lists of band names.

    Args:
        img_dir_path (str): Path to directory containing bands
        band_name_list_10m (list of str): List of names of 10m bands to load.
        band_name_list_20m (list of str): List of names of 20m bands to load.
        band_name_list_60m (list of str): List of names of 60m bands to load.

    Returns:
        np.ndarray: N x M x b_10m image containing 10m bands
        np.ndarray: N // 2 x M // 2 x b_20m image containing 20m bands
        np.ndarray: N // 6 x M // 6 x b_60m image containing 60m bands
        list of str: List of names of 10m bands that were loaded
        list of str: List of names of 20m bands that were loaded
        list of str: List of names of 60m bands that were loaded
    """
    all_bands_name_list = []

    # ######
    # Open 10m bands
    # ######
    band_list_10m = []
    for band in band_name_list_10m:
        img = im_open(os.path.join(img_dir_path, band))
        band_list_10m.append(img)
        name = band[0:3]
        all_bands_name_list.append(name)
    y_10m = np.zeros((band_list_10m[0].shape[0], band_list_10m[0].shape[1], len(band_name_list_10m)))
    for i in range(len(band_list_10m)):
        y_10m[:, :, i] = band_list_10m[i].squeeze()

    # ######
    # Open 20m bands
    # ######
    band_list_20m = []
    for band in band_name_list_20m:
        img = im_open(os.path.join(img_dir_path, band))
        band_list_20m.append(img)
        name = band[0:3]
        all_bands_name_list.append(name)
    y_20m = np.zeros((band_list_20m[0].shape[0], band_list_20m[0].shape[1], len(band_name_list_20m)))
    for i in range(len(band_list_20m)):
        y_20m[:, :, i] = band_list_20m[i].squeeze()

    # ######
    # Open 60m bands
    # ######
    band_list_60m = []
    for band in band_name_list_60m:
        img = im_open(os.path.join(img_dir_path, band))
        band_list_60m.append(img)
        name = band[0:3]
        all_bands_name_list.append(name)
    y_60m = np.zeros((band_list_60m[0].shape[0], band_list_60m[0].shape[1], len(band_name_list_60m)))
    for i in range(len(band_list_60m)):
        y_60m[:, :, i] = band_list_60m[i].squeeze()

    return (
        y_10m,
        y_20m,
        y_60m,
        band_name_list_10m,
        band_name_list_20m,
        band_name_list_60m,
    )

def load_images(img_dir_path,
                hr_band_name_list = ['B02_10m.jp2', 'B03_10m.jp2', 'B04_10m.jp2', 'B08_10m.jp2'], 
                lr_band_name_list = ['B05_20m.jp2', 'B06_20m.jp2', 'B07_20m.jp2', 'B8A_20m.jp2', 'B11_20m.jp2', 'B12_20m.jp2']):
    """
    Load high-resolution and low-resolution MSI bands contained in img_dir_path as specificed by the lists of band names.
    
    Args:
        img_dir_path (str): Path to directory containing bands
        hr_band_name_list (list or str): List of names of HR bands to load
        lr_band_name_list (list of str): List of names of LR images to load
    
    Returns:
        np.ndarray: N x M x B image containing HR bands 
        np.ndarray: n x m x b image containing LR bands
        list of str: List of names of HR images that were loaded
        list of str: List of names of LR images that were loaded
    """
    all_bands_name_list = []

    # ######
    # Open HR bands
    # ######
    hr_band_list = []
    for band in hr_band_name_list:
        img = im_open(os.path.join(img_dir_path, band))
        hr_band_list.append(img)
        name = band[0:3]
        all_bands_name_list.append(name)
    y_hr = np.zeros((hr_band_list[0].shape[0], hr_band_list[0].shape[1], len(hr_band_name_list)))
    for i in range(len(hr_band_list)):
        y_hr[:, :, i] = hr_band_list[i].squeeze()

    # ######
    # Open LR bands
    # ######
    lr_band_list = []
    for band in lr_band_name_list:
        img = im_open(os.path.join(img_dir_path, band))
        lr_band_list.append(img)
        name = band[0:3]
        all_bands_name_list.append(name)
    y_lr = np.zeros((lr_band_list[0].shape[0], lr_band_list[0].shape[1], len(lr_band_name_list)))
    for i in range(len(lr_band_list)):
        y_lr[:, :, i] = lr_band_list[i].squeeze()

    return y_hr, y_lr, hr_band_name_list, lr_band_name_list

def rasterize_img(img):
    """
    Rasterize image array by stacking columns. 

    Args:
        img (np.ndarray): N x M x C image

    Returns: 
        np.ndarray: NM x C rasterized image
    """
    if np.ndim(img) == 2:
        img = img[..., None]
    img_vec = img.reshape(-1, img.shape[-1])
    return img_vec

def unrasterize_img(img_vec, cols):
    """
    Un-rasterize by unstacking columns.

    Args:
        img_vec (np.ndarray): NM x C rasterized image
        cols (int): M

    Returns:
        np.ndarray: N x M x C unrasterized image

    """
    
    if np.ndim(img_vec) == 1:
        img_vec = img_vec[..., None]
    (pixels, bands) = img_vec.shape
    rows = int(pixels / cols)
    img = np.reshape(img_vec, (rows, cols, bands))
    return img

def im_open(im_path, nchannels=1, norm_factor=0):
    """
    Load image from path. If image is a JPEG2000, load using GDAL. Otherwise, load using PIL.

    Args:
        im_path (str): Path to image
        nchannels (int): number of channels in image (default: 1)
        norm_factor (float): factor to scale image by (default: 65535.0 for JPEG2000 and 255.0 else)

    Returns:
        np.ndarray: N x M X C image array with values in [0,1]
    """

    if im_path.endswith("jp2"):
        if norm_factor == 0:
            norm_factor = 65535.0
        img_gdal = gdal.Open(im_path)
        channels_list = []
        for i in range(nchannels):
            channel = img_gdal.GetRasterBand(i + 1).ReadAsArray() / norm_factor
            channel = np.float64(channel)
            channels_list.append(channel)
        rows = channels_list[0].shape[0]
        cols = channels_list[0].shape[1]
        img = np.zeros((rows, cols, nchannels))
        for j in range(len(channels_list)):
            img[:, :, j] = channels_list[j]
    else:
        if norm_factor == 0:
            norm_factor = 255.0
        img = Image.open(os.path.join(im_path))
        if nchannels == 1:
            img = ImageOps.grayscale(img)
            img = np.asarray(img) / norm_factor
            img = np.expand_dims(img, 2)
        else:
            img = img.convert("RGB")
            img = np.asarray(img) / norm_factor
        img = np.float64(img)

    return img

def normalize_image(img, perc=98, nchannels=1):
    """
    Normalize image using the perc-th percentile of the image.

    Args:
        img (np.ndarray): M x N x C image to normalize
        perc (int): percentile to use as high value (default: 98)
        nchannels (int): number of channels (default: 1)

    Returns:
        np.ndarray: M x N x C image normalized using the perc-th percentile of input image
        float32: perc-th percentile of input imgg
        float32: (100-perc)-th percentile of input images
    """
    if img.ndim < 3:
        img = np.expand_dims(img, 2)
        squeeze_flag = True
    else: 
        squeeze_flag = False

    img_norm = np.zeros_like(img)
    img_high = np.zeros(nchannels)
    img_low = np.zeros(nchannels)
    for i in range(nchannels):
        img_high[i] = np.percentile(img[:, :, i], perc)
        img_low[i] = np.percentile(img[:, :, i], 100 - perc)
        img_range = img_high[i] - img_low[i]
        img_norm[:, :, i] = (img[:, :, i] - img_low[i]) / img_range
    
    if squeeze_flag:
        img_norm = np.squeeze(img_norm)

    return img_norm, img_high, img_low

def unnormalize_image(img_norm, img_low, img_high):
    """
    Unnormalize image.

    Args:
        img_norm (np.ndarray):  M x N x C normalized image to unnormalize
        img_high (float): High value used to normalize image
        img_low (float): Low value used to normalize image

    Returns:
        np.ndarray: M x N x C Unnormalized image using img_high and img_low
    """
    img_range = img_high - img_low

    img = (img_norm * img_range) + img_low
    return img

def add_AWGN(img, sigma):
    """
    Add white Gaussian noise to image with standard deviation sigma scaled by the range of the image

    Args:
        img (np.ndarray): image to add noise to
        sigma (float): Standard deviation of AWGN to add to normalized image (value from 0 to 1)

    Returns:
        np.ndarray: image with AWGN
    """

    e_size = img.shape
    img_range = np.max(img) - np.min(img)
    e = np.random.normal(0, sigma * img_range, e_size)
    noisy_img = img + e

    return noisy_img

def im_save(img, im_path, nchannels=1, norm_factor=0):
    """
    Save image to path. If image is a JPEG2000, save using GDAL. Otherwise, save using cv2.

    Args:
        img (np.ndarray): N x M x C image tosave
        im_path (str): Path to save location
        nchannels (int): Number of channels (default: 1)
        norm_factor (float): factor to scale image by (default: 65535.0 for JPEG2000 and 255.0 else)

    Returns:
        None
    """
    if im_path.endswith("jp2"):
        if norm_factor == 0:
            norm_factor = 65535.0
        rows = img.shape[0]
        cols = img.shape[1]
        driver = gdal.GetDriverByName("GTiff")
        dataset_out = driver.Create(im_path, cols, rows, nchannels, gdal.GDT_UInt16)
        if nchannels == 1:
            dataset_out.GetRasterBand(1).WriteArray(img[:, :] * norm_factor)
        else:
            for i in range(nchannels):
                dataset_out.GetRasterBand(i + 1).WriteArray(img[:, :, i] * norm_factor)
    else:
        if norm_factor == 0:
            norm_factor = 255.0
        if img.ndim < 3:
            img_bgr = img
        elif img.ndim == 3:
            img_bgr = np.zeros_like(img)
            img_bgr[:, :, 0] = img[:, :, 2]
            img_bgr[:, :, 1] = img[:, :, 1]
            img_bgr[:, :, 2] = img[:, :, 0]
        cv2.imwrite(im_path, img_bgr * norm_factor)

def im_show(
    imgs,
    title,
    subplot_rows,
    subplot_cols,
    figure_size=(8, 8),
    save_path="",
    display_range=None,
    cmap="gray",
    dpi=192,
    show_flag = False
):
    """
    Args:
        img (dictionary of np.ndarray): Dictionary containing images to display
        title (str): Title of entire figure
        subplot_rows (int): Number of rows in figure (rows * cols must equal the number of images)
        subplot_cols (int): Number of cols in figure (rows * cols must equal the number of images)
        figure_size (tuple of ints): Size of figure (default: (8, 8))
        save_path (str): If given, path to save figure. If '', do not save. (default: '')
        display_range (tuple of floats): Range of display (default: (min(img), max(img)))
        cmap (str): Color map to be used (default: 'gray')
        nchannels (int): Number of channels (default: 1)
        dpi (int): Display DPI (default: 192)

    """
    if subplot_rows * subplot_cols != len(imgs):
        print("Dimension of subplot inconsistent with number of images.")
        exit

    # Create plt figure
    fig, axs = plt.subplots(subplot_rows, subplot_cols, figsize=figure_size)

    # If unspecified, set the display range to be max and min of the images being displayed
    if display_range is None:
        imgs_min = 1
        imgs_max = 0
        for img in imgs.values():
            if np.min(img) < imgs_min:
                imgs_min = np.min(img)
            if np.max(img) > imgs_max:
                imgs_max = np.max(img)
    else:
        imgs_min = display_range[0]
        imgs_max = display_range[1]

    i = 0
    j = 0
    for key in imgs:
        metrics = (
            "Min: "
            + str(round(np.min(imgs[key]), 3))
            + "\n Max: "
            + str(round(np.max(imgs[key]), 3))
            + "\n Mean: "
            + str(round(np.mean(imgs[key]), 3))
            + "\n STD: "
            + str(round(np.std(imgs[key]), 3))
        )

        # If only displaying one image
        if len(imgs) == 1:
            im = axs.imshow(imgs[key], cmap=cmap, vmin=imgs_min, vmax=imgs_max)
            axs.title.set_text(key)
            axs.title.set_text(key)
            axs.set_xlabel(metrics)
        # If only displaying one row
        elif subplot_rows == 1:
            im = axs[j].imshow(imgs[key], cmap=cmap, vmin=imgs_min, vmax=imgs_max)
            axs[j].title.set_text(key)
            axs[j].set_xlabel(metrics)
        # If only displaying one column
        elif subplot_cols == 1:
            im = axs[i].imshow(imgs[key], cmap=cmap, vmin=imgs_min, vmax=imgs_max)
            axs[i].title.set_text(key)
            axs[i].set_xlabel(metrics)
        # If displaying multiple rows and columns
        else:
            im = axs[i, j].imshow(imgs[key], cmap=cmap, vmin=imgs_min, vmax=imgs_max)
            axs[i, j].title.set_text(key)
            axs[i, j].set_xlabel(metrics)
        # Move to the next column if not on the edge of the figure
        if j < subplot_cols - 1:
            j += 1
        # If on the edge of the figure, move to the next row
        else:
            j = 0
            i += 1
    # Add title to figure
    fig.suptitle(title)
    # fig.tight_layout()
    if subplot_rows + subplot_cols > 2:
        fig.colorbar(im, ax=axs.ravel().tolist(), shrink=2 * 1 / (subplot_cols + subplot_rows))
    else:
        fig.colorbar(im)

    # Save figure if location is specified (with corresponding dpi)
    if save_path != "":
        plt.savefig(save_path, dpi=dpi)

    # Show figure
    if show_flag:
        plt.show()
    plt.close()
