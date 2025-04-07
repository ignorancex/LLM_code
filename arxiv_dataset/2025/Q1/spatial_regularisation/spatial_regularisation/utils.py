import os
import glob
import torch
import itertools
import numpy as np
import nibabel as nib


def list_images_in_folder(path_dir, include_single_image=True, check_if_empty=True):
    """List all files with extension nii, nii.gz, mgz, or npz within a folder."""
    basename = os.path.basename(path_dir)
    if include_single_image & \
            (('.nii.gz' in basename) | ('.nii' in basename) | ('.mgz' in basename) | ('.npz' in basename)):
        assert os.path.isfile(path_dir), 'file %s does not exist' % path_dir
        list_images = [path_dir]
    else:
        if os.path.isdir(path_dir):
            list_images = sorted(glob.glob(os.path.join(path_dir, '*nii.gz')) +
                                 glob.glob(os.path.join(path_dir, '*nii')) +
                                 glob.glob(os.path.join(path_dir, '*.mgz')) +
                                 glob.glob(os.path.join(path_dir, '*.npz')))
        else:
            raise Exception('Folder does not exist: %s' % path_dir)
        if check_if_empty:
            assert len(list_images) > 0, 'no .nii, .nii.gz, .mgz or .npz image could be found in %s' % path_dir
    return list_images


def load_volume(filename, return_aff=False, dtype=None):
    """Read a nifty volume from a path. Can return the corresponding affine matrix if necessary."""
    x = nib.load(filename)
    vol = x.get_fdata()
    if vol.ndim == 3:
        vol = add_axis(vol, -1)
    if dtype is not None:
        if 'int' in dtype:
            vol = np.round(vol)
        vol = vol.astype(dtype=dtype)
    if return_aff:
        aff = x.affine
        return vol, aff
    else:
        return vol


def save_tensor(tensor, path, moveaxis=False, aff=None):
    """
    Save a torch tensor.
    :param tensor: volume to save
    :param moveaxis: whether to swap first and last axis (after squeezing)
    :param path: path where to save the volume.
    :param aff: affine matrix to use to save the tensor
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if aff is None:
        aff = np.eye(4)
    elif torch.is_tensor(aff):
        aff = aff.detach().cpu().numpy().squeeze()

    volume = tensor.detach().cpu().numpy().squeeze()
    if moveaxis:
        volume = np.moveaxis(volume, 0, -1)

    nib.save(nib.Nifti1Image(volume, aff, nib.Nifti1Header()), path)


def build_subject_dict(im_dir, lab_dir=None):
    """Build a dictionary of the form
    {'im basename': [path_im, path_lab]} if lab_dir is given
    {'im basename': [path_im]} if lab_dir is None.
    Corresponding images and label maps should e sorted in the same order in their own folder.
    """
    data_dict = {}
    path_images = list_images_in_folder(im_dir)
    path_labels = list_images_in_folder(lab_dir) if lab_dir is not None else [None] * len(path_images)
    assert len(path_images) == len(path_labels), 'not the same number of images and labels'
    for path_im, path_lab in zip(path_images, path_labels):
        if path_lab is not None:
            data_dict[os.path.basename(path_im).replace('.nii.gz', '')] = [path_im, path_lab]
        else:
            data_dict[os.path.basename(path_im).replace('.nii.gz', '')] = [path_im]
    return data_dict


def build_subject_dict_series(main_dir, name_image_dir, name_labels_dir=None):
    """This function expects time-series data to be organised as follows:
    time_series_1_dir: name_image_dir: image_time_frame_0
                                       image_time_frame_1
                                       ...
                       name_label_dir: label_time_frame_0
                                       label_time_frame_1
                                       ...
    To gather all the data, this function will loop over all the subfolders of main_dir. It will build a dictionary:
    {'time_series_1_dir': [[image_0, label_0], [image_1, label_1], ...]} if name_labels_dir is given
    {'time_series_1_dir': [[image_0], [image_1], ...]} if lab_dir is None.
    :param main_dir: path to the main directory
    :param name_image_dir: name of the subfolder containing the images in each time-series subfolder
    :param name_labels_dir: (optional) same as above but for labels
    """

    subj_dict = {os.path.basename(main_dir): []}
    im_dir = os.path.join(main_dir, name_image_dir)
    lab_dir = os.path.join(main_dir, name_labels_dir) if name_labels_dir is not None else None
    path_images = list_images_in_folder(im_dir)
    if lab_dir is not None:
        path_labels = list_images_in_folder(lab_dir)
        assert len(path_images) == len(path_labels), 'not the same number of images and labels'
    else:
        path_labels = [None] * len(path_images)
    for path_im, path_lab in zip(path_images, path_labels):
        if path_lab is not None:
            subj_dict[os.path.basename(main_dir)].append((path_im, path_lab))
        else:
            subj_dict[os.path.basename(main_dir)].append([path_im])
    return subj_dict


def build_xfm_dict(xfm_dir):
    """Build a dictionary of the form {'im basename': path_gt_xfm}."""
    data_dict = {}
    path_xfms = sorted(glob.glob(os.path.join(xfm_dir, '*.npy')))
    for path_xfm in path_xfms:
        subjID_frameID = os.path.basename(path_xfm).split('.')[0]
        data_dict[subjID_frameID] = path_xfm  # each entry of the dict is a list with a list of 1
    return data_dict


def build_xfm_dict_series(main_dir, name_xfm_dir):
    """Build a dictionary of the form {'im basename': [path_xfm_1, path_xfm_2, ...]}."""
    return {os.path.basename(main_dir): sorted(glob.glob(os.path.join(main_dir, name_xfm_dir, '*.npy')))}


def resize_zero_padding_nd(vol, shape, pad=True, crop=True):
    """This function takes the volume to the correct shape by zero padding and then by cropping.
    shape: tuple of the desired shape. In 3D, it can either be [H, W, D] or [H, W, D, C]
    """

    n_dims = len(vol.shape)

    dxyz = []
    dxyz_mod = []
    for i in range(n_dims):
        if i < len(shape) and shape[i] is not None:
            dxyz.append(np.abs(shape[i] - vol.shape[i]) // 2)
            dxyz_mod.append((shape[i] - vol.shape[i]) % 2)
        else:
            dxyz.append(None)
            dxyz_mod.append(None)

    crop_limits = []
    zpad_limits = []
    for i, (diff, diff_mod) in enumerate(zip(dxyz, dxyz_mod)):

        # padding index
        if i < len(shape) and diff is not None and shape[i] - vol.shape[i] > 0:
            zpad_limits.append((diff, diff + diff_mod))
            crop_limits.append((0, shape[i]))
        # cropping index
        elif i < len(shape) and diff is not None and shape[i] - vol.shape[i] < 0:
            zpad_limits.append((0, 0))
            crop_limits.append((diff, vol.shape[i] - diff - diff_mod))
        # no change, either None or diff == 0
        else:
            zpad_limits.append((0, 0))
            crop_limits.append((0, vol.shape[i]))

    if pad:
        resized = np.pad(vol, tuple(zpad_limits), 'constant')
    else:
        resized = vol

    if crop:
        slices = tuple(slice(int(start), int(stop)) for start, stop in crop_limits)
        resized = resized[slices]

    return resized


def preprocess(files,
               normalise=True,
               min_perc=0.01,
               max_perc=99.99,
               mean_05=True,
               resize=None,
               labels_lut=None,
               return_aff=False,
               dtype='float32'):
    """Load a volume, normalise its intensities and resize to the desired shape with zero-padding/cropping.
    :param files: can be the path to a volume, or a list of paths, or even a volume itself (i.e., a numpy array ).
    :param normalise: whether to normalise the intensities between 0 and 1 with min-max normalisation.
    :param min_perc: clip values to the given bottom percentile before rescaling.
    :param max_perc: clip values to the given top percentile before rescaling.
    :param mean_05: whether to make sure the mean intensity of non-zero voxels is 0.5 (after normalising to 0,1)
    :param resize: whether to resize the volume to a given shape (can either be [H, W, D] or [H, W, D, C]).
    This is done by zero-padding and clipping.
    :param labels_lut: look up table to convert labels into other values. Can only be used if loaded volume is int.
    :param return_aff: whether to return the affine matrix associated with the loaded volume.
    :param dtype: numpy dtype to convert volume to.
    """

    # read volume
    aff = None
    if isinstance(files, str):
        if return_aff:
            vol, aff = load_volume(files, return_aff=True, dtype=dtype)  # [H, W, D, C]
        else:
            vol = load_volume(files, dtype=dtype)
    elif isinstance(files, list):
        vol = np.concatenate([load_volume(f, dtype=dtype) for f in files], axis=-1)
    else:
        vol = files

    if normalise:
        vol = np.clip(vol, 0, None)
        m, M = np.percentile(vol.flat, [min_perc, max_perc])
        vol = np.clip(vol, m, M)
        vol = (vol - m) / (M - m)
        if mean_05:
            vol = vol / np.mean(vol[vol > 0]) * 0.5

    # resize
    if resize is not None:
        vol = resize_zero_padding_nd(vol, resize)

    if labels_lut is not None:
        vol = labels_lut[vol]

    if return_aff:
        return vol, aff  # [H, W, D, C]
    else:
        return vol


def create_transform(rx, ry, rz,
                     tx, ty, tz,
                     sx=1, sy=1, sz=1,
                     shxy=0, shxz=0, shyx=0, shyz=0, shzx=0, shzy=0,
                     ordering='txyzsh', input_angle_unit='degrees'):
    """create transformation matrix for rotation (degrees) and translation (pixels).
    ordering indicates the order of the operations (translation, then rotation around axis x, etc.)"""

    if torch.is_tensor(rx):

        if input_angle_unit == 'degrees':
            if len(rx.shape) == 1:
                rx, ry, rz = torch.split(torch.cat([rx, ry, rz], dim=0) / 180 * np.pi, 1, dim=0)
            else:
                rx, ry, rz = torch.split(torch.cat([rx, ry, rz], dim=1) / 180 * np.pi, 1, dim=1)

        one = torch.ones_like(rx)
        zero = torch.zeros_like(rx)

        Rx = torch.cat([torch.stack([one, zero, zero, zero], dim=-1),
                        torch.stack([zero, torch.cos(rx), -torch.sin(rx), zero], dim=-1),
                        torch.stack([zero, torch.sin(rx), torch.cos(rx), zero], dim=-1),
                        torch.stack([zero, zero, zero, one], dim=-1)],
                       dim=-2)

        Ry = torch.cat([torch.stack([torch.cos(ry), zero, torch.sin(ry), zero], dim=-1),
                        torch.stack([zero, one, zero, zero], dim=-1),
                        torch.stack([-torch.sin(ry), zero, torch.cos(ry), zero], dim=-1),
                        torch.stack([zero, zero, zero, one], dim=-1)],
                       dim=-2)

        Rz = torch.cat([torch.stack([torch.cos(rz), -torch.sin(rz), zero, zero], dim=-1),
                        torch.stack([torch.sin(rz), torch.cos(rz), zero, zero], dim=-1),
                        torch.stack([zero, zero, one, zero], dim=-1),
                        torch.stack([zero, zero, zero, one], dim=-1)],
                       dim=-2)

        T = torch.cat([torch.stack([one, zero, zero, tx], dim=-1),
                       torch.stack([zero, one, zero, ty], dim=-1),
                       torch.stack([zero, zero, one, tz], dim=-1),
                       torch.stack([zero, zero, zero, one], dim=-1)],
                      dim=-2)

        S = torch.cat([torch.stack([sx, zero, zero, zero], dim=-1),
                       torch.stack([zero, sy, zero, zero], dim=-1),
                       torch.stack([zero, zero, sz, zero], dim=-1),
                       torch.stack([zero, zero, zero, one], dim=-1)],
                      dim=-2)

        Sh = torch.cat([torch.stack([one, shxy, shxz, zero], dim=-1),
                        torch.stack([shyx, one, shyz, zero], dim=-1),
                        torch.stack([shzx, shzy, one, zero], dim=-1),
                        torch.stack([zero, zero, zero, one], dim=-1)],
                       dim=-2)

        transform_matrix = torch.cat([torch.stack([one, zero, zero, zero], dim=-1),
                                      torch.stack([zero, one, zero, zero], dim=-1),
                                      torch.stack([zero, zero, one, zero], dim=-1),
                                      torch.stack([zero, zero, zero, one], dim=-1)],
                                     dim=-2)

    else:

        if input_angle_unit == 'degrees':
            rx, ry, rz = np.array([rx, ry, rz]) * np.pi / 180

        if len(rx.shape) == 0:
            rx, ry, rz, tx, ty, tz = add_axis(np.array([rx, ry, rz, tx, ty, tz]), -1)
            shxy, shxz, shyx, shyz, shzx, shzy = add_axis(np.array([shxy, shxz, shyx, shyz, shzx, shzy]), -1)

        one = np.ones_like(rx)
        zero = np.zeros_like(rx)

        Rx = np.concatenate([np.stack([one, zero, zero, zero], axis=-1),
                             np.stack([zero, np.cos(rx), -np.sin(rx), zero], axis=-1),
                             np.stack([zero, np.sin(rx), np.cos(rx), zero], axis=-1),
                             np.stack([zero, zero, zero, one], axis=-1)],
                            axis=-2)

        Ry = np.concatenate([np.stack([np.cos(ry), zero, np.sin(ry), zero], axis=-1),
                             np.stack([zero, one, zero, zero], axis=-1),
                             np.stack([-np.sin(ry), zero, np.cos(ry), zero], axis=-1),
                             np.stack([zero, zero, zero, one], axis=-1)],
                            axis=-2)

        Rz = np.concatenate([np.stack([np.cos(rz), -np.sin(rz), zero, zero], axis=-1),
                             np.stack([np.sin(rz), np.cos(rz), zero, zero], axis=-1),
                             np.stack([zero, zero, one, zero], axis=-1),
                             np.stack([zero, zero, zero, one], axis=-1)],
                            axis=-2)

        T = np.concatenate([np.stack([one, zero, zero, tx], axis=-1),
                            np.stack([zero, one, zero, ty], axis=-1),
                            np.stack([zero, zero, one, tz], axis=-1),
                            np.stack([zero, zero, zero, one], axis=-1)],
                           axis=-2)

        S = np.diag([sx, sy, sz, 1])

        Sh = np.concatenate([np.stack([one, shxy, shxz, zero], axis=-1),
                             np.stack([shyx, one, shyz, zero], axis=-1),
                             np.stack([shzx, shzy, one, zero], axis=-1),
                             np.stack([zero, zero, zero, one], axis=-1)],
                            axis=-2)

        transform_matrix = np.eye(4)

    dd = {'x': Rx, 'y': Ry, 'z': Rz, 't': T, 's': S, 'h': Sh}
    for ll in ordering[::-1]:
        transform_matrix = dd[ll] @ transform_matrix

    return transform_matrix


def add_axis(x, axis=None):
    """Add axis to a numpy array or pytorch tensor.
    :param x: input array/tensor
    :param axis: index of the new axis to add. Can also be a list of indices to add several axes at the same time."""
    func = torch.unsqueeze if torch.is_tensor(x) else np.expand_dims
    axis = 0 if axis is None else axis
    if not isinstance(axis, list):
        axis = [axis]
    for ax in axis:
        x = func(x, ax)
    return x


def pts_to_xfm_numerical(means_moving, means_fixed, im_shape, weights=None, scaling_type=None):
    """gets rigid optimal transform to go from means_moving to means_fixed
    :param: means_moving, means_fixed: torch tensors of size [batch, n_channels, 3]
    :param: im_shape: torch tensor of size [3]
    :param: weights: torch tensor of size [batch, n_channels, 1] indicating the weight of each point in computing the
    transform. Doesn't need to be normalised (i.e., sum up to 1), because this is done internally.
    This parameter is interesting when points are virtually outside the image shape. In this case, it's good to
    set their weight to 0, such that they are not taken into account (otherwise they mess up the estimation).
    :param scaling_type: if scaling_type is not None, this function will also try to estimate a scaling between the two
    point-clouds. Chose between isotropic and anisotropic scaling.
    """

    # shift feature coordinates to center of the image
    half_im_shape = add_axis((im_shape - 1) / 2.0, [0, 0])  # [1, 1, 3]
    means_moving = means_moving - half_im_shape  # [B, K, 3]
    means_fixed = means_fixed - half_im_shape  # [B, K, 3]

    # correct for centroids
    if weights is not None:
        weights = weights / weights.sum(dim=1, keepdim=True)
        centroid_moving = (means_moving * weights).sum(dim=1, keepdims=True)  # [B, 1, 3]
        centroid_fixed = (means_fixed * weights).sum(dim=1, keepdims=True)  # [B, 1, 3]
    else:
        centroid_moving = torch.mean(means_moving, dim=1, keepdim=True)  # [B, 1, 3]
        centroid_fixed = torch.mean(means_fixed, dim=1, keepdim=True)  # [B, 1, 3]
    means_moving = means_moving - centroid_moving  # [B, K, 3]
    means_fixed = means_fixed - centroid_fixed  # [B, K, 3]

    if scaling_type is None:

        # SVD decomposition
        if weights is not None:
            H = (means_moving * weights).transpose(1, 2) @ means_fixed  # [B, 3, 3]
        else:
            H = means_moving.transpose(1, 2) @ means_fixed  # [B, 3, 3]
        U, S, V = torch.svd(H, compute_uv=True)  # U, V: [B, 3, 3],  S: [B, 3]

        # special reflection case
        dets = add_axis(torch.det(U) * torch.det(V), -1)  # [B, 1]
        dets = torch.stack([torch.ones_like(dets), torch.ones_like(dets), dets], dim=1)  # [B, 3, 1]
        xfm = V * torch.sign(dets) @ U.transpose(1, 2)  # [B, 3, 3]

    elif scaling_type == 'isotropic':
        assert weights is None, "weights interfere with scaling, so these two parameters cannot be used together"

        # SVD decomposition
        H = means_moving.transpose(1, 2) @ means_fixed  # [B, 3, 3]
        U, S, V = torch.svd(H, compute_uv=True)  # U, V: [B, 3, 3],  S: [B, 3]

        # special reflection case
        dets = add_axis(torch.det(U) * torch.det(V), -1)  # [B, 1]
        dets = torch.stack([torch.ones_like(dets), torch.ones_like(dets), dets], dim=1)  # [B, 3, 1]
        R = V * torch.sign(dets) @ U.transpose(1, 2)  # [B, 3, 3]

        # find scaling
        scaling = S.sum(dim=1) / torch.einsum("...ii", means_moving.transpose(1, 2) @ means_moving)
        scaling = add_axis(scaling, [1, 2]) * add_axis(torch.eye(3))
        xfm = R @ scaling

    elif scaling_type == 'anisotropic':

        assert weights is None, "weights interfere with scaling, so these two parameters cannot be used together"

        # SVD decomposition
        H = means_fixed.transpose(1, 2) @ means_moving @ torch.linalg.inv(means_moving.transpose(1, 2) @ means_moving)
        U, S, V = torch.svd(H.transpose(1, 2), compute_uv=True)

        # special reflection case
        dets = add_axis(torch.det(U) * torch.det(V), -1)  # [B, 1]
        dets = torch.stack([torch.ones_like(dets), torch.ones_like(dets), dets], dim=1)  # [B, 3, 1]
        R = V * torch.sign(dets) @ U.transpose(1, 2)  # [B, 3, 3]

        # find scaling
        scaling = list()
        for i in range(3):
            lambda_i = torch.zeros_like(R)
            lambda_i[:, i, i] = torch.ones(1)
            numerator = torch.einsum("...ii", means_fixed @ R @ lambda_i @ means_moving.transpose(1, 2))
            denominator = torch.einsum("...ii", means_moving @ lambda_i @ means_moving.transpose(1, 2))
            scaling.append(numerator / denominator)
        scaling = torch.diag_embed(torch.stack(scaling, dim=-1))
        xfm = R @ scaling

    else:
        raise ValueError("Scaling type should be None, 'isotropic' or 'anisotropic', had %s" % scaling_type)

    # find translation
    T = centroid_fixed.transpose(1, 2) - xfm @ centroid_moving.transpose(1, 2)   # [B, 3, 1]

    # reconstruct affine matrix
    xfm = torch.cat([xfm, T], dim=-1)  # [B, 3, 4]
    last_row = add_axis(torch.tensor([0] * means_moving.shape[-1] + [1]), [0, 0]).repeat(xfm.shape[0], 1, 1)
    xfm = torch.cat([xfm, last_row.to(device=xfm.device, dtype=xfm.dtype)], dim=1)  # [B, 4, 4]

    return xfm


def pts_to_xfm_analytical(means_moving, means_fixed, im_shape):
    """Finds the optimal transform between two point clouds, gotten from an image of shape im_shape."""

    # shift feature coordinates to center of the image
    half_im_shape = add_axis((im_shape - 1) / 2.0, [0, 0])  # [1, 1, 3]
    means_moving = means_moving - half_im_shape  # [B, K, 3]
    means_fixed = means_fixed - half_im_shape  # [B, K, 3]

    # convert to homogeneous coordinates
    means_moving = means_moving.permute(0, 2, 1)  # [B, 3, K]
    means_fixed = means_fixed.permute(0, 2, 1)  # [B, 3, K]
    ones = torch.ones(means_moving.shape[0], 1, means_moving.shape[2]).float().to(means_moving.device)  # [B, 1, K]
    means_moving = torch.cat([means_moving, ones], 1)  # [B, 4, K]

    # reconstruct affine matrix
    xfm = torch.bmm(means_moving, torch.transpose(means_moving, -2, -1))  # [B, 4, 4]
    xfm = torch.inverse(xfm)  # [B, 4, 4]
    xfm = torch.bmm(torch.transpose(means_moving, -2, -1), xfm)  # [B, K, 4]
    xfm = torch.bmm(means_fixed, xfm)

    # add last row to make it [B, 4, 4]
    last_row = add_axis(torch.tensor([0] * means_fixed.shape[-2] + [1]), [0, 0]).repeat(xfm.shape[0], 1, 1)
    xfm = torch.cat([xfm, last_row.to(device=xfm.device, dtype=xfm.dtype)], dim=1)  # [B, 4, 4]

    return xfm


def make_gaussian(mean, covariance_matrix, im_shape, gaussian_type='anisotropic'):
    """This function returns a torch tensor of size [B, C, *im_shape] representing Gaussians of specified means and
    covariance matrices. The covariance matrices can be given as tensors of shape [B, C] (isotropic Gaussians) or
     [B, C, n_dims, n_dims] (anisotropic Gaussians).
    :param mean: means of the Gaussians. size is [B, C, n_dims]
    :param covariance_matrix: covariance matrix of the Gaussians. size is [B, C] (isotropic Gaussians) or
    [B, C, n_dims, n_dims] (anisotropic Gaussians).
    :param im_shape: dimensions of the Gaussians to create, for example [H, W] (2D case) or [H, W, D] (3D case).
    :param gaussian_type: type of the Gaussian to create.
    """

    # prepare coordinate grid
    n_dims = len(im_shape)
    coords_list = [torch.arange(shape, dtype=mean.dtype, device=mean.device) for shape in im_shape]
    coords_grid = add_axis(torch.stack(torch.meshgrid(*coords_list), -1), [0, 1])  # [1, 1, *, n_dims]

    # simplified version for isotropic (no matrix inversion, no reshaping)
    if gaussian_type == 'isotropic':

        mean = add_axis(mean, axis=[2] * n_dims)
        covariance_matrix = add_axis(covariance_matrix, [-1] * n_dims)
        exp_term = - torch.sum((coords_grid - mean) ** 2, dim=-1) / (2 * covariance_matrix)
        gaussian = torch.exp(exp_term) / torch.sqrt((2 * torch.pi * covariance_matrix) ** n_dims)

    elif gaussian_type == 'anisotropic':

        # linear algebra on covariance matrix
        det = torch.linalg.det(covariance_matrix)
        try:
            covariance_matrix_inv = torch.linalg.inv(covariance_matrix)

        # In case there's a singular matrix (it happens when the features are a single point, so the covariance matrix
        # is very small everywhere) we split along batch and channel to invert the cov one by one. We replace the
        # singular covariance matrices by 1e-8 * Identity
        except RuntimeError:
            covariance_matrix_inv = list()
            id_3x3 = add_axis(torch.eye(3, device=covariance_matrix.device), [0, 1])
            for cov_batch in torch.split(covariance_matrix, split_size_or_sections=1, dim=0):
                cov_channel_inv = list()
                for cov_channel in torch.split(cov_batch, split_size_or_sections=1, dim=1):
                    if torch.max(cov_channel) < 1e-6:
                        cov_channel_inv.append(torch.linalg.inv(1e-6 * id_3x3))
                    elif torch.linalg.det(cov_channel) == 0:
                        cov_channel_inv.append(torch.linalg.inv(torch.clamp(cov_channel, min=1e-6) * id_3x3))
                    else:
                        cov_channel_inv.append(torch.linalg.inv(cov_channel))
                cov_channel_inv = torch.concat(cov_channel_inv, dim=1)
                covariance_matrix_inv.append(cov_channel_inv)
            covariance_matrix_inv = torch.concat(covariance_matrix_inv, dim=0)
            det = 1 / torch.linalg.det(covariance_matrix_inv)

        # prepare exp term
        coords_grid = coords_grid - add_axis(mean, axis=[2] * n_dims)
        coords_grid = torch.reshape(coords_grid, list(coords_grid.shape[:2]) + [np.prod(im_shape), n_dims])
        exp_term = torch.moveaxis(torch.matmul(covariance_matrix_inv, torch.moveaxis(coords_grid, -2, -1)), -2, -1)
        exp_term = -0.5 * (coords_grid * exp_term).sum(dim=-1)

        # make gaussian
        gaussian = torch.exp(exp_term) / torch.sqrt(add_axis(det, axis=-1) * (2 * torch.pi) ** n_dims)
        gaussian = torch.reshape(gaussian, list(coords_grid.shape[:2]) + list(im_shape))

    else:
        raise ValueError('gaussian_type must be anisotropic or anisotropic, had %s' % gaussian_type)

    # make gaussian probabilistic, just in case (we can have funny problems because of field-of-view)
    return gaussian / gaussian.sum(dim=list(range(2, 2 + n_dims)), keepdim=True)  # [B, C, *]


def aff_to_field(affine_matrix, image_size, invert_affine=False, rescale_values=False, keep_centred=False):
    """Build a deformation field out of a transformation matrix. Can handle inputs with/without batch dim.
    :param affine_matrix: torch tensor or numpy array of size [B, n_dims + 1, n_dims + 1] or [n_dims + 1, n_dims + 1].
    :param image_size: size of the images that we will deform with the returned field. This excludes batch or channel
    dimensions, so it must be [H, W] (2D case) or [W, H, D] (3D case).
    :param invert_affine: whether to invert the affine matrix before computing the field. Useful for pull transforms.
    :param rescale_values: whether to rescale all the values of the field between [-1, 1], where [-1, -1] would be the
    top left corner and [1, 1] the bottom right corner for a 2d square. (useful for torch grid_sampler)
    :param keep_centred: whether to keep the center of coordinates at the center of the field.
    returns: a tensor of shape [B, *image_size, n_dims]"""

    n_dims = len(image_size)
    includes_batch_dim = len(affine_matrix.shape) == 3

    # make sure affine_matrix is float32 tensor
    if not torch.is_tensor(affine_matrix):
        affine_matrix = torch.tensor(affine_matrix, dtype=torch.float32)
    if affine_matrix.dtype != torch.float32:
        affine_matrix = affine_matrix.to(dtype=torch.float32)

    # meshgrid of coordinates
    coords = torch.meshgrid(*[torch.arange(s, device=affine_matrix.device, dtype=torch.float32) for s in image_size])

    # shift to centre of image
    offset = [(image_size[f] - 1) / 2 for f in range(len(image_size))]
    coords = [coords[f] - offset[f] for f in range(len(image_size))]
    if rescale_values | (not keep_centred):
        offset = add_axis(torch.tensor(offset, device=affine_matrix.device), [0] * (len(affine_matrix.shape) - 1))

    # add an all-ones entry (for homogeneous coordinates) and reshape into a list of points
    coords = [torch.flatten(f) for f in coords]
    coords.append(torch.ones_like(coords[0]))
    coords = torch.transpose(torch.stack(coords, dim=1), 0, 1)  # n_dims + 1 x n_voxels
    if includes_batch_dim:
        coords = add_axis(coords)

    # compute transform of each point
    if invert_affine:
        affine_matrix = torch.linalg.inv(affine_matrix)
    field = torch.matmul(affine_matrix, coords)  # n_dims + 1 x n_voxels
    if includes_batch_dim:
        field = torch.transpose(field, 1, 2)[..., :n_dims]  # n_voxels x n_dims
    else:
        field = torch.transpose(field, 0, 1)[..., :n_dims]  # n_voxels x n_dims

    # rescale values in [-1, 1]
    if rescale_values:
        field /= offset

    if not keep_centred:
        field += offset

    # reshape field to grid
    if includes_batch_dim:
        new_shape = [field.shape[0]] + list(image_size) + [n_dims]
    else:
        new_shape = list(image_size) + [n_dims]
    field = torch.reshape(field, new_shape)  # *volshape x n_dims

    return field


def interpolate(vol, loc, batch_size=0, method='linear', vol_dtype=torch.float32):
    """Perform interpolation of provided volume based on the given voxel locations.

    :param vol: volume to interpolate. torch tensor or numpy array of size [dim1, dim2, ..., channel] or
    [B, dim1, dim2, ..., channel].
    WARNING!! if there's a batch dimension, please specify it in corresponding parameter.
    :param loc: locations to interpolate from. torch tensor or numpy array of size [dim1, dim2, ..., n_dims] or
    [B, dim1, dim2, ..., n_dims].
    :param batch_size: batch size of the provided vol and loc. Put 0 if these two tensors don't have a batch dimension.
    :param method: either "nearest" or "linear"
    :param vol_dtype: dtype of vol if we need to convert it from numpy array to torch tensor.

    returns: a pytorch tensor with the same shape as vol, where, for nearest interpolation in 3d we have
    output[i, j, k] = vol[loc[i, j, k, 0], loc[i, j, k, 1], loc[i, j, k, 2]]
    """

    # convert to tensor
    if not torch.is_tensor(vol):
        vol = torch.tensor(vol, dtype=vol_dtype)
    if not torch.is_tensor(loc):
        loc = torch.tensor(loc, device=vol.device, dtype=torch.float32)

    # get dimensions
    vol_shape_all_dims = list(vol.shape)
    vol_shape = vol_shape_all_dims[1:-1] if batch_size > 0 else vol_shape_all_dims[:-1]
    n_dims = loc.shape[-1]
    n_channels = vol_shape_all_dims[-1]
    vol = torch.reshape(vol, [-1, n_channels])

    if method == 'nearest':

        # round location values
        round_loc = torch.round(loc).to(torch.int32)

        # clip location values to volume shape
        max_loc = torch.tensor(vol_shape, device=vol.device, dtype=torch.int32) - 1
        max_loc = add_axis(max_loc, [0] * len(round_loc.shape[:-1]))
        round_loc = torch.clamp(round_loc, torch.zeros_like(max_loc), max_loc)

        # get values
        indices = coords_to_indices(round_loc, vol_shape, n_channels, batch=batch_size)
        vol_interp = torch.gather(vol, 0, indices).reshape(vol_shape_all_dims)

    elif method == 'linear':

        # get lower locations of the cube
        loc0 = torch.floor(loc)

        # clip location values to volume shape
        max_loc = [ss - 1 for ss in vol_shape]
        clipped_loc = [torch.clamp(loc[..., d], 0, max_loc[d]) for d in range(n_dims)]
        loc0_list = [torch.clamp(loc0[..., d], 0, max_loc[d]) for d in range(n_dims)]

        # get other end of point cube
        loc1_list = [torch.clamp(loc0_list[d] + 1, 0, max_loc[d]) for d in range(n_dims)]
        locs = [[tens.to(torch.int32) for tens in loc0_list], [tens.to(torch.int32) for tens in loc1_list]]

        # compute distances between points and upper and lower points of the cube
        dist_loc1 = [loc1_list[d] - clipped_loc[d] for d in range(n_dims)]
        dist_loc0 = [1 - d for d in dist_loc1]
        weights_loc = [dist_loc1, dist_loc0]  # note reverse ordering since weights are inverse of distances

        # go through all the cube corners, indexed by a binary vector
        vol_interp = 0
        cube_pts = list(itertools.product([0, 1], repeat=n_dims))
        for c in cube_pts:

            # get locations for this cube point
            tmp_loc = torch.stack([locs[c[d]][d] for d in range(n_dims)], -1)

            # get values for this cube point
            tmp_indices = coords_to_indices(tmp_loc, vol_shape, n_channels, batch=batch_size)
            tmp_vol_interp = torch.gather(vol, 0, tmp_indices).reshape(vol_shape_all_dims)

            # get weights for this cube point: if c[d] is 0, weight = dist_loc1, else weight = dist_loc0
            tmp_weights = [weights_loc[c[d]][d] for d in range(n_dims)]
            tmp_weights = torch.prod(torch.stack(tmp_weights, -1), dim=-1, keepdim=True)

            # compute final weighted value for each cube corner
            vol_interp = vol_interp + tmp_weights * tmp_vol_interp

    else:
        raise ValueError('method should be nearest or linear, had %s' % method)

    return vol_interp


def coords_to_indices(coords, vol_shape, n_channels, batch=0):
    cum_prod_shape = np.flip(np.cumprod([1] + list(vol_shape[::-1])))[1:]
    cum_prod_shape = add_axis(torch.tensor(cum_prod_shape.copy(), device=coords.device), [0] * len(vol_shape))
    if batch > 0:
        cum_prod_shape = add_axis(cum_prod_shape)
    indices = torch.sum(coords * cum_prod_shape, dim=-1)
    if batch > 0:
        batch_correction = torch.tensor(np.arange(batch) * np.prod(vol_shape), device=coords.device)
        indices += add_axis(batch_correction, [-1] * len(vol_shape))
    indices = add_axis(indices.flatten(), -1).repeat(1, n_channels)
    return indices


def matrix_to_angles(matrix, convention='XYZ'):
    """
    From PyTorch3d.transforms
    Convert rotations given as rotation matrices to Euler angles in radians.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
        convention: Convention string of three uppercase letters.

    Returns:
        Euler angles in radians as tensor of shape (..., 3).
    """
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")
    i0 = _index_from_letter(convention[0])
    i2 = _index_from_letter(convention[2])
    tait_bryan = i0 != i2
    if tait_bryan:
        central_angle = torch.asin(
            torch.clamp(matrix[..., i0, i2] * (-1.0 if i0 - i2 in [-1, 2] else 1.0), min=-1, max=1)
        )
    else:
        central_angle = torch.acos(torch.clamp(matrix[..., i0, i0], min=-1, max=1))

    o = (
        _angle_from_tan(
            convention[0], convention[1], matrix[..., i2], False, tait_bryan
        ),
        central_angle,
        _angle_from_tan(
            convention[2], convention[1], matrix[..., i0, :], True, tait_bryan
        ),
    )
    return torch.stack(o, -1)


def _index_from_letter(letter: str) -> int:
    """From PyTorch3d.transforms"""
    if letter == "X":
        return 0
    if letter == "Y":
        return 1
    if letter == "Z":
        return 2
    raise ValueError("letter must be either X, Y or Z.")


def _angle_from_tan(
    axis: str, other_axis: str, data, horizontal: bool, tait_bryan: bool
) -> torch.Tensor:
    """
    From PyTorch3d.transforms
    Extract the first or third Euler angle from the two members of
    the matrix which are positive constant times its sine and cosine.

    Args:
        axis: Axis label "X" or "Y or "Z" for the angle we are finding.
        other_axis: Axis label "X" or "Y or "Z" for the middle axis in the
            convention.
        data: Rotation matrices as tensor of shape (..., 3, 3).
        horizontal: Whether we are looking for the angle for the third axis,
            which means the relevant entries are in the same row of the
            rotation matrix. If not, they are in the same column.
        tait_bryan: Whether the first and third axes in the convention differ.

    Returns:
        Euler Angles in radians for each matrix in data as a tensor
        of shape (...).
    """

    i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
    if horizontal:
        i2, i1 = i1, i2
    even = (axis + other_axis) in ["XY", "YZ", "ZX"]
    if horizontal == even:
        return torch.atan2(data[..., i1], data[..., i2])
    if tait_bryan:
        return torch.atan2(-data[..., i2], data[..., i1])
    return torch.atan2(data[..., i2], -data[..., i1])
