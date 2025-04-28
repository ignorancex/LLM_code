"""
Helper functions specific to the baseline variability code

@author: Ishaan Bhat
@email: ishaan@isi.uu.nl

"""
import torch
import os
from metrics import dice_score
import numpy as np
import SimpleITK as sitk
from torchviz import make_dot
from random import sample, shuffle
import random
import pandas as pd
from scipy.ndimage import rotate
from scipy.ndimage.filters import gaussian_filter
from distutils.dir_util import copy_tree
import shutil

failed_registrations = [1, 49, 73, 86, 3, 6, 18, 19, 34, 35, 40, 50, 74]
LITS_TEST_PATS = 18
LITS_VAL_PATS = 10

def clean_up_ray_trials(exp_dir=None,
                        best_trial_dir=None,
                        new_dir=None):
    """
    Function to remove trial directories apart from the "winner"

    """
    assert(exp_dir is not None)
    assert(best_trial_dir is not None)

    ray_exp_dir = [f.path for f in os.scandir(exp_dir) if f.is_dir()]
    assert(len(ray_exp_dir) == 1)

    ray_trial_dirs = [f.path for f in os.scandir(ray_exp_dir[0]) if f.is_dir()]

#    for trial_dir in ray_trial_dirs:
#        if trial_dir != best_trial_dir:
#            shutil.rmtree(trial_dir)

    # Using shutil.move() function to change the checkpoint directory caused
    # a corruption in the file due to which the model could not be loaded
    # during inference. This has been fixed by use the copy_tree function
    # from the distutils package
    if new_dir is not None: # Copy the best trial directory to the new place
        if os.path.exists(new_dir) is False:
            os.makedirs(new_dir)
        copy_tree(os.path.join(best_trial_dir, 'checkpoints'), os.path.join(new_dir, 'checkpoints'))
        copy_tree(os.path.join(best_trial_dir, 'logs'), os.path.join(new_dir, 'logs'))

#        shutil.rmtree(ray_exp_dir[0])

def set_mcd_eval_mode(model):
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def calculate_num_patches_in_slice(slice_size=256, patch_size=128, stride=64):

    # Calculate patches in 1 direction
    num_patches_single_dim = 1
    patch_limit = patch_size # Track the end of the patch
    while(patch_limit<=slice_size):
        patch_limit +=  stride
        num_patches_single_dim += 1

    # For 2-D symmetric slices
    num_patches = num_patches_single_dim*num_patches_single_dim

    return num_patches

def get_image_paths(image_dir):

    image_paths = [os.path.join(image_dir, img_path) for img_path in os.listdir(image_dir) if
                   (os.path.isfile(os.path.join(image_dir, img_path)) and img_path.split('.')[-1] == 'gz' and
                    img_path.split('.')[-2] == 'nii' and img_path.split('.')[-3].split('_')[-1] == 'image')]
    return image_paths


def get_patient_dirs(data_dir):

    pat_dirs = [os.path.join(data_dir, pat_dir) for pat_dir in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, pat_dir))]
    if os.path.exists(os.path.join(data_dir, 'DCEDWI')):
        pat_dirs.remove(os.path.join(data_dir,'DCEDWI'))

    for excluded_patient in failed_registrations:
        try:
            pat_dirs.remove(os.path.join(data_dir, str(excluded_patient)))
        except ValueError:
            continue

    return pat_dirs

def get_train_and_val_dirs(data_dir, n_val=5):
    pat_dirs = get_patient_dirs(data_dir)
    shuffle(pat_dirs)
    val_dirs = pat_dirs[0:n_val]
    train_dirs = pat_dirs[n_val:]
    return train_dirs, val_dirs

def create_split_for_lits(seed=42, num_patients=131,skip_patients=None):

    patient_ids = [i for i in range(num_patients)]
    if skip_patients is not None:
        patient_ids = list(set(patient_ids) - set(skip_patients)) # Compute the set difference between all patient IDs and the ones without a tumor


    print('Number of patients in dataset = {}'.format(len(patient_ids)))
    # Shuffle to create the test - (train, val) split
    random.seed(seed)
    random.shuffle(patient_ids)
    test_patients = patient_ids[:LITS_TEST_PATS]
    train_val_patients = patient_ids[LITS_TEST_PATS:]

    # Shuffle the list with train and val patients
    random.shuffle(train_val_patients)
    val_patients = train_val_patients[:LITS_VAL_PATS]
    train_patients = train_val_patients[LITS_VAL_PATS:]

    print('Number of training patients = {}'.format(len(train_patients)))
    print('Number of validation patients = {}'.format(len(val_patients)))
    print('Number of test patients = {}'.format(len(test_patients)))

    return train_patients, val_patients, test_patients




def create_model_graph(model=None, writer=None):
    assert(model is not None)
    assert(writer is not None)
    dummy_images = torch.randn(1, 1, 64, 128, 128)
    writer.add_graph(model, dummy_images)
    return writer

def calculate_dice_score(preds, gt):
    """
    Calculates dice overlap for a pair of tensors
    Args:
        preds: (torch.Tensor)
        gt: (torch.Tensor)

    Returns:
        dice_coeff: (float) Dice overlap co-efficient

    """
    if isinstance(preds, torch.Tensor):
        if preds.device != torch.device('cpu'):
            preds = preds.cpu().numpy()
        else:
            preds = preds.numpy()

    if isinstance(gt, torch.Tensor):
        if gt.device != torch.device('cpu'):
            gt = gt.cpu().numpy()
        else:
            gt = gt.numpy()

    # Add fake batch axis
    preds = np.expand_dims(preds, axis=0)
    gt = np.expand_dims(gt, axis=0)
    dice_coeff = dice_score(preds.astype(np.float32), gt.astype(np.float32))
    dice_coeff = np.mean(dice_coeff)

    return dice_coeff


def calculate_mean_for_list(pylist):
    """
    Function to compute mean of a python list

    Args:
        pylist: (list)

    Returns:
        mean: (float)
    """
    assert (isinstance(pylist, list))
    pyndarray = np.asarray(pylist, dtype=np.float32)
    return np.mean(pyndarray)


def calculate_weights_per_class(labels, num_classes=3):
    """
    Calculate weights per class for a given batch.
    Class-balancing idea from  Christ et al. (2017) URL: http://arxiv.org/abs/1702.05970

    Args:
        labels: (torch.Tensor) Shape -- (B, D, H, W)
        num_classes: (int) Providing this explicity because not all patches may contain examples from all 3 classes
    Returns:
        class_weights: (torch.Tensor) -- vector of length of C

    """
    assert (isinstance(labels, torch.Tensor))
    assert (labels.dim() == 3)
    eps = 1e-4
    class_weights = torch.zeros(size=(num_classes,), dtype=torch.float32)
    total_num_voxels = labels.shape[0]*labels.shape[1]*labels.shape[2]
    for class_id in range(num_classes):
        num_voxels_in_class = torch.sum(torch.where(labels == class_id, torch.ones_like(labels), torch.zeros_like(labels))).item()
        frac_voxels = num_voxels_in_class/total_num_voxels

        class_weights[class_id] = 1/np.log(frac_voxels+1.2)

    return class_weights

def get_slices_from_indices(indices_dict, batch_size=4):
    """
    Function to extract indices of 3D volume(s) from a dictionary

    Args:
        indices_dict: (Python dictionary)
    Returns:
        slices: (Python list) List of slices, each element of the list corr. to one item in the batch

    """
    assert(isinstance(indices_dict, dict))

    # Extract tensors from dictionary
    z_indices = indices_dict['z']
    y_indices = indices_dict['y']
    x_indices = indices_dict['x']

    z_slice_list = construct_slice_for_axis(z_indices)
    y_slice_list = construct_slice_for_axis(y_indices)
    x_slice_list = construct_slice_for_axis(x_indices)

    slices = []

    # Create a tuple of 3 elems (one for each patch in the batch)
    for z_slice, y_slice, x_slice in zip(z_slice_list, y_slice_list, x_slice_list):
        slice_tuple = (z_slice, y_slice, x_slice)
        slices.append(slice_tuple)
    assert(len(slices) == batch_size)

    return slices

def construct_slice_for_axis(axis_indices):
    """
    Construct slices for a given axis/direction from a list of tensors

    """

    assert(isinstance(axis_indices, list))
    start_indices = axis_indices[0]
    end_indices = axis_indices[1]
    assert(start_indices.shape == end_indices.shape)

    slice_list = []
    for idx in range(start_indices.shape[0]):
        patch_start = start_indices[idx].item()
        patch_end = end_indices[idx].item()
        slice_list.append(slice(patch_start, patch_end))
    return slice_list

def save_itk_image(data, fname, z_first=False, metadata=None, dtype=None):
    """
    Save tensor/numpy array as an ITK image.
    3D and 4D images supported

    For 4-D images, the expected order of the input is : C x H x W x D
    For 3-D images, the expected order of the input is : H x W x D
    """
    if (isinstance(data, torch.Tensor)):
        data = data.numpy()

    if dtype is not None:
        data = np.array(data, dtype=dtype)

    if data.ndim == 3:
        if z_first is False:
            data = np.transpose(data, (2, 0, 1))  # Put the slice axis first
        itk_img = sitk.GetImageFromArray(data)

        if metadata is not None:
            itk_img.SetSpacing(metadata['spacing'])
            itk_img.SetDirection(metadata['direction'])
            itk_img.SetOrigin(metadata['origin'])

    elif data.ndim == 4:
        data = np.transpose(data, (0, 3, 1, 2))
        n_channels = data.shape[0]
        series = []
        for channel in range(n_channels):
            series.append(sitk.GetImageFromArray(data[channel, :, :, :]))

        itk_img = sitk.JoinSeries(series)

    else:
        print('Cannot save image, dim={} not supported'.format(data.ndim))

    sitk.WriteImage(itk_img, fname)

def save_metric_dict(metric_dict=None, fname=None):
    """
    Save the lesion counts dictionary (capturing TP/FP/FN) as a csv using
    the Pandas dataframe as an intermediate data structure

    :param counts: (list) List of per-patient dictionaries
    :param fname: (str) Filename for the CSV file

    """

    assert(isinstance(metric_dict, dict))

    ext = fname.split('.')[-1]
    assert(ext == 'csv')

    df = pd.DataFrame.from_dict(data=metric_dict)

    print(df)

    print('Mean number of false positives = {}'.format(df['False Positives'].mean()))
    print('Mean number of false negatives = {}'.format(df['False Negatives'].mean()))

    df.to_csv(fname)

def calculate_entropy(pred_mean):
    """
    Function to caclulate entropy of predictions
    :param pred_distribution: 3-D mean prediction obtained from multiple MC-Dropout passes
                              Shape : C x H x W x D

    For formula see: Kendall, Alex, and Yarin Gal. “What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?” ArXiv:1703.04977 [Cs], October 5, 2017. http://arxiv.org/abs/1703.04977.

    """

    if isinstance(pred_mean, torch.Tensor):
        if pred_mean.device != torch.device('cpu'):
            pred_mean = pred_mean.cpu().numpy()
        else:
            pred_mean = pred_mean.numpy()

    pred_mean_fg = pred_mean[1, :, :, :]
    pred_mean_bg = pred_mean[0, :, :, :]

    pred_entropy = -(pred_mean_bg*np.log(pred_mean_bg + 1e-8) + pred_mean_fg*np.log(pred_mean_fg + 1e-8))
    pred_entropy = np.where(pred_entropy < 0, 0, pred_entropy) # Weirdly, the minimum is sometimes -0.0 :(

    return pred_entropy

def rotate_tensor(img=None, rot_angle=None, order=1, transpose=True, verbose=False):
    """
    Function to rotate a (batch of ) tensor
        img: (torch.Tensor or np.ndarray) Tensor of the shape [N, C, H, W] or [S, N, C, H, W]
        rot_angle: (float) Angle of rotation
        Returns:
            rot_img: (torch.Tensor) rotated image

    """

    if rot_angle == 0.0:

        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)

        return img

    if isinstance(img, torch.Tensor):
        img_np = img.numpy()
    elif isinstance(img, np.ndarray):
        img_np = img
    else:
        raise ValueError('{} type is not supported'.format(type(img)))

    if transpose is True:
        if img_np.ndim == 5:
            img_np = img_np.transpose((3, 4, 2, 1, 0)) # [H, W, C, N, S]
        elif img_np.ndim == 4:
            img_np = img_np.transpose((2, 3, 1, 0)) # [H, W, C, N]
        elif img_np.ndim == 3:
            img_np = img_np.transpose((2, 0, 1))

    h = img_np.shape[0]
    w = img_np.shape[1]

    # Rotate the numpy ndarray
    if verbose is True:
        print('Image shape before rotation : {}'.format(img_np.shape))

    rot_img_np = rotate(input = img_np,
                        angle=rot_angle,
                        axes=(1,0),
                        reshape=False,
                        order=order,
                        mode='constant',
                        cval=0.0)
    if verbose is True:
        print('Image shape after rotation : {}'.format(rot_img_np.shape))

    # Re-arrange the axes to batch-size first [(S), N, C, H, W]
    if transpose is True:
        if rot_img_np.ndim == 5:
            rot_img_np = rot_img_np.transpose((4, 3, 2, 0, 1))
        elif rot_img_np.ndim == 4:
            rot_img_np = rot_img_np.transpose((3, 2, 0, 1))
        elif rot_img_np.ndim == 3:
            rot_img_np = rot_img_np.transpose((2, 0, 1))

    # Convert to torch.Tensor
    rot_img = torch.from_numpy(rot_img_np)

    if verbose is True:
        print('Image shape after conversion to tensor : {}'.format(rot_img.shape))

    return rot_img

def create_gaussian_importance_map(patch_size):
    """
    Create a gaussian importance map, used to merge patch predictions.
    Center voxels in the prediction given a higher weight
    See code at: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/network_architecture/neural_network.py (Line 250)

    """
    tmp = np.zeros((patch_size, patch_size), dtype=np.float32)
    sigmas = [patch_size*(1/8), patch_size*(1/8)]
    center_coords = np.array([patch_size//2, patch_size//2])
    tmp[center_coords] = 1
    gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
    gaussian_importance_map = gaussian_importance_map/np.amax(gaussian_importance_map)
    gaussian_importance_map = gaussian_importance_map.astype(np.float32)
    min_non_zero_value = np.amin(gaussian_importance_map[gaussian_importance_map != 0])
    gaussian_importance_map[gaussian_importance_map == 0] = min_non_zero_value
    assert(np.amin(gaussian_importance_map) > 0)

    return gaussian_importance_map



# Code taken from: https://github.com/google-research/google-research/blob/9bd249201efac2ac60d8fe93a29c0058be544b5a/uq_benchmark_2019/metrics_lib.py#L31
def bin_predictions_and_accuracies(probabilities, ground_truth, bins=10):
    """A helper function which histograms a vector of probabilities into bins.
    Args:
    probabilities: A numpy vector of N probabilities assigned to each prediction
    ground_truth: A numpy vector of N ground truth labels in {0,1}
    bins: Number of equal width bins to bin predictions into in [0, 1], or an
      array representing bin edges.
    Returns:
    bin_edges: Numpy vector of floats containing the edges of the bins
      (including leftmost and rightmost).
    accuracies: Numpy vector of floats for the average accuracy of the
      predictions in each bin.
    counts: Numpy vector of ints containing the number of examples per bin.
    """

    if len(probabilities) != len(ground_truth):
        raise ValueError(
            'Probabilies and ground truth must have the same number of elements.')

    if [v for v in ground_truth if v not in [0., 1., True, False]]:
        raise ValueError(
            'Ground truth must contain binary labels {0,1} or {False, True}.')

    if isinstance(bins, int):
        num_bins = bins
    else:
        num_bins = bins.size - 1

    # Ensure probabilities are never 0, since the bins in np.digitize are open on
    # one side.
    probabilities = np.where(probabilities == 0, 1e-8, probabilities)
    counts, bin_edges = np.histogram(probabilities, bins=bins, range=[0., 1.])
    indices = np.digitize(probabilities, bin_edges, right=True)
    accuracies = np.array([np.mean(ground_truth[indices == i])
                         for i in range(1, num_bins + 1)])
    return bin_edges, accuracies, counts


def bin_centers_of_mass(probabilities, bin_edges):
    probabilities = np.where(probabilities == 0, 1e-8, probabilities)
    indices = np.digitize(probabilities, bin_edges, right=True)
    return np.array([np.mean(probabilities[indices == i])
                   for i in range(1, len(bin_edges))])


def expected_calibration_error(probabilities, ground_truth, bins=15):
    """Compute the expected calibration error of a set of preditions in [0, 1].
    Args:
    probabilities: A numpy vector of N probabilities assigned to each prediction
    ground_truth: A numpy vector of N ground truth labels in {0,1, True, False}
    bins: Number of equal width bins to bin predictions into in [0, 1], or
      an array representing bin edges.
    Returns:
    Float: the expected calibration error.
    """

    probabilities = probabilities.flatten()
    ground_truth = ground_truth.flatten()
    bin_edges, accuracies, counts = bin_predictions_and_accuracies(
      probabilities, ground_truth, bins)
    bin_centers = bin_centers_of_mass(probabilities, bin_edges)
    num_examples = np.sum(counts)

    ece = np.sum([(counts[i] / float(num_examples)) * np.sum(
      np.abs(bin_centers[i] - accuracies[i]))
            for i in range(bin_centers.size) if counts[i] > 0])

    return ece


