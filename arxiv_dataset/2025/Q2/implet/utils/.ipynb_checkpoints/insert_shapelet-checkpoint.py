import os
import os.path
import numpy as np
import torch
from sklearn.metrics import mean_squared_error
from scipy.interpolate import make_interp_spline

import utils
from utils.model_utils import get_pred_with_acc
from utils.explaination_utils import explain, get_xai_ref
from utils.visualization import plot_multiple_images_with_attribution
from utils.utils import pickle_save_to_file, pickle_load_from_file

from aeon.classification.shapelet_based import ShapeletTransformClassifier


def insert_blender(instance, shape1, starting, blend_length):
    """
    blend the shapelet into instance
    :param instance:
    :param shape1:
    :param starting:
    :param blend_length:
    :return:
    """
    shapelet_length = len(shape1)
    length = len(instance)

    ending = starting + shapelet_length

    # Create blending weights
    blend_end = np.linspace(0, 1, blend_length)
    blend_start = np.linspace(1, 0, blend_length)
    # shape1
    # Insert sequence with blending
    result_sequence = instance.copy()
    result_sequence[starting:ending] = shape1

    # Apply blending at the start
    result_sequence[starting - blend_length:starting] \
        = instance[starting - blend_length] * blend_start + shape1[0] * blend_end

    # Apply blending at the end
    result_sequence[ending:ending + blend_length] = \
        instance[ending: ending + blend_length] * blend_end + shape1[-1] * blend_start

    return result_sequence


def insert_best(instance, shape1, quantile: float = None):
    """
    Insert the shapelet into the best location where casue the least steep changes
    if with quantile, then randomly choose a potition within the least percentiles else choose the least one

    :param instance:
    :param shape1:
    :param quantile:
    :return: inserted instance
    """

    shapelet_length = len(shape1)
    length = len(instance)
    shapelet_diff = shape1[0] - shape1[-1]
    instance_diff = instance[:-(shapelet_length)] - instance[shapelet_length:]
    # print(shapelet_diff, instance_diff)
    insert_diff = np.abs(instance_diff - shapelet_diff)
    if quantile is None:
        starting = np.argmin(insert_diff)

    else:
        threshold = np.quantile(insert_diff, quantile)
        indices = np.where(insert_diff <= threshold)[0]
        starting = np.random.choice(indices)
        # print(indices, insert_diff[indices], insert_diff[indices])
    best_diff = instance_diff[starting] - shapelet_diff
    # print(instance[starting] - shape1[0])
    instance[starting:starting + shapelet_length] = shape1 + (instance[starting] - shape1[0] - best_diff / 2)

    return instance, starting


def insert_shapelet(data, shape1, is_add=False, is_blending=False, blend_length=5, is_best_insert=False):
    shapelet_length = len(shape1)

    num = data.shape[0]
    length = data.shape[-1]
    c1 = np.zeros((num, 1, length))
    startings = []
    for i in range(num):
        a = data[i].flatten()
        if is_add:
            a[starting:starting + shapelet_length] = a[starting:starting + shapelet_length] + shape1
        elif is_blending:
            starting = np.random.randint(blend_length, length - shapelet_length - blend_length)
            a = insert_blender(a, shape1, starting, blend_length)
        elif is_best_insert:
            a, starting = insert_best(a, shape1, quantile=0.05)
        else:
            starting = np.random.randint(length - shapelet_length)
            a[starting:starting + shapelet_length] = shape1
        c1[i] = a.reshape(1, length)
        startings.append([starting])
    return c1, startings


def insert_fixed(data, shape1, starting, is_add=False, is_blending=False, blend_length=5, shift=True):
    shapelet_length = len(shape1)

    num = data.shape[0]
    length = data.shape[-1]
    c1 = np.zeros((num, 1, length))
    startings = [starting] * num
    for i in range(num):
        a = data[i].flatten()
        if is_add:
            a[starting:starting + shapelet_length] = a[starting:starting + shapelet_length] + shape1
        elif shift:
            diff_a = a[starting] - a[starting + shapelet_length]
            diff_shape = shape1[0] - shape1[-1]
            diff = diff_a - diff_shape
            a[starting:starting + shapelet_length] = shape1 + (a[starting] - shape1[0] - diff / 2)
        elif is_blending:
            a = insert_blender(a, shape1, starting, blend_length)
        else:
            a[starting:starting + shapelet_length] = shape1
        c1[i] = a.reshape(1, length)

    return c1, startings


def multiple_insert_shapelet(length, num_shapelet, shapelet_length, is_blending=False, blend_length=5):
    if is_blending:
        if length < (num_shapelet - 1) * (shapelet_length + 2 * blend_length):
            raise ValueError("Not enough space")
        startings = []
        while len(startings) < num_shapelet:
            _starting = np.random.randint(blend_length, length - shapelet_length - blend_length)

            if all(abs(_starting - val) >= (shapelet_length + 2 * blend_length) for val in startings):
                startings.append(_starting)
    else:
        if length < (num_shapelet - 1) * shapelet_length:
            raise ValueError("Not enough space")
        startings = []
        while len(startings) < num_shapelet:
            _starting = np.random.randint(length - shapelet_length)
            if all(abs(_starting - val) >= shapelet_length for val in startings):
                startings.append(_starting)
    return sorted(startings)


def data_given_env_multiple_shapelet(data, shape1=None, num_shapelet=1, is_add=True, is_blending=False, blend_length=5,
                                     is_best_insert=False):
    num = data.shape[0]
    length = data.shape[-1]
    c1 = np.zeros((num, 1, length))
    shapelet_length = len(shape1)

    if num_shapelet == 1 or is_best_insert:
        c1, startings = insert_shapelet(data, shape1, is_add=is_add, is_blending=is_blending, blend_length=5,
                                        is_best_insert=is_best_insert)

    else:
        c1 = np.zeros((num, 1, length))
        startings = []
        shapelet_length = len(shape1)
        for i in range(num):
            instance_startings = multiple_insert_shapelet(length, num_shapelet, shapelet_length, is_blending,
                                                          blend_length)
            a = data[i].flatten()

            for starting in instance_startings:

                if is_add:
                    a[starting:starting + shapelet_length] = a[starting:starting + shapelet_length] + shape1
                elif is_blending:

                    a = insert_blender(a, shape1, starting, blend_length)
                else:
                    a[starting:starting + shapelet_length] = shape1
            c1[i] = a.reshape(1, length)
            startings.append(instance_startings)
    return c1, startings


def interpolate_along_last_axis(data, inst_length=150):
    data = data.reshape(-1, data.shape[-1])
    input_shape = data.shape
    input_length = input_shape[-1]
    output_shape = input_shape[:-1] + (inst_length,)

    interpolated_data = np.zeros(output_shape)
    original_indices = np.linspace(0, 1, input_length)
    target_indices = np.linspace(0, 1, inst_length)

    for i in range(input_shape[0]):
        interpolated_data[i] = np.interp(target_indices, original_indices, data[i])
    interpolated_data = interpolated_data.reshape(-1, 1, inst_length)
    return interpolated_data


def closest_gt_mse(shapelet_attr, gt_attr):
    mses = []
    for i in range(len(gt_attr)):
        mses.append(mean_squared_error(shapelet_attr, gt_attr[i], squared=True))
    return np.min(mses)


def compute_attributions(model, data, xai_name, target_class, device='cuda', startings=None, length=None, repeats=None):
    """
    Compute attributions for a dataset using the specified XAI method.
    :param model:
    :param data:
    :param xai_name:
    :param target_class:
    :param device:
    :param startings: starting position of the shapelet
    :param length: the length of the shapelet
    :param repeats: first number of repeat instance
    :return:
    """

    xai_ref = get_xai_ref(xai_name)
    xai = xai_ref(model)
    model.to(device)

    if repeats is None:
        repeats = len(data)
    attributions = np.zeros((repeats, data.shape[-1]))
    if startings is None:
        attribution_shapelets = None
    elif isinstance(length, int):
        attribution_shapelets = np.zeros((repeats, length))
    else:
        attribution_shapelets = []

    for i in range(repeats):
        sample = torch.from_numpy(data[i].reshape(1, -1, data.shape[-1])).float().to(device)
        target_label = target_class if target_class is not None else torch.argmax(model(sample)).item()
        exp = explain(xai, xai_name, sample, target_label, sliding_window=(1, 5), baselines=None)
        exp = exp.detach().cpu().numpy().flatten()
        attributions[i] = exp
        if startings is not None:
            attribution_shapelets.append(exp[startings[i]:startings[i] + length])

    return attributions, np.array(attribution_shapelets, dtype=object)


def prepare_dataset(dataset, inst_length, is_z_norm=True, repeat_max=None):
    """
    Prepares the dataset by interpolating and normalizing.
    """
    train_ds, test_x, train_y, test_y, enc1 = utils.read_UCR_UEA(dataset=dataset, UCR_UEA_dataloader=None)

    if repeat_max is None:
        repeat_max = len(train_ds)

    train_ds = utils.interpolate_along_last_axis(train_ds[:repeat_max], inst_length=inst_length)
    if is_z_norm:
        train_ds = utils.z_normalization(train_ds)

    return train_ds


def insert_data_to_env(model, shapelet, selected_datasets, inst_length, num_shapelet=1, is_add=True,
                       xai_name='DeepLift', num_attribution_each=None,
                       target_class=0, repeat_max=100,
                       is_z_norm=True,
                       is_blending=False, blend_length=5,
                       is_best_insert=False,
                       img_path=None,
                       save_dir=None, device='cuda', is_plot=False, is_verbose=False):
    insert_shapelet_percentage = {}
    dataset_attr = {}

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if (xai_name is not None) and (not os.path.exists(os.path.join(save_dir, xai_name))):
        os.makedirs(os.path.join(save_dir, xai_name))

    length = len(shapelet)

    for ds in selected_datasets:
        if not os.path.exists(os.path.join(save_dir, f'{ds}.pkl')):
            train_ds = prepare_dataset(ds, inst_length, is_z_norm=True, repeat_max=repeat_max)
            train_ds, startings = data_given_env_multiple_shapelet(train_ds, shape1=shapelet, num_shapelet=num_shapelet,
                                                                   is_add=is_add, is_blending=is_blending,
                                                                   is_best_insert=is_best_insert,
                                                                   blend_length=blend_length)
            # if is_z_norm:
            #     train_ds = z_normalization(train_ds)
            preds, percentage = get_pred_with_acc(model, train_ds, target_class, device)

            insert_shapelet_percentage[ds] = percentage
            # model.cpu()

            if is_verbose:
                print(f'----------------------------')
                # print(f'{ds} mean: {train_ds.mean():.2f}'))
                print(f'percentage {percentage:.2f}')

            data_save = {
                'ds_name': ds,
                'shapelet': shapelet,
                'length': length,
                'train_ds': train_ds,
                'startings': startings,
                'num_sample': len(train_ds),
                'preds': preds,
                'target_class': target_class,
                'percentage': percentage
            }

            pickle_save_to_file(data=data_save, file_path=os.path.join(save_dir, f'{ds}.pkl'))
        else:
            data_save = pickle_load_from_file(file_path=os.path.join(save_dir, f'{ds}.pkl'))

            _, _, _, train_ds, startings, _, preds, _, percentage = data_save.values()
            insert_shapelet_percentage[ds] = percentage

        if xai_name is None:
            continue

        attributions, attribution_shapelets = compute_attributions(
            model, train_ds, xai_name, target_class, device, startings=startings, length=len(shapelet),
            repeats=num_attribution_each or len(train_ds)
        )

        if is_plot:
            plot_multiple_images_with_attribution(train_ds, preds, 4, (12, 6),
                                                  use_attribution=True,
                                                  attributions=attributions,
                                                  normalize_attribution=True,
                                                  save_path=f'{img_path}_{ds}')
        attribution_save = {
            'xai_name': xai_name,
            'attributions': attributions,
            'attribution_shapelet': attribution_shapelets,
        }
        dataset_attr[ds] = attributions
        pickle_save_to_file(data=attribution_save, file_path=os.path.join(save_dir, xai_name, f'{ds}.pkl'))
    return dataset_attr, insert_shapelet_percentage


def get_bg_pred(model, selected_datasets, inst_length, target_class, device='cuda', is_z_norm=True, repeat_max=100):
    bg_per = {}
    model.to(device)
    for ds in selected_datasets:
        train_ds = prepare_dataset(ds, inst_length, is_z_norm=True, repeat_max=repeat_max)

        preds, percentage = get_pred_with_acc(model, train_ds, target_class, device)
        bg_per[ds] = percentage

    return bg_per


def get_attr(model, data, startings, length, save_dir, xai_name='DeepLift', target_class=None, repeats=None,
             device='cuda'):
    if save_dir is not None:
        path_only = os.path.split(save_dir)[0]
        if path_only and not os.path.isdir(path_only):
            os.makedirs(path_only, exist_ok=True)
    attributions, attribution_shapelets = compute_attributions(
        model, data, xai_name, target_class, device, startings, length, repeats
    )

    if save_dir is not None:
        attr = {
            'target_class': target_class or 'pred',
            'xai_name': xai_name,
            'attributions': attributions,
            'attribution_shapelet': attribution_shapelets
        }
        pickle_save_to_file(data=attr, file_path=save_dir)
    return attributions, attribution_shapelets  # ,  attr,


def insert_multiple_shapelets(train_ds, shapelets, is_add, is_blending, blend_length, is_best_insert):
    """
    Inserts multiple shapelets into the training dataset.

    :param train_ds: Training dataset (numpy array)
    :param shapelets: List of shapelets to insert
    :param is_add: Whether to add the shapelet or replace data with it
    :param is_blending: Whether to blend shapelet into the data
    :param is_best_insert: Whether to insert at the best location
    :param blend_length: Blending length
    :return: Modified training dataset and list of starting positions
    """
    len_train = len(train_ds)
    num_shapelets = len(shapelets)
    # Determine shapelet assignment
    if num_shapelets == 1:
        shapelet_order = np.array(shapelets * len_train)

    elif len_train > num_shapelets:
        # Assign each shapelet approximately equal times
        repetitions = len_train // num_shapelets
        remaining = len_train % num_shapelets
        shapelet_order = (
            np.concatenate((shapelets * repetitions, np.random.choice(shapelets, remaining, replace=False)))
        )
    else:
        # Randomly assign shapelets to instances
        shapelet_order = np.random.choice(shapelets, size=len_train, replace=True)

    # Shuffle shapelet indices to ensure randomness
    np.random.shuffle(shapelet_order)

    # Insert shapelets

    startings_data = []
    shapelet_lengths = []
    for i, shapelet in enumerate(shapelet_order):
        shapelet_length = len(shapelet)
        # print(shapelet_length, shapelet)
        train_ds[i], starting_pos = data_given_env_multiple_shapelet(
            train_ds[i:i + 1],  # Process one instance at a time
            shape1=shapelet,
            num_shapelet=1,  # Always insert 1 shapelet to one instance
            is_add=is_add,
            is_blending=is_blending,
            is_best_insert=is_best_insert,
            blend_length=blend_length
        )
        startings_data.append(starting_pos)
        shapelet_lengths.append(shapelet_length)
    return train_ds, startings_data, shapelet_lengths


def get_pdata(
        shapelet, selected_datasets, inst_length, num_shapelet=1, is_add=False,
        repeat_max=None, is_z_norm=True, is_blending=False, blend_length=5, insert_fixed_starting=None,
        is_best_insert=False, save_dir='probe/GunPoint'
):
    """
    Inserts a shapelet into selected datasets and creates datasets with and without the shapelet.
    """
    pdata_with_shapelet = []
    pdata_without_shapelet = []
    startings = []

    for dataset in selected_datasets:
        # Prepare dataset
        train_ds = prepare_dataset(dataset, inst_length, is_z_norm, repeat_max)

        # Save without shapelet
        pdata_without_shapelet.append(train_ds.copy())

        # Insert shapelet
        if isinstance(shapelet, list):
            train_ds_with_shapelet, startings_data, shapelet_lengths = insert_multiple_shapelets(
                train_ds,
                shapelet,
                is_add=is_add,
                is_blending=is_blending,
                is_best_insert=is_best_insert,
                blend_length=blend_length,
            )
        elif insert_fixed_starting is not None:
            shift = True
            train_ds_with_shapelet, startings_data = insert_fixed(
                train_ds,
                shapelet,
                insert_fixed_starting,
                is_add=is_add,
                is_blending=is_blending,
                blend_length=blend_length,
                shift=shift,
            )
            shapelet_lengths = len(shapelet)
        else:
            train_ds_with_shapelet, startings_data = data_given_env_multiple_shapelet(
                train_ds,
                shape1=shapelet,
                num_shapelet=num_shapelet,
                is_add=is_add,
                is_blending=is_blending,
                is_best_insert=is_best_insert,
                blend_length=blend_length
            )
            shapelet_lengths = len(shapelet)
        # Normalize after insertion
        if is_z_norm:
            train_ds_with_shapelet = utils.z_normalization(train_ds_with_shapelet)

        pdata_with_shapelet.append(train_ds_with_shapelet)
        startings.extend(startings_data)

    # Concatenate all datasets
    pdata_with_shapelet = np.concatenate(pdata_with_shapelet, axis=0)
    pdata_without_shapelet = np.concatenate(pdata_without_shapelet, axis=0)

    # Save to file
    pdata = {
        'pdata_ws': pdata_with_shapelet,
        'pdata_wos': pdata_without_shapelet,
        'startings': startings,
        'shapelet': shapelet,
        'length': shapelet_lengths
    }

    # save_dir = os.path.join(save_dir, 'pdata.pkl')
    pickle_save_to_file(pdata, save_dir)
    return pdata


def _generate_smooth_signal(start, end, start_gradient, end_gradient, y_min=None, y_max=None,
                           num_points=100, random_seed=None):
    """
    Generate a random, smooth 1D signal.

    Parameters:
        start (float): Starting value of the signal.
        end (float): Ending value of the signal.
        start_gradient (float): Gradient of the signal at the start.
        end_gradient (float): Gradient of the signal at the end.
        y_min (float): Minimum value of the signal.
        y_max (float): Maximum value of the signal.
        num_points (int): Number of points in the generated signal.
        random_seed (int): Seed for random number generator (optional).

    Returns:
        np.ndarray: Smooth 1D signal.
    """
    if y_min is None:
        y_min = min(start, end)

    if y_max is None:
        y_max = max(start, end)

    y_mean = (y_min + y_max) / 2
    y_std = (y_max - y_mean) / 2

    if random_seed is not None:
        np.random.seed(random_seed)

    # Generate random control points
    num_control_points = max(num_points // 10, 4)  # Ensure at least 3 control points
    x_control = np.linspace(0, 1, num_control_points)
    y_control = y_std * np.random.rand(num_control_points) + y_mean

    # Adjust the first and last points to match the start and end values
    y_control[0] = start
    y_control[-1] = end

    # Create a cubic spline with boundary conditions for gradients
    x_fine = np.linspace(0, 1, num_points)
    spline = make_interp_spline(
        x_control, y_control, k=3, bc_type=[[(1, start_gradient)], [(1, end_gradient)]]
    )
    y_fine = spline(x_fine)

    return y_fine


def overwrite_shaplet_random(instance, start, length, blend_length=5, mode='smooth'):
    """
    Insert a random signal of LENGTH into INSTANCE at a given location.
    """
    end = start + length - 1

    val_start = instance[start]
    val_end = instance[end]

    grad_start_len = max(start - blend_length, 0) - start
    grad_start = (instance[start] - instance[max(start - blend_length, 0)]) / grad_start_len \
        if grad_start_len != 0 else 0

    grad_end_len = min(end + blend_length, len(instance) - 1) - end
    grad_end = (instance[min(end + blend_length, len(instance) - 1)] - instance[end]) / grad_end_len \
        if grad_end_len != 0 else 0

    y_min = np.min(instance[start:end + 1])
    y_max = np.max(instance[start:end + 1])

    match mode:
        case 'smooth':
            segment_rand = _generate_smooth_signal(val_start, val_end,
                                                   grad_start, grad_end,
                                                   y_min, y_max,
                                                   length)
        case 'rand_uniform':
            segment_rand = np.random.uniform(y_min, y_max, length)
        case 'rand_gauss':
            segment_rand = np.random.normal(np.mean(instance), np.std(instance), length)
        case 'zero':
            segment_rand = np.zeros(length)
        case 'mean':
            segment_rand = np.ones(length) * np.mean(instance)
        case _:
            raise ValueError()

    result = instance.copy()
    result[start:end + 1] = segment_rand
    # result = insert_blender(instance, segment_rand, start, blend_length)
    return result


def insert_random(instance, length, mode='smooth'):
    """
    Insert a random signal of LENGTH into INSTANCE at a random location.
    """
    start = np.random.randint(len(instance) - length + 1)
    return overwrite_shaplet_random(instance, start, length, mode=mode)
