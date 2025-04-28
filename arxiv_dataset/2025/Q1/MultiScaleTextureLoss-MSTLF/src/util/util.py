"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
import csv
import matplotlib.pyplot as plt
import json

def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def tensor2im2(input_image):
    return input_image[0][0].cpu().float().numpy()


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)


def save_plots(losses, path, title):
    # Calculate the number of rows and columns needed for the subplots
    num_losses = len(losses)
    num_rows = (num_losses // 2) + (num_losses % 2)
    num_cols = 2 if num_losses > 1 else 1

    # Create the subplots
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(10, 10))
    axs = axs.flatten()

    if not is_ordered_dict_nested(losses):

        # Plot each loss function in a subplot and add a legend with the loss names
        for i, (loss_name, loss_data) in enumerate(losses.items()):
            axs[i].plot(loss_data, label=loss_name)
            axs[i].set_title(loss_name)
            axs[i].legend()

        # Remove any unused subplots
        for i in range(num_losses, len(axs)):
            fig.delaxes(axs[i])

        # Add a title to the figure
        fig.suptitle(f'{title}')

        # Save the figure
        plt.savefig(path)
        plt.cla()
        plt.close(fig)
    else:
        for i, metric in enumerate(losses.keys()):
            axs[i].plot(losses[metric]['mean'], label=metric)
            axs[i].set_title(metric)
            axs[i].legend()

        # Remove any unused subplots
        for i in range(num_losses, len(axs)):
            fig.delaxes(axs[i])

        # Add a title to the figure
        fig.suptitle(f'{title}')

        # Save the figure
        plt.savefig(path)
        plt.cla()
        plt.close(fig)


def empty_dictionary(dictionary):
    for key in dictionary.keys():
        dictionary[key] = []


def save_ordered_dict_as_csv(data, filename):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        flatten_and_write(data, writer)


def flatten_and_write(data, writer, parent_key=''):
    for key, value in data.items():
        new_key = f"{parent_key}.{key}" if parent_key else key
        if isinstance(value, dict):
            flatten_and_write(value, writer, new_key)
        else:
            writer.writerow([new_key, value])


def is_ordered_dict_nested(data):
    for value in data.values():
        if isinstance(value, dict):
            return True
    return False


def save_list_to_csv(data, filename):
    # Open the CSV file in write mode
    with open(filename, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)

        # Write the header row
        writer.writerow(['id_d', 'id_theta'])

        # Write the data rows
        writer.writerows(data)


def load_saved_weights(model, saved_weights_path, device):
    saved_weights = torch.load(saved_weights_path, map_location=device)
    model.load_state_dict(saved_weights["state_dict"])
    return model


def frobenious_dist(t1):
        dot_prod = t1*t1
        return torch.sqrt(torch.sum(dot_prod, dim=1))


def save_json(data, path):
    file_json = json.dumps(data, indent=6)
    with open(f"{path}.json", 'w') as f:
            f.write(file_json)