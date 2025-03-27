import os
from PIL import Image
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt


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


def array_to_image(name_list, array: np.ndarray, target_dir: str):
    for i in range(array.shape[0]):
        # array[i][0] = array[i][0] * std[0] + mean[0]
        # array[i][1] = array[i][1] * std[1] + mean[1]
        # array[i][2] = array[i][2] * std[2] + mean[2]
        # array[i][0] = (array[i][0] - np.min(array[i][0])) / (np.max(array[i][0]) - np.min(array[i][0]))
        # array[i][1] = (array[i][1] - np.min(array[i][1])) / (np.max(array[i][1]) - np.min(array[i][1]))
        # array[i][2] = (array[i][2] - np.min(array[i][2])) / (np.max(array[i][2]) - np.min(array[i][2]))
        # image = (array[i] - np.min(array[i])) / (np.max(array[i]) - np.min(array[i]))
        image = (array[i] + 1) / 2.0
        image = image.transpose((1, 2, 0))
        image = image * 255
        # print(image)
        target_path = os.path.join(target_dir, name_list[i])
        if not os.path.exists(os.path.split(target_path)[0]):
            os.makedirs(os.path.split(target_path)[0])
        # plt.imsave(target_path, image)
        # print(target_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(target_path, image)


if __name__ == '__main__':
    pass
