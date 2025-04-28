import numpy as np
from skimage.transform import resize
import cv2


def create_visualization(image, segmentation, labels=None, colors=None, remove_oos=False,contour=False,
                         eps=1e-6):
    '''
    Create a visualization of the ultrasound image with the segmentation overlayed
    :param image: ndarray
        ndarray with shape (width,height) containing the pixel values of the grayscale ultrasound image
    :param segmentation:  ndarray
        ndarray with shape (labels,width,height) containing the segmentation (one-hot encoded) or
        ndarray with shape (width,height) containing the segmentation (not one-hot encoded)
    :param labels: list of int, optional
        the labels of the segments to visualize
        If not specified, default value is [1, 2, 3, 4, 5, 6, 7]
    :param colors: ndarray, optional
        ndarray with shape (n,3) containing the colors to use for the segments
        If not specified, default value is np.array([(1,0,0),(0,0,1),(0,1,0),(1,1,0),(0,1,1),(1,0,1),(1,1,1),
                                      (0.55,0.27,0.07),(1,0.55,0)])
    :param remove_oos
        remove out of sector parts
        If True, parts of the segmentation outside the sector will be removed
    :param eps: float
        small value to avoid division by zero
    :param contour: bool
        If True, the contours of each region will be plotted instead of the filled regions.
        The same colors will be used for the contours as for the filled regions.
    :return: ndarray
        the visualization of the ultrasound image with the segmentation overlayed with values in range [0,255]
    '''
    image_rescaled = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255
    if labels is None:
        labels = [1, 2, 3, 4, 5, 6, 7]
    if colors is None:
        colors = np.array([(1, 0, 0), (0, 0, 1), (0, 1, 0), (1, 1, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1),
                           (0.55, 0.27, 0.07), (1, 0.55, 0)])

    if len(labels) > len(colors):
        raise ValueError('Not enough colors for plotting')

    if len(segmentation.shape) == 3:
        one_hot = True
        if max(labels) > len(segmentation):
            raise ValueError('Labels should be in range of the number of channels in the segmentation')
        resize_shape = (segmentation.shape[1], segmentation.shape[2])
    else:
        one_hot = False
        resize_shape = (segmentation.shape[0], segmentation.shape[1])
    # resize image to the same size as the segmentation
    image_resized = resize(image_rescaled, resize_shape, preserve_range=True)
    oos_mask = image_resized < eps

    result = np.zeros((image_resized.shape[0], image_resized.shape[1], 3))
    for i in range(3):
        result[:, :, i] = image_resized / 255
    for i, label in enumerate(labels):
        if one_hot:
            label_semgentation = segmentation[label, ...] > 0.5
        else:
            label_semgentation = segmentation == label

        if contour:
            label_semgentation = label_semgentation.astype(np.uint8)
            contours, _ = cv2.findContours(label_semgentation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(result, contours, -1, colors[i], 1)

        else:
            if colors[i, 0] != 0:
                result[label_semgentation, 0] = np.clip(colors[i, 0] * (0.35 +
                                                                        result[label_semgentation, 0]), 0.0, 1.0)
            if colors[i, 1] != 0:
                result[label_semgentation, 1] = np.clip(colors[i, 1] * (0.35 +
                                                                        result[label_semgentation, 1]), 0.0, 1.0)
            if colors[i, 2] != 0:
                result[label_semgentation, 2] = np.clip(colors[i, 2] * (0.35 +
                                                                        result[label_semgentation, 2]), 0.0, 1.0)
    if remove_oos:
        result[oos_mask] = 0 # set out of sector parts to black
    return (result * 255).astype(np.uint8)

