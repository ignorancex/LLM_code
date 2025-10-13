""" Collection of utility functions.
Mask R-CNN functions are taken/modified from https://debuggercafe.com/instance-segmentation-with-pytorch-and-mask-r-cnn/ """

import os
import numpy as np
import torch
import cv2

from utils.coco_names import COCO_INSTANCE_CATEGORY_NAMES as coco_names
# from coco_names import COCO_INSTANCE_CATEGORY_NAMES as coco_names
# create a different color for each class
COLORS = np.random.uniform(0, 255, size=(len(coco_names), 3))


def get_filenames(dataset_dir, extension):
    """ Returns a list of filenames.

    Args:
      dataset_dir: A directory containing a set of files.
      extension: extension of the file to return.

    Returns:
      A list of file paths, relative to `dataset_dir and depending on `extension.
    """

    filenames = []
    for filename in os.listdir(dataset_dir):
        path = os.path.join(dataset_dir, filename)
        if path.endswith(extension):
            filenames.append(path)

    return filenames


def get_outputs(image, model, threshold):
    """ Performs inference phase and thresholds the outputs.

    :param image: image as ndarray
    :param model: model which performs the inference
    :param threshold: detection threshold
    :return: lists of masks, boxes, classes and final scores
    """
    with torch.no_grad():
        # forward pass of the image through the modle
        outputs = model(image)

    # get all the scores
    scores = list(outputs[0]['scores'].detach().cpu().numpy())
    # index of those scores which are above a certain threshold
    thresholded_preds_indices = [scores.index(i) for i in scores if i > threshold]
    thresholded_preds_count = len(thresholded_preds_indices)
    # get the masks
    masks = (outputs[0]['masks'] > 0.5).squeeze().detach().cpu().numpy()
    # discard masks for objects which are below threshold
    masks = masks[:thresholded_preds_count]
    # get the bounding boxes, in (x1, y1), (x2, y2) format
    boxes = [[(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))] for i in outputs[0]['boxes'].detach().cpu()]
    # discard bounding boxes below threshold value
    boxes = boxes[:thresholded_preds_count]
    # get the classes labels
    classes_temp = [i for i in outputs[0]['labels'].cpu().numpy()]
    classes = [classes_temp[i] for i in range(0, len(classes_temp)) if i in thresholded_preds_indices]
    # get scores
    final_scores = [i for i in scores if i > threshold]
    return masks, boxes, classes, final_scores


def draw_segmentation_map(image, masks, boxes, classes, scores):
    """ Draws segmentation map.

    :param image: image as ndarray
    :param masks: list of masks
    :param boxes: list of boxes
    :param classes: list of classes
    :param scores: list of scores
    :return: the image with boxes and mask overprinted
    """
    alpha = 1
    beta = 0.8  # transparency for the segmentation map
    gamma = 0  # scalar added to each sum
    for i in range(len(masks)):
        red_map = np.zeros_like(masks[i]).astype(np.uint8)
        green_map = np.zeros_like(masks[i]).astype(np.uint8)
        blue_map = np.zeros_like(masks[i]).astype(np.uint8)
        # apply a randon color mask to each object
        color = COLORS[classes[i]]
        red_map[masks[i] == 1], green_map[masks[i] == 1], blue_map[masks[i] == 1] = color
        # combine all the masks into a single image
        segmentation_map = np.stack([red_map, green_map, blue_map], axis=2)
        # apply mask on the image
        cv2.addWeighted(image, alpha, segmentation_map, beta, gamma, image)
        # draw the bounding boxes around the objects
        cv2.rectangle(image, boxes[i][0], boxes[i][1], color=color,
                      thickness=2)
        # put the label text above the objects
        cv2.putText(image, coco_names[classes[i]] + ": {:.2}".format(scores[i]), (boxes[i][0][0], boxes[i][0][1] - 10),
                    cv2.FONT_HERSHEY_PLAIN, 1, color,
                    thickness=2)

    return image


def filter_results(masks, boxes, cls, scores):
    """ Filters outputs of Mask R-CNN with COCO weights. Considers only 'cup', 'book', 'wine glass', 'bottle'.

    :param masks: list of output masks
    :param boxes: list of output boxes
    :param cls: list of output classes
    :param scores: list of output scores
    :return: filtered masks, boxes, classes and scores
    """
    indx_to_remove = [i for i in range(0, len(cls))]

    # select classes to keep
    indx_to_keep0 = [index for index in range(0, len(cls)) if cls[index] == coco_names.index('cup')]
    indx_to_keep1 = [index for index in range(0, len(cls)) if cls[index] == coco_names.index('book')]
    indx_to_keep2 = [index for index in range(0, len(cls)) if cls[index] == coco_names.index('wine glass')]
    indx_to_keep3 = [index for index in range(0, len(cls)) if cls[index] == coco_names.index('bottle')]
    indx_to_keep = (indx_to_keep0 + indx_to_keep1 + indx_to_keep2 + indx_to_keep3)

    # sort indices to keep
    indx_to_keep.sort()

    # select indices to remove
    [indx_to_remove.remove(i) for i in indx_to_keep]

    # delete corresponding arrays
    cls = np.delete(np.asarray(cls), indx_to_remove)
    scores = np.delete(np.asarray(scores), indx_to_remove)
    masks = np.delete(np.asarray(masks), indx_to_remove, axis=0)
    boxes = np.delete(np.asarray(boxes), indx_to_remove, axis=0)
    return masks, boxes, cls, scores
