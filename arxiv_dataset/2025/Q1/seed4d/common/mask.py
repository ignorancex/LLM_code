# Copyright (C) 2025 co-pace GmbH (subsidiary of Continental AG).
# Licensed under the BSD-3-Clause License.
# @author: Marius Kästingschäfer and Théo Gieruc
# ==============================================================================

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import json
import os
import scipy


def isolate_mask(image_sem):
    
    '''
    Obtain a mask of the desired objects from the semantic segmentation image.
    CARLA segmentation colors: https://carla.readthedocs.io/en/latest/ref_sensors/
    using cityScape palette: https://github.com/carla-simulator/carla/blob/master/LibCarla/source/carla/image/CityScapesPalette.h
    RGB vehicle:  0, 0, 142 --> BGR: 142, 0, 0
    
    Parameters:
        image_sem (np.array): The semantic segmentation image.
    
    Returns:
        image_sem (np.array): The semantic segmentation image with the desired objects isolated.
    '''

    # 

    car = 14
    bicycle = 19
    rider = 13
    motorcycle = 18
    truck = 15
    bus = 16

    # Create a mask to isolate the desired color range
    car_mask = cv2.inRange(
        image_sem, np.array([car, car, car]), np.array([car, car, car])
    )
    rider_mask = cv2.inRange(
        image_sem, np.array([rider, rider, rider]), np.array([rider, rider, rider])
    )
    bicycle_mask1 = cv2.inRange(
        image_sem,
        np.array([bicycle, bicycle, bicycle]),
        np.array([bicycle, bicycle, bicycle]),
    )
    bicycle_mask2 = cv2.inRange(
        image_sem, np.array([118, 10, 31]), np.array([120, 12, 33])
    )  # bicylce
    motorcycle_mask = cv2.inRange(
        image_sem,
        np.array([motorcycle, motorcycle, motorcycle]),
        np.array([motorcycle, motorcycle, motorcycle]),
    )
    truck_mask = cv2.inRange(
        image_sem, np.array([truck, truck, truck]), np.array([truck, truck, truck])
    )
    bus_mask = cv2.inRange(
        image_sem, np.array([bus, bus, bus]), np.array([bus, bus, bus])
    )

    # Set the pixels outside the desired color range to white
    mask = (
        car_mask
        + rider_mask
        + bicycle_mask1
        + bicycle_mask2
        + motorcycle_mask
        + truck_mask
        + bus_mask
    )

    image_sem[mask != 255] = (255, 255, 255)

    return image_sem


def dilate_wb_mask(bw_mask, iterations=3):

    # dilate image
    res = scipy.ndimage.binary_dilation(bw_mask, iterations=iterations)
    # revert format into something PIL can work with +
    # that give returns a binary img
    res = Image.fromarray(res)
    res = res.convert("RGB")
    res = res.convert("L")
    wb_mask = res.point(lambda x: 255 if x < 128 else 0, "1")

    return wb_mask


def turn_mask_in_bw(mask, iterations):

    # make the mask black-and-white
    mask = Image.fromarray(mask)
    mask = mask.convert("RGB")
    gray_mask = mask.convert("L")
    bw_mask = gray_mask.point(lambda x: 255 if x < 128 else 0, "1")

    # suitable for masking
    mask = np.array(bw_mask)
    mask = mask.astype(np.uint8)

    # increase contour thickness / dilate mask
    wb_mask = dilate_wb_mask(bw_mask, iterations)

    return wb_mask, mask


def apply_mask(img, mask):

    result = cv2.bitwise_and(img, img, mask=mask)
    # change background from black to white
    result[np.where((result == [0, 0, 0]).all(axis=2))] = [255, 255, 255]

    return result


def load_json(path):
    with open(path, "r") as file:
        data = json.load(file)

    return data


def obtain_wb_mask_and_object_only(path, iterations):
    '''
    Obtain white-black mask and object-only image from semantic segmentation image.
    
    Parameters:
        path (str): The path to the dataset.
        iterations (int): The number of iterations to binary_dilation iterations for the white-black mask.    
    '''

    json_path = path + "/transforms/transforms.json"
    data = load_json(json_path)

    for idx in range(len(data["frames"])):
        # read rgb and semantic img paths from disk
        image_rgb_path = data["frames"][idx]["file_path"][2:]
        image_sem_path = data["frames"][idx]["semantic_segmentation_file_path"][2:]

        # append sys path
        image_rgb_path = path + image_rgb_path
        image_sem_path = path + image_sem_path

        # read images
        image_rgb = cv2.imread(image_rgb_path)
        image_sem = cv2.imread(image_sem_path)

        # 1. obtain semantic seg. of interest
        # 2. make into bw and wb mask (wb_mask used for learning background)
        # 3. obtain final image with bw mask
        colored_mask = isolate_mask(image_sem)
        wb_mask, bw_mask = turn_mask_in_bw(colored_mask, iterations)
        # final_image = apply_mask(image_rgb, bw_mask)

        # write white-black mask to disk
        wb_mask.save(
            path + "/sensors/" + str(idx) + "_mask.png",
        )
        # new method
        mask = cv2.imread(
            path + "/sensors/" + str(idx) + "_mask.png", cv2.IMREAD_GRAYSCALE
        )
        wb_mask = cv2.bitwise_not(mask)

        # Ensure the mask is binary (either 0 or 255)
        ret, wb_mask = cv2.threshold(wb_mask, 127, 255, cv2.THRESH_BINARY)
        wb_mask_np = np.array(wb_mask)
        # Expand the mask to have 3 channels for RGB
        mask_rgb = cv2.merge([wb_mask_np, wb_mask_np, wb_mask_np])

        # Apply the mask to each channel of the RGB image individually
        result = cv2.bitwise_and(image_rgb, mask_rgb)

        overlay = np.ones_like(image_rgb) * 255
        result += cv2.bitwise_and(overlay, overlay, mask=cv2.bitwise_not(wb_mask_np))

        # write object only to disk
        cv2.imwrite(path + "/sensors/" + str(idx) + "_obj.png", result)


def write_transform_jsons(path):
    '''
    Creates a object, background and normalized background transform json files per timestep.
    Object transform - contains the path to the object-only image.
    Background transform - contains the path to the full images and the vehicle masks.
    Normalized background transform - Same as background transform but with the normalized poses. 
    
    
    Parameters:
        path (str): The path to the dataset.
    '''

    # write object transform (obj file in image path)
    json_path = path + "/transforms/transforms.json"
    data = load_json(json_path)

    for idx in range(len(data["frames"])):
        data["frames"][idx]["file_path"] = "../sensors/" + str(idx) + "_obj.png"

    with open(path + "/transforms/transforms_obj.json", "w") as file:
        json.dump(data, file)

    # write background transform
    json_path = path + "/transforms/transforms.json"
    data = load_json(json_path)

    for idx in range(len(data["frames"])):
        items_list = list(data["frames"][idx].items())
        items_list.insert(-1, ("mask_path", "../sensors/" + str(idx) + "_mask.png"))
        data["frames"][idx] = dict(items_list)

    with open(path + "/transforms/transforms_background.json", "w") as file:
        json.dump(data, file)

    # write normalized background transform
    #json_path = path + "/transforms/transforms_normalized.json"
    #data = load_json(json_path)
#
    #for idx in range(len(data["frames"])):
    #    items_list = list(data["frames"][idx].items())
    #    items_list.insert(-1, ("mask_path", "../sensors/" + str(idx) + "_mask.png"))
    #    data["frames"][idx] = dict(items_list)
#
    #with open(path + "/transforms/transforms_normalized_background.json", "w") as file:
    #    json.dump(data, file)
