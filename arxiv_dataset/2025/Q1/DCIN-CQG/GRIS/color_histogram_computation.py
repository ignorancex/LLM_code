import json
import os
from tqdm import tqdm
from PIL import Image, ImageOps
import numpy as np
import cv2
from scipy.spatial.distance import cdist


def load_images_and_paths(json_path, image_dir):
    # load the JSON file
    with open(json_path) as f:
        data = json.load(f)

    images_data = data['images']
    images = []
    image_paths = []

    for img_data in tqdm(images_data, desc="LOADING IMAGES"):
        # load and process image
        img_path = os.path.join(image_dir, img_data['file_name'])
        image = Image.open(img_path).convert("RGB")
        image = ImageOps.exif_transpose(image)
        image = np.array(image)

        # append to lists
        images.append(image)
        image_paths.append(img_path)

    return images, image_paths


def generate_color_histograms(images, bins=(8, 8, 8)):
    histograms = []

    for image in tqdm(images, desc="GENERATING HISTOGRAMS"):
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        hist = cv2.calcHist([image_bgr], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        histograms.append(hist)

    return histograms


def select_most_representative_image(features):
    print("SELECTING IMAGE")
    
    # compute the pairwise distance matrix (using Euclidean distance)
    distance_matrix = cdist(features, features, metric='euclidean')

    # calculate the average distance to all other images for each image
    avg_distances = distance_matrix.mean(axis=1)

    # find the index of the image with the minimum average distance
    most_representative_idx = np.argmin(avg_distances)

    # return the path of the most representative image
    return most_representative_idx