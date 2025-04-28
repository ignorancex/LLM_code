# create a function that takes the yolo output labels
# from individual files and converts them to coco output format

# import the necessary packages
import json
import os
import glob
import argparse
import numpy as np
import cv2

from tqdm import tqdm


# the yolo output laels are in the following format
# <object-class> <x> <y> <width> <height> <confidence>

# the coco output labels are in the following format
# [
#   {
#    "image_id": int,
#   "category_id": int,
#  "bbox": [x,y,width,height],
# "score": float,
# }
# ]

# the function takes the yolo output labels and converts them to coco output format
def yolo_to_coco(yolo_label, image_id, image_width, image_height):
    # print(yolo_label)
    coco_label = {}
    coco_label["image_id"] = image_id
    coco_label["category_id"] = int(yolo_label[0])

    # convert the x, y, width, height to coco format
    coco_label["bbox"] = [
        yolo_label[1] * image_width,
        yolo_label[2] * image_height,
        yolo_label[3] * image_width,
        yolo_label[4] * image_height,
    ]

    coco_label["score"] = yolo_label[5]
    return coco_label


# the function takes the yolo output labels from individual files and converts them to coco output format
def yolo_outputs_to_coco_format(yolo_output_dir, coco_output_dir, images_dir):
    # get the list of yolo output files
    yolo_output_files = glob.glob(os.path.join(yolo_output_dir, "*.txt"))

    # create a list to store the coco output labels
    coco_output_labels = []

    # loop over the yolo output files
    for yolo_output_file in tqdm(yolo_output_files):
        # get the image id from the file name
        image_id = os.path.basename(yolo_output_file).split(".")[0]

        # read the yolo output labels from the file
        yolo_output_labels = np.loadtxt(yolo_output_file)

        # loop over the yolo output labels
        # load image to get the image width and height
        image = cv2.imread(os.path.join(images_dir, image_id + ".jpg"))
        image_height, image_width, _ = image.shape
        if any(isinstance(i, np.ndarray) for i in yolo_output_labels):
            for yolo_output_label in yolo_output_labels:

                # convert the yolo output labels to coco output labels
                coco_output_label = yolo_to_coco(
                    yolo_output_label, image_id, image_width, image_height
                )
                coco_output_labels.append(coco_output_label)
        else:
            coco_output_label = yolo_to_coco(
                yolo_output_labels, image_id, image_width, image_height
            )
            coco_output_labels.append(coco_output_label)

    # write the coco output labels to a json file
    with open(os.path.join(coco_output_dir, "results.bbox.json"), "w") as f:
        json.dump(coco_output_labels, f)


# construct the argument parser and parse the arguments

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-y", "--yolo_output_dir", required=True, help="path to yolo output directory"
    )
    ap.add_argument(
        "-c", "--coco_output_dir", required=True, help="path to coco output directory"
    )
    ap.add_argument(
        "-i", "--images_dir", required=True, help="path to images directory"
    )
    args = vars(ap.parse_args())
    yolo_outputs_to_coco_format(
        args["yolo_output_dir"], args["coco_output_dir"], args["images_dir"]
    )
