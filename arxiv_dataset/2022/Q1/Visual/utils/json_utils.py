""" This file contains utilities for to store info in JSON file.
"""
import json
import os


def fill_annotation(gt_json, image_name, cont_id, ar_w, ar_h, avg_d, wt, wb, h, cap, m):
    """ Fills the annotation field in a disctionary.

    :param gt_json: dictionary of the eventual json file
    :param image_name: name of the image
    :param cont_id: container id
    :param ar_w: aspect ratio width
    :param ar_h: aspect ratio height
    :param avg_d: average distance
    :param wt: width at the top
    :param wb: width at the bottom
    :param h: height
    :param cap: capacity
    :param m: mass
    :return: None
    """
    gt_json["annotations"].append({
        "image_name": image_name,
        "container id": cont_id,
        "aspect ratio width": ar_w,
        "aspect ratio height": ar_h,
        "average distance": avg_d,
        "width top": wt,
        "width bottom": wb,
        "height": h,
        "capacity": cap,
        "mass": m
    })


def save_json(path_to_dest, gt_json):
    """ Saves json file.

    :param path_to_dest: path to the destination file (including the name of the file)
    :param gt_json: dictionary of the eventual json
    :return: None
    """
    with open(path_to_dest, 'w') as outfile:
        json.dump(gt_json, outfile)
    print("{} successfully created!!".format(os.path.basename(path_to_dest)))
