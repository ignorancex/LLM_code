""" This script splits data into training and validation set based on a prcentage value """
import argparse
import shutil
import os
from random import random
import random
import glob
from utils.annotation_parser import JsonParser
from utils.csv_utils import CsvResults
from utils.json_utils import fill_annotation, save_json
import pandas as pd
random.seed(7)


def save_set(rgb_dir, depth_dir, rgb_dest_dir, dpt_dest_dir, annotations, annotationfile):
    """
    """
    gt = pd.read_csv(annotationfile, sep=',')
    col_id = gt['id'].tolist()
    col_container_id = gt['container id'].tolist()
    col_scenario = gt['scenario'].tolist()
    col_background = gt['background'].tolist()
    col_illumination = gt['illumination'].tolist()
    col_wt = gt['width at the top'].tolist()
    col_wb = gt['width at the bottom'].tolist()
    col_h = gt['height'].tolist()
    col_d = gt['depth'].tolist()
    col_cap = gt['container capacity'].tolist()
    col_m = gt['container mass'].tolist()
    col_ft = gt['filling type'].tolist()
    col_fl = gt['filling level'].tolist()
    col_fd = gt['filling density'].tolist()
    col_fm = gt['filling mass'].tolist()
    col_obj_m = gt['object mass'].tolist()
    col_sf = gt['handover starting frame'].tolist()
    col_st = gt['handover start timestamp'].tolist()
    col_hh = gt['handover hand'].tolist()
    csv_res = CsvResults(
        ['id', 'container id', 'scenario', 'background', 'illumination', 'width at the top', 'width at the bottom',
         'height', 'depth', 'container capacity', 'container mass', 'filling type', 'filling level', 'filling density',
         'filling mass', 'object mass', 'handover starting frame', 'handover start timestamp', 'handover hand'])
    names_list = annotations.image_name
    names_list.sort()
    for j in range(0, len(names_list)):
        image_name = names_list[j].split(".png")[0]
        video_name = image_name.split("_")[0] + ".mp4"
        src_rgb = os.path.join(rgb_dir, video_name)
        dst_rgb = os.path.join(rgb_dest_dir, video_name)
        if int(image_name.split("_")[1]) == 0:
            try:
                shutil.copy(src_rgb, dst_rgb)
                src_dpt = os.path.join(depth_dir, video_name.split(".mp4")[0])
                dst_dir = os.path.join(dpt_dest_dir, video_name.split(".mp4")[0])
                shutil.copytree(src_dpt, dst_dir)
                i = int(video_name.split(".mp4")[0])
                csv_res.fill_entry('id', i)
                csv_res.fill_entry('container id', int(col_container_id[i]))
                csv_res.fill_entry('scenario', int(col_scenario[i]))
                csv_res.fill_entry('background', int(col_background[i]))
                csv_res.fill_entry('illumination', int(col_illumination[i]))
                csv_res.fill_entry('width at the top', round((col_wt[i]),1))
                csv_res.fill_entry('width at the bottom', round((col_wb[i]), 1))
                csv_res.fill_entry('height', round((col_h[i]), 1))
                csv_res.fill_entry('depth', round((col_d[i]), 1))
                csv_res.fill_entry('container capacity', round((col_cap[i]), 3))
                csv_res.fill_entry('container mass', round((col_m[i]), 1))
                csv_res.fill_entry('filling type', int((col_ft[i])))
                csv_res.fill_entry('filling level', int((col_fl[i])))
                csv_res.fill_entry('filling density', round((col_fd[i]), 2))
                csv_res.fill_entry('filling mass', round((col_fm[i]), 1))
                csv_res.fill_entry('object mass', round((col_obj_m[i]), 1))
                csv_res.fill_entry('handover starting frame', -1)
                csv_res.fill_entry('handover start timestamp', -1)
                csv_res.fill_entry('handover hand', -1)

            except IOError as e:
                print("Unable to copy file. %s" % e)
    csv_res.save_csv("Val_annotations_fold2.csv")


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(prog='split_training_validation',
                                     usage='%(prog)s --path_to_annotations <PATH_TO_ANN> --path_to_src_dir <PATH_TO_SRC_DIR>')
    parser.add_argument('--path_to_annotations', type=str,
                        default="/media/sealab-ws/Hard Disk/CORSMAL challenge/train_patches/annotations_val_2.json")

    parser.add_argument('--path_to_src_dir', type=str,
                        default="/media/sealab-ws/Hard Disk/CORSMAL challenge/train/view3/rgb")
    parser.add_argument('--path_to_dpt_dir', type=str,
                        default="/media/sealab-ws/Hard Disk/CORSMAL challenge/train/view3/depth")
    args = parser.parse_args()

    # Assertions
    assert args.path_to_annotations is not None, "Please, provide path to annotations json file"
    assert os.path.exists(args.path_to_annotations), "path_to_annotations does not exist"
    assert args.path_to_src_dir is not None, "Please, provide path to source rgb images directory"
    assert os.path.exists(args.path_to_src_dir), "path_to_src_dir does not exist"

    # Load annotations
    path_to_json = args.path_to_annotations
    annotations = JsonParser()
    annotations.load_json(path_to_json)

    ann_dir = os.path.dirname(path_to_json)
    rgb_dir = args.path_to_src_dir
    depth_dir = args.path_to_dpt_dir
    annotationfile = "/home/sealab-ws/Desktop/Visual/demo/ccm_train_annotation.csv"
    # Create folder for fold
    dest_dir = os.path.join(os.path.dirname(rgb_dir), "val_fold_2")
    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)

    # Create validation folder
    rgb_dest_dir = os.path.join(dest_dir, "rgb")
    if not os.path.exists(rgb_dest_dir):
        os.mkdir(rgb_dest_dir)
    # Create validation folder
    dpt_dest_dir = os.path.join(dest_dir, "depth")
    if not os.path.exists(dpt_dest_dir):
        os.mkdir(dpt_dest_dir)

    # Save the validation set of the current folder
    save_set(rgb_dir, depth_dir, rgb_dest_dir, dpt_dest_dir, annotations, annotationfile)
    print('Finished!')
