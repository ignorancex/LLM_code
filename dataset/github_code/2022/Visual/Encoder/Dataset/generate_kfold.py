""" This script generates folds and splits data into training and validation set. """
import argparse
import shutil
import os

from utils.annotation_parser import JsonParser
from utils.json_utils import fill_annotation, save_json


def save_set(src_dir, dest_dir, annotations, indices, set, fold):
    """ Saves training and validation set in fold directories and create new json annotations for the splits.

    :param src_dir: source directory (containing files to split)
    :param dest_dir: directory where to store the splits
    :param annotations: original annotations file from where the function takes the labels
    :param indices: indices of the containers id in the current fold
    :param set: indicates in which set e.g. training or validation data are destined
    :param fold: number of the current fold
    :return: None
    """
    keys = {"annotations"}
    gt_json = {key: [] for key in keys}
    for i in range(0, len(indices)):
        file_name = annotations.image_name[indices[i]]
        src = os.path.join(src_dir, file_name)
        dst = os.path.join(dest_dir, file_name)
        try:
            shutil.copy(src, dst)
            fill_annotation(gt_json,
                            image_name=annotations.image_name[indices[i]],
                            cont_id=annotations.container_id[indices[i]],
                            ar_w=annotations.ar_w[indices[i]],
                            ar_h=annotations.ar_h[indices[i]],
                            avg_d=annotations.avg_d[indices[i]],
                            wt=annotations.wt[indices[i]],
                            wb=annotations.wb[indices[i]],
                            h=annotations.height[indices[i]],
                            cap=annotations.capacity[indices[i]],
                            m=annotations.mass[indices[i]])
        except IOError as e:
            print("Unable to copy file. %s" % e)
    save_json(os.path.join(os.path.curdir, "annotations_{}_{}.json".format(set, fold)), gt_json)


def prepare_fold(annotations, train_ids):
    indices = []
    for id in train_ids:
        ind_list = [i for i in range(0, len(annotations.container_id)) if annotations.container_id[i] == id]
        indices += ind_list
    indices.sort()
    return indices


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(prog='generate_kfold',
                                     usage='%(prog)s --path_to_annotations <PATH_TO_ANN> --path_to_src_dir <PATH_TO_SRC_DIR>')
    parser.add_argument('--path_to_annotations', type=str,
                        default="/media/sealab-ws/Hard Disk/CORSMAL challenge/train_patches/dataset_pulito/annotations.json")
    parser.add_argument('--path_to_src_dir', type=str,
                        default="/media/sealab-ws/Hard Disk/CORSMAL challenge/train_patches/dataset_pulito/rgb")
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

    # Dictionary with ids of containers for the folds
    folds = {"train": [[1, 2, 4, 5, 7, 8],
                       [1, 3, 4, 6, 7, 9],
                       [2, 3, 5, 6, 8, 9]],
             "val": [[3, 6, 9],
                     [2, 5, 8],
                     [1, 4, 7]]}

    ann_dir = os.path.dirname(path_to_json)
    src_dir = args.path_to_src_dir
    for i in range(len(folds["train"])):
        # Create folder for fold
        dest_dir = os.path.join(ann_dir, "fold_{}".format(i))
        if not os.path.exists(dest_dir):
            os.mkdir(dest_dir)
        # Create training folder
        train_dest_dir = os.path.join(dest_dir, "train")
        if not os.path.exists(train_dest_dir):
            os.mkdir(train_dest_dir)
        # Create validation folder
        val_dest_dir = os.path.join(dest_dir, "val")
        if not os.path.exists(val_dest_dir):
            os.mkdir(val_dest_dir)
        # Retrieve the indices for the current fold (training)
        train_indices = prepare_fold(annotations, folds["train"][i])
        # Save the training set of the current fold
        save_set(src_dir, train_dest_dir, annotations, train_indices, "train", i)
        # Retrieve the indices for the current fold (validation)
        val_indices = prepare_fold(annotations, folds["val"][i])
        # Save the validation set of the current fold
        save_set(src_dir, val_dest_dir, annotations, val_indices, "val", i)
    print('Finished!')