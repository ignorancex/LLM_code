# Copyright (c) Facebook, Inc. and its affiliates.
import json
import logging
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.builtin_meta import CITYSCAPES_CATEGORIES, _get_builtin_metadata
from detectron2.data.datasets.coco import load_coco_json, register_coco_instances
from detectron2.data.datasets import load_sem_seg
from detectron2.utils.file_io import PathManager

"""
This file contains functions to register the ACDC panoptic dataset to the DatasetCatalog.
"""


logger = logging.getLogger(__name__)


_RAW_ACDC_INSTANCE_SPLITS = {
    "acdc_instance_train": (
        "acdc/rgb_anon/",
        "acdc/gt_detection/instancesonly_train_gt_detection.json",
    ),
    "acdc_instance_val": (
        "acdc/rgb_anon/",
        "acdc/gt_detection/instancesonly_val_gt_detection.json",
    ),
    "acdc_instance_test": (
        "acdc/rgb_anon/",
        "acdc/gt_detection/instancesonly_test_gt_detection.json",
    ),
    # "foggy_cityscapes_fine_panoptic_test": not supported yet
}


def _get_acdc_instances_meta():
    thing_ids = [k["id"] for k in CITYSCAPES_CATEGORIES if k["isthing"]==1]
    # assert len(thing_ids) == 100, len(thing_ids)
    # Mapping from the incontiguous ADE category id to an id in [0, 99]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in CITYSCAPES_CATEGORIES if k["isthing"]==1]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
    }
    return ret


def register_all_acdc_instance(root):
    for key, (image_root, json_file) in _RAW_ACDC_INSTANCE_SPLITS.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_coco_instances(
            key,
            _get_acdc_instances_meta(),
            os.path.join(root, json_file),
            os.path.join(root, image_root),
        )


_RAW_ACDC_SEM_SEG_SPLITS = {
    "acdc_sem_seg_train": (
        "train",
        "acdc/rgb_anon/",
        "acdc/gt/",
    ),
    "acdc_sem_seg_val": (
        "val",
        "acdc/rgb_anon/",
        "acdc/gt/",
    ),
    "acdc_sem_seg_test": (
        "test",
        "acdc/rgb_anon/",
        "acdc/gt/",
    ),
}


def _get_acdc_files(image_dir, gt_dir, split):
    files = []
    # scan through the directory
    conditions = PathManager.ls(image_dir)
    logger.info(f"{len(conditions)} conditions found in '{image_dir}'.")
    for cond in conditions:
        cond_img_dir = os.path.join(image_dir, cond, split)
        cond_gt_dir = os.path.join(gt_dir, cond, split)
        cities = PathManager.ls(cond_img_dir)
        for city in cities:
            city_cond_img_dir = os.path.join(cond_img_dir, city)
            city_cond_gt_dir = os.path.join(cond_gt_dir, city)
            for basename in PathManager.ls(city_cond_img_dir):
                image_file = os.path.join(city_cond_img_dir, basename)
                suffix = "_rgb_anon.png"
                assert basename.endswith(suffix), basename
                basename = basename[: -len(suffix)]
                label_file = os.path.join(city_cond_gt_dir, basename + "_gt_labelIds.png")

                files.append((image_file, label_file))
    logger.info(f"{len(files)} images found in split '{split}'.")
    assert len(files), "No images found in {}".format(image_dir)
    for f in files[0]:
        assert PathManager.isfile(f), f
    return files


def load_acdc_semantic(image_dir, gt_dir, split):
    """
    Args:
        image_dir (str): path to the raw dataset. e.g., "~/cityscapes/leftImg8bit/train".
        gt_dir (str): path to the raw annotations. e.g., "~/cityscapes/gtFine/train".

    Returns:
        list[dict]: a list of dict, each has "file_name" and
            "sem_seg_file_name".
    """
    ret = []
    # gt_dir is small and contain many small files. make sense to fetch to local first
    gt_dir = PathManager.get_local_path(gt_dir)
    for image_file, label_file in _get_acdc_files(image_dir, gt_dir, split):
        label_file = label_file.replace("labelIds", "labelTrainIds")

        # with PathManager.open(json_file, "r") as f:
        #     jsonobj = json.load(f)
        ret.append(
            {
                "file_name": image_file,
                "sem_seg_file_name": label_file,
                # "height": jsonobj["imgHeight"],
                # "width": jsonobj["imgWidth"],
            }
        )
    assert len(ret), f"No images found in {image_dir}!"
    assert PathManager.isfile(
        ret[0]["sem_seg_file_name"]
    ), "Please generate labelTrainIds.png with cityscapesscripts/preparation/createTrainIdLabelImgs.py"  # noqa
    return ret


def register_all_acdc_sem_seg(root):
    meta = _get_builtin_metadata("cityscapes")
    for key, (split, image_dir, gt_dir) in _RAW_ACDC_SEM_SEG_SPLITS.items():
        image_dir = os.path.join(root, image_dir)
        gt_dir = os.path.join(root, gt_dir)
        DatasetCatalog.register(
            key, lambda x=image_dir, y=gt_dir, z=split: load_acdc_semantic(x, y, z)
        )
        MetadataCatalog.get(key).set(
                image_dir=image_dir,
                gt_dir=gt_dir,
                # evaluator_type="cityscapes_sem_seg",
                evaluator_type="sem_seg",
                ignore_label=255,
                **meta,
            )
    

# def register_all_coco_stuff_10k(root):
#     root = os.path.join(root, "coco", "coco_stuff_10k")
#     meta = _get_coco_stuff_meta()
#     for name, image_dirname, sem_seg_dirname in [
#         ("train", "images_detectron2/train", "annotations_detectron2/train"),
#         ("test", "images_detectron2/test", "annotations_detectron2/test"),
#     ]:
#         image_dir = os.path.join(root, image_dirname)
#         gt_dir = os.path.join(root, sem_seg_dirname)
#         name = f"coco_2017_{name}_stuff_10k_sem_seg"
#         DatasetCatalog.register(
#             name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="_gt_labelTrainIds.png", image_ext="_rgb_anon.png")
#         )
#         MetadataCatalog.get(name).set(
#             image_root=image_dir,
#             sem_seg_root=gt_dir,
#             evaluator_type="sem_seg",
#             ignore_label=255,
#             **meta,
#         )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_acdc_instance(_root)
register_all_acdc_sem_seg(_root)
