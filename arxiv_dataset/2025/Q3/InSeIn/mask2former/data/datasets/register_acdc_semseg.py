import os

from detectron2.data import DatasetCatalog, MetadataCatalog
#from detectron2.data.datasets import load_sem_seg

CITYSCAPES_CATEGORIES = [
    {"color": (128, 64, 128), "isthing": 0, "id": 7, "trainId": 0, "name": "road"},
    {"color": (244, 35, 232), "isthing": 0, "id": 8, "trainId": 1, "name": "sidewalk"},
    {"color": (70, 70, 70), "isthing": 0, "id": 11, "trainId": 2, "name": "building"},
    {"color": (102, 102, 156), "isthing": 0, "id": 12, "trainId": 3, "name": "wall"},
    {"color": (190, 153, 153), "isthing": 0, "id": 13, "trainId": 4, "name": "fence"},
    {"color": (153, 153, 153), "isthing": 0, "id": 17, "trainId": 5, "name": "pole"},
    {"color": (250, 170, 30), "isthing": 0, "id": 19, "trainId": 6, "name": "traffic light"},
    {"color": (220, 220, 0), "isthing": 0, "id": 20, "trainId": 7, "name": "traffic sign"},
    {"color": (107, 142, 35), "isthing": 0, "id": 21, "trainId": 8, "name": "vegetation"},
    {"color": (152, 251, 152), "isthing": 0, "id": 22, "trainId": 9, "name": "terrain"},
    {"color": (70, 130, 180), "isthing": 0, "id": 23, "trainId": 10, "name": "sky"},
    {"color": (220, 20, 60), "isthing": 1, "id": 24, "trainId": 11, "name": "person"},
    {"color": (255, 0, 0), "isthing": 1, "id": 25, "trainId": 12, "name": "rider"},
    {"color": (0, 0, 142), "isthing": 1, "id": 26, "trainId": 13, "name": "car"},
    {"color": (0, 0, 70), "isthing": 1, "id": 27, "trainId": 14, "name": "truck"},
    {"color": (0, 60, 100), "isthing": 1, "id": 28, "trainId": 15, "name": "bus"},
    {"color": (0, 80, 100), "isthing": 1, "id": 31, "trainId": 16, "name": "train"},
    {"color": (0, 0, 230), "isthing": 1, "id": 32, "trainId": 17, "name": "motorcycle"},
    {"color": (119, 11, 32), "isthing": 1, "id": 33, "trainId": 18, "name": "bicycle"},
]

def load_sem_seg(gt_root, image_root):
    """
    Load semantic segmentation datasets. All files under "gt_root" with "gt_ext" extension are
    treated as ground truth annotations and all files under "image_root" with "image_ext" extension
    as input images. Ground truth and input images are matched using file paths relative to
    "gt_root" and "image_root" respectively without taking into account file extensions.
    This works for COCO as well as some other datasets.

    Args:
        gt_root (str): full path to ground truth semantic segmentation files. Semantic segmentation
            annotations are stored as images with integer values in pixels that represent
            corresponding semantic labels.
        image_root (str): the directory where the input images are.
        gt_ext (str): file extension for ground truth annotations.
        image_ext (str): file extension for input images.

    Returns:
        list[dict]:
            a list of dicts in detectron2 standard format without instance-level
            annotation.

    Notes:
        1. This function does not read the image and ground truth files.
           The results do not have the "image" and "sem_seg" fields.
    """

    # We match input images with ground truth based on their relative filepaths (without file
    # extensions) starting from 'image_root' and 'gt_root' respectively.

    split = ('train','val')
    input_train_path = image_root
    target_train_path = gt_root
    conditions = ['fog','night','rain','snow']
    file_paths = []
    for condition in conditions:
        for folders in os.listdir(input_train_path+condition+'/'+split[0]):
            for files in os.listdir(input_train_path+condition+'/'+split[0]+'/'+folders):
                image_path = input_train_path+condition+'/'+split[0]+'/'+folders+'/'+files
                target_path = target_train_path+condition+'/'+split[0]+'/'+folders+'/'+files.replace('rgb_anon.png','gt_labelTrainIds.png')
                file_paths.append({"file_name":image_path,"sem_seg_file_name": target_path})
        for folders in os.listdir(input_train_path+condition+'/'+split[1]):
            for files in os.listdir(input_train_path+condition+'/'+split[1]+'/'+folders):
                image_path = input_train_path+condition+'/'+split[1]+'/'+folders+'/'+files
                target_path = target_train_path+condition+'/'+split[1]+'/'+folders+'/'+files.replace('rgb_anon.png','gt_labelTrainIds.png')
                file_paths.append({"file_name":image_path,"sem_seg_file_name":target_path})

    # def file2id(folder_path, file_path):
    #     # extract relative path starting from `folder_path`
    #     image_id = os.path.normpath(os.path.relpath(file_path, start=folder_path))
    #     # remove file extension
    #     image_id = os.path.splitext(image_id)[0]
    #     return image_id
    #
    # input_files = sorted(
    #     (os.path.join(image_root, f) for f in PathManager.ls(image_root) if f.endswith(image_ext)),
    #     key=lambda file_path: file2id(image_root, file_path),
    # )
    # gt_files = sorted(
    #     (os.path.join(gt_root, f) for f in PathManager.ls(gt_root) if f.endswith(gt_ext)),
    #     key=lambda file_path: file2id(gt_root, file_path),
    # )
    #
    # assert len(gt_files) > 0, "No annotations found in {}.".format(gt_root)
    #
    # # Use the intersection, so that val2017_100 annotations can run smoothly with val2017 images
    # # if len(input_files) != len(gt_files):
    # #     logger.warn(
    # #         "Directory {} and {} has {} and {} files, respectively.".format(
    # #             image_root, gt_root, len(input_files), len(gt_files)
    # #         )
    # #     )
    #     input_basenames = [os.path.basename(f)[: -len(image_ext)] for f in input_files]
    #     gt_basenames = [os.path.basename(f)[: -len(gt_ext)] for f in gt_files]
    #     intersect = list(set(input_basenames) & set(gt_basenames))
    #     # sort, otherwise each worker may obtain a list[dict] in different order
    #     intersect = sorted(intersect)
    #     #logger.warn("Will use their intersection of {} files.".format(len(intersect)))
    #     input_files = [os.path.join(image_root, f + image_ext) for f in intersect]
    #     gt_files = [os.path.join(gt_root, f + gt_ext) for f in intersect]
    #
    # # logger.info(
    # #     "Loaded {} images with semantic segmentation from {}".format(len(input_files), image_root)
    # # )
    #
    # dataset_dicts = []
    # for img_path, gt_path in zip(images, targets):
    #     record = {}
    #     record["file_name"] = img_path
    #     record["sem_seg_file_name"] = gt_path
    #     dataset_dicts.append(record)

    return file_paths


def _get_acdc_meta():
    CITYSCAPES_THING_CLASSES = [
            "person", "rider", "car", "truck",
            "bus", "train", "motorcycle", "bicycle",
        ]
    CITYSCAPES_STUFF_CLASSES = [
        "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
        "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car",
        "truck", "bus", "train", "motorcycle", "bicycle",
    ]
        # fmt: on
    return {
        "thing_classes": CITYSCAPES_THING_CLASSES,
        "stuff_classes": CITYSCAPES_STUFF_CLASSES,
    }

def register_acdc(root):

    meta = _get_acdc_meta()

    image_dir = os.path.join(root, 'rgb_anon')
    gt_dir = os.path.join(root, 'gt')
    name = "acdc_sem_seg_trainval"
    DatasetCatalog.register(
        name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x)
    )
    MetadataCatalog.get(name).set(
        image_root=image_dir,
        sem_seg_root=gt_dir,
        evaluator_type="acdc_sem_seg",
        ignore_label=255,  # different from other datasets, Mapillary Vistas sets ignore_label to 65
        **meta,
    )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_acdc(_root)

# print(load_sem_seg('D:/Thesis/ACDC/gt','D:/Thesis/ACDC/rgb_anon/'))
