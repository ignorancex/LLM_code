# DATA PATHS
DATA_PATH = "/data"

TRAIN_DIR = f"{DATA_PATH}/throat_instance_segmentation"
TRAIN_PATH = f"{TRAIN_DIR}/annotations/throat_inst_seg_4cls_COCO_train.json"
VAL_PATH = f"{TRAIN_DIR}/annotations/throat_inst_seg_4cls_COCO_val.json"
IMG_DIR_HQ = f"{TRAIN_DIR}/images"

LQ_DIR = f"{DATA_PATH}/old_camera_image_sdg"
LQ_PATH = f"{LQ_DIR}/instances_default.json"
IMG_DIR_LQ = LQ_DIR

SP_DIR = f"{DATA_PATH}/smartphone-throat-images"
SP_PATH = f"{SP_DIR}/annotations/instances_default.json"
IMG_DIR_SP = SP_DIR

# MAPPINGS
INT_TO_LABEL = {0: 'other', 1: 'tonsil', 2: 'uvula', 3: 'tongue', 4: 'back_wall'}
LABEL_TO_INT = {'other': 0, 'tonsil': 1, 'uvula': 2, 'tongue': 3, 'back_wall': 4}