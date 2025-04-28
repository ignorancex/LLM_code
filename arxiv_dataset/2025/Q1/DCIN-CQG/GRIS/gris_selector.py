from color_histogram_computation import *

# PATHS
DATA_PATH = "/data"
TRAIN_DIR = f"{DATA_PATH}/throat_instance_segmentation"
JSON_PATH = f"{TRAIN_DIR}/annotations/throat_inst_seg_4cls_COCO_train.json"
IMG_DIR = f"{TRAIN_DIR}/images"

# GENERATE HISTOGRAMS
images, paths = load_images_and_paths(JSON_PATH, IMG_DIR)
features = generate_color_histograms(images, bins=(8, 8, 8))

# REPRESENTATIVE IMAGE
best_idx = select_most_representative_image(features)
print(paths[best_idx])
print()