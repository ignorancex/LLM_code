from glob import glob
import numpy as np
from PIL import Image

# Get all images under the folder
save_folder = "outputs/mar_large_arinar_6_w1024_d1/ariter256-gnum6-linearcfg3.7-image50000_ema_evaluate/*.*"
image_files = glob(save_folder)
print(len(image_files))

all_images = [np.array(Image.open(file)) for file in image_files]

# Save the images as numpy
all_images = np.array(all_images)
print(all_images.shape)
np.savez("./outputs/new_arinar_large_to_evaluate_t1_0_cfg_3_7.npz", all_images)