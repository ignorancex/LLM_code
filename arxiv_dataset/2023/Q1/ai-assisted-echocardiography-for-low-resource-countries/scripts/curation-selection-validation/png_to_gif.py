import glob
from PIL import Image
# REFERENCE: https://stackoverflow.com/questions/753190/

# filepaths
# fp_in = "/home/mx19/datasets/vital-us/echocardiography/preprocessed-datasets/01NVb-003-072/T1/cropped_us_image/clip001/*.png"
# fp_out = "/home/mx19/datasets/vital-us/echocardiography/preprocessed-datasets/01NVb-003-072/T1/cropped_us_image/animations/clip001.gif"

# fp_in = "/home/mx19/datasets/vital-us/echocardiography/preprocessed-datasets/01NVb-003-072/T2/cropped_us_image/clip001/*.png"
# fp_out = "/home/mx19/datasets/vital-us/echocardiography/preprocessed-datasets/01NVb-003-072/T2/cropped_us_image/animations/clip001.gif"

# fp_in = "/home/mx19/datasets/vital-us/echocardiography/preprocessed-datasets/01NVb-003-072/T2/cropped_us_image/clip002/*.png"
# fp_out = "/home/mx19/datasets/vital-us/echocardiography/preprocessed-datasets/01NVb-003-072/T2/cropped_us_image/animations/clip002.gif"

# fp_in = "/home/mx19/datasets/vital-us/echocardiography/preprocessed-datasets/01NVb-003-072/T2/cropped_us_image/clip003/*.png"
# fp_out = "/home/mx19/datasets/vital-us/echocardiography/preprocessed-datasets/01NVb-003-072/T2/cropped_us_image/animations/clip003.gif"


# https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
img.save(fp=fp_out, format='GIF', append_images=imgs, save_all=True, duration=200, loop=0)