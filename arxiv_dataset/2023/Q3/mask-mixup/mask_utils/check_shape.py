import glob
import mmcv
from PIL import Image
import json
import numpy as np
from tqdm import tqdm

print("parsing images...")
# images = glob.glob('./MERGE_OCEAN/Train/**/**/*')

print("checking images...")
# inverted = []
# for f in tqdm(images):
#     if str(mmcv.imread(f).shape) != str(np.array(Image.open(f)).shape):
#         inverted.append(f)
# print(len(inverted), "bad images :", inverted)

# for f in tqdm(images):
#     image = Image.open(f)

#     # next 3 lines strip exif
#     data = list(image.getdata())
#     image_without_exif = Image.new(image.mode, image.size)
#     image_without_exif.putdata(data)

#     image_without_exif.save(f)
img = np.array(Image.open('./mmdetection/mask.png'))
print(img.shape)
print(np.unique(img))
mask = mmcv.impad(img, shape=(800, 1184), pad_val=0)
