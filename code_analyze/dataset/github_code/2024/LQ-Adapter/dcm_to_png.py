import cv2
import os
import pydicom
from tqdm import tqdm

# Replace this line with your choice: 'train' or 'test'
conversion_type = 'train'

if conversion_type == 'train':
    print('Converting train images from .dcm to .jpg...')
    inputdir = 'data/stage_2_train_images/'
    outdir = 'data/images'
elif conversion_type == 'test':
    print('Converting test images from .dcm to .jpg...')
    inputdir = 'data/stage_2_test_images/'
    outdir = 'data/samples'
os.makedirs(outdir, exist_ok=True)

train_list = [f for f in os.listdir(inputdir)]

for i, f in tqdm(enumerate(train_list[:]), total=len(train_list)):
    ds = pydicom.read_file(inputdir + f)  # read dicom image
    img = ds.pixel_array  # get image array
    # img = cv2.resize(img, (416, 416))
    cv2.imwrite(os.path.join(outdir, f.replace('.dcm', '.png')), img)  # write jpg image