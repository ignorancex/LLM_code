import cv2
from PIL import Image
import glob
import os

def crop(out_path, in_path, fname):
    img = cv2.imread(in_path+fname)
    for r in range(0, img.shape[0], 256):
        for c in range(0, img.shape[1], 256):
            cv2.imwrite(out_path+f"{r}_{c}_"+fname, img[r:r + 256, c:c + 256, :])


def crop_and_resize(resize_out_path, crop_out_path, org_path):
    # crop
    for f in glob.glob(org_path + "*.png"):
        fname = os.path.basename(f)
        img = cv2.imread(org_path+fname)
        for r in range(0, img.shape[0], 500):
            for c in range(0, img.shape[1], 500):
                cv2.imwrite(crop_out_path+f"{r}_{c}_"+fname, img[r:r + 500, c:c + 500, :])
                #only_out_filename = crop_out_path+f"{r}_{c}_"+fname

    # resize
    for file in glob.glob(crop_out_path + "*.png"):
        print(file)
        only_filename = os.path.basename(file)
        image = Image.open(file)
        image = image.resize((256, 256), Image.ANTIALIAS)
        image.save(fp=resize_out_path + only_filename)

        print(resize_out_path + only_filename)


org_path = '/home/shong/image_inpainting_model_for_lane_geomery_discovery/util/'
crop_out_path = '/home/shong/image_inpainting_model_for_lane_geomery_discovery/util/crop/'
resize_out_path = '/home/shong/image_inpainting_model_for_lane_geomery_discovery/util/resize/'
crop_and_resize(resize_out_path, crop_out_path, org_path)

