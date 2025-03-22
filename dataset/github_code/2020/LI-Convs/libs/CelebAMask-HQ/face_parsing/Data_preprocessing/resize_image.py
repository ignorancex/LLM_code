# resize original image to the size of segmentation labels
# generate list for evaluation

import os
import argparse
import cv2
import glob

LB_SIZE = (512, 512)
DATA_ROOT = '../../../../../dataset/CelebAMask-HQ'

parser = argparse.ArgumentParser()
parser.add_argument('--data_root',default=DATA_ROOT, help='Data root path of CelebAMask-HQ')
args = parser.parse_args()

def main():
    data_root = args.data_root.strip()
    image_folder = os.path.join(data_root, 'CelebA-HQ-img')
    save_folder = os.path.join(data_root, 'CelebA-HQ-img-resize')

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    img_list = glob.glob(os.path.join(image_folder,'*.jpg'))
    assert len(img_list) == 30000, 'Incorrect number of images: {}, should be: 30000'.format(len(img_list))

    print('Resizing images to label size ...')

    for idx,img_path in enumerate(img_list):
        this_image = cv2.imread(img_path)
        this_image_resize = cv2.resize(this_image, LB_SIZE, interpolation=cv2.INTER_LINEAR)
        img_name = os.path.split(img_path)[1]
        save_path = os.path.join(save_folder, img_name)
        rt = cv2.imwrite(save_path, this_image_resize)
        if not rt:
            raise ValueError('Insuccessful image write at: {}'.format(save_path))
        print('Origin: {}, Saving to: {}'.format(img_path,save_path))


if __name__=="__main__":
    main()

