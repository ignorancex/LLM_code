import os
import ete3
import glob
import time
import shutil
import tarfile
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path',
                        type=str,
                        default='data/CUB_200_2011/images',
                        help='Path to images folder under CUB-200-2011 dataset')
    parser.add_argument('--phylogeny_path',
                        type=str,
                        default='data/phlyogenyCUB/1_tree-consensus-Hacket-AllSpecies-modified_cub-names_v1.phy',
                        help='Path to cub190 phlyogeny dataset')
    parser.add_argument('--cub190_path',
                        type=str,
                        default='data/CUB_200_2011/images_cub190',
                        help='Path to save the filtered cub190 set')
    parser.add_argument('--image_size',
                        type=int,
                        default=224,
                        help='Size of the output image')
    parser.add_argument('--segment',
                        action='store_true', 
                        help='Segments images based on segmentation masks. Assumes "data/cub_meta/segmentations.tgz" is available')
    return parser.parse_args()


def filter_cub200_to_cub190(args):
                    
    tree = ete3.Tree(args.phylogeny_path)
    leaf_names = [leaf.name for leaf in tree.get_leaves()]
    folder_names = leaf_names

    os.makedirs(args.cub190_path, exist_ok=True)
    print('Filtering CUB200 into CUB190')
    copied_count = 0
    for folder in tqdm(os.listdir(args.path), total=len(os.listdir(args.path))):
        # target folder name in the same format as in the cub phylogeny
        target_folder = 'cub_' + folder[:3] + '_' + folder[4:]
        if target_folder in leaf_names:
            folder_path = os.path.join(args.path, folder)
            if os.path.isdir(folder_path):
                shutil.copytree(folder_path, os.path.join(args.cub190_path, target_folder), dirs_exist_ok=True)
                copied_count += 1
    print(f'Copied {copied_count} folders')


def make_squared(args, img, padding='white'):

    imageDimension = args.image_size

    img_H = img.size[0]
    img_W = img.size[1]
    if not imageDimension:
        imageDimension = max(img_H, img_W)
    smaller_dimension = 0 if img_H < img_W else 1
    larger_dimension = 1 if img_H < img_W else 0
    if (imageDimension != img_H or imageDimension != img_W):
        new_smaller_dimension = int(imageDimension * img.size[smaller_dimension] / img.size[larger_dimension])
        if smaller_dimension == 1:
            img = transforms.functional.resize(img, (new_smaller_dimension, imageDimension))
        else:
            img = transforms.functional.resize(img, (imageDimension, new_smaller_dimension))

        diff = imageDimension - new_smaller_dimension
        pad_1 = int(diff/2)
        pad_2 = diff - pad_1

        if padding == 'imagenet':
            mean = np.asarray([ 0.485, 0.456, 0.406 ])
            fill = tuple([int(round(mean[0]*255)), int(round(mean[1]*255)), int(round(mean[2]*255))])
        elif padding=='black':
            fill = tuple([0, 0, 0])
        else:
            fill = tuple([255, 255, 255])

        if smaller_dimension == 0:
            img = transforms.functional.pad(img, (pad_1, 0, pad_2, 0), padding_mode='constant', fill = fill)
        else:
            img = transforms.functional.pad(img, (0, pad_1, 0, pad_2), padding_mode='constant', fill = fill)

    return img    


def alter_name_for_cub_phylogeny(name):
    return 'cub_' + name[:3] + '_' + name[4:]


def apply_segmentation_mask_with_imagenet_mean(image_path, segmentation_mask_path):
    image = Image.open(image_path).convert("RGBA")    
    mask_path = segmentation_mask_path
    mask = Image.open(mask_path).convert("L")  # Convert to grayscale
    imagenet_mean_color = (124, 116, 104)  # Approximate ImageNet mean RGB values rounded to nearest integer
    mean_color_image = Image.new("RGBA", image.size, (*imagenet_mean_color, 255))
    segmented_image = Image.composite(image, mean_color_image, mask)
    return segmented_image


def split_segment_reshape_cub190(args):

    """
    Creates train_crop, train, test_crop, test folders
    train_crop -> train images are segmented, cropped and reshaped
    train -> train images are segmented and reshaped
    Same process done for test images
    """

    segmentation_tgz_path = "data/cub_meta/segmentations.tgz"
    segmentation_path = os.path.join(os.path.dirname(segmentation_tgz_path), 'segmentations')

    if args.segment and not os.path.exists(segmentation_tgz_path) and not os.path.exists(segmentation_path):
        raise Exception(f'CUB-200-2011 segmentations.tgz not found. Download and save at {os.path.dirname(segmentation_tgz_path)}')
    
    if args.segment and not os.path.exists(segmentation_path):
        start_ = time.time()
        print(f'Extracting {segmentation_tgz_path}')
        with tarfile.open(segmentation_tgz_path, "r:gz") as tar:
            tar.extractall(path=os.path.dirname(segmentation_tgz_path))
        print('Extraction completed. Elapsed time', f'{round(time.time() - start_)} s')

    path = os.path.dirname(args.path)
    images_path = args.cub190_path # "images_cub190/"

    time_start = time.time()

    path_images = os.path.join(path,'images.txt')
    path_split = os.path.join(path,'train_test_split.txt')
    train_save_path = os.path.join(path,'dataset_cub190/train_crop/')
    test_save_path = os.path.join(path,'dataset_cub190/test_crop/')
    bbox_path = os.path.join(path, 'bounding_boxes.txt')

    images = []
    with open(path_images,'r') as f:
        for line in f:
            images.append(list(line.strip('\n').split(',')))
    split = []
    with open(path_split, 'r') as f_:
        for line in f_:
            split.append(list(line.strip('\n').split(',')))

    bboxes = dict()
    with open(bbox_path, 'r') as bf:
        for line in bf:
            id, x, y, w, h = tuple(map(float, line.split(' ')))
            bboxes[int(id)]=(x, y, w, h)

    print('Creating cropped images')
    num = len(images)
    folders_skipped = set()
    for k in tqdm(range(num), total=num):
        id, fn = images[k][0].split(' ')
        id = int(id)
        original_file_name = fn.split('/')[0]
        file_name = alter_name_for_cub_phylogeny(original_file_name)

        if not os.path.exists(os.path.join(images_path, file_name)):
            folders_skipped.add(file_name)
            continue
        
        if args.segment:
            img = apply_segmentation_mask_with_imagenet_mean(os.path.join(images_path, file_name, images[k][0].split(' ')[1].split('/')[1]), 
                                                            os.path.join(segmentation_path, original_file_name, images[k][0].split(' ')[1].split('/')[1].split('.')[0] + '.png')).convert('RGB')
        else:
            img = Image.open(os.path.join(images_path, file_name, images[k][0].split(' ')[1].split('/')[1])).convert('RGB')
        x, y, w, h = bboxes[id]
        cropped_img = img.crop((x, y, x+w, y+h))
        cropped_img = make_squared(args, cropped_img, padding='imagenet')

        if int(split[k][0][-1]) == 1:
            os.makedirs(os.path.join(train_save_path, file_name), exist_ok=True)
            cropped_img.save(os.path.join(os.path.join(train_save_path,file_name),images[k][0].split(' ')[1].split('/')[1]))
        else:
            os.makedirs(os.path.join(test_save_path,file_name), exist_ok=True)
            cropped_img.save(os.path.join(os.path.join(test_save_path,file_name),images[k][0].split(' ')[1].split('/')[1]))

    print('Done saving cropped images')
    print(f'{len(folders_skipped)} folders skipped (If this is 10 it is normal for creating cub190 from cub200)')

    train_save_path = os.path.join(path,'dataset_cub190/train/')
    test_save_path = os.path.join(path,'dataset_cub190/test/')
    
    print('Saving full-sized images')
    folders_skipped = set()
    num = len(images)
    for k in tqdm(range(num), total=num):
        id, fn = images[k][0].split(' ')
        id = int(id)
        original_file_name = fn.split('/')[0]
        file_name = alter_name_for_cub_phylogeny(original_file_name)

        if not os.path.exists(os.path.join(images_path, file_name)):
            folders_skipped.add(file_name)
            continue

        if args.segment:
            img = apply_segmentation_mask_with_imagenet_mean(os.path.join(images_path, file_name, images[k][0].split(' ')[1].split('/')[1]), 
                                                            os.path.join(segmentation_path, original_file_name, images[k][0].split(' ')[1].split('/')[1].split('.')[0] + '.png')).convert('RGB')
        else:
            img = Image.open(os.path.join(images_path, file_name, images[k][0].split(' ')[1].split('/')[1])).convert('RGB')
        img = make_squared(args, img, padding='imagenet')

        if int(split[k][0][-1]) == 1:
            os.makedirs(os.path.join(train_save_path,file_name), exist_ok=True)
            img.save(os.path.join(os.path.join(train_save_path,file_name),images[k][0].split(' ')[1].split('/')[1]))
        else:
            os.makedirs(os.path.join(test_save_path,file_name), exist_ok=True)
            img.save(os.path.join(os.path.join(test_save_path,file_name),images[k][0].split(' ')[1].split('/')[1]))

    time_end = time.time()
    print('Done saving full-sized images')
    print(f'{len(folders_skipped)} folders skipped (If this is 10 it is normal for creating cub190 from cub200)')


if __name__ == "__main__":
    args = get_args()

    filter_cub200_to_cub190(args)

    split_segment_reshape_cub190(args)


