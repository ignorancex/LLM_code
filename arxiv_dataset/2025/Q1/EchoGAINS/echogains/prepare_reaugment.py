import os
import echogains.utils as utils
import numpy as np
import matplotlib.pyplot as plt
import copy
import shutil
from tqdm import tqdm

def prepare_depth_augment(image, label, depth_increase):
    """
    Prepare a B-mode depth augmentation by increasing the depth of the image and label by depth_increase
    The image and label are resized and take up the top part of the sector.
    The bottom part is filled with zeros.
    :param image: np.array of shape (H, W)
        The image to augment.
    :param label: np.array of shape (H, W)
        The segmentation mask corresponding to the image.
    :param depth_increase: int
        The depth increase to apply.
    :return: tuple
        bmode: np.array of shape (H, W)
            The transformed image, ready for repainting.
        gt: np.array of shape (H, W)
            The transformed mask.
    """
    bmode, gt = utils.get_repaint_input_depth_increase(image,label,depth_increase)
    return bmode, gt


def prepare_rotation_augment(image, label, rotation):
    '''
    Prepare a B-mode rotation augmentation by rotating the image and label by rotation degrees.
    :param image: np.array of shape (H, W)
        The image to augment.
    :param label: np.array of shape (H, W)
        The segmentation mask corresponding to the image.
    :param rotation: int
        The rotation to apply.
    :return: tuple
        bmode: np.array of shape (H, W)
            The transformed image, ready for repainting.
        gt: np.array of shape (H, W)
            The transformed mask.
    '''
    bmode_resized, gt_resized = utils.get_repaint_input_rotation(image, label, rotation)
    return bmode_resized, gt_resized


def prepare_sector_width_augment(image, label, random_width_factor):
    '''
    Prepare a B-mode sector width augmentation by changing the sector width of the image and label by random_width_factor.
    :param image: np.array of shape (H, W)
        The image to augment.
    :param label: np.array of shape (H, W)
        The segmentation mask corresponding to the image.
    :param random_width_factor: float
        The width factor to apply.
    :return: tuple
        bmode: np.array of shape (H, W)
            The transformed image, ready for repainting.
        gt: np.array of shape (H, W)
            The transformed mask.
    '''
    bmode, gt = utils.get_repaint_input_sector_width(image, label,random_width_factor)
    return bmode, gt


def prepare_translation_augment(image, label, random_translation):
    '''
    Prepare a B-mode translation augmentation by translating the image and label by random_translation.
    :param image: np.array of shape (H, W)
        The image to augment.
    :param label: np.array of shape (H, W)
        The segmentation mask corresponding to the image.
    :param random_translation: tuple
        The translation to apply.
        The tuple contains the x and y translation.
    :return: tuple
        bmode: np.array of shape (H, W)
            The transformed image, ready for repainting.
        gt: np.array of shape (H, W)
            The transformed mask.
    '''
    bmode, gt = utils.get_repaint_input_translation(image,
                                                    label,
                                                    random_translation)
    return bmode, gt


def prepare_gen_aug_seg_sample(image, label, augmentation_params, nb_augmentations=5, repaint_border_thickness=10,
                               max_nb_tries=30):
    """
    Prepare an augmentation by applying a series of augmentations to the image and label.
    The augmentations are defined in augmentation_params and are applied sequentially.
    The function returns a list of nb_augmentations augmented images, labels and repaint masks.
    The repaint mask is a binary mask indicating the part of the image that should be repainted.
    This mask will include all 'missing pixels', plus a border of repaint_border_thickness pixels
    that includes the border of the original image. This border helps reduce artefacts at the border of the sector.
    The function will try to generate a valid augmentation max_nb_tries times.
    An augmentation is considered valid if at least 50% of the foreground pixels (nonzero value)
    in the label are in the sector.
    :param image: np.array of shape (H, W)
        The image to augment.
    :param label: np.array of shape (H, W)
        The segmentation mask corresponding to the image.
    :param augmentation_params: list of dicts
        Each dict contains the parameters for an augmentation.
        Each dict should have the following keys:
        - 'type': str
            The type of augmentation to apply. Can be 'depth', 'rotation', 'sector_width', or 'translation'.
        - 'prob': float
            The probability of applying the augmentation.
        The rest of the keys depend on the type of augmentation:
        - When the type is 'depth':
            - 'depth_increase_range': list of 2 ints
                The range of the depth increase to apply.
                A random value in this range is chosen for each augmentation.
        - When the type is 'rotation':
            - 'rotation_range': list of 2 ints
                The range of the rotation to apply.
                A random value in this range is chosen for each augmentation.
        - When the type is 'sector_width':
            - 'width_factor': list of 2 floats
                The range of the width factor to apply.
                A random value in this range is chosen for each augmentation.
        - When the type is 'translation':
            - 'displacement_radius_range': list of 2 ints
                The range of the displacement radius to apply.
                A random value in this range is chosen for each augmentation.
    :param nb_augmentations: int
        The number of augmentations to generate.
        Default is 5.
    :param repaint_border_thickness: int
        The mask of the border of the repaint mask.
        Default is 10.
    :param max_nb_tries: int
        The maximum number of times to try to generate a valid augmentation.
    :return: list of tuples
        Each tuple contains the augmented image, label and repaint mask of 1 augmentation.
        augmented_image: np.array of shape (H, W)
            The augmented image.
        augmented_label: np.array of shape (H, W)
            The augmented segmentation mask.
        repaint_mask: np.array of shape (H, W)
            The binary mask indicating the part of the image that should be repainted.
    """
    result = []
    for i in range(nb_augmentations):
        augmented_image = copy.deepcopy(image)
        augmented_label = copy.deepcopy(label)
        original_sector = image != 0
        augmentation_valid = False
        nb_tries = 0
        while not augmentation_valid and nb_tries < max_nb_tries:
            for aug_param in augmentation_params:
                if np.random.rand() < aug_param['prob']:
                    if aug_param['type'] == 'depth':
                        random_depth_increase = np.random.randint(aug_param['depth_increase_range'][0],
                                                                  aug_param['depth_increase_range'][1])
                        augmented_image, augmented_label = prepare_depth_augment(augmented_image,
                                                                                 augmented_label,
                                                                                 random_depth_increase)
                    elif aug_param['type'] == 'rotation':
                        random_rotation = np.random.randint(aug_param['rotation_range'][0],
                                                            aug_param['rotation_range'][1])
                        augmented_image, augmented_label = prepare_rotation_augment(augmented_image,
                                                                                    augmented_label,
                                                                                    random_rotation)
                    elif aug_param['type'] == 'sector_width':
                        random_width_factor = np.random.uniform(aug_param['width_factor'][0],
                                                                    aug_param['width_factor'][1])
                        augmented_image, augmented_label = prepare_sector_width_augment(augmented_image,
                                                                                        augmented_label,
                                                                                        random_width_factor)
                    elif aug_param['type'] == 'translation':
                        random_radius = np.random.randint(aug_param['displacement_radius_range'][0],
                                                            aug_param['displacement_radius_range'][1])
                        random_angle = np.random.uniform(0, 2 * np.pi)
                        random_translation = (random_radius * np.cos(random_angle),
                                                random_radius * np.sin(random_angle))
                        augmented_image, augmented_label = prepare_translation_augment(augmented_image,
                                                                                       augmented_label,
                                                                                       random_translation)

                    else:
                        raise ValueError(f'Unknown augmentation type: {aug_param["type"]}')

            # calculate overlap between label and original sector
            labelled_part = augmented_label != 0
            overlap = np.logical_and(labelled_part, original_sector)
            nb_overlapping_pixels = np.sum(overlap)
            total_nb_pixels = np.sum(labelled_part)
            if nb_overlapping_pixels < 0.5 * total_nb_pixels:
                # less than 50% of the label is in the sector
                # this augmentation is invalid, try again
                augmentation_valid = False
                nb_tries+=1
                augmented_image = copy.deepcopy(image)
                augmented_label = copy.deepcopy(label)
            else:
                augmentation_valid = True


        # mask augmented image and label by original sector.
        # They might have values outside of the original sector because of rotation and/or translation
        augmented_image = 255 * augmented_image


        augmented_image = augmented_image * original_sector

        repaint_mask = utils.get_repaint_mask(augmented_image, repaint_border_thickness)

        result.append((augmented_image, augmented_label, repaint_mask))
    return result


def prepare_gen_aug_seg(input_dataset_path,
                        output_dataset_path,
                        repaint_keep_path,
                        repaint_masks_path,
                        augmentation_params,
                        nb_augmentations,
                        include_original=True):
    """
    Prepare an augmentation of a segmentation dataset.
    The idea is that output_dataset_path will eventually contain the augmented dataset.
    This function only creates the augmented labels. The augmented images need to be created separately using
    echo.run_repaint.
    The function reads the images and labels from input_dir_path and applies the augmentations defined in
    augmentation_params to each image and label.
    The input directory, given by input_dir_path, should have the following structure:
    input_dir_path
    ├── images
    │   ├── frame1.png
    │   ├── frame2.png
    │   ├── ...
    ├── labels
    │   ├── frame1.png
    │   ├── frame2.png
    │   ├── ...
    The label, image-to-be-repainted, and repaint mask are saved in
    output_dataset_path/'labels', repaint_keep_path and repaint_masks_path respectively, in the following structure:
    output_dataset_path/'labels' | repaint_keep_path | repaint_masks_path
    ├── frame1_0.png
    ├── frame1_1.png
    ├── ...
    ├── frame1_{nb_augmentations}_0000.png
    ├── ...
    ├── frame2_0.png
    ├── frame2_1.png
    ├── ...
    ├── frame2_{nb_augmentations}_0000.png
    ├── ...
    output_dataset_path/'labels' contains the augmented labels.
    If include_original is True, the original images and labels are copied to output_dataset_path/'images' and
    output_dataset_path/'labels' respectively.
    :param input_dataset_path: str
        The path to the input dataset.
    :param output_dataset_path: str
        The path to the output dataset.
    :param repaint_keep_path: str
        The path to the directory where the images to be repainted will be saved.
    :param repaint_masks_path: str
        The path to the directory where the repaint masks will be saved.
    :param augmentation_params: list of dicts
        Each dict contains the parameters for an augmentation.
        See prepare_gen_aug_seg_sample for more information.
    :param nb_augmentations: int
        The number of augmented images to generate for each image in the input dataset.
    :param include_original: bool
        If True, the original images and labels are copied to the output directory.
    """
    images_path = os.path.join(input_dataset_path, 'images')
    labels_path = os.path.join(input_dataset_path, 'labels')

    # first copy the contents of the input directory to the output directory
    out_path_labels = os.path.join(output_dataset_path, 'labels')

    for path in [output_dataset_path, repaint_keep_path, repaint_masks_path, out_path_labels]:
        if not os.path.exists(path):
            os.makedirs(path)

    if include_original:
        shutil.copytree(input_dataset_path, output_dataset_path, dirs_exist_ok=True)


    for image_name in tqdm(os.listdir(images_path)):
        # image_name is of form frane_name.png
        # corresponding label name is frame_name.png
        # output paths for segmentation:
        image_path = os.path.join(images_path, image_name)
        label_path = os.path.join(labels_path, image_name)

        image = plt.imread(image_path)
        label = (255*plt.imread(label_path)).astype(np.uint8)
        result = prepare_gen_aug_seg_sample(image, label, augmentation_params, nb_augmentations)
        for i, (image, label, repaint_mask) in enumerate(result):
            save_path_label = os.path.join(out_path_labels, f'{image_name[:-4]}_aug{i}.png')
            utils.save_as_image(label, save_path_label)
            save_path_image = os.path.join(repaint_keep_path, f'{image_name[:-4]}_aug{i}.png')
            utils.save_as_image(image, save_path_image)
            save_path_repaint_mask = os.path.join(repaint_masks_path, f'{image_name[:-4]}_aug{i}.png')
            utils.save_as_image(repaint_mask, save_path_repaint_mask)

