import numpy as np
import skimage
import cv2
from PIL import Image
from skimage.transform import resize


def extend_image(image, nb_rows_to_add, preserve_square=True):
    """
    Extend the image by adding black rows at the bottom.
    :param image: np.array with shape (nb_rows, nb_columns)
        the image to extend
    :param nb_rows_to_add: int
        the number of rows to add
    :param preserve_square: bool
        if True, the image is padded equally to the left and right to make it square again after adding the rows
    :return: np.array with shape (nb_rows + nb_rows_to_add, nb_columns) or (nb_rows + nb_rows_to_add, nb_rows + nb_rows_to_add)
        the extended image
    """
    original_shape = image.shape
    new_depth = image.shape[0] + nb_rows_to_add
    # width stays the same
    new_image = np.zeros((new_depth, image.shape[1]))
    new_image[:image.shape[0], :] = image
    if preserve_square:
        # make the new image square by padding equally to the left and right
        padding_amount = (new_depth - original_shape[1]) // 2
        new_image_rectangular = new_image
        new_image = np.zeros((new_depth, new_depth))
        new_image[:, padding_amount:padding_amount + original_shape[1]] = new_image_rectangular
    return new_image


def get_top_padding(sector):
    """
    Count the number of completely black rows at the top of the sector
    :param sector: np.array with shape (nb_rows, nb_columns)
        the sector as a boolean array. It is True at the locations where the sector is
    :return: int
        the number of black rows at the top
    """
    top_padding = 0
    for row in sector:
        if np.all(row == 0):
            top_padding += 1
        else:
            break
    return top_padding


def add_top_padding(image, nb_top_padding_rows):
    """
    Add black rows at the top of the image
    This will change the shape of the image
    :param image: np.array with shape (nb_rows, nb_columns)
        the image to add padding to
    :param nb_top_padding_rows: int
        the number of rows to add
    :return: np.array with shape (nb_rows + nb_top_padding_rows, nb_columns)
        the image with the added padding
    """
    padding = np.zeros((nb_top_padding_rows, image.shape[1]))
    return np.vstack((padding, image))


def adjust_padding(image, nb_top_padding_rows):
    """
    Adjust the padding of the image to have the given number of completely top black padding rows at the top
    :param image: np.array with shape (nb_rows, nb_columns)
        the image to adjust the padding of
    :param nb_top_padding_rows: int
        the number of black rows at the top
    :return: np.array with shape (nb_rows, nb_columns)
        the image with the adjusted padding
    """
    target_size = image.shape
    original_nb_top_padding_rows = get_top_padding(image)
    image_no_padding = image[original_nb_top_padding_rows:]
    image = add_top_padding(image_no_padding, nb_top_padding_rows)
    output = np.zeros(target_size)
    nb_rows_to_fill = np.min((image.shape[0], target_size[0]))
    # fill the output images with the resized images. This takes care of the bottom padding
    output[:nb_rows_to_fill, :] = image[:nb_rows_to_fill, :]
    return output


def resize_label(label_mask, new_size, nb_labels=4):
    # resize the reference_mask for each label separately to avoid interpolation artifacts
    if len(label_mask.shape) == 3: # one-hot encoded
        reference_resized = np.zeros((nb_labels, new_size[0], new_size[1]))
        for i in range(nb_labels):
            mask = label_mask[i]
            resized_mask = skimage.transform.resize(mask, new_size)
            reference_resized[i] = resized_mask
    elif len(label_mask.shape) == 2: # label encoded
        reference_resized = np.zeros(new_size)
        for i in range(nb_labels):
            mask = label_mask == i
            resized_mask = skimage.transform.resize(mask, new_size)
            reference_resized[resized_mask] = i
    else:
        raise ValueError(f'reference_mask has unexpected shape {label_mask.shape}')

    return reference_resized


def resize_annotated_bmode(bmode, reference_mask, new_size, nb_labels=4):
    """
    Resize the bmode and the reference mask to the new size without interpolation artifacts
    :param bmode: np.array with shape (nb_rows, nb_columns)
        the bmode image
    :param reference_mask: np.array with shape (nb_rows, nb_columns) or (nb_labels, nb_rows, nb_columns)
        the reference mask.
        If it is one-hot encoded, it has shape (nb_labels, nb_rows, nb_columns)
        If it is label encoded, it has shape (nb_rows, nb_columns)
    :param new_size: tuple with 2 ints
        the new size of the images (nb_rows, nb_columns)
    """
    # resize the bmode
    bmode_resized = skimage.transform.resize(bmode, new_size)
    reference_resized = resize_label(reference_mask, new_size, nb_labels)


    reference_resized = reference_resized.astype(np.uint8)
    return bmode_resized, reference_resized


def get_repaint_input_depth_increase(bmode, reference_mask, depth_increase):
    """
    Extend the bmode and reference mask, resize them, and compute the repaint mask
    for a depth increase repaint augmentation
    :param bmode: np.array with shape (nb_rows, nb_columns)
        The bmode image.
    :param reference_mask: np.array with shape (nb_rows, nb_columns)
        The reference mask.
    :param depth_increase: int
        The number of rows to add to the bottom of the image.
    :param border_thickness: int
        The thickness of the border for the repaint mask.
    :return: tuple of 2 np.arrays with shape (256, 256)
        bmode, reference mask
    """
    if not bmode.shape == reference_mask.shape:
        raise ValueError('bmode and reference mask must have the same shape')
    shape = bmode.shape
    # Extend the bmode and reference mask
    bmode_extended = extend_image(bmode, depth_increase)
    reference_mask_extended = extend_image(reference_mask, depth_increase)

    # Resize the extended bmode and reference mask back to the original size
    bmode, reference = resize_annotated_bmode(bmode_extended, reference_mask_extended, shape)

    return bmode, reference


def get_repaint_mask(image, border_thickness=2):
    """
    Get the mask that indicates which pixels should be inpainted.
    We inpaint all pixels that are not in the sector and the border of the pixels inside the sector.
    We do this to avoid artefacts at the border as much as possible.
    :param: image: np.array with shape (nb_rows, nb_columns)
        the image to get the repaint mask for
    :param border_thickness: int
        the thickness of the border of the sector to repaint
    """
    repaint_mask = ~(image != 0).astype(bool)
    kernel = np.ones((border_thickness, border_thickness), np.uint8)
    dilated_mask = cv2.dilate(repaint_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    return ~dilated_mask


def save_as_image(np_array,path):
    """
    Save a 2D numpy array as a grayscale image (only 1 channel)
    :param np_array: np.array of size (W,H)
        The array to save
    :param path: str
        The path to sae the array to
    """
    img_data = Image.fromarray(np_array)
    img_data = img_data.convert("L")
    img_data.save(path)


def rotate_reference(reference_mask, rotation, center, nb_labels = 4):
    """
    Rotate the reference mask label by label to avoid interpolation artifacts
    :param reference_mask: np.array with shape (nb_rows, nb_columns)
        the reference mask to rotate
    :param rotation: float
        the rotation angle in degrees
    :param center: tuple with 2 ints
        the center of the rotation
    :param nb_labels: int
        the number of labels in the reference mask
    :return: np.array with shape (nb_rows, nb_columns)
        the rotated reference mask
    """
    # rotate each label separately to avoid interpolation artifacts
    if len(reference_mask.shape) == 3: # one-hot encoded
        raise NotImplementedError("Rotation for one-hot encoded reference mask is not implemented")
    elif len(reference_mask.shape) == 2: # label encoded
        result = np.zeros(reference_mask.shape)
        for i in range(1,nb_labels):
            mask = reference_mask == i
            rotated_mask = skimage.transform.rotate(mask, rotation, center=center)
            result[rotated_mask] = i
    else:
        raise ValueError(f'reference_mask has unexpected shape {reference_mask.shape}')
    return result


def get_repaint_input_rotation(bmode, reference_mask, rotation):
    """
    Rotate the bmode and reference mask around the top middle point
    :param bmode: np.array with shape (nb_rows, nb_columns)
        the bmode image
    :param reference_mask: np.array with shape (nb_rows, nb_columns)
        the reference mask
    :param rotation: float
        the rotation angle in degrees
    :return: tuple of 2 np.arrays with shape (nb_rows, nb_columns)
        the rotated bmode and reference mask
    """
    # rotate the bmode and reference mask around top middle point
    rotation_cetner = (bmode.shape[1]//2, 0)
    bmode_rotated = skimage.transform.rotate(bmode, rotation, center=rotation_cetner)
    reference_mask_rotated = rotate_reference(reference_mask, rotation, rotation_cetner)
    return bmode_rotated, reference_mask_rotated


def get_repaint_input_sector_width(bmode, reference_mask, width_factor):
    """
    Resize the bmode and reference mask to the original width * width_factor
    :param bmode: np.array with shape (nb_rows, nb_columns)
        the bmode image
    :param reference_mask: np.array
        the reference mask
    :param width_factor: float
        the factor to resize the image width by
    :return: tuple of 2 np.arrays with shape (nb_rows, nb_columns)
        the resized bmode and reference mask
    """
    original_shape = bmode.shape
    # first resize the image to original_width * random_squeeze_factor
    new_width = int(bmode.shape[1] * width_factor)
    new_size = (bmode.shape[0], new_width)
    bmode_resized = skimage.transform.resize(bmode, new_size)
    reference_resized = resize_label(reference_mask, new_size)

    if width_factor < 1:
        # add padding to the left and right to return to the original width
        padding_left = (original_shape[1] - new_width) // 2
        padding_right = original_shape[1] - new_width - padding_left
        bmode_resized = np.pad(bmode_resized, ((0, 0), (padding_left, padding_right)), mode='constant')
        reference_resized = np.pad(reference_resized, ((0, 0), (padding_left, padding_right)), mode='constant')
    else:
        # crop the image to the original width
        margin_left = (bmode_resized.shape[1] - original_shape[1]) // 2
        margin_right = bmode_resized.shape[1] - original_shape[1] - margin_left
        if margin_right == 0:
            bmode_resized = bmode_resized[:, margin_left:]
            reference_resized = reference_resized[:, margin_left:]
        else:
            bmode_resized = bmode_resized[:, margin_left:-margin_right]
            reference_resized = reference_resized[:, margin_left:-margin_right]

    assert bmode_resized.shape == original_shape

    return bmode_resized, reference_resized


def translate_reference(reference_mask, translation, nb_labels=4):
    """
    Translate the reference mask label by label to avoid interpolation artifacts
    :param reference_mask: np.array with shape (nb_rows, nb_columns)
        the reference mask to translate
    :param translation: tuple with 2 ints
        the translation in x and y direction
    :param nb_labels: int
        the number of labels in the reference mask
    :return: np.array with shape (nb_rows, nb_columns)
        the translated reference mask
    """
    # translate reference mask label by label to avoid interpolation artifacts
    x_translation, y_translation = translation
    result = np.zeros(reference_mask.shape)
    if len(reference_mask.shape) == 3: # one-hot encoded
        for i in range(1, reference_mask.shape[0]):
            mask = reference_mask[i]
            translated_mask = skimage.transform.warp(mask, skimage.transform.AffineTransform(
                translation=(x_translation, y_translation)))
            result[i] = translated_mask
    elif len(reference_mask.shape) == 2: # label encoded
        for i in range(1, nb_labels):
            mask = reference_mask == i
            translated_mask = skimage.transform.warp(mask, skimage.transform.AffineTransform(
                translation=(x_translation, y_translation)))
            result[translated_mask] = i
    else:
        raise ValueError(f'reference_mask has unexpected shape {reference_mask.shape}')
    return result


def get_repaint_input_translation(bmode, reference_mask, random_translation):
    """
    Translate the bmode and reference mask
    :param bmode: np.array with shape (nb_rows, nb_columns)
        the bmode image
    :param reference_mask: np.array with shape (nb_rows, nb_columns)
        the reference mask
    :param random_translation: tuple with 2 ints
        the translation in x and y direction
    :return: tuple of 2 np.arrays with shape (nb_rows, nb_columns)
        the translated bmode and reference mask
    """
    # translate the image and reference mask
    x_translation, y_translation = random_translation
    bmode_translated = skimage.transform.warp(bmode, skimage.transform.AffineTransform(
        translation=(x_translation, y_translation)))
    reference_mask_translated = translate_reference(reference_mask, random_translation)

    return bmode_translated, reference_mask_translated


def create_segmentation_visualization(image, segmentation, labels=None, colors=None, remove_oos=False,eps=1e-6):
    '''
    Create a visualization of the ultrasound image with the segmentation overlayed
    :param image: ndarray
        ndarray with shape (width,height) containing the pixel values of the grayscale ultrasound image
    :param segmentation:  ndarray
        ndarray with shape (labels,width,height) containing the segmentation (one-hot encoded) or
        ndarray with shape (width,height) containing the segmentation (not one-hot encoded)
    :param labels: list of int, optional
        the labels of the segments to visualize
        If not specified, default value is [0, 1, 2, 3, 4, 5, 6, 7]
    :param colors: ndarray, optional
        ndarray with shape (n,3) containing the colors to use for the segments
        If not specified, default value is np.array([(1,0,0),(0,0,1),(0,1,0),(1,1,0),(0,1,1),(1,0,1),(1,1,1),
                                      (0.55,0.27,0.07),(1,0.55,0)])
    :param remove_oos
        remove out of sector parts
        If True, parts of the segmentation outside the sector will be removed
    :return: ndarray
        the visualization of the ultrasound image with the segmentation overlayed with values in range [0,255]
    '''
    image_rescaled = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255
    if labels is None:
        labels = [1, 2, 3, 4, 5, 6, 7]
    if colors is None:
        colors = np.array([(1, 0, 0), (0, 0, 1), (0, 1, 0), (1, 1, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1),
                           (0.55, 0.27, 0.07), (1, 0.55, 0)])

    if len(labels) > len(colors):
        raise ValueError('Not enough colors for plotting')

    if len(segmentation.shape) == 3:
        one_hot = True
        if max(labels) > len(segmentation):
            raise ValueError('Labels should be in range of the number of channels in the segmentation')
        resize_shape = (segmentation.shape[1], segmentation.shape[2])
    else:
        one_hot = False
        resize_shape = (segmentation.shape[0], segmentation.shape[1])
    # resize image to the same size as the segmentation
    image_resized = resize(image_rescaled, resize_shape, preserve_range=True)
    oos_mask = image_resized < eps

    result = np.zeros((image_resized.shape[0], image_resized.shape[1], 3))
    for i in range(3):
        result[:, :, i] = image_resized / 255
    for i, label in enumerate(labels):
        if one_hot:
            label_semgentation = segmentation[label, ...] > 0.5
        else:
            label_semgentation = segmentation == label

        if colors[i, 0] != 0:
            result[label_semgentation, 0] = np.clip(colors[i, 0] * (0.35 +
                                                                    result[label_semgentation, 0]), 0.0, 1.0)
        if colors[i, 1] != 0:
            result[label_semgentation, 1] = np.clip(colors[i, 1] * (0.35 +
                                                                    result[label_semgentation, 1]), 0.0, 1.0)
        if colors[i, 2] != 0:
            result[label_semgentation, 2] = np.clip(colors[i, 2] * (0.35 +
                                                                    result[label_semgentation, 2]), 0.0, 1.0)
    if remove_oos:
        result[oos_mask] = 0 # set out of sector parts to black
    return (result * 255).astype(np.uint8)

