import glob
import gzip
import os.path
import shutil
from glob import glob

import nibabel as nib
import numpy as np
from numpy import ndarray
from tqdm import tqdm


def calc_z_score(img: ndarray) -> ndarray:
    """
    Standardize the image data using the zscore (z = (x-μ)/σ).

    :param img: Image data with shape components of (width, height, depth).
    :return: standardized image.
    """
    avg_pixel_value = np.sum(img) / np.count_nonzero(img)
    sd_pixel_value = np.std(img[np.nonzero(img)])

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(img.shape[2]):
                if img[i, j, k] != 0:
                    img[i, j, k] = (img[i, j, k] - avg_pixel_value) / sd_pixel_value

    return img


def change_mask_shape(mask: ndarray):
    """
    Reshape mask to dimensions of 128 x 128 x 128 x 4.

    :param mask: Mask data to reshape.
    :return: reshaped mask.
    """
    if mask.shape == (128, 128, 128, 4):
        raise ValueError(
            f"Mask shape is already (128, 128, 128, 4)")

    new_mask = np.zeros((128, 128, 128, 4))
    for i in range(128):
        for j in range(128):
            for k in range(128):
                new_mask[i, j, k, mask[i, j, k]] = 1

    return new_mask


def normalize_mri_data(t1: ndarray, t1ce: ndarray, t2: ndarray, flair: ndarray, mask: ndarray) \
        -> tuple[ndarray, ndarray]:
    """
    Normalize the MRI data from the dataset using the z-score of the MRI data.

    :param t1: T1-Weighted MRI data.
    :param t1ce: T1-Weighted Contrast Enhanced MRI data.
    :param t2: T2-Weighted MRI data.
    :param flair: Flair MRI data.
    :param mask: Segmented mask data.
    :return: Stacked MRI data and segmented mask data.
    """
    t2 = t2[56:184, 56:184, 13:141]
    t2 = t2.reshape(-1, t2.shape[-1]).reshape(t2.shape)
    t2 = calc_z_score(t2)
    t1ce = t1ce[56:184, 56:184, 13:141]
    t1ce = t1ce.reshape(-1, t1ce.shape[-1]).reshape(t1ce.shape)
    t1ce = calc_z_score(t1ce)

    flair = flair[56:184, 56:184, 13:141]
    flair = flair.reshape(-1, flair.shape[-1]).reshape(flair.shape)
    flair = calc_z_score(flair)

    t1 = t1[56:184, 56:184, 13:141]
    t1 = t1.reshape(-1, t1.shape[-1]).reshape(t1.shape)
    t1 = calc_z_score(t1)

    mask = mask.astype(np.uint8)
    mask[mask == 4] = 3
    mask = mask[56:184, 56:184, 13:141]

    data = np.stack([flair, t1ce, t1, t2], axis=3)

    mask = change_mask_shape(mask)

    return data, mask


def get_mri_data_from_directory(patient_directory: str, t1: str, t1ce: str, t2: str, flair: str, mask: str) \
        -> type[ndarray, ndarray]:
    """
    Load MRI data from .nii files in a patient directory.

    :param patient_directory: parent patient directory.
    :param t1: t1 .nii file.
    :param t1ce: t1ce .nii file.
    :param t2: t2 .nii file.
    :param flair: flair .nii file.
    :param mask: mask .nii file.
    :return: normalized MRI data (stacked MRI data, mask data).
    """
    t1_data = nib.load(os.path.join(patient_directory, t1)).get_fdata()
    t1ce_data = nib.load(os.path.join(patient_directory, t1ce)).get_fdata()
    t2_data = nib.load(os.path.join(patient_directory, t2)).get_fdata()
    flair_data = nib.load(os.path.join(patient_directory, flair)).get_fdata()
    mask_data = nib.load(os.path.join(patient_directory, mask)).get_fdata()

    return normalize_mri_data(t1_data, t1ce_data, t2_data, flair_data, mask_data)


def roi_crop(img: ndarray, mask: ndarray, model) -> tuple[ndarray, ndarray]:
    """
    Crop the image and mask using the binary mask model.

    :param img: image to crop.
    :param mask: mask to crop.
    :param model: model to create the binary mask from the image.
    :return: cropped image and mask.
    """
    img_input = np.expand_dims(img, axis=0)

    binary_mask = model.predict(img_input)
    binary_mask = binary_mask[0, :, :, :, 0]
    binary_mask = np.expand_dims(binary_mask, -1)
    loc = np.where(binary_mask == 1)

    thresh = 12
    a = max(0, np.amin(loc[0]) - thresh)
    b = min(128, np.amax(loc[0]) + thresh)
    c = max(0, np.amin(loc[1]) - thresh)
    d = min(128, np.amax(loc[1]) + thresh)

    while abs(b - a) < 48:
        a = max(0, a - 1)
        b = min(128, b + 1)

    while abs(d - c) < 48:
        c = max(0, c - 1)
        d = min(128, d + 1)

    img1 = np.concatenate((img[a:b, c:d], binary_mask[a:b, c:d]), axis=-1)
    return img1, mask[a:b, c:d]


def global_extraction(input_images: ndarray | list[ndarray], input_masks: ndarray | list[ndarray]) -> tuple[ndarray, ndarray]:
    """
    Crops the image to 48 x 48 x 128 x C.

    :param input_images:
    :param input_masks:
    """
    images = []
    masks = []
    for img, mask in zip(input_images, input_masks):
        r = np.random.randint(0, img.shape[0] - 47)
        c = np.random.randint(0, img.shape[1] - 47)

        img_temp = img[r:r + 48, c:c + 48, :, :]
        mask_temp = mask[r:r + 48, c:c + 48, :, :]

        images.append(img_temp)
        masks.append(mask_temp)

    stacked_images = np.stack(images, axis=0)
    stacked_masks = np.stack(masks, axis=0)
    return stacked_images, stacked_masks


def mask_to_binary_mask(mask: ndarray) -> ndarray:
    """
    Convert the mask to a binary mask by checking the contents of the voxel.

    :param mask: mask to convert.
    :return: a binary version of the mask.
    """
    new_mask = np.zeros(mask.shape[:-1])
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            for k in range(mask.shape[2]):
                if mask[i][j][k][0] != 1:
                    new_mask[i][j][k] = 1

    return new_mask


def decompress_patient_folders(patient_folders_directory: str, delete_archives=True) -> None:
    """
    Decompress all the .gz files in a patient folder.

    :param patient_folders_directory: patients directory.
    :param delete_archives: whether to delete the residual archive file.
    """
    if not os.path.isdir(patient_folders_directory):
        raise FileNotFoundError("The patient folders directory is not a valid directory")

    for patient_folder_path in tqdm(os.listdir(patient_folders_directory)):
        patient_folder_path = os.path.join(patient_folders_directory, patient_folder_path)
        if not os.path.isdir(patient_folder_path):
            continue

        os.chdir(patient_folder_path)
        for mri_file in os.listdir(patient_folder_path):
            if not mri_file.endswith(".gz"):
                continue

            archive_path = os.path.abspath(mri_file)
            mri_file_name = os.path.basename(mri_file).rsplit('.', 1)[0]

            with gzip.open(archive_path, "rb") as archive_file, open(mri_file_name, "wb") as output_mri_file:
                shutil.copyfileobj(archive_file, output_mri_file)

            if delete_archives:
                # Remove the old compressed file
                os.remove(archive_path)


def random_train_test_split(data: list, train_split=0.8) -> tuple[list, list]:
    """
    Create a Test-Train split by randomly shuffling data then splitting by a provided value.

    :param data: data to split.
    :param train_split: percent of data to allocate as train data.
    :return: the data split into train and val data.
    """
    if train_split >= 1:
        raise ValueError("Train Split has to be less than 1")

    np.random.shuffle(data)
    data_len = len(data)

    return data[:int(data_len * train_split)], data[int(data_len * train_split):]


def create_split_from_lists(data: list, train_files_list_path: str, val_files_list_path: str) -> tuple[list, list]:
    """
    Create a Test-Train from lists of train and val images.

    :param data: data to split.
    :param train_files_list_path: text file path with train files.
    :param val_files_list_path:  text file path with val files.
    :return: the data split into train and val data.
    """
    train_data = []
    with open(train_files_list_path, "r") as train_files_list:
        for file_name in train_files_list:
            patient_index = file_name.strip()[-5:]
            # Conduct a linear search over the data to find one with the same id. Remove it from list as there is guaranteed to be only one instance.
            for i in range(len(data)):
                if data[i][0] == patient_index:
                    train_data.append(data.pop(i))
                    break
            else:
                raise ValueError(f"Data didn't contain a patient with id: {patient_index}")

    val_data = []
    with open(val_files_list_path, "r") as val_files_list:
        for file_name in val_files_list:
            patient_index = file_name.strip()[-5:]
            # Conduct a linear search over the data to find one with the same id. Remove it from list as there is guaranteed to be only one instance.
            for i in range(len(data)):
                if data[i][0] == patient_index:
                    val_data.append(data.pop(i))
                    break
            else:
                raise ValueError(f"Data didn't contain a patient with id: {patient_index}")

    return train_data, val_data


def create_dataset_from_patients_directory(patients_directory: str, output_dataset_directory: str) -> None:
    """
    Create uncropped dataset from the parent directory of all the patient directories.

    :param patients_directory: directory of all the patient directories.
    :param output_dataset_directory: path of where to save the dataset.
    """
    if not os.path.isdir(patients_directory):
        raise NotADirectoryError("The patients directory is not a valid directory")

    # All MRI data from .nii files as (patient_index, image_data, mask_data)
    all_mri_data = []
    print("Loading MRI Data")
    for patient_directory_name in tqdm(os.listdir(patients_directory)):
        patient_path = os.path.join(patients_directory, patient_directory_name)

        if os.path.isdir(patient_path):
            mri_data = {}
            for file_name in os.listdir(patient_path):
                if file_name.endswith(".gz"):
                    raise TypeError("The MRI modality files are still compressed, uncompress them first")

                if "_t1." in file_name:
                    mri_data["t1"] = file_name
                elif "_t1ce." in file_name:
                    mri_data["t1ce"] = file_name
                elif "_t2." in file_name:
                    mri_data["t2"] = file_name
                elif "_flair." in file_name:
                    mri_data["flair"] = file_name
                elif "_seg." in file_name:
                    mri_data["mask"] = file_name

            # Extract the patient's id from the directory name
            patient_index = patient_directory_name[-5:]
            image, mask = get_mri_data_from_directory(patient_path, **mri_data)
            all_mri_data.append((patient_index, image, mask))

    print("Shuffling MRI Data into Train (80%) and Validation (20%) splits")
    train_mri_data, val_mri_data = random_train_test_split(all_mri_data)

    for category, data in [("train", train_mri_data), ("val", val_mri_data)]:
        output_images_directory = os.path.join(output_dataset_directory, category, "images")
        output_masks_directory = os.path.join(output_dataset_directory, category, "masks")

        if not os.path.isdir(output_images_directory):
            os.makedirs(output_images_directory)

        if not os.path.isdir(output_masks_directory):
            os.makedirs(output_masks_directory)

        print(f"Saving {category.title()} MRI Data")
        for patient_index, image, mask in tqdm(data):
            np.save(os.path.join(output_images_directory, f"image-{patient_index}.npy"), image)
            np.save(os.path.join(output_masks_directory, f"mask-{patient_index}.npy"), mask)


def create_binary_dataset_from_dataset(input_dataset: str, output_dataset_directory: str) -> None:
    """
    Create the binary dataset from the uncropped dataset.

    :param input_dataset: path of the uncropped dataset.
    :param output_dataset_directory: path of where to save the binary dataset.
    """
    if not os.path.isdir(input_dataset):
        raise NotADirectoryError("The dataset directory is not a valid directory")

    for category in ["train", "val"]:
        input_images_directory = os.path.join(input_dataset, category, "images")
        input_masks_directory = os.path.join(input_dataset, category, "masks")

        output_images_directory = os.path.join(output_dataset_directory, category, "images")
        output_masks_directory = os.path.join(output_dataset_directory, category, "masks")

        if not os.path.isdir(output_masks_directory):
            os.makedirs(output_masks_directory)

        print(f"Copying images over for the {category} category")
        shutil.copytree(input_images_directory, output_images_directory)

        print(f"Converting masks to binary masks for the {category} category")
        for mask_path in tqdm(glob(os.path.join(input_masks_directory, "*.npy"))):
            mask = np.load(mask_path)

            binary_mask = mask_to_binary_mask(mask)

            np.save(os.path.join(output_masks_directory, os.path.basename(mask_path)), binary_mask)


def create_cropped_dataset_from_dataset(dataset_directory: str, model, output_dataset_directory: str) -> None:
    """
    Create the cropped dataset from the uncropped dataset using the binary model. The model should already have the weights loaded.

    :param dataset_directory: path of the uncropped dataset.
    :param model: the binary model with the weights already loaded.
    :param output_dataset_directory: path of where to save the cropped dataset.
    """
    if not os.path.isdir(dataset_directory):
        raise NotADirectoryError("The dataset directory is not a valid directory")

    for category in ["train", "val"]:
        all_images = glob(os.path.join(dataset_directory, category, 'images', "*.npy"))
        all_masks = glob(os.path.join(dataset_directory, category, 'masks', "*.npy"))

        if len(all_images) != len(all_masks):
            raise ValueError(f"There are not the same number of images and masks in the {category} category")

        output_images_directory = os.path.join(output_dataset_directory, category, "images")
        output_masks_directory = os.path.join(output_dataset_directory, category, "masks")

        if not os.path.isdir(output_images_directory):
            os.makedirs(output_images_directory)

        if not os.path.isdir(output_masks_directory):
            os.makedirs(output_masks_directory)

        print(f"Cropping and saving dataset for the {category} category")
        for img_path, mask_path in tqdm(zip(all_images, all_masks), total=len(all_images)):
            img_data = np.load(img_path)
            mask_data = np.load(mask_path)

            cropped_image, cropped_mask = roi_crop(img_data, mask_data, model)

            np.save(os.path.join(output_images_directory, os.path.basename(img_path)), cropped_image)
            np.save(os.path.join(output_masks_directory, os.path.basename(mask_path)), cropped_mask)
