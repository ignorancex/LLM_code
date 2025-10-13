import numpy as np
import nibabel as nib
import csv
from skimage.transform import resize
import torch


FULL_INDEX = ['29', '30', '32', '33', '35', '37', '40', '42', '43', '46', '52', '57', '59', '60', '61', '63', '65',
              '67', '68', '70', '72', '74', '76', '78', '80', '81', '82', '83']

DEFAULT_AAPM_CT_TRAIN_INDEX = ['29', '30', '32', '33', '35', '37', '40']
DEFAULT_AAPM_CT_VALID_INDEX = ['42']
DEFAULT_AAPM_CT_TEST_INDEX = ['43', '46']


def read_nii(data_path, csv_file, idx):
    """read spacings and image indices in CT-ORG"""

    assert idx in FULL_INDEX

    with open('{}/{}'.format(data_path, csv_file), 'rt') as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:

            if row['volume'] == idx:

                print("[aapm ct] loading volume {}".format(row['volume']))

                img = np.array(
                    nib.load('{}/volume-{}.nii.gz'.format(data_path, int(row['volume']) - 1)).get_data()).astype(
                    np.float32)
                slice_start = int(row['slice_start'])
                slice_end = int(row['slice_end'])

                img = img[:, :, slice_start - 1:slice_end - 1]
                img = np.transpose(img, [2, 0, 1])

                return img


def get_data_via_idx(
        volume_idx,
        output_shape: int = -1,
        root_path: str = '/opt/dataset/CT_ORG'
):

    x = None
    for idx in volume_idx:
        img = read_nii(root_path, 'CT_ORG_train.csv', idx)

        if x is None:
            x = img

        else:
            x = np.concatenate([x, img], axis=0)

    num_img = x.shape[0]

    if output_shape > 0:

        x_reshape = np.zeros(shape=[num_img, output_shape, output_shape], dtype=np.float32)
        for i in range(num_img):
            x_reshape[i] = resize(x[i], [output_shape, output_shape])

        x = x_reshape

    for i in range(num_img):
        x[i] = (x[i] - np.amin(x[i])) / (np.amax(x[i]) - np.amin(x[i]))

    return {
        'x': torch.from_numpy(x)
    }


def get_data(
        mode,
        output_shape,
        root_path,
):

    if mode == 'tra':
        volume_idx = DEFAULT_AAPM_CT_TRAIN_INDEX

    elif mode == 'val':
        volume_idx = DEFAULT_AAPM_CT_VALID_INDEX

    elif mode == 'tst':
        volume_idx = DEFAULT_AAPM_CT_TEST_INDEX

    else:
        raise ValueError()

    return get_data_via_idx(volume_idx, output_shape, root_path)
