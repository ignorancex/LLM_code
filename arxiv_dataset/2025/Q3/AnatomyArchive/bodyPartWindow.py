import warnings
import numpy as np
import pydicom
import segManager
from typing import Union


dict_window = {
    'Head': {
        'PosteriorFossa': {
            'Width': 150,
            'Level': 40
        },
        'Brain': {
            'Width': 100,
            'Level': 30
        },
        'TemporalBone': {
            'Width': 2800,
            'Level': 600
        }
    },
    'Chest': {
        'Mediastinum': {
            'Width': 350,
            'Level': 50
        },
        'Lung': {
            'Width': 1500,
            'Level': -600
        },
    },
    'Abdomen': {
        'SoftTissue': {
            'Width': 350,
            'Level': 50
        },
        'Liver': {
            'Width': 150,
            'Level': 30
        },
    },
    'Pelvis': {
        'SoftTissue': {
            'Width': 400,
            'Level': 50
        },
        'Bone': {
            'Width': 1800,
            'Level': 400
        }
    },
    'Spine': {
        'SoftTissue': {
            'Width': 250,
            'Level': 50
        },
        'Bone': {
            'Width': 1800,
            'Level': 400
        },
    }
}


triple_window_dict = {
    'Lung': {
            'Width': 1500,
            'Level': -600
    },
    'Bone': {
            'Width': 1800,
            'Level': 400
    },
    'SoftTissue': {
            'Width': 350,
            'Level': 50
    }
}


def get_window_dict_by_key(key):
    def gen_parent_dict_by_key(_dict_window, _key):
        body_part_names = dict_window.keys()
        if key in body_part_names:
            yield key
        else:
            for k, v in _dict_window.items():
                if k == _key:
                    yield k
                if isinstance(v, dict):
                    for result in gen_parent_dict_by_key(v, _key):
                        yield k

    dict_gen = gen_parent_dict_by_key(dict_window, key)
    body_part = next(dict_gen)
    new_window_dict = {body_part: dict_window[body_part]}
    # Add spline window settings if the body part belongs to Chest, Abdomen or Pelvis
    if body_part in ['Chest', 'Abdomen', 'Pelvis']:
        new_window_dict = new_window_dict | {'Spine': dict_window['Spine']}
    # Reconstruct the window setting dictionaries for simplicity
    body_window_dict_res = {}
    for k_part, v_part in new_window_dict.items():
        for k_tissue, v_tissue in v_part.items():
            body_window_dict_res.update({k_part + ' ' + k_tissue: [v_tissue['Width'], v_tissue['Level']]})
    return body_window_dict_res


def get_window_dict_from_dicom(dicom_attr):
    window_center = dicom_attr.WindowCenter
    window_width = dicom_attr.WindowWidth
    try:
        window_name = list(dicom_attr.WindowCenterWidthExplanation)
    except AttributeError:
        # noinspection PyUnresolvedReferences
        if isinstance(dicom_attr.WindowCenter, pydicom.multival.MultiValue):
            window_name = list(np.arange(len(window_center))+1)
            window_name = ['Window ' + str(el) for el in window_name]
        else:
            # noinspection PyUnresolvedReferences
            assert isinstance(dicom_attr.WindowCenter, pydicom.valuerep.DSfloat)
            window_name = 'Window 1'
    window_dict = {}
    if isinstance(window_name, list):
        for i in range(len(window_center)):
            window_dict.update({window_name[i]: [int(window_width[i]), int(window_center[i])]})
    else:
        window_dict = {window_name: [int(window_width), int(window_center)]}
    # Add window width and level for bones as a default setting
    window_dict.update({'Bone': [1800, 400]})
    return window_dict


def get_window_dict(dicom_attributes, import_window_ref=False):
    if isinstance(dicom_attributes, pydicom.dataset.FileDataset):
        if import_window_ref is True:
            # Adapt the body part naming stored in the DICOM files. Currently only customized solution is made for
            # 'lever' (which is the swedish word for 'liver').
            if 'lever' in dicom_attributes.BodyPartExamined:
                body_window_dict = get_window_dict_by_key('Liver')
            else:
                # For future implementation, local dictionary needs to be created for downstream process.
                raise ValueError('No downstream window settings for the examined body part specified!')
        else:
            # Load window settings from locally stored dicom attributes
            body_window_dict = get_window_dict_from_dicom(dicom_attributes)
    else:
        raise TypeError('Input variable has to be a pydicom.dataset.FileDataset type!')
    return body_window_dict


def get_window_min_max_from_dict(_dict):
    _min = _dict['Level'] - _dict['Width'] / 2
    _max = _dict['Level'] + _dict['Width'] / 2
    return _min, _max


# noinspection SpellCheckingInspection
def get_window_image(img: np.ndarray, window_width_level: Union[list, tuple, dict, None] = None,
                     dicom_rescale_attr=None, print_message=False, get_8bit=False, **kwargs) -> np.ndarray:
    """
    :param img: numpy array
    :param window_width_level: list or tuple of window_width, window_level or dict.
                               For dict, the current implementation is case-sensitive, and the keys should be named as
                               'Width' and 'Level'.
    :param dicom_rescale_attr: list or tuple of slope, intercept.
    :param print_message: print min and max settings.
    :param get_8bit: bool on whether to rescale the image to 8 bit.
    :param kwargs: window_min_max: [window_min, window_max] in case window_settings are not provided.
                   If neither window_width_level nor window_min_max is provided, load window settings for soft tissue
                   in abdomen.
    :return: img_new, nummpy array of the same type as the input.
    """
    # The method works for both scaled and non-scaled DICOM image using slope and intercept.
    # For non-scaled DICOM image, the 'dicom_rescale_attr' must be provided!
    # slope = dicom_rescale_attr[0] and intercept = dicom_rescale_attr[1]

    # The DICOM image is usually an uint12 array, with a minimum of 0, while negative values are common in
    # Hounsfield Units (HU) for CT images.
    # To match the units of the window settings, one way is to linearly transform the original pixel array
    # using y = slope * x + intercept.
    # deepcopy is used because otherwise the input image will also be modified as python follows the principle
    # of 'pass by object'.
    # Shallow copy cannot solve the problem as it will simply copy the object reference.
    # Alternative one can reload the corresponding dicom slice everytime when computing the pixel values to
    # avoid deepcopy.

    img_new = img.copy()
    if np.min(img_new) >= 0:
        if dicom_rescale_attr is None:
            raise NameError('DICOM rescale parameters, i.e., slope and rescale attributes not provided!!!')
        else:
            slope = dicom_rescale_attr[0]
            intercept = dicom_rescale_attr[1]
            img_new = img_new * slope + intercept  # Perform linear transformation of the original image
    if window_width_level is None:
        try:
            img_min, img_max = kwargs['window_min_max']
        except KeyError:
            warnings.warn("Neither 'window_width_level' nor 'window_min_max' is provided.\n"
                          "Load default window settings for soft tissue")
            img_min, img_max = get_window_min_max_from_dict(triple_window_dict['SoftTissue'])
    else:
        if isinstance(window_width_level, (list, tuple)):
            img_min = window_width_level[1] - window_width_level[0] / 2  # measured in HU
            img_max = window_width_level[1] + window_width_level[0] / 2  # measured in HU
        else:
            img_min, img_max = get_window_min_max_from_dict(window_width_level)

    img_new[img_new < img_min] = img_min
    img_new[img_new > img_max] = img_max
    if print_message:
        print('Window settings - min:', img_min, '; max: ', img_max)
    if get_8bit:
        # Image is rescaled to 8-bit
        img_new = ((img_new - img_min) / (img_max - img_min) * 255).round().astype('uint8')
    return img_new


# noinspection SpellCheckingInspection
def get_window_min_max_for_display(window_width_level, image_scaled: bool = True,
                                   dicom_rescale_attr: Union[tuple, list, None] = None):
    # The method works for both scaled and non-scaled DICOM image using slope and intercept.
    # If the DICOM image is already scaled, do not run the method with 'dicom_rescale_attr' otherwise the window
    # setting will be wrongly scaled.
    # slope = dicom_rescale_attr[0] and intercept = dicom_rescale_attr[1]

    # The non-scaled raw DICOM image is usually an uint16 array, with a minimum of 0, while negative values are
    # common in Hounsfield Units (HU) for CT images.
    # To match the units of the window settings, one way is to linearly transform the original pixel array
    # using y = slope * x + intercept.
    # Alternatively, one can transform the window min and max by using x = (y - intercept)/slope, so that
    # the image keeps its uint16 values.
    # This is preferred as no re-computation is needed for visualization such that neither reloading of the
    # DICOM file nor deepcopy is required.
    # This method is not suitable for window setting fusion as it only computes the value bounds for later
    # visualization in matplotlib.

    # slope = dicom_attr.RescaleSlope
    # intercept = dicom_attr.RescaleIntercept

    img_min = window_width_level[1] - window_width_level[0] / 2  # measured in HU
    img_max = window_width_level[1] + window_width_level[0] / 2  # measured in HU
    if not image_scaled:
        if dicom_rescale_attr is not None:
            slope = dicom_rescale_attr[0]
            intercept = dicom_rescale_attr[1]
            img_min = (img_min - intercept) / slope  # in uint16
            img_max = (img_max - intercept) / slope  # in uint16
        else:
            raise ValueError(f"For non-scaled array, 'dicom_rescale_attr' has to be provided each as tuple or list!")
    print('Min:', img_min, '; Max: ', img_max)
    return img_min, img_max


def get_window_min_max_by_name(_obj_name: str, image_scaled: bool = True,
                               dicom_rescale_attr: Union[tuple, list, None] = None):
    list_bones = segManager.get_preset_anatomy_groups()['bone']
    _obj_name = segManager.anatomy_name_standardization(_obj_name)
    if 'lung' in _obj_name:
        _dict_window = triple_window_dict['Lung']
    elif len([e for e in list_bones if e in _obj_name]) != 0:
        _dict_window = triple_window_dict['Bone']
    else:
        _dict_window = triple_window_dict['SoftTissue']

