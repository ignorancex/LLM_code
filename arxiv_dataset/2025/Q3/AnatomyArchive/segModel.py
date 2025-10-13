import os
import re
import time
import shutil
import subprocess
import warnings
import cv2 as cv
import numpy as np
import nibabel as nib
import matplotlib
import matplotlib.pyplot as plt
from scipy import ndimage
from tqdm import tqdm
from pathlib import Path
from scipy import ndimage as ndi
from typing import Union, List
from numpy.typing import NDArray
from skimage import filters
from skimage.segmentation import watershed
from skimage.filters import threshold_multiotsu
from skimage.measure import regionprops
from totalsegmentator import nnunet
from genericImageIO import dicom_to_nifti
from totalsegmentator import postprocessing
from totalsegmentator.resampling import resample_img
from totalsegmentator.map_to_binary import class_map
from genericImageIO import convert_nifti_to_numpy
from util import print_highlighted_text, normalized_image_to_8bit
from skimage.morphology import disk, remove_small_objects, skeletonize


v_totalsegmentator = None
v_str = None
matplotlib.use('QtAgg')


# noinspection SpellCheckingInspection
def get_segmentator_version():
    global v_totalsegmentator
    global v_str
    if v_totalsegmentator is None:
        from totalsegmentator.config import setup_nnunet
        setup_nnunet()
        import importlib
        # noinspection PyUnresolvedReferences
        v_str = importlib.metadata.distribution('totalsegmentator').version
        print(f'Installed TotalSegmentator version: {v_str}')
        v_totalsegmentator = int(v_str.split('.')[0])


get_segmentator_version()


# noinspection SpellCheckingInspection
segmentation_settings = {
    'body': {
        'coarse': {
            'task_id_v1': 269,
            'task_id_v2': 300,
            'task_name': 'body',
            'trainer': 'default',
            'voxel_size': 6.0,
            'crop': None,
        },
        'fine': {
            'task_id_v1': 273,
            'task_id_v2': 299,
            'task_name': 'body',
            'trainer': 'default',
            'voxel_size': 1.5,
            'crop': None
        }
    },
    'total': {
        'coarse': {
            'task_id_v1': 256,
            'task_id_v2': 297,
            'task_name': 'total',
            'trainer_v1': 'nnUNetTrainerV2_ep8000_nomirror',
            'trainer_v2': 'nnUNetTrainer_4000epochs_NoMirroring',
            'voxel_size': 3.0,
            'crop': None,
        },
        'fine': {
            'task_id_v1': [251, 252, 253, 254, 255],
            'task_id_v2': [291, 292, 293, 294, 295],
            'task_name': 'total',
            'trainer_v1': 'nnUNetTrainerV2_ep4000_nomirror',
            'trainer_v2': 'nnUNetTrainerNoMirroring',
            'voxel_size': 1.5,
            'crop': None
        }
    },
    'total_mr': {
        'coarse': {
            'task_id': 732,
            'task_name': 'total_mr',
            'trainer': 'nnUNetTrainer_DASegOrd0_NoMirroring',
            'voxel_size': 3.0
        },
        'fine': {
            'task_id': [730, 731],
            'task_name': 'total_mr',
            'trainer': 'nnUNetTrainer_DASegOrd0_NoMirroring',
            'voxel_size': 1.5
        }
    },
    'liver_components': {
        'task_id': 8,
        'task_name': 'liver_vessels',
        'trainer': 'default',
        'voxel_size': None,
        'crop': 'liver',
        'crop_addon': (20, 20, 20)
    },
    'lung_components': {
        'task_id': 258,
        'task_name': 'lung_vessels',
        'trainer': 'default',
        'voxel_size': None,
        'crop_v1': 'lung',
        'crop_v2': ['lung_upper_lobe_left', 'lung_lower_lobe_left', 'lung_upper_lobe_right',
                    'lung_middle_lobe_right', 'lung_lower_lobe_right']
    },
    'kidney_cysts': {
        'task_id_v2': 789,
        'task_name': 'kidney_cysts',
        'trainer': 'nnUNetTrainer_DASegOrd0_NoMirroring',
        'voxel_size': 1.5,
        'crop': ['kidney_left', 'kidney_right', 'liver', 'spleen', 'colon'],
        'crop_addon': (10, 10, 10),
        'remove_auxiliary': False
    },
    'heartchambers_highres': {
        'task_id_v2': 301,
        'task_name': 'heartchambers_highres',
        'trainer': 'default',
        'voxel_size': None,
        'crop': ['heart'],
        'crop_addon': (5, 5, 5)
    },
    'vertebrae_body': {
        'task_name': 'vertebrae_body',
        'task_id_v2': 302,
        'trainer': 'default',
        'voxel_size': 1.5
    },
    'tissue_types': {
        'task_name': 'tissue_types',
        'task_id_v1': 278,
        'task_id_v2': 481,
        'trainer_v1': 'nnUNetTrainerV2_ep4000_nomirror',
        'trainer_v2': 'default',
        'voxel_size': 1.5,
        'crop': None
    },
    'tissue_types_mr': {
        'task_name': 'tissue_types_mr',
        'task_id': 734,
        'trainer': 'nnUNetTrainer_DASegOrd0_NoMirroring',
        'voxel_size': 1.5
    },
    'appendicular_bones': {
        'task_name': 'appendicular_bones',
        'task_id_v2': 304,
        'trainer': 'nnUNetTrainerNoMirroring',
        'voxel_size': 1.5,
        'remove_auxiliary': False
    }
}


# noinspection SpellCheckingInspection
def image_resample(data: Union[np.ndarray, nib.nifti1.Nifti1Image], voxel_size: Union[tuple, np.ndarray, None] = None,
                   target_voxel_size: Union[float, tuple, None] = None, order=0, remove_negative=False):

    if isinstance(data, np.ndarray):
        assert voxel_size is not None
        is_array = True
        _dtype = data.dtype
        if isinstance(voxel_size, tuple):
            voxel_size = np.array(voxel_size)
    else:
        is_array = False
        affine_new = np.copy(data.affine)
        _dtype = data.get_data_dtype()
        data, voxel_size = convert_nifti_to_numpy(data)

    if target_voxel_size is None:
        # Take voxel size on x axis
        voxel_size_new = np.ones(3) * voxel_size[0]
    elif isinstance(target_voxel_size, float):
        voxel_size_new = np.ones(3) * target_voxel_size
    else:
        voxel_size_new = np.array(target_voxel_size)
    zoom = voxel_size / voxel_size_new
    data_new = resample_img(data, zoom=zoom, order=order)
    if remove_negative:
        data_new[data_new < 1e-4] = 0
    if is_array is False:
        # noinspection PyUnboundLocalVariable
        affine_new[:3, 0] = affine_new[:3, 0] / zoom[0]
        affine_new[:3, 1] = affine_new[:3, 1] / zoom[1]
        affine_new[:3, 2] = affine_new[:3, 2] / zoom[2]
        return nib.Nifti1Image(data_new.astype(_dtype), affine_new)
    else:
        return data_new.astype(_dtype)


def modify_task_name(task_name: str):
    if '_' in task_name:
        list_strs = task_name.split('_')
        return list_strs[0] + ''.join(_sub.capitalize() for _sub in list_strs[1:])
    else:
        return task_name


def set_seg_file_name(file_in: str, seg_config: dict):
    task_name_s = modify_task_name(seg_config['task_name'])
    file_out = f"{file_in.split('.')[0]}_{task_name_s}_seg_v{v_totalsegmentator}.nii.gz"
    return file_out


# noinspection SpellCheckingInspection
def get_seg_config_by_task_name(task_name: str = 'total', coarse: Union[bool, None] = None):
    def _get_default_trainer():
        _default_trainer = 'nnUNetTrainer'
        if v_tt == 1:
            return f'{_default_trainer}V2'
        else:
            return _default_trainer

    # noinspection SpellCheckingInspection
    def _get_v_dependent_setting(_str):
        if _str not in seg_config_new.keys():
            _versions = [int(re.search('[vV](\d)', _k).group(1)) for _k in seg_config_new.keys() if _str in _k and
                         re.search('[vV](\d)', _k)]
            if len(_versions) > 0:
                # Either there exist task ids for version 1 and 2 or only for version 2.
                if v_tt in _versions:
                    seg_config_new[_str] = seg_config_new[f'{_str}_v{v_tt}']
                    for _v in _versions:
                        seg_config_new.pop(f'{_str}_v{_v}')
                else:
                    # Task id for version 2 exists but the current installed version is 1.
                    raise ValueError(error_mismatch)
        if _str in seg_config_new.keys() and seg_config_new[_str] == 'default':
            seg_config_new[_str] = _get_default_trainer()

    v_tt = v_totalsegmentator
    if 'coarse' in segmentation_settings[task_name].keys():
        if coarse is not True:
            seg_config = segmentation_settings[task_name]['fine']
        else:
            seg_config = segmentation_settings[task_name]['coarse']
    else:
        if v_tt == 1 and (task_name == 'appendicular_bones' or 'tissue' in task_name):
            # For version 1, the model for tissue types can generate masks for appendicular bones as well.
            task_name = 'tissue_types'
        seg_config = segmentation_settings[task_name]
    seg_config_new = seg_config.copy()
    error_mismatch = f"ERROR: The task {seg_config_new['task_name']} can only run using TotalSegmentator of " \
                     f"version 2 but {v_str} is installed!"
    if 'mr' in seg_config_new['task_name']:
        if v_tt == 1:
            raise ValueError(error_mismatch)
    _get_v_dependent_setting('trainer')
    _get_v_dependent_setting('task_id')
    _get_v_dependent_setting('crop')
    return seg_config_new


def get_anatomy_synonym():
    synonym_dict = dict()
    synonym_dict['hip'] = 'pelvic'
    synonym_dict['autochthon'] = 'spinal_erectors'
    return synonym_dict


def standardize_anatomy_names_in_cls_map(_cls_map: dict):
    synonym_dict = get_anatomy_synonym()
    anatomy_names = [v.replace(v.split('_')[0], synonym_dict[v.split('_')[0]])
                     if v.split('_')[0] in synonym_dict.keys() else v for v in _cls_map.values()]
    return dict(zip(_cls_map.keys(), anatomy_names))


def anatomy_name_standardization(anatomy_types: Union[dict, List[str], str]) -> List[str]:
    synonym_dict = get_anatomy_synonym()
    # Rename some tissues/organs by their synonyms
    if isinstance(anatomy_types, dict):
        list_of_anatomies = [v.replace(v.split('_')[0], synonym_dict[v.split('_')[0]])
                             if v.split('_')[0] in synonym_dict.keys() else v for v in anatomy_types.values()]
    elif isinstance(anatomy_types, list):
        list_of_anatomies = [v.replace(v.split('_')[0], synonym_dict[v.split('_')[0]])
                             if v.split('_')[0] in synonym_dict.keys() else v for v in anatomy_types]
    else:
        list_of_anatomies = anatomy_types.replace(anatomy_types.split('_')[0],
                                                  synonym_dict[anatomy_types.split('_')[0]]) if \
                                                  anatomy_types.split('_')[0] in synonym_dict.keys() else anatomy_types
    return list_of_anatomies


# noinspection SpellCheckingInspection
def get_v_dependent_cls_map(task_name: str, full_name: bool = True, return_version: bool = False):
    v_tt = v_totalsegmentator
    if task_name not in class_map.keys():
        if full_name:
            if v_tt == 1 and task_name == 'appendicular_bones':
                task_name_new = 'bones_tissue_test'
                warnings.warn(f"The class map for TotalSegmentator of version {v_str} that is similar to"
                              f" to '{task_name}' is '{task_name_new}'.")
                cls_map = class_map[task_name_new]
            else:
                raise ValueError(f"Installed TotalSegmentator version: {v_str}\nMake sure class map name {task_name}"
                                 f" is defined in the map_to_binary.py file!")
        else:
            _cls_map_name = [_k for _k in class_map.keys() if task_name in _k]
            if len(_cls_map_name) == 0:
                raise ValueError(f"Installed TotalSegmentator version: {v_str}\nMake sure class map name containing"
                                 f" keyword{task_name} is defined in the map_to_binary.py file!")
            elif len(_cls_map_name) > 1:
                raise ValueError(f"Installed TotalSegmentator version: {v_str}\nMake sure class map name containing"
                                 f" keyword{task_name} is unique in the map_to_binary.py file!")
            else:
                cls_map = class_map[_cls_map_name[0]]
    else:
        if v_tt == 2 and f'{task_name}_auxiliary' in class_map.keys():
            class_map_aux = class_map[f'{task_name}_auxiliary']
            class_map[task_name].update(class_map_aux)
            cls_map = class_map[task_name]
        else:
            cls_map = class_map[task_name]
    cls_map = standardize_anatomy_names_in_cls_map(cls_map)
    if not return_version:
        return cls_map
    else:
        return cls_map, v_tt


def check_mask_file_and_load(file_in: str, seg_config: dict):
    file_mask = set_seg_file_name(file_in, seg_config)
    if not os.path.isfile(file_mask):
        mask = perform_segmentation_generic(file_in, seg_config, return_numpy=True)
    else:
        # noinspection PyTypeChecker
        mask, *_ = convert_nifti_to_numpy(nib.load(file_mask))
    return mask


# noinspection SpellCheckingInspection
def validate_nifti_image_dtype(image: nib.Nifti1Image, return_dtype: bool = False):
    img_dtype = image.get_data_dtype()
    if img_dtype.fields is not None:
        raise TypeError(f"Invalid dtype {img_dtype}. Expected a simple dtype, not a structured one.")
    if return_dtype:
        return img_dtype


# noinspection SpellCheckingInspection
def nnUNet_predict_image(file_in: Union[str, Path, nib.Nifti1Image], file_out: Union[str, Path, None], task_id,
                         model="3d_fullres", folds=None, trainer="nnUNetTrainerV2", tta=False,
                         multilabel_image=True, resample=None, crop=None, crop_path=None, task_name="total",
                         nora_tag: Union[str, None] = None, preview=False, nr_threads_resampling=1,
                         nr_threads_saving=6, force_split=False, crop_addon=(3, 3, 3), roi_subset=None,
                         output_type="nifti", quiet=False, verbose=False, test=0, skip_saving=False, device="cuda",
                         no_derived_masks=False, v1_order=False, remove_auxiliary: Union[bool, None] = None,
                         remove_small_blobs=False, save_label_to_nifti: Union[bool, None] = None):

    """
    crop: string or a nibabel image
    resample: None or float (target spacing for all dimensions) or list of floats
    """

    def _save_and_return_result(_img_out: nib.Nifti1Image, _label_map: dict):
        if _img_out.get_fdata().sum() == 0:
            if not quiet:
                print("INFO: Crop is empty. Returning empty segmentation.")
            return -1
        else:
            if file_out is not None and skip_saving is False:
                if not quiet:
                    print("Saving segmentations...")
                if roi_subset is not None:
                    _label_map = {k: v for k, v in _label_map.items() if v in roi_subset}
                _t0 = time.time()
                if output_type == "dicom":
                    file_out.mkdir(exist_ok=True, parents=True)
                    from totalsegmentator.dicom_io import save_mask_as_rtstruct
                    save_mask_as_rtstruct(img_data, _label_map, file_in_dcm, file_out / "segmentations.dcm")
                else:
                    file_out.parent.mkdir(exist_ok=True, parents=True)
                    nib.save(_img_out, file_out)
                    if os.name != 'nt':
                        if isinstance(nora_tag, str) and nora_tag != "None":
                            subprocess.call(f"/opt/nora/src/node/nora -p {nora_tag} --add {file_out} --addtag atlas",
                                            shell=True)
                if not quiet:
                    print(f"  Saved in {time.time() - _t0:.2f}s")
            return _img_out

    def _postprocess_multilabel(data: np.ndarray, _label_map: dict, rois: List[str], func_name: str, interval=None):
        """
        Keep the largest blob for the classes defined in rois.

        data: multilabel image (np.array)
        class_map: class map {label_idx: label_name}
        rois: list of labels where to filter for the largest blob

        return multilabel image (np.array)
        """
        assert func_name in ('remove_small_blobs', 'keep_largest_blob')
        _label_map_inv = {v: k for k, v in _label_map.items()}
        for roi in tqdm(rois, disable=quiet):
            _label = _label_map_inv[roi]
            data_roi = data == _label
            if func_name == 'remove_small_blobs':
                try:
                    assert interval is not None
                except AssertionError:
                    NameError("ERROR: To remove small blobs, size interval should be supplied!")
                cleaned_roi = postprocessing.remove_small_blobs(data_roi, interval=interval) > 0.5
            else:
                cleaned_roi = postprocessing.keep_largest_blob(data_roi) > 0.5
            data[data_roi] = 0  # Clear the original ROI in data
            data[cleaned_roi] = _label  # Write back the cleaned ROI into data
        return data

    if isinstance(file_in, (str, Path)):
        if isinstance(file_in, Path):
            str_file_in = str(file_in)
        else:
            str_file_in = file_in
            file_in = Path(file_in)
        if str_file_in.endswith(".nii") or str_file_in.endswith(".nii.gz"):
            img_type = "nifti"
        else:
            img_type = "dicom"
        if not file_in.exists():
            raise FileNotFoundError
    else:
        if isinstance(file_in, nib.Nifti1Image):
            img_type = "nifti"
        else:
            raise ValueError("Unsupported image format.")
    multimodel = type(task_id) is list
    if output_type == "dicom" and img_type != "dicom":
        raise ValueError("To use output type dicom you also have to use a Dicom image as input.")

    if type(resample) is float:
        resample = [resample, resample, resample]
    v_tt = v_totalsegmentator
    with nnunet.tempfile.TemporaryDirectory(prefix="nnunet_tmp_") as tmp_folder:
        tmp_dir = Path(tmp_folder)
        if verbose:
            print(f"tmp_dir: {tmp_dir}")
        if img_type == "dicom":
            if not quiet:
                print("Converting dicom to nifti...")
            (tmp_dir / "dcm").mkdir()  # make subdir otherwise this file would be included by nnUNet_predict
            dicom_to_nifti(file_in, tmp_dir / "dcm" / "converted_dcm.nii.gz")
            file_in_dcm = file_in
            file_in = tmp_dir / "dcm" / "converted_dcm.nii.gz"
            if not quiet:
                # noinspection PyUnresolvedReferences
                print(f"  found image with shape {nib.load(file_in).shape}")
        if isinstance(file_in, nib.Nifti1Image):
            img_in_orig = file_in
        else:
            img_in_orig = nib.load(file_in)
        if len(img_in_orig.shape) == 2:
            raise ValueError("TotalSegmentator does not work for 2D images. Use a 3D image.")
        if len(img_in_orig.shape) > 3:
            print(f"WARNING: Input image has {len(img_in_orig.shape)} dimensions. Only using first three dimensions.")
            img_in_orig = nib.Nifti1Image(img_in_orig.get_fdata()[:, :, :, 0], img_in_orig.affine)
        validate_nifti_image_dtype(img_in_orig)
        # takes ~0.9s for medium image
        img_in = nib.Nifti1Image(img_in_orig.get_fdata(), img_in_orig.affine)  # copy img_in_orig
        if crop is not None:
            if type(crop) is str:
                if v_tt == 1:
                    if crop == "lung" or crop == "pelvis" or crop == "heart":
                        nnunet.combine_masks(crop_path, crop_path / f"{crop}.nii.gz", crop)
                else:
                    if crop == "lung" or crop == "pelvis":
                        # noinspection PyArgumentList
                        nnunet.combine_masks(crop_path, crop)
                crop_mask_img = nib.load(crop_path / f"{crop}.nii.gz")
            else:
                crop_mask_img = crop
            img_in, bbox = nnunet.crop_to_mask(img_in, crop_mask_img,
                                               addon=crop_addon, dtype=np.int32,
                                               verbose=verbose)
            if not quiet:
                print(f"  cropping from {crop_mask_img.shape} to {img_in.shape}")

        img_in = nnunet.as_closest_canonical(img_in)
        if resample is not None:
            if not quiet:
                print("Resampling...")
            st = time.time()
            img_in_shape = img_in.shape
            img_in_rsp = nnunet.change_spacing(img_in, resample,
                                               order=3, dtype=np.int32,
                                               nr_cpus=nr_threads_resampling)  # 4 cpus instead of 1 makes it slower
            if verbose:
                print(f"  from shape {img_in.shape} to shape {img_in_rsp.shape}")
            if not quiet:
                print(f"  Resampled in {time.time() - st:.2f}s")
        else:
            img_in_rsp = img_in
        nib.save(img_in_rsp, tmp_dir / "s01_0000.nii.gz")

        # todo important: change
        nr_voxels_thr = 512 * 512 * 900
        img_parts = ["s01"]
        ss = img_in_rsp.shape
        # If image to big then split into 3 parts along z axis. Also make sure that z-axis is at least 200px otherwise
        # splitting along it does not really make sense.
        do_triple_split = np.prod(ss) > nr_voxels_thr and ss[2] > 200 and multimodel
        if force_split:
            do_triple_split = True
        if do_triple_split:
            if not quiet:
                print("Splitting into subparts...")
            img_parts = ["s01", "s02", "s03"]
            third = img_in_rsp.shape[2] // 3
            margin = 20  # set margin with fixed values to avoid rounding problem if using percentage of third
            img_in_rsp_data = img_in_rsp.get_fdata()
            nib.save(nib.Nifti1Image(img_in_rsp_data[:, :, :third + margin], img_in_rsp.affine),
                     tmp_dir / "s01_0000.nii.gz")
            nib.save(nib.Nifti1Image(img_in_rsp_data[:, :, third + 1 - margin:third * 2 + margin], img_in_rsp.affine),
                     tmp_dir / "s02_0000.nii.gz")
            nib.save(nib.Nifti1Image(img_in_rsp_data[:, :, third * 2 + 1 - margin:], img_in_rsp.affine),
                     tmp_dir / "s03_0000.nii.gz")

        if v_tt == 2:
            if task_name == "total" and resample is not None and resample[0] < 3.0:
                step_size = 0.8
            else:
                step_size = 0.5
        else:
            step_size = None
        st = time.time()
        if multimodel:  # if running multiple models
            if v_tt == 1:
                class_map_parts = nnunet.class_map_5_parts
                map_taskid_to_partname = nnunet.map_taskid_to_partname
            else:
                if task_name == "total":
                    class_map_parts = nnunet.class_map_5_parts
                    # noinspection PyUnresolvedReferences
                    map_taskid_to_partname = nnunet.map_taskid_to_partname_ct
                elif task_name == "total_mr":
                    # noinspection PyUnresolvedReferences
                    class_map_parts = nnunet.class_map_parts_mr
                    # noinspection PyUnresolvedReferences
                    map_taskid_to_partname = nnunet.map_taskid_to_partname_mr
                elif task_name == "headneck_muscles":
                    # noinspection PyUnresolvedReferences
                    class_map_parts = nnunet.class_map_parts_headneck_muscles
                    # noinspection PyUnresolvedReferences
                    map_taskid_to_partname = nnunet.map_taskid_to_partname_headneck_muscles
            # only compute model parts containing the roi subset
            if roi_subset is not None:
                part_names = []
                new_task_id = []
                for part_name, part_map in class_map_parts.items():
                    if any(organ in roi_subset for organ in part_map.values()):
                        # get taskid associated to model part_name
                        map_partname_to_taskid = {v: k for k, v in map_taskid_to_partname.items()}
                        new_task_id.append(map_partname_to_taskid[part_name])
                        part_names.append(part_name)
                task_id = new_task_id
                if verbose:
                    print(f"Computing parts: {part_names} based on the provided roi_subset")

            if test == 0:
                class_map_inv = {v: k for k, v in class_map[task_name].items()}
                (tmp_dir / "parts").mkdir(exist_ok=True)
                seg_combined = {}
                # iterate over subparts of image
                for img_part in img_parts:
                    # noinspection PyUnresolvedReferences
                    img_shape = nib.load(tmp_dir / f"{img_part}_0000.nii.gz").shape
                    seg_combined[img_part] = np.zeros(img_shape, dtype=np.uint8)
                # Run several tasks and combine results into one segmentation
                for idx, tid in enumerate(task_id):
                    if not quiet: print(f"Predicting part {idx +1} of {len(task_id)} ...")
                    with nnunet.nostdout(verbose):
                        if v_tt == 1:
                            nnunet.nnUNet_predict(tmp_dir, tmp_dir, tid, model, folds, trainer, tta,
                                                  nr_threads_resampling, nr_threads_saving)
                        else:
                            # noinspection PyUnresolvedReferences
                            nnunet.nnUNetv2_predict(tmp_dir, tmp_dir, tid, model, folds, trainer, tta,
                                                    nr_threads_resampling, nr_threads_saving,
                                                    device=device, quiet=quiet, step_size=step_size)
                    # iterate over models (different sets of classes)
                    for img_part in img_parts:
                        (tmp_dir / f"{img_part}.nii.gz").rename(tmp_dir / "parts" / f"{img_part}_{tid}.nii.gz")
                        # noinspection PyUnresolvedReferences
                        seg = nib.load(tmp_dir / "parts" / f"{img_part}_{tid}.nii.gz").get_fdata()
                        for jdx, class_name in class_map_parts[map_taskid_to_partname[tid]].items():
                            seg_combined[img_part][seg == jdx] = class_map_inv[class_name]
                # iterate over subparts of image
                for img_part in img_parts:
                    nib.save(nib.Nifti1Image(seg_combined[img_part], img_in_rsp.affine), tmp_dir / f"{img_part}.nii.gz")
            elif test == 1:
                print("WARNING: Using reference seg instead of prediction for testing.")
                shutil.copy(Path("tests") / "reference_files" / "example_seg.nii.gz", tmp_dir / "s01.nii.gz")
        else:
            if not quiet:
                print("Predicting...")
            if test == 0:
                with nnunet.nostdout(verbose):
                    if v_tt == 1:
                        nnunet.nnUNet_predict(tmp_dir, tmp_dir, task_id, model, folds, trainer, tta,
                                              nr_threads_resampling, nr_threads_saving)
                    else:
                        # noinspection PyUnresolvedReferences
                        nnunet.nnUNetv2_predict(tmp_dir, tmp_dir, task_id, model,
                                                folds, trainer, tta,
                                                nr_threads_resampling,
                                                nr_threads_saving,
                                                device=device, quiet=quiet,
                                                step_size=step_size)
            elif test == 3:
                print("WARNING: Using reference seg instead of prediction for testing.")
                shutil.copy(Path("tests") / "reference_files" / "example_seg_lung_vessels.nii.gz", tmp_dir / "s01.nii.gz")
        if not quiet:
            print(f"Predicted in {time.time() - st:.2f}s")

        # Combine image subparts back to one image
        if do_triple_split:
            combined_img = np.zeros(img_in_rsp.shape, dtype=np.uint8)
            # noinspection PyUnresolvedReferences
            combined_img[:, :, :third] = nib.load(tmp_dir / "s01.nii.gz").get_fdata()[:, :, :-margin]
            # noinspection PyUnresolvedReferences
            combined_img[:, :, third:third * 2] = nib.load(tmp_dir / "s02.nii.gz").get_fdata()[:, :, margin - 1:-margin]
            # noinspection PyUnresolvedReferences
            combined_img[:, :, third * 2:] = nib.load(tmp_dir / "s03.nii.gz").get_fdata()[:, :, margin - 1:]
            nib.save(nib.Nifti1Image(combined_img, img_in_rsp.affine), tmp_dir / "s01.nii.gz")

        img_pred = nib.load(tmp_dir / "s01.nii.gz")
        if v_tt == 2 and remove_auxiliary is True and f'{task_name}_auxiliary' in class_map.keys():
            # noinspection PyUnresolvedReferences
            img_pred = nnunet.remove_auxiliary_labels(img_pred, task_name)
        img_data = img_pred.get_fdata().astype(np.uint8)

        # Reorder labels if needed
        if v1_order and task_name == "total" and v_tt == 2:
            # noinspection PyUnresolvedReferences
            img_data = nnunet.reorder_multilabel_like_v1(img_data, class_map["total"], class_map["total_v1"])
            label_map = class_map["total_v1"]
        else:
            label_map = class_map[task_name]

        # Keep only voxel values corresponding to the roi_subset
        if roi_subset is not None:
            label_map = {k: v for k, v in label_map.items() if v in roi_subset}
            img_data *= np.isin(img_data, list(label_map.keys()))
        # Postprocessing multilabel (run here on lower resolution)
        if task_name == 'body':
            img_pred_pp = _postprocess_multilabel(img_data, label_map, ["body_trunc"], 'keep_largest_blob')
            img_pred = nib.Nifti1Image(img_pred_pp, img_pred.affine)
        if remove_small_blobs:
            # noinspection PyUnresolvedReferences
            vox_vol = np.prod(img_pred.header.get_zooms())
            if task_name == 'body':
                size_thr_mm3 = 50000 / vox_vol
                list_objs = ["body_extremities"]
            else:
                size_thr_mm3 = 200
                list_objs = list(label_map.values())
            img_pred_pp = _postprocess_multilabel(img_data, label_map, list_objs, 'remove_small_blobs',
                                                  interval=[size_thr_mm3, 1e10])
            img_pred = nib.Nifti1Image(img_pred_pp, img_pred.affine)

        if preview:
            from totalsegmentator.preview import generate_preview
            # Generate preview before upsampling, so it is faster and still in canonical space
            # for better orientation.
            if not quiet:
                print("Generating preview...")
            st = time.time()
            smoothing = 20
            preview_dir = file_out.parent if multilabel_image else file_out
            generate_preview(img_in_rsp, preview_dir / f"preview_{task_name}.png", img_pred.get_fdata(), smoothing,
                             task_name)
            if not quiet:
                print(f"Generated in {time.time() - st:.2f}s")

        if resample is not None:
            if not quiet:
                print("Resampling...")
            if verbose:
                print(f"  back to original shape: {img_in_shape}")
            # Use force_affine otherwise output affine sometimes slightly off (which then is even increased
            # by undo_canonical)
            img_pred = nnunet.change_spacing(img_pred, resample, img_in_shape,
                                             order=0, dtype=np.uint8,
                                             nr_cpus=nr_threads_resampling,
                                             force_affine=img_in.affine)

        if verbose:
            print("Undoing canonical...")
        img_pred = nnunet.undo_canonical(img_pred, img_in_orig)

        if crop is not None:
            if verbose:
                print("Undoing cropping...")
            img_pred = nnunet.undo_crop(img_pred, img_in_orig, bbox)

        nnunet.check_if_shape_and_affine_identical(img_in_orig, img_pred)

        # Prepare output nifti
        # Copy header to make output header exactly the same as input. But change dtype otherwise it will be
        # float or int and therefore the masks will need a lot more space.
        # (infos on header: https://nipy.org/nibabel/nifti_images.html)
        new_header = img_in_orig.header.copy()
        # noinspection PyUnresolvedReferences
        new_header.set_data_dtype(np.uint8)
        img_out = nib.Nifti1Image(img_pred.get_fdata().astype(np.uint8), img_pred.affine, new_header)
        if save_label_to_nifti is not False:
            from totalsegmentator.nifti_ext_header import add_label_map_to_nifti
            img_out = add_label_map_to_nifti(img_out, label_map)
        _save_and_return_result(img_out, label_map)
        if file_out is not None and skip_saving is False:
            if task_name == "body" and not multilabel_image and not no_derived_masks:
                if not quiet:
                    print("Creating body.nii.gz")
                if v_tt == 1:
                    nnunet.combine_masks(file_out, file_out / "body.nii.gz", "body")
                else:
                    # noinspection PyArgumentList
                    body_img = nnunet.combine_masks(file_out, "body")
                    nib.save(body_img, file_out / "body.nii.gz")
                if not quiet:
                    print("Creating skin.nii.gz")
                skin = nnunet.extract_skin(img_in_orig, nib.load(file_out / "body.nii.gz"))
                nib.save(skin, file_out / "skin.nii.gz")
    return img_out


# noinspection SpellCheckingInspection
def perform_segmentation_generic(file_in: str, seg_config: dict, save_output: bool = True,
                                 return_numpy: bool = False,) -> Union[nib.Nifti1Image, np.ndarray]:
    # For the time being, it is not recommended to save segmentation results after postfix directly as more
    # sub-functions will be added to improve the postfix method.
    """
    :param file_in: input file directory of image in nifti format;
    :param seg_config: a dict that contains information about 'task_id', 'trainer', 'voxel_size', 'crop' and might
                       include 'crop_addon' as well;
    :param save_output: bool on whether to save the output image;
    :param return_numpy: return segmentation result as numpy array or as a nifti image;
    :return: result: either as an image in nifti format or as a numpy array
    """
    file_out = set_seg_file_name(file_in, seg_config)
    if not os.path.isfile(file_out):
        crop_addon = [3, 3, 3] if 'crop_addon' not in seg_config else seg_config['crop_addon']
        multilabel = True if 'multilabel' not in seg_config else seg_config['multilabel']
        trainer = seg_config['trainer']
        task_id = seg_config['task_id']
        voxel_size = seg_config['voxel_size']
        task_name = seg_config['task_name']
        remove_auxiliary = seg_config.get('remove_auxiliary', None)
        result = nnUNet_predict_image(file_in, None, task_id=task_id, model='3d_fullres', folds=[0],
                                      trainer=trainer, tta=False, multilabel_image=multilabel, resample=voxel_size,
                                      crop=None, crop_path=None, task_name=task_name, nora_tag=None, preview=False,
                                      nr_threads_resampling=1, nr_threads_saving=6, force_split=False,
                                      crop_addon=crop_addon, roi_subset=None, output_type='nifti', quiet=False,
                                      verbose=False, remove_auxiliary=remove_auxiliary, test=0)
        if save_output:
            nib.save(result, file_out)
    else:
        result = nib.load(file_out)
    if return_numpy:
        result, *_ = convert_nifti_to_numpy(result)
    return result


def remove_bed_with_model(file_image: str, save_output: bool = True, return_numpy: bool = False):
    seg_config = get_seg_config_by_task_name('body')
    image_arr_body = perform_segmentation_generic(file_image, seg_config, save_output=save_output, return_numpy=True)
    img_nii = nib.load(file_image)
    # noinspection PyTypeChecker
    image_arr, affine, _ = convert_nifti_to_numpy(img_nii)
    mask_other = image_arr_body == 0
    image_arr[mask_other] = image_arr.min()
    if return_numpy:
        return image_arr
    else:
        img_nii_new = nib.Nifti1Image(image_arr, affine)
        if save_output:
            nib.save(img_nii_new, file_image)
        return img_nii_new


def remove_bed_with_scipy(img_3d: np.ndarray, threshold=-50, struct_size=(15, 15), fill_value=-1024):
    img_axial = np.max(img_3d, axis=2)
    img_axial[img_axial > threshold] = 1
    img_axial[img_axial <= threshold] = 0  # fill_value
    img_axial = (ndimage.binary_opening(img_axial, structure=np.ones(struct_size))).astype(int)
    mask_body = np.repeat(img_axial[:, :, np.newaxis], repeats=img_3d.shape[2], axis=2)
    _img = img_3d.copy()
    _img[mask_body == 0] = fill_value
    return _img


# noinspection SpellCheckingInspection
def remove_segments_with_size_limit(img: np.ndarray, dict_size_limit: Union[dict, None] = None) -> np.ndarray:
    """ Modified from 'remove_small_blobs' provided by TotalSegmentator.

    Find segments of the same label. Remove all segments which don't fulfil the size limit.
    Args:
        img: Binary image.
        dict_size_limit: dict defining size limits, which should include either 'LB', 'UB' or both.
    Returns:
        Detected segments.
    """
    if dict_size_limit is not None:
        assert len(dict_size_limit) > 0 and all(k in ['LB', 'UB'] for k in dict_size_limit.keys())
    else:
        dict_size_limit = {'LB': 1000}
    mask, num_segments = ndimage.label(img)
    counts = np.bincount(mask.flatten())  # number of voxels in each segment
    if num_segments > 1:
        if 'LB' in dict_size_limit.keys() and 'UB' not in dict_size_limit.keys():
            remove = np.where((counts <= dict_size_limit['LB']), True, False)
        elif 'LB' in dict_size_limit.keys() and 'UB' in dict_size_limit.keys():
            remove = np.where((counts <= dict_size_limit['LB']) | (counts > dict_size_limit['UB']), True, False)
        else:
            remove = np.where((counts > dict_size_limit['UB']), True, False)
        if np.sum(remove) > 0:
            print(f'{np.sum(remove)} objects out of {num_segments} are to be removed.')
        remove_idx = np.nonzero(remove)[0]
        mask[np.isin(mask, remove_idx)] = 0
        mask[mask > 0] = 1
        return mask.astype(bool)
    else:
        # If only one segment, there is nothing to remove.
        if len(counts) == 0:
            print_highlighted_text('Input mask image is blank!')
        return img


def segment_bright_objects_init(img_2d):
    # Use edge enhancing filter as an initial step to segment bright objects
    img_mag_farid = filters.farid(img_2d)
    # noinspection PyTypeChecker
    markers = np.zeros_like(img_mag_farid)
    markers[img_2d < 900] = 1
    # noinspection PyTypeChecker
    markers[img_mag_farid > 120 / np.iinfo(img_2d.dtype).max] = 2
    img_seg = watershed(img_mag_farid, markers)
    # In most cases the following two lines are enough to segment out bright objects. However, because gradient-based
    # methods detect also edges of dark objects in the neighborhood of bright objects, 'fill hole' operation will
    # probably get part of dark objects in the returned segmentations. This is why this method is only the initial step.
    img_seg = ndi.binary_fill_holes(img_seg - 1)
    img_seg = ndi.binary_opening(img_seg, disk(2))
    return img_seg


def segment_bright_objects(img_2d, area_threshold=200):
    img_seg = segment_bright_objects_init(img_2d)
    if len(np.unique(img_seg)) > 1:
        img_2d_t = np.multiply(img_seg, img_2d)
        # print('Median intensity is {}'.format(np.median(img_2d_t[img_2d_t > 0])))
        img_edge = filters.farid(img_2d_t)
        threshold_edge = threshold_multiotsu(img_edge)
        # noinspection PyUnresolvedReferences
        seg_idx_0 = img_edge < threshold_edge[0]
        seg_idx_1 = img_2d_t < np.median(img_2d_t[img_2d_t > 0])
        # noinspection PyUnresolvedReferences
        img_edge[np.multiply(seg_idx_0, seg_idx_1)] = 0
        # noinspection PyUnresolvedReferences
        # noinspection PyTypeChecker
        img_edge[img_edge > 0] = 1
        # noinspection PyTypeChecker
        img_edge = ndi.binary_closing(img_edge, disk(3))
        img_seg = ndi.binary_fill_holes(img_edge)
        img_seg = ndi.binary_opening(img_seg, disk(1))
        img_seg_int = np.zeros_like(img_2d)
        img_seg_int[img_2d > 900] = 1
        img_seg = np.multiply(img_seg_int, img_seg)
        img_seg, _ = ndi.label(img_seg)
        img_seg = remove_small_objects(img_seg, min_size=area_threshold)
        img_seg[img_seg > 0] = 1
    return img_seg


def get_objects_at_border(mask_2d: Union[np.ndarray, NDArray[np.bool_]], selected_border='all', buffer_size=0, bg_val=0,
                          in_place=False, mask=None, out=None):
    # It does the opposite to skimage.segmentation.clear_border to detect objects at borders

    if any((buffer_size >= s for s in mask_2d.shape)) and mask is None:
        # ignore buffer_size if mask
        raise ValueError("buffer size may not be greater than labels size")

    if out is not None:
        np.copyto(out, mask_2d, casting='no')
        in_place = True

    if not in_place:
        out = mask_2d.copy()
    elif out is None:
        out = mask_2d

    if mask is not None:
        err_msg = (f'labels and mask should have the same shape but '
                   f'are {out.shape} and {mask.shape}')
        if out.shape != mask.shape:
            raise (ValueError, err_msg)
        if mask.dtype != bool:
            raise TypeError("mask should be of type bool.")
        borders = ~mask
    else:
        # create borders with buffer_size
        borders = np.zeros_like(out, dtype=bool)
        ext = buffer_size + 1
        sl_end = slice(-ext, None)
        slices = [slice(None) for _ in out.shape]
        if selected_border == 'all':
            sl_start = slice(ext)
            for d in range(out.ndim):
                slices[d] = sl_start
                borders[tuple(slices)] = True
                slices[d] = sl_end
                borders[tuple(slices)] = True
                slices[d] = slice(None)
        elif selected_border == 'low':
            slices[0] = sl_end
            borders[tuple(slices)] = True
        else:
            raise Exception("Currently only objects at borders for 'all' and 'low' are implemented!")

    # Re-label, in case we are dealing with a binary out
    # and to get consistent labeling
    labels, number = ndi.label(out)

    # determine all objects that are connected to borders
    borders_indices = np.unique(labels[borders])
    indices = np.arange(number + 1)
    # mask all label indices that are connected to borders
    label_mask = np.in1d(indices, borders_indices)
    # create mask for pixels at borders
    # noinspection PyUnresolvedReferences
    mask = label_mask[labels.reshape(-1)].reshape(labels.shape)
    out[~mask] = bg_val
    return out


def get_bbx_from_2d_mask(mask_2d: NDArray[np.bool_]):
    mask_2d_new = normalized_image_to_8bit(mask_2d)
    cnt, _ = cv.findContours(mask_2d_new, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contour = cnt[0]
    x, y, w, h = cv.boundingRect(contour)
    return [x, y, w, h], contour


def get_object_width_at_bottom(mask_2d: NDArray[np.bool_], position=0.2):
    bbx, _ = get_bbx_from_2d_mask(mask_2d)
    obj_bottom = int(round(bbx[3] * (1 - position)))
    mask_2d_cropped = mask_2d.copy()
    mask_2d_cropped[bbx[1]:bbx[1] + obj_bottom + 1, :] = 0
    mask_2d_new = normalized_image_to_8bit(mask_2d_cropped)
    distance = cv.distanceTransform(mask_2d_new, distanceType=cv.DIST_L2, maskSize=3).astype(np.float32)
    skeleton = skeletonize(mask_2d_cropped).astype(np.float32)
    thickness = cv.multiply(distance, skeleton)
    thickness = 2 * thickness[skeleton != 0]
    thickness_median = np.median(thickness)
    return thickness_median


def get_lb_bright_objects(img_2d, idx_patient=-1, patient_id='', aspect=1, area_threshold=200, prosthesis_min=1000,
                           show_image=False, file_path=''):

    def show_image_seg(images_subplot, title_subplots, aspect_ratio):
        figure, axs = plt.subplots(1, len(images_subplot), layout='constrained')
        for i in range(len(images_subplot)):
            if i == 0:
                axs[i].imshow(images_subplot[i], vmin=-500, vmax=1300, cmap='gray', aspect=aspect_ratio)
            else:
                axs[i].imshow(images_subplot[i], aspect=aspect_ratio)
            axs[i].axis('off')
            axs[i].set_title(title_subplots[i])
        plt.show()
        if file_path:
            figure.savefig(os.path.join(file_path, patient_id + '_detected_bright_objects.png'))

    def get_hip_prosthesis(seg_labeled_implants, size_threshold=prosthesis_min):
        seg_labeled_at_border = get_objects_at_border(seg_labeled_implants, selected_border='low')
        props = regionprops(seg_labeled_at_border)
        if len(props) >= 1:
            seg_labeled_prosthesis = seg_labeled_at_border.copy()
            seg_prosthesis_num = 0
            for i in range(len(props)):
                seg_labeled_region = np.asarray(seg_labeled_prosthesis == props[i].label)
                obj_width = get_object_width_at_bottom(seg_labeled_region)
                if props[i].area > size_threshold and obj_width > 10:
                    print('Detected prosthesis with label of {} has a median width of {}'.format(props[i].label,
                                                                                                 obj_width))
                    seg_prosthesis_num += 1
                else:
                    seg_labeled_prosthesis[seg_labeled_region] = 0
            if seg_labeled_prosthesis.max() == 0:
                seg_labeled_prosthesis = -1
            else:
                print('{} detected hip prosthesis with sizes larger than {}!'.format(seg_prosthesis_num,
                                                                                     size_threshold))
        else:
            seg_labeled_prosthesis = -1

        return seg_labeled_prosthesis

    def determine_what_to_plot(seg, seg_at_border):
        disp_seg = True
        disp_seg_at_border = False
        if isinstance(seg_at_border, int) is False:
            prop_seg = regionprops(seg)
            prop_seg_border = regionprops(seg_at_border)
            if len(prop_seg_border) > 0:
                if len(prop_seg_border) == len(prop_seg):
                    disp_seg = False
                    disp_seg_at_border = True
                else:
                    disp_seg = True
                    disp_seg_at_border = True
        return disp_seg, disp_seg_at_border

    img_seg = segment_bright_objects(img_2d, area_threshold)
    # To remove possible dark objects in the segmented image, additional segmentation operation is therefore computed.

    seg_labeled, num_obj = ndi.label(img_seg)
    if num_obj > 0:
        print('{} implant(s) with area(s) larger than {} pixels detected!'.format(num_obj, area_threshold))
        seg_prosthesis = get_hip_prosthesis(seg_labeled)
        if show_image is True:
            if (not patient_id) and (idx_patient != -1):
                img_title_ori = '%03d: ' % idx_patient + patient_id
            else:
                img_title_ori = 'Original coronal view'
            display_seg, display_prosthesis = determine_what_to_plot(seg_labeled, seg_prosthesis)
            if display_seg is True and display_prosthesis is True:
                img_disp = [img_2d, seg_labeled, seg_prosthesis]
                img_title = [img_title_ori, 'Detected bright objects', 'Hip prosthesis']
                show_image_seg(img_disp, img_title, aspect)
            elif display_seg is True and display_prosthesis is False:
                img_disp = [img_2d, seg_labeled]
                img_title = [img_title_ori, 'Detected bright objects']
                show_image_seg(img_disp, img_title, aspect)
            elif display_seg is False and display_prosthesis is True:
                img_disp = [img_2d, seg_prosthesis]
                img_title = [img_title_ori, 'Hip prosthesis']
                show_image_seg(img_disp, img_title, aspect)
    else:
        seg_labeled = -1
        seg_prosthesis = -1
        if show_image is True:
            if (not patient_id) and (idx_patient != -1):
                img_title = '%03d: ' % idx_patient + patient_id
            else:
                img_title = 'Original coronal view'
            fig, ax = plt.subplots(1, 1)
            ax.imshow(img_2d, vmin=-500, vmax=1300, cmap='gray', aspect=aspect)
            ax.axis('off')
            ax.set_title(img_title)
            fig.show()

    return seg_labeled, seg_prosthesis





