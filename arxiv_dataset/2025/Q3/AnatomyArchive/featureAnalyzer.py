import os
import re
import six
import json
import pandas as pd
import numpy as np
import nibabel as nib
# noinspection SpellCheckingInspection
import SimpleITK as sitk
from p_tqdm import p_map
from typing import Union, List
from volStandardizer import *
import matplotlib.pyplot as plt
from radiomics import featureextractor
from sklearn.preprocessing import StandardScaler
from util import NestedDict, get_files_in_folder, print_highlighted_text
from genericImageIO import convert_nifti_to_numpy, convert_sitk_image_to_numpy
from segManager import reconstruct_data_from_coo_components
from segModel import get_seg_config_by_task_name, perform_segmentation_generic
from segModel import image_resample, remove_bed_with_scipy, get_v_dependent_cls_map
from segManager import anatomy_name_standardization, segmentation_postfix, recover_3d_mask_array_from_coord
from simpleStats import get_index_of_plane_with_largest_mask_area, estimate_kernel_density, get_optimal_hist_bin_width


def get_body_component_attributes(img_3d: np.ndarray, img_3d_seg: np.ndarray, data_id: str,
                                  result_dict: NestedDict, cls_map_name: str, selected_objects: list,
                                  voxel_size: Union[tuple, list], dict_bounds: Union[dict, None] = None,
                                  index_plane: Union[int, None] = None, mask_dict: Union[dict, None] = None,
                                  hist_dict: Union[NestedDict, None] = None, bin_edges: Union[np.ndarray, None] = None,
                                  data_class: Union[int, None] = None, dataset_tag: Union[NestedDict, None] = None):
    def get_attrs(_obj_name, _mask):
        if np.sum(_mask) == 0:
            print_highlighted_text(f"Volume for {_obj_name} is zero. Please double-check the segmentation for data "
                                   f"with ID of '{data_id}'!")
            if dataset_tag is not None:
                add_tag_to_data(dataset_tag, f'{_obj_name}Missing', data_id, 'Warning')
            result_dict[data_id][_obj_name + '_vol_cm3'] = np.nan
            result_dict[data_id][_obj_name + '_vol_median_HU'] = np.nan
            result_dict[data_id][_obj_name + '_vol_mean_HU'] = np.nan
            if index_plane is not None:
                result_dict[data_id][_obj_name + '_area_cm2'] = np.nan
                result_dict[data_id][_obj_name + '_area_median_HU'] = np.nan
                result_dict[data_id][_obj_name + '_area_mean_HU'] = np.nan
        else:
            if dict_bounds is not None:
                _mask_obj = _mask * mask_clipped
            else:
                _mask_obj = _mask

            if np.sum(_mask_obj) > 0:
                if isinstance(hist_dict, NestedDict):
                    hist = np.histogram(img_3d[_mask_obj], bins=bin_edges, density=True)[0]
                    print(f"Add histogram {data_id}: {_obj_name} to 'hist_dict'")
                    if data_class is not None:
                        if 'DataIDs' not in hist_dict[_obj_name][data_class].keys():
                            hist_dict[_obj_name][data_class]['DataIDs'] = list()
                        hist_dict[_obj_name][data_class]['DataIDs'].append(data_id)
                        if 'All' not in hist_dict[_obj_name][data_class].keys():
                            hist_dict[_obj_name][data_class]['All'] = hist
                            hist_dict[_obj_name][data_class]['Count'] = 1
                        else:
                            hist_dict[_obj_name][data_class]['All'] = (hist_dict[_obj_name][data_class]['Count'] *
                                                                       hist_dict[_obj_name][data_class]['All']+hist) / \
                                                                      (hist_dict[_obj_name][data_class]['Count'] + 1)
                            hist_dict[_obj_name][data_class]['Count'] += 1
                    else:
                        hist_dict[data_id][_obj_name] = hist
                # convert volume to mL or cm^3
                result_dict[data_id][_obj_name + '_vol_cm3'] = round(voxel_vol * np.sum(_mask_obj) / 1000, 2)
                result_dict[data_id][_obj_name + '_vol_median_HU'] = np.median(img_3d[_mask_obj])
                result_dict[data_id][_obj_name + '_vol_mean_HU'] = np.mean(img_3d[_mask_obj])
            if index_plane is not None:
                # index plane is obtained from the original image without volume cut. Therefore, use the original _mask
                # variable instead of _mask_obj after clipping.
                area_obj = get_object_area_at_indexed_plane(_mask, voxel_size, index_plane=index_plane)
                result_dict[data_id][_obj_name + '_area_cm2'] = round(area_obj, 2)
                img_2d = img_3d[:, :, index_plane]
                mask_2d = _mask[:, :, index_plane]
                result_dict[data_id][_obj_name + '_area_median_HU'] = np.median(img_2d[mask_2d])
                result_dict[data_id][_obj_name + '_area_mean_HU'] = np.mean(img_2d[mask_2d])

    if isinstance(hist_dict, dict):
        if bin_edges is None:
            bin_edges = np.linspace(-1000, 1600, 261, endpoint=True)
    voxel_vol = np.prod(voxel_size)  # mm^3
    cls_map = get_v_dependent_cls_map(cls_map_name)
    list_labels = list(cls_map.keys())
    list_anatomies = anatomy_name_standardization(cls_map)
    selected_objects = anatomy_name_standardization(selected_objects)
    assert all([_e in list_anatomies for _e in selected_objects])
    if dict_bounds is not None:
        mask_clipped = np.zeros_like(img_3d_seg)
        mask_clipped[:, :, dict_bounds['LB']:dict_bounds['UB']] = 1
    if isinstance(mask_dict, dict):
        segmentation_postfix(img_3d, img_3d_seg, cls_map_name, mask_dict, selected_objects)
        for _obj in mask_dict['SelectedAnatomies']:
            get_attrs(_obj, mask_dict[_obj])
    else:
        for _obj in selected_objects:
            mask = img_3d_seg == list_labels[list_anatomies.index(_obj)]
            get_attrs(_obj, mask)


def body_component_analysis(dir_input, result_dict: NestedDict, target_eva_config: Union[dict, None] = None,
                            csv_file_name: Union[str, None] = None, inspect_plot_as_nifti: bool = False,
                            dataset_tag: Union[NestedDict, None] = None, hist_dict: Union[NestedDict, None] = None):

    def _analyze_components(_file_in):
        _file_in = os.path.join(dir_input, _file_in)
        skip = False
        data_id = str(_file_in.split('.')[0]).split('_')[-1]
        img_nii = nib.load(_file_in)
        # noinspection PyTypeChecker
        img_3d, affine, voxel_size = convert_nifti_to_numpy(img_nii)
        mask_dict = dict()
        img_3d_seg_ref = perform_segmentation_generic(_file_in, seg_config_ref, return_numpy=True)
        index_plane = None
        if body_comp_type == 0:
            obj_ref = None
            obj_ref_name = target_eva_config[cls_map_name_ref]['refObj']
            if prosthesis_detection:
                obj_ref_lb = {'LB': 'pelvic'}
                define_volume_bounds_by_objects(img_3d_seg_ref, obj_ref_lb, cls_map_name_ref, data_id=data_id,
                                                dataset_tag=dataset_tag)
                prosthesis_detected = prosthesis_detection_at_lower_bound(img_3d, obj_ref_lb, voxel_size=voxel_size,
                                                                          img_3d_seg_ref=img_3d_seg_ref,
                                                                          data_id=data_id, dataset_tag=dataset_tag)
                if prosthesis_detected != 0:
                    skip = True
                else:
                    obj_ref_plane = {obj_ref_name: None}
                    fig_ref_plane = os.path.join(dir_input, 'ResultInspection',
                                                 f"{_file_in.split('.')[0]}_{obj_ref_name}_referencePlane.png")
                    set_central_ref_plane(img_3d, img_3d_seg_ref, voxel_size, cls_map_name_ref, obj_ref_plane,
                                          plot_image=True, save_fig_name=fig_ref_plane,
                                          save_ref_plane_mask=inspect_plot_as_nifti)
                    index_plane = obj_ref_plane[obj_ref_name]
                    if inspect_plot_as_nifti:
                        mask_save = obj_ref_plane['centralPlaneMask'][:, :, np.newaxis]
                        img_mask = nib.Nifti1Image(mask_save, affine)
                        # # Double check if the transpose is needed after update.
                        # img_mask = nib.Nifti1Image(mask_save.transpose((1, 0, 2)), affine)
                        # noinspection PyUnresolvedReferences
                        img_mask.header.set_zooms(tuple(voxel_size))
                        mask_file_name = os.path.join(dir_input, 'ResultInspection',
                                                      f"{_file_in.split('.')[0]}_{obj_ref_name}CentralPlane.nii.gz")
                        img_mask.to_filename(mask_file_name)
                    if analyze_tissue_types:
                        # Needed for body cropping detection in case body compositions are to be analyzed.
                        obj_ref = {'UB': target_dict_main['refObj'], 'LB': target_dict_main['refObj']}
        else:
            if body_comp_type == 1:
                obj_ref = {'UB': target_dict_main['refObjUB'], 'LB': target_dict_main['refObjLB']}
                define_volume_bounds_by_objects(img_3d_seg_ref, obj_ref, cls_map_name_ref, data_id=data_id,
                                                dataset_tag=dataset_tag)
                if obj_ref['UB']['boundPlane'] < 0 or obj_ref['UB']['boundPlane'] < 0:
                    skip = True
                if prosthesis_detection:
                    if obj_ref['LB']['Name'] == 'hip':
                        prosthesis_detected = prosthesis_detection_at_lower_bound(img_3d, obj_ref, cls_map_name_ref,
                                                                                  dataset_tag)
                        if prosthesis_detected != 0:
                            skip = True
                    else:
                        print_highlighted_text(f"Prosthesis detection is only enabled if hip bones are set as lower "
                                               f"bound, which however is set to be {obj_ref['LB']['Name']}. "
                                               f"Therefore, prosthesis detection is skipped")
            else:
                obj_ref = None
        if not skip:
            dict_bounds = {k: v['boundPlane'] for k, v in obj_ref.items()} if body_comp_type == 1 else None
            if objs_sel_main is not None and len(objs_sel_main) > 0:
                get_body_component_attributes(img_3d, img_3d_seg_ref, data_id, result_dict, cls_map_name_ref,
                                              objs_sel_main, voxel_size, dict_bounds=dict_bounds,
                                              index_plane=index_plane, mask_dict=mask_dict, hist_dict=hist_dict)
            for _cls_name in cls_map_names:
                seg_config = get_seg_config_by_task_name(_cls_name)
                img_3d_seg = perform_segmentation_generic(_file_in, seg_config, return_numpy=True)
                selected_objects = target_eva_config[_cls_name]['selectedObjs']
                if 'tissue' in _cls_name and body_comp_type >= 0 and analyze_tissue_types:
                    seg_config_body = get_seg_config_by_task_name('body', coarse)
                    img_seg_body = perform_segmentation_generic(_file_in, seg_config_body, return_numpy=True)
                    cls_map_app_bones, v_tt = get_v_dependent_cls_map('appendicular_bones', return_version=True)
                    if v_tt == 2:
                        seg_config_bones = get_seg_config_by_task_name('appendicular_bones')
                        img_3d_seg_bones = perform_segmentation_generic(_file_in, seg_config_bones, return_numpy=True)
                    else:
                        img_3d_seg_bones = img_3d_seg

                    fig_body_det = os.path.join(dir_input, 'ResultInspection',
                                                f"{_file_in.split('.')[0]}_arm&BodyCroppingDetection.png")
                    _dict_result = separate_arms_and_legs(img_seg_body, img_3d_seg_bones, cls_map_ref=cls_map_app_bones,
                                                          plot_image=True, obj_ref=obj_ref, all_within_bounds=False,
                                                          img_3d=img_3d, voxel_size=voxel_size,
                                                          save_fig_name=fig_body_det, body_cropping_detection=True,
                                                          data_id=data_id, dataset_tag=dataset_tag)
                    if _dict_result['bodyCroppedFlag'] > 0:
                        skip = True
                        continue
                    if body_comp_type == 1:
                        img_3d_seg[~_dict_result['bodyWithoutArms']] = 0
                if not skip:
                    get_body_component_attributes(img_3d, img_3d_seg, data_id, result_dict, _cls_name, selected_objects,
                                                  voxel_size, dict_bounds=dict_bounds, index_plane=index_plane,
                                                  mask_dict=mask_dict, hist_dict=hist_dict)

    if target_eva_config is None:
        with open(os.path.join(dir_input, 'TargetEvaConfig.json'), 'r') as h_config:
            target_eva_config = json.load(h_config)
    cls_map_name_ref = target_eva_config.get('refClassMap', 'total')
    # Need to add code to ensure that the refClassMap contains bones
    files = get_files_in_folder(dir_input, 'nii.gz', 'seg')
    cls_map_names = list(target_eva_config.keys())
    cls_map_names.remove(cls_map_name_ref)
    if 'refClassMap' in cls_map_names:
        cls_map_names.remove('refClassMap')
    target_dict_main = target_eva_config[cls_map_name_ref]
    objs_sel_main = target_dict_main.get('selectedObjs', None)
    coarse = target_dict_main.get('coarse', None)
    seg_config_ref = get_seg_config_by_task_name(cls_map_name_ref, coarse)
    prosthesis_detection = target_dict_main.get('exclude_prosthesis_samples', None)
    list_obj_ref = [v for k, v in target_dict_main.items() if k in 'refObj' and v is not None]
    analyze_tissue_types = any(['tissue' in k for k, v in target_eva_config.items() if
                                v['SelectedObjs'] is not None and any(['muscle' in _e for _e in v['SelectedObjs']])])
    if len(list_obj_ref) == 1:
        body_comp_type = 0
    elif len(list_obj_ref) == 2:
        body_comp_type = 1
    else:
        body_comp_type = -1
    if not os.path.isdir(os.path.join(dir_input, 'ResultInspection')):
        os.mkdir(os.path.join(dir_input, 'ResultInspection'))
    # for i in tqdm(range(len(files)), desc=f'Performing body component analysis for {len(files)} patients.'):
    #     file = files[i]
    print(f'Performing body component analysis for {len(files)} patients.')
    p_map(_analyze_components, files)
    df = pd.DataFrame.from_dict(result_dict, orient='index')
    if csv_file_name is None:
        csv_file_name = 'BCA_results.csv'
    df.to_csv(os.path.join(dir_input, csv_file_name))
    if hist_dict is not None:
        np.savez_compressed(os.path.join(dir_input, 'Hist_dict.npz'), hist_dict=hist_dict)


def illustrative_figure_bca(img_3d: np.ndarray, mask_dict: dict, color_dict: dict,
                            x: Union[int, None] = None, z: Union[int, None] = None, fig_name: Union[str, None] = None,
                            voxel_size=(1, 1, 1), apply_tissue_window: bool = True, apply_rgba_cmap=False):
    import matplotlib as mpl
    from functools import reduce
    import operator

    def get_color(_obj_name):
        try:
            _color = color_dict[_obj_name]
        except KeyError:
            if 'fatty' in _obj_name:
                _color = [_v for _k, _v in color_dict.items() if 'fatty' in _k][0]
            elif 'fatty' not in _obj_name and 'fat' in _obj_name:
                _color = [_v for _k, _v in color_dict.items() if 'fat' in _k and 'fatty' not in _k][0]
            else:
                raise Exception(f'No matched color for {_obj_name}')
        return _color

    def set_label(_obj_name, _mask):
        # Used for the case without setting RGBA colors to img_3d_label
        try:
            _label = label_dict[_obj_name]
        except KeyError:
            if 'fatty' in _obj_name:
                _key = [_k for _k in color_dict.keys() if 'fatty' in _k][0]
                img_3d_label[_mask] = label_dict[_key]
            elif 'fatty' not in _obj_name and 'fat' in _obj_name:
                _key = [_k for _k in color_dict.keys() if 'fat' in _k and 'fatty' not in _k][0]
                img_3d_label[_mask] = label_dict[_key]
            else:
                _label = -1
        else:
            img_3d_label[_mask] = _label

    def set_rgb_color(_mask, _color):
        # Convert color string to RGBA values.
        _rgba_value = to_rgba(_color)
        for _i, _v in enumerate(_rgba_value):
            img_3d_label[_mask, _i] = _v

    def add_sub_figure_in_subplot(_ax):
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(_ax)
        _ax_l = divider.append_axes('bottom', size='50%')
        _fig = _ax.get_figure()
        _fig.delaxes(_ax)
        return _ax_l

    if color_dict is None:
        color_dict = dict()
        color_dict['subcutaneous_fat'] = 'yellow'
        color_dict['torso_fat'] = 'orange'
        color_dict['fatty_muscle'] = 'blue'
        color_dict['skeletal_muscle'] = 'red'

    img_shape = img_3d.shape
    if not apply_rgba_cmap:
        img_3d_label = np.zeros_like(img_3d).astype(np.uint8)
        colors = ['black'] + list(color_dict.values())
        cmap = mpl.colors.LinearSegmentedColormap.from_list('', colors)
    else:
        # Add RGB channels plus an alpha mask to the label array.
        img_3d_label = np.zeros(tuple(list(img_shape) + [4]))
        from matplotlib.colors import to_rgba

    label_dict = dict()
    for i, k in enumerate(color_dict.keys()):
        label_dict[k] = i + 1

    aspect_ratios = np.array(voxel_size)/voxel_size[0]
    px = 1 / plt.rcParams['figure.dpi']
    aspect_ratio_z = round(aspect_ratios[2] * img_shape[2] / img_shape[0], 2)
    if x is None:
        x = get_index_of_plane_with_largest_mask_area(mask_dict, 'x', aspect_ratios=aspect_ratios)
    if z is None:
        z = get_index_of_plane_with_largest_mask_area(mask_dict, 'z', aspect_ratios=aspect_ratios)

    fig, axs = plt.subplots(2, 3, figsize=(1000 * px, 600 * px), constrained_layout=True,
                            gridspec_kw={'width_ratios': [aspect_ratio_z, 1, aspect_ratio_z],
                                         'height_ratios': [1, 1]})
    alpha_value = 0.8 if apply_rgba_cmap else 0.5
    mask_original = list()
    for k, v in mask_dict.items():
        if isinstance(v, np.ndarray):
            mask_original.append(v)
            _edges, _dens, _hist = estimate_kernel_density(img_3d, v)
            _color_k = get_color(k)
            axs[0, 2].stairs(_hist, edges=_edges, fill=True, alpha=alpha_value,
                             orientation='horizontal', color=_color_k)
            axs[0, 2].plot(np.exp(_dens), _edges, linestyle='--', color=_color_k if _color_k != 'yellow' else 'orange')
            if not apply_rgba_cmap:
                set_label(k, v)
            else:
                set_rgb_color(v, _color_k)

    mask = reduce(operator.xor, mask_original)
    _edges, _dens, _hist = estimate_kernel_density(img_3d, mask)
    axs[0, 2].stairs(_hist, edges=_edges, fill=True, alpha=alpha_value, orientation='horizontal', color='gray',
                     label='Original without split')
    axs[0, 2].plot(np.exp(_dens), _edges, color='k', linestyle='--', label='Estimated kernel density')
    axs[0, 2].legend()
    axs[0, 2].set_ylabel('Gray value in HU')
    axs[0, 2].set_xlabel('Probability density')

    ax_l = add_sub_figure_in_subplot(axs[1, 2])
    voxel_vol = np.prod(np.array(voxel_size))
    vol = [round(voxel_vol * np.sum(v) / 1000, 2) for v in mask_dict.values() if isinstance(v, np.ndarray)
           and v.dtype == np.bool_]  # cm3
    pos_bar = range(len(vol))
    alpha_value = 1 if apply_rgba_cmap else 0.25
    bars_0 = ax_l.barh(pos_bar, vol, alpha=alpha_value)
    ax_new = ax_l.twiny()
    vol_pct = np.around(np.array(vol) / sum(vol) * 100, decimals=1)
    bars_1 = ax_new.barh(pos_bar, vol_pct, alpha=alpha_value)
    for i, k in enumerate(mask_dict.keys()):
        if isinstance(mask_dict[k], np.ndarray):
            _color_k = get_color(k)
            bars_0[i].set_color(_color_k)
            bars_0[i].set_edgecolor(_color_k if _color_k != 'yellow' else 'orange')
            bars_1[i].set_color(_color_k)
            bars_1[i].set_edgecolor(_color_k if _color_k != 'yellow' else 'orange')

    ax_l.set_xlabel('Volume ($cm^{3}$)')
    ax_l.tick_params(axis='x', colors='k')
    ax_l.tick_params(axis='y', left=False, labelleft=False)
    ax_new.set_xlabel('Volume fraction (%)')
    ax_new.tick_params(axis='x', colors='k')
    if apply_tissue_window:
        import bodyPartWindow
        dict_window = bodyPartWindow.triple_window_dict['SoftTissue']
        img_3d = bodyPartWindow.get_window_image(img_3d, dict_window)
    for r in range(2):
        for c in range(2):
            axs[r, c].set_axis_off()
            if c == 0:
                axs[r, c].imshow(np.fliplr(img_3d[:, :, z]), cmap='gray', aspect=aspect_ratios[0])
            else:
                axs[r, c].imshow(np.flipud(img_3d[x, :, :].T), cmap='gray', aspect=aspect_ratios[2])
            if r == 1:
                if c == 0:
                    if not apply_rgba_cmap:
                        # noinspection PyUnboundLocalVariable
                        axs[r, c].imshow(np.fliplr(img_3d_label[:, :, z]), cmap=cmap, alpha=0.4,
                                         aspect=aspect_ratios[0], interpolation='none', vmin=0.05)
                    else:
                        axs[r, c].imshow(np.fliplr(img_3d_label[:, :, z, :]), aspect=aspect_ratios[0])
                else:
                    if not apply_rgba_cmap:
                        axs[r, c].imshow(np.flipud(img_3d_label[x, :, :].T), cmap=cmap, alpha=0.4,
                                         aspect=aspect_ratios[2], interpolation='none', vmin=0.05)
                    else:
                        axs[r, c].imshow(np.flipud(np.rollaxis(img_3d_label[x, :, :, :], 1, 0)),
                                         aspect=aspect_ratios[2])

    from matplotlib import patches
    _patches = list()
    alpha_value = 1 if apply_rgba_cmap else 0.5
    for k, v in color_dict.items():
        _patches.append(patches.Patch(facecolor=v, label=k.replace('_', ' '), alpha=alpha_value,
                                      edgecolor=v if v != 'yellow' else 'orange'))
    ax_new.legend(handles=_patches, bbox_to_anchor=(0.5, 3.2), loc='upper center')
    if fig_name is not None:
        fig.savefig(fig_name)
    else:
        plt.show()


def set_type_bin_width(image_arr: np.ndarray, mask_arr: np.ndarray, selected_labels: List[int],
                       bin_width_opt: Union[np.ndarray, None] = None):

    def round_bin_width(_bin_width):
        ratio = _bin_width / bin_width_opt
        idx = (np.abs(ratio - 1)).argmin()
        return bin_width_opt[idx]

    if bin_width_opt is None:
        bin_width_opt = np.array([2, 5, 10, 20, 40, 50])
    preset_bin_widths = dict()
    for _l in selected_labels:
        mask = mask_arr == _l
        bin_width = round_bin_width(get_optimal_hist_bin_width(image_arr[mask], round_to_int=True))
        preset_bin_widths[_l] = bin_width
    return preset_bin_widths


def set_type_bin_width_batch(files_image: list, files_mask: list, selected_labels: List[int],
                             bin_width_opt: np.ndarray | None = None, n_samples: int = 10):
    # Randomly sample a set of images for computing type specific bin widths
    def round_bin_width(_bin_width):
        ratio = _bin_width / bin_width_opt
        idx = (np.abs(ratio - 1)).argmin()
        return bin_width_opt[idx]

    assert len(files_image) == len(files_mask) >= n_samples
    np.random.seed(10)
    selected_idx = np.random.choice(len(files_image), n_samples, replace=False)
    files_image_sel = [files_image[i] for i in selected_idx]
    files_mask_sel = [files_image[i] for i in selected_idx]
    if bin_width_opt is None:
        bin_width_opt = np.array([2, 5, 10, 20, 40, 50])
    dict_data = {_k: [] for _k in selected_labels}
    dict_bin_widths = {_k: 0 for _k in selected_labels}
    for i in range(n_samples):
        # noinspection PyUnresolvedReferences
        image_arr = nib.load(files_image_sel[i]).get_fdata().astype(np.int16)
        # noinspection PyUnresolvedReferences
        mask_arr = nib.load(files_mask_sel[i]).get_fdata().astype(np.uint8)
        for _l in selected_labels:
            mask = mask_arr == _l
            dict_data[_l].append(image_arr[mask])
    for _l in selected_labels:
        dict_bin_widths[_l] = round_bin_width(get_optimal_hist_bin_width(np.concatenate(dict_data[_l]),
                                                                         round_to_int=True))
    return dict_bin_widths


# noinspection SpellCheckingInspection
def get_feature_extractor(file_params: str, voxel_based: bool = True, **kwargs):
    """
    :param file_params: full file name for parameter configuration that specifies the non-tissue specific settings;
    :param voxel_based: bool, with default value of True;
    :param kwargs: print_params (bool), whether to print feature extraction settings and enabled features;
                   kernel_radius (int, >=1), in case the preset value in the config file is to be replaced;
                   bin_width (int), in case the preset value in the config file is to be replaced;
                   geometry_tolerance (float, default 0.001 if not specified). This is a correction for geometry
                                 mismatch due to, for instance, rounding error in voxel size/spacing.

    :return: extractor
    """
    extractor = featureextractor.RadiomicsFeatureExtractor(file_params)
    settings = extractor.settings
    # Setting an initial value for voxel-based features to NaN has a risk of getting NaN values for all voxels.
    if settings['initValue'] == np.NaN:
        print_highlighted_text('Setting initial value to NaN will lead to numeric errors. It is reset to 0 instead!')
        settings['initValue'] = 0
    if 'kernel_radius' in kwargs.keys():
        if voxel_based:
            if kwargs['kernel_radius'] <= 0:
                print_highlighted_text(f'Reset the kernel radius to 1, to fulfil the minimum value requirement for '
                                       f'voxel-based feature extraction.')
                settings['kernelRadius'] = 1
            else:
                settings['kernelRadius'] = kwargs['kernel_radius']
        else:
            settings['kernelRadius'] = 0
    if 'bin_width' in kwargs.keys():
        settings['binWidth'] = kwargs['bin_width']
    settings['geometryTolerance'] = kwargs.get('geometry_tolerance', 0.001)
    print_params = kwargs.get('print_params', False)
    features_enabled = extractor.enabledFeatures
    # To make implicit enabled features become explicit.
    feature_classes_unspecified = [k for k, v in features_enabled.items() if v is None or len(v) == 0]
    if len(feature_classes_unspecified) > 0:
        from radiomics import getFeatureClasses
        feature_classes = getFeatureClasses()
        for _f_class in feature_classes_unspecified:
            _features = [f for f, deprecated in six.iteritems(feature_classes[_f_class].getFeatureNames()) if
                         not deprecated]
            if _f_class == 'firstorder':
                print_highlighted_text('TotalEnergy is just a voxel-size scaled version of Energy, therefore they are '
                                       'highly correlated, providing no new information. Remove TotalEngergy from the '
                                       'feature list.')
                if 'TotalEnergy' in _features and 'Energy' in _features:
                    _features.remove('TotalEngergy')
            elif _f_class == 'glcm':
                print_highlighted_text('GLCM is symmetrical, and SumAverage=2*JointAverage. As JointAverage will be '
                                       'included, remove SumAverage from its feature list.')
                _features.remove('SumAverage')
            features_enabled[_f_class] = _features
    if print_params:
        print('Extraction parameters:')
        for _param_name, _value in settings.items():
            print(f'\t{_param_name[0].upper() + _param_name[1:]}: {_value}')
        print('Enabled features:')
        for _feature_class, _feature_names in features_enabled.items():
            print(f'\t{_feature_class.upper()}:\n\t\t' + '\n\t\t'.join(_feature_names))
    return extractor


def check_label_presence(mask: Union[str, nib.Nifti1Image, np.ndarray], selected_labels: List[int]):
    if isinstance(mask, str):
        # noinspection PyUnresolvedReferences
        mask = nib.load(file_mask).get_fdata().astype(np.uint8)
    elif isinstance(mask, nib.Nifti1Image):
        mask = mask.get_fdata().astype(np.uint8)
    labels = np.unique(mask[mask > 0])
    labels_absent = [_l for _l in selected_labels if _l not in labels]
    labels_caution = [_l for _l in labels if 0 < np.sum(mask == _l) <= 100]
    if len(labels_absent) == 0 and len(labels_caution) == 0:
        result = None
    elif len(labels_absent) != 0:
        result = {'Absence': labels_caution}
    else:
        result = {'Caution': labels_caution}
    return result


# noinspection SpellCheckingInspection
def get_voxel_based_radiomic_features(extractor: Union[str, featureextractor.RadiomicsFeatureExtractor],
                                      file_image: str, file_mask: str, selected_labels: List[int],
                                      dict_feature_map: NestedDict, cls_map: Union[str, dict] = 'total',
                                      voxel_norm: bool = True, show_feature_map: Union[str, None] = None,
                                      remove_bed: int = 0, preset_bin_widths: Union[List[int], None] = None,
                                      **kwargs):
    """
    This function is supposed to work for any SITK-supported format. However, nifti image has been tested. A potential
    risk with other image format could be that the image orientation differs from one's expectation.

    :param extractor: file name of the parameter configuration file that specifies the non-tissue specific settings,
                      or extractor engine generated from the parameter configuration file;
    :param file_image: original image file name of any SITK-supported format;
    :param file_mask: segmentation file name of any SITK-supported format;
    :param selected_labels: Make sure that the labels are contained in the segmentation result from the file_mask,
                            as label 1 can represent completely different anatomy in TotalSegmentator;
    :param cls_map: string name of class map that specifies what each integer label represents (default: 'total') or
                    a class map dict.
    :param voxel_norm: bool on whether to perform voxel number normalization to improve robustness of
                       voxel-based feature maps, default True;
    :param dict_feature_map: nested dictionary used to store the extracted feature maps;
                             Inspite of being a nested dict, it takes a flat hierarchy for storing the anatomies. It is
                             a key difference from the graph representation. An associated function for cross-reference
                             to graph representation is needed!!! THIS HOWEVER IS NOT IMPLEMENTED YET!!!
    :param show_feature_map: str '2d' or '3d' that specifies how the visualization should be performed, or not at all
                             in case it is set to None, which is the default value.
    :param remove_bed: int on method to remove bed from image array. Valid only if show_feature_map is not None.
                       0: no bed removal
                       1: remove bed with TotalSegmentator
                       2: remove bed using inbuilt method for numpy array
    :param preset_bin_widths: list of bin widths that has one-to-one correspondance with selected_labels.
                              If None, it is computed from the data.
    :param kwargs: print_params (bool), whether to print feature extraction settings and enabled features;
                   target_voxel_size (float), in case resized image is to be used for extracting features;
                   kernel_radius (int), in case a different value than the preset value contained in the parameter
                                        configuration file is to be used.
                   bin_width_opt (np.ndarray), in case a different array of optional bin widths based which the
                   tissue-type specific bin width will be calculated.
    :return:
    """

    # noinspection SpellCheckingInspection
    def resize_sitk_image(_image: sitk.Image, is_mask=False):
        filter_resample = sitk.ResampleImageFilter()
        _voxel_size = np.array(_image.GetSpacing())
        _size = np.array(_image.GetSize())
        _size_new = [int(e) for e in np.round(_size*_voxel_size/target_voxel_size)]
        filter_resample.SetOutputSpacing([target_voxel_size] * 3)
        filter_resample.SetSize(_size_new)
        filter_resample.SetOutputOrigin(_image.GetOrigin())
        filter_resample.SetOutputDirection(_image.GetDirection())
        filter_resample.SetTransform(sitk.Transform())
        if is_mask:
            filter_resample.SetInterpolator(sitk.sitkNearestNeighbor)
        else:
            filter_resample.SetInterpolator(sitk.sitkBSpline)
        return filter_resample.Execute(_image)

    # noinspection SpellCheckingInspection
    def get_feature_map(_feature_map_sitk, _num_mask_voxels):
        x_0, y_0, z_0 = np.min(mask_coord, axis=0)
        x_s, y_s, z_s = np.max(mask_coord, axis=0) - np.min(mask_coord, axis=0) + 1
        _feature_map, _ = convert_sitk_image_to_numpy(_feature_map_sitk)
        if voxel_norm:
            # Original voxel-based feature maps are found to be voxel number dependent. Normalize the feature
            # map by number of voxels of VOI improves feature robustness.
            _feature_map /= np.log10(_num_mask_voxels)

        # The feature map returned by radiomics package is cropped by a bounding box and padded with additional
        # margins that correspond to the kernel radius.
        # Put cropped feature map back to original size. Anything outside the bbx is set to zero.
        _feature_map_full = np.zeros(dict_feature_map['Shape'])
        # We remove these paddings to ensure that there will be no overlaps between different segments.
        _feature_map_full[x_0:x_0 + x_s, y_0:y_0 + y_s, z_0:z_0 + z_s] = _feature_map[ks:-ks, ks:-ks, ks:-ks]
        return _feature_map_full

    if 'target_voxel_size' in kwargs.keys():
        resize = True
        target_voxel_size = kwargs['target_voxel_size']
    else:
        resize = False
    if preset_bin_widths is None:
        bin_width_opt = kwargs.get('bin_width_opt', np.array([2, 5, 10, 20, 40, 50]))
        if 'bin_width_opt' in kwargs.keys():
            kwargs.pop('bin_width_opt')
    else:
        assert len(preset_bin_widths) == len(selected_labels)
        bin_width_opt = None
    if isinstance(cls_map, str):
        cls_map = get_v_dependent_cls_map(cls_map)
    if isinstance(extractor, str):
        extractor = get_feature_extractor(extractor, True, **kwargs)
    else:
        if 'kernel_radius' in kwargs.keys():
            if kwargs['kernel_radius'] <= 0:
                print_highlighted_text(f'Reset the kernel radius to 1, to fulfil the minimum value requirement for '
                                       f'voxel-based feature extraction.')
                extractor.settings['kernelRadius'] = 1
            else:
                extractor.settings['kernelRadius'] = kwargs['kernel_radius']
    # Instead of taking image and mask file names as input to extractor, we take the loaded images. This is prefered
    # in case multiple labels will be processed, so we can avoid repeatedly loading them.
    image = sitk.ReadImage(file_image)
    mask = sitk.ReadImage(file_mask)
    if not all([abs(a - b) <= extractor.settings['geometryTolerance'] for a, b in
                zip(image.GetSpacing(), mask.GetSpacing())]):
        seg_config = get_seg_config_by_task_name('total')
        _ = perform_segmentation_generic(file_image, seg_config)
        mask = sitk.ReadImage(file_mask)
    if resize:
        image = resize_sitk_image(image, False)
        mask = resize_sitk_image(mask, True)
    image_arr, _, voxel_size = convert_sitk_image_to_numpy(image)
    mask_arr, _ = convert_sitk_image_to_numpy(mask)
    if show_feature_map is not None:
        if remove_bed > 0:
            if remove_bed == 1:
                seg_config = get_seg_config_by_task_name('body')
                img_body = perform_segmentation_generic(file_image, seg_config)
                if resize:
                    # noinspection PyUnboundLocalVariable
                    img_body = image_resample(img_body, target_voxel_size=target_voxel_size, remove_negative=True)
                image_arr_body, _ = convert_nifti_to_numpy(img_body)
                mask_body = image_arr_body == 1
                image_arr[~mask_body] = image_arr.min()
            elif remove_bed == 2:
                image_arr = remove_bed_with_scipy(image_arr)
    if all([s == voxel_size[0] for s in voxel_size]):
        dict_feature_map['VoxelSize'] = voxel_size[0]
    else:
        voxel_aspect_ratios = np.array(voxel_size)/voxel_size[0]
        if all([np.isclose(x, 1.0, atol=0.001) for x in voxel_aspect_ratios]):
            dict_feature_map['VoxelSize'] = voxel_size[0]
        else:
            dict_feature_map['VoxelSize'] = voxel_size
    dict_feature_map['Shape'] = image_arr.shape
    if preset_bin_widths is None:
        preset_bin_widths = set_type_bin_width(image_arr, mask_arr, selected_labels, bin_width_opt)
    for i, _label in enumerate(selected_labels):
        extractor.settings['binWidth'] = preset_bin_widths[_label]
        features_extracted = extractor.execute(image, mask, label=_label, voxelBased=True)
        mask_binary = mask_arr == _label
        num_voxels = np.sum(mask_binary)
        mask_coord = np.argwhere(mask_binary).astype(np.int16)
        dict_feature_map[cls_map[_label]]['MaskCoord'] = mask_coord
        ks = extractor.settings['kernelRadius']
        # Not needed really for subsequent calcuation but for documentation only, in case that different anatomies are
        # set to have different kernel radius.
        dict_feature_map[cls_map[_label]]['KernelRadius'] = ks
        for k, v in six.iteritems(features_extracted):
            if isinstance(v, sitk.Image):
                _feature_cls, _feature_name = k.split('_')[1:]
                print(f'Extract feature: {_feature_cls} - {_feature_name}')
                feature_map = get_feature_map(v, num_voxels)
                dict_feature_map[cls_map[_label]][_feature_cls][_feature_name] = feature_map[mask_binary]
                if show_feature_map is not None:
                    if show_feature_map == '2d':
                        # Split the name to words by finding capital letters in feature name string
                        _feature_name = ' '.join([e.lower() for e in re.findall('[A-Z][^A-Z]*', _feature_name)])
                        _anatomy_name_substr = cls_map[_label].split('_')
                        if len(_anatomy_name_substr) > 1:
                            if _anatomy_name_substr[-1] in ['left', 'right']:
                                _anatomy_name_substr.insert(0, _anatomy_name_substr.pop())
                            _anatomy_name = ' '.join(_anatomy_name_substr).capitalize()
                        else:
                            _anatomy_name = _anatomy_name_substr.capitalize()
                        z0 = np.min(mask_coord[:, 2])
                        zs = z0 + (np.max(mask_coord[:, 2]) - z0 + 1) // 2
                        plt.imshow(image_arr[:, :, zs], cmap='gray')
                        # noinspection PyUnboundLocalVariable
                        feature_map_to_plot = get_voxel_feature_maps_for_plot(feature_map, image_arr.shape,
                                                                              dict_feature_map[cls_map[_label]]
                                                                              ['MaskCoord'])
                        plt.imshow(feature_map_to_plot[:, :, zs, :])
                        plt.title(f"{_anatomy_name} - {_feature_cls}: {_feature_name}")
                        plt.axis('off')
                        plt.colorbar()
                        plt.show()


def get_robustness_improved_voxel_based_features(extractor: Union[str, featureextractor.RadiomicsFeatureExtractor],
                                                 file_image: str, file_mask: str, selected_labels: List[int],
                                                 target_kr_range: list, dict_feature_map: NestedDict,
                                                 cls_map: Union[str, dict] = 'total', include_default=True,
                                                 preset_bin_widths: Union[List[int], None] = None):
    if isinstance(cls_map, str):
        cls_map = get_v_dependent_cls_map(cls_map)
    if isinstance(extractor, str):
        extractor = get_feature_extractor(extractor)
    kr = extractor.settings['kernelRadius']
    if include_default:
        if kr not in target_kr_range:
            target_kr_range.insert(0, kr)
    features_enabled = extractor.enabledFeatures
    dict_feature_maps = NestedDict()
    for _l in selected_labels:
        obj_str = cls_map[_l]
        for _fc in features_enabled.keys():
            for _fn in features_enabled[_fc]:
                dict_feature_maps[obj_str][_fc][_fn] = list()
    scaler = StandardScaler()
    for i, _kr in enumerate(target_kr_range):
        _dict_fmap = NestedDict()
        extractor.settings['kernelRadius'] = _kr
        get_voxel_based_radiomic_features(extractor, file_image, file_mask, selected_labels, _dict_fmap, cls_map,
                                          preset_bin_widths=preset_bin_widths)
        for _l in selected_labels:
            obj_str = cls_map[_l]
            if i == 0:
                if 'Shape' not in dict_feature_map.keys():
                    dict_feature_map['Shape'] = _dict_fmap['Shape']
                if obj_str not in dict_feature_map.keys():
                    dict_feature_map[obj_str]['MaskCoord'] = _dict_fmap[obj_str]['MaskCoord']
            for _fc in features_enabled.keys():
                for _fn in features_enabled[_fc]:
                    _fmap = scaler.fit_transform(_dict_fmap[obj_str][_fc][_fn].reshape(-1, 1))
                    dict_feature_maps[obj_str][_fc][_fn].append(_fmap)

    for _l in selected_labels:
        obj_str = cls_map[_l]
        for _fc in features_enabled.keys():
            for _fn in features_enabled[_fc]:
                _fmap = np.mean(scaler.fit_transform(np.hstack(dict_feature_maps[obj_str][_fc][_fn])), axis=1)
                dict_feature_map[obj_str][_fc][_fn] = _fmap


def concatenate_feature_maps_in_dict(dict_feature_map: NestedDict, flatten_array: bool = True):
    def concatenate_feature_maps(anatomy_name):
        _list_feature_cls = list(set(dict_feature_map[anatomy_name].keys()) - {'MaskCoord', 'KernelRadius'})
        _list_feature_vec = list()
        for _fc in _list_feature_cls:
            for _fn, _fmap in dict_feature_map[anatomy_name][_fc].items():
                print(f'Append feature: {_fc}-{_fmap} to feature maps')
                _list_feature_vec.append(_fmap)
        return _list_feature_vec

    def get_feature_number():
        _anatomy = anatomies_selected[0]
        _list_fc = list(set(dict_feature_map[_anatomy].keys()) - {'MaskCoord', 'KernelRadius'})
        _n_features = sum([len(dict_feature_map[_anatomy][_fc]) for _fc in _list_fc])
        return _n_features

    anatomies_selected = [k for k, v in dict_feature_map.items() if isinstance(v, dict)]
    list_mask_coord = list()
    n_features = get_feature_number()
    list_feature_vec_all = [[] for i in range(n_features)]
    scaler = StandardScaler()
    for k in anatomies_selected:
        mask_coord = dict_feature_map[k]['MaskCoord']
        list_mask_coord.append(mask_coord)
        list_feature_vec = concatenate_feature_maps(k)
        feature_vec = scaler.fit_transform(np.vstack(list_feature_vec))
        # Using max and min to normalize the feature vector to value between 1 and 255.
        feature_vec = (feature_vec - np.min(feature_vec)) / (np.max(feature_vec) - np.min(feature_vec)) * 254 + 1
        # Cast the value to int first and then normalize by 255 to value between 1/255 and 1.
        feature_vec = feature_vec.astype(np.uint8).astype(np.float32)/255
        for i in range(n_features):
            list_feature_vec_all[i].append(feature_vec[i, :])
    mask_coord_all = np.vstack(list_mask_coord)
    data_shape = dict_feature_map['Shape']
    fmap_new = np.zeros((np.prod(data_shape), n_features))
    for i in range(n_features):
        # Values outside of mask will be filled with zeros.
        fmap_new[:, i] = reconstruct_data_from_coo_components(mask_coord_all, np.ravel(list_feature_vec_all[i]),
                                                              data_shape, 0).ravel()
    if flatten_array:
        return fmap_new.ravel().reshape(1, -1)
    else:
        return fmap_new


def get_voxel_feature_maps_for_plot(feature_maps: np.ndarray, data_shape: tuple,
                                    mask_coord: Union[np.ndarray, None] = None):
    # It is important to note that it does not only append an alpha mask layer but also normalize the features to
    # values between 1 and 255. Zeros are filled to non-ROI regions.
    try:
        assert feature_maps.shape[3] <= 3
    except IndexError:
        feature_maps = feature_maps[..., np.newaxis]
    except AssertionError:
        raise Exception(f'The input feature maps has {feature_maps.shape[3]} channels, however since only RGB color '
                        f'is supported by matplotlib, the visualization is only possible for <= 3 channels!')
    if mask_coord is None:
        # After applying standard scaler, the feature maps contain both negative and positive values. The regions of
        # zeros are outside where the feature maps were calculated.
        mask_binary = feature_maps[..., 0] != 0
    else:
        mask_binary = recover_3d_mask_array_from_coord(mask_coord, data_shape)
    n_components = feature_maps.shape[3]
    if n_components > 1:
        mask_channels = np.repeat(mask_binary[..., np.newaxis], n_components, axis=3)
    else:
        # Faster for repeats = 1 compared with the expression above, though results are identical.
        mask_channels = mask_binary[..., np.newaxis]
    feature_maps_ma = np.ma.masked_array(feature_maps, mask=~mask_channels)
    for i in range(n_components):
        feature_maps_ma[..., i] = (feature_maps_ma[..., i] - np.ma.min(feature_maps_ma[..., i])) / \
                                  (np.ma.max(feature_maps_ma[..., i]) - np.ma.min(feature_maps_ma[..., i])) * 254 + 1
    feature_maps_ma = feature_maps_ma.filled(fill_value=0)
    if n_components == 1:
        # Append green and blue channels of zeros and an alpha channel
        feature_maps_ma = np.concatenate((feature_maps_ma / 255, np.zeros(tuple(list(data_shape) + [2])),
                                          1 * mask_binary[..., np.newaxis]), axis=3)
    elif n_components == 2:
        # Append blue channel of zeros and an alpha channel
        feature_maps_ma = np.concatenate((feature_maps_ma / 255, 0 * mask_binary[..., np.newaxis],
                                          1 * mask_binary[..., np.newaxis]), axis=3)
    else:
        # Append an alpha channel
        feature_maps_ma = np.concatenate((feature_maps_ma / 255, 1 * mask_binary[..., np.newaxis]), axis=3)
    return feature_maps_ma


def visualize_voxel_based_features_after_alpha_appending(image_array: np.ndarray, feature_maps: np.ndarray,
                                                         n_components: int = 2, x: Union[int, None] = None,
                                                         z: Union[int, None] = None, print_subplot_title: bool = True,
                                                         feature_names: Union[List[str], str, None] = None,
                                                         apply_tissue_window: bool = True,
                                                         voxel_size: tuple = (1, 1, 1)):
    assert n_components in [1, 2, 3]
    col = n_components // 2 + 2
    if isinstance(feature_names, list):
        try:
            assert len(feature_names) == n_components
        except AssertionError:
            raise Exception("The provided number of feature names should be identical to 'n_components'!")
    if apply_tissue_window:
        import bodyPartWindow
        dict_window = bodyPartWindow.dict_window['Abdomen']['SoftTissue']
        image_array = bodyPartWindow.get_window_image(image_array, dict_window)

    img_shape = image_array.shape
    if n_components == 1:
        mask = feature_maps[..., 0] > 0
    else:
        mask = feature_maps[..., 1] > 0
    if x is None or z is None:
        x_0, _, z_0 = np.mean(np.argwhere(mask), axis=0)
        x = int(x_0)
    if z is None:
        # noinspection PyUnboundLocalVariable
        z = int(z_0)
    px = 1 / plt.rcParams['figure.dpi']
    aspect_ratios = np.array(voxel_size) / voxel_size[0]
    aspect_ratio_z = round(aspect_ratios[2] * img_shape[2] / img_shape[0], 2)
    fig, axs = plt.subplots(2, col, figsize=(1000 * px, 650 * px), constrained_layout=True,
                            gridspec_kw={'width_ratios': [aspect_ratio_z, 1],
                                         'height_ratios': [1, 1]} if col == 2 else
                            {'width_ratios': [aspect_ratio_z, aspect_ratio_z, 1],
                                         'height_ratios': [1, 1]}
                            )

    if n_components == 1:
        if print_subplot_title:
            subplot_titles = ['Axial view image', 'Coronal view image', 'Axial feature map', 'Coronal feature map']
            if feature_names is not None:
                for i in [2, 3]:
                    if isinstance(feature_names, str):
                        subplot_titles[i] += f' - {feature_names}'
                    else:
                        subplot_titles[i] += f' - {feature_names[0]}'
        for r in range(2):
            for c in range(2):
                axs[r, c].set_axis_off()
                if print_subplot_title:
                    # noinspection PyUnboundLocalVariable
                    axs[r, c].title.set_text(subplot_titles[2*r+c])
                if c == 0:
                    axs[r, c].imshow(np.fliplr(image_array[:, :, z]), cmap='gray', aspect=aspect_ratios[0])
                else:
                    axs[r, c].imshow(np.flipud(image_array[x, :, :].T), aspect=aspect_ratios[2])
                if r == 1:
                    if c == 0:
                        axs[r, c].imshow(np.fliplr(feature_maps[:, :, z, :]), aspect=aspect_ratios[0])
                    else:
                        # Switch x and y-axis as matrix transpose, however without affecting the channel dimension.
                        axs[r, c].imshow(np.flipud(np.rollaxis(feature_maps[x, :, :, :], 1, 0)),
                                         aspect=aspect_ratios[2])

    else:
        if print_subplot_title:
            if n_components == 2:
                subplot_titles = ['Axial view image', 'Axial feature map 1', 'Axial feature map 2']
            else:
                subplot_titles = ['Axial view image', 'Axial feature map 1', 'Axial feature map 2',
                                  'Axial feature map 3']
            if isinstance(feature_names, list):
                for i in range(len(feature_names)):
                    subplot_titles[i + 1] += f': {feature_names[i]}'
            if n_components == 2:
                subplot_titles.append('Composite axial feature maps')
            subplot_titles.extend(['Coronal view image', 'Composite coronal feature maps'])
        # Plot colum 0 and 1:
        for i in range(4):
            axs[i // 2, i % 2].imshow(np.fliplr(image_array[:, :, z]), cmap='gray', aspect=aspect_ratios[0])
            axs[i // 2, i % 2].set_axis_off()
            if print_subplot_title:
                # noinspection PyUnboundLocalVariable
                axs[i // 2, i % 2].title.set_text(subplot_titles[i])
            if i > 0:
                feature_arr_channel = feature_maps.copy()
                if n_components == 2:
                    if i in [1, 2]:  # No modification will be made to the last composite feature maps.
                        if i == 1:
                            # Set green channel to zero
                            feature_arr_channel[..., 1] = 0
                        else:
                            # Set red channel to zero
                            feature_arr_channel[..., 0] = 0
                else:  # n_components == 3
                    for _i in list(set(range(3)) - {i-1}):
                        feature_arr_channel[..., _i] = 0
                axs[i // 2, i % 2].imshow(np.fliplr(feature_arr_channel[:, :, z, :]), aspect=aspect_ratios[0])

        # Plot colum 2 of all rows:
        for i in range(2):
            axs[i, 2].imshow(np.flipud(image_array[x, :, :].T), cmap='gray', aspect=aspect_ratios[2])
            if i > 0:
                axs[i, 2].imshow(np.flipud(np.rollaxis(feature_maps[x, :, :, :], 1, 0)), aspect=aspect_ratios[2])
            axs[i, 2].set_axis_off()
            if print_subplot_title:
                axs[i, 2].title.set_text(subplot_titles[i+4])
    plt.show()

