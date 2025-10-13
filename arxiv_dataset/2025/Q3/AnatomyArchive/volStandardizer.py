import operator
import numpy as np
from typing import Union
import volumeViewerMPL
from functools import reduce
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from loggerConfig import logger
from skimage.measure import find_contours
from util import NestedDict, print_highlighted_text
from volStandardizer_reg import get_lb_bright_objects
from simpleStats import get_index_of_plane_with_largest_mask_area
from segModel import remove_segments_with_size_limit, get_v_dependent_cls_map
from simpleGeometry import get_2d_oriented_bbox_from_mask, get_side_from_closed_curve, get_2d_bool_mask_boundary_points


def add_tag_to_data(dataset_tag: NestedDict, tag: str, data_id: str, tag_type: str = 'Error'):
    if tag not in dataset_tag[tag_type].keys():
        dataset_tag[tag_type][tag] = list()
    if data_id not in dataset_tag[tag_type][tag]:
        dataset_tag[tag_type][tag].append(data_id)


def get_object_area_at_indexed_plane(_mask: Union[np.ndarray, bool], voxel_size: Union[tuple, list, np.ndarray],
                                     half_width: int = 1, label: Union[int, None] = None,
                                     index_plane: Union[int, None] = None):
    """

    :param _mask: mask array, can be a bool array or int-typed array;
    :param voxel_size: voxel size in 3d with unit of mm;
    :param half_width: half-width including self, therefore must be >=1. If >1, the returned area is an average over
                       half-width -1 below and above the central plane defined by the centroid of the mask.
    :param label: int or None. It is used to get the mask coordinates in case the input mask array is int-typed.
    :param index_plane: int or None.
    :return: obj_area: in unit cm^2 or also index_plane.
    """
    assert half_width >= 1
    if _mask.dtype == bool:
        mask_coord = np.argwhere(_mask)
    else:
        assert isinstance(label, int)
        mask_coord = np.argwhere(_mask == label)
    if index_plane is None:
        return_index = True
        index_plane = np.round(np.mean(mask_coord, axis=0)).astype(np.int16)[2]
    else:
        return_index = False
    voxel_area = voxel_size[0] * voxel_size[1]  # mm^2 in axial plane
    if half_width == 1:
        obj_area = np.sum(mask_coord[:, 2] == index_plane) * voxel_area / 100
    else:
        obj_area = np.sum(np.logical_and(index_plane + 1 - half_width <= mask_coord[:, 2], mask_coord[:, 2] <=
                                         index_plane + half_width - 1)) / (2 * half_width - 1) * voxel_area / 100
    if return_index:
        return obj_area, index_plane
    else:
        return obj_area


def set_central_ref_plane(img_3d: np.ndarray, mask_3d: np.ndarray, voxel_size: Union[list, np.ndarray, tuple],
                          cls_map_name: str, obj_ref: dict, plot_image: bool = False,
                          save_fig_name: Union[str, None] = None, dicom_id: Union[str, None] = None,
                          save_ref_plane_mask: bool = False):
    cls_map = get_v_dependent_cls_map(cls_map_name)
    list_anatomies = list(cls_map.values())
    list_labels = list(cls_map.keys())
    obj_ref_name = list(obj_ref.keys())[0] if 'Name' not in obj_ref.keys() else obj_ref['Name']
    mask = mask_3d == list_labels[list_anatomies.index(obj_ref_name)]
    area_obj, index_plane = get_object_area_at_indexed_plane(mask, voxel_size)
    ratio = np.array(voxel_size) / voxel_size[0]
    obj_ref[obj_ref_name] = index_plane
    print(f'Reference central plane index of {obj_ref} for all selected anatomies is {index_plane}')
    if save_ref_plane_mask:
        obj_ref['centralPlaneMask'] = mask[:, :, index_plane].astype(np.uint8)
    if plot_image:
        fig, ax = plt.subplots()
        # noinspection PyTypeChecker
        idx_sag = get_index_of_plane_with_largest_mask_area(mask * 1, dim='x', aspect_ratios=ratio)
        ax.imshow(np.flipud(img_3d[idx_sag, :, :].T), cmap='gray', aspect=ratio[2])
        ax.imshow(np.flipud(mask[idx_sag, :, :].T) * 1.0, cmap='Reds', alpha=0.5, aspect=ratio[2])
        ax.plot(np.ones(img_3d.shape[0]) * (img_3d.shape[2] - index_plane), color='r')
        if dicom_id is not None:
            plt.title(dicom_id)
        plt.axis('off')
        if save_fig_name is not None:
            plt.savefig(save_fig_name)
            plt.close(fig)
        else:
            plt.show()


# noinspection SpellCheckingInspection
def define_volume_bounds_by_objects(img_3d_seg: np.ndarray, obj_ref: dict, cls_map_name: str = 'total',
                                    data_id: Union[str, None] = None, dataset_tag: Union[NestedDict, None] = None):
    """
    :param img_3d_seg: 3D segmented numpy array.
    :param obj_ref: dict that specifies which bones are used to set the upper- and lower-bound. Bones that have left
                    and right counterparts are supported by taking the common string without left and right
                    specification, for example 'pelvic'. However, ribs are not supported yet.
    :param cls_map_name: name of class map that should contain the objects which define the upper- and lower-bound.
    :param data_id: str or None. It is useful for logging especially if it is used for deselection of images.
    :param dataset_tag: NestedDict or None. It is useful for logging and accumulating deselected the image data and the
                        reasons.
    :return: bounds: dict with keys of 'UB' & 'LB' with value ranging from 0 to z_max, if both bounds can be properly
                     set. Otherwise, negative values are returned as flags to indicate failure due to different reasons.
                     -1: Bound-defining object missing.
                         If any -1 case occurs, the entire bound calculations will be skipped.
                     -2: Bound-defining object cropped.
                     If 'pelvic' is used to define the lower boundary, this function tries to detect whether prosthesis
                     is present, and additional information with keyword 'ProsthesisDetected' will be added.

    """

    def verify_ref_obj(_key: str):
        _label_obj = [k for k, v in cls_map.items() if obj_ref[_key] in v]
        if len(_label_obj) != 1:
            if len(_label_obj) == 0:
                raise ValueError(f'Reference object {obj_ref[_key]} is not contained in the class map with name of '
                                 f'{cls_map_name}!\nPlease double check the setting!')
            elif len(_label_obj) == 2:
                obj_names = [cls_map[k] for k in _label_obj]
                try:
                    assert any([_sub in obj_names[0] for _sub in ['left', 'right']])
                except AssertionError:
                    raise ValueError(f'If non-uniquely defined reference {obj_ref[_key]} is used, only bones with left '
                                     f'and right counterparts are supported. However, the provided reference objects '
                                     f'are {[cls_map[k] for k in _label_obj]} according to class map {cls_map_name}!')
                return {'Name': obj_ref[_key], 'Label': _label_obj}
            else:
                raise ValueError(f'If non-uniquely defined reference {obj_ref[_key]} is used, only bones with left '
                                 f'and right counterparts are supported. However, {[cls_map[k] for k in _label_obj]} '
                                 f'according to class map {cls_map_name} are used to define {_key}!')
        else:
            return {'Name': obj_ref[_key], 'Label': _label_obj[0]}

    def get_vert_boundary_line(_label: int, side: str):
        _mask = mask_vert == _label
        _mask_sag = np.flipud(np.max(_mask[:, y_index - slab_hw:y_index + slab_hw, :], axis=1).T)
        bbox, _, _flag = get_2d_oriented_bbox_from_mask(_mask_sag)
        _line = get_side_from_closed_curve(bbox, side)
        # Get the left and right boundary points from _line.
        # Caution: this is not a generic solution for a non-straight line which contains many points.
        points = np.vstack((_line[np.argmin(_line[:, 0]), :], _line[np.argmax(_line[:, 0]), :]))
        return points, _flag

    def get_median_label_dist(_labels):
        idx_cont = np.where(np.ediff1d(_labels) == 1)[0]
        n_pair = len(idx_cont)
        if n_pair == 0:
            dist = -1
            if dataset_tag is not None:
                add_tag_to_data(dataset_tag, 'VertSegError', data_id)
            logger.error(f'No continuous labels found in {_labels}!\nPlease double check the input segments!')
        else:
            dist_arr = np.zeros(n_pair)
            for _i in range(n_pair):
                _line_upper, _ = get_vert_boundary_line(_labels[idx_cont[_i]] + 1, 'b')
                _line_lower, _ = get_vert_boundary_line(_labels[idx_cont[_i]], 't')
                dist_arr[_i] = np.mean(np.linalg.norm(_line_upper - _line_lower, axis=1))
            dist = np.median(dist_arr)/2
        return dist

    # noinspection SpellCheckingInspection
    def get_bounds_non_vert_ref_obj(_mask):
        z_coord = np.argwhere(_mask)[:, 2]
        if _bound == 'LB':
            thre = 0.02
        else:
            thre = 99.98
        return np.percentile(z_coord, thre), z_coord

    def _append_data_id_str():
        return f" for data {data_id}" if isinstance(data_id, str) else ""

    if dataset_tag is not None:
        assert data_id is not None
    cls_map = get_v_dependent_cls_map(cls_map_name)
    # Update the obj_ref such that it becomes a nested dict.
    if 'UB' in obj_ref:
        obj_ref['UB'] = verify_ref_obj('UB')
    obj_ref['LB'] = verify_ref_obj('LB')
    labels_found = np.unique(img_3d_seg[img_3d_seg > 0])
    for _bound, _obj_ref in obj_ref.items():
        _obj_label = _obj_ref['Label']
        if isinstance(_obj_label, int):
            if _obj_label not in labels_found:
                obj_ref[_bound]['boundPlane'] = -1
                if dataset_tag is not None:
                    add_tag_to_data(dataset_tag, f"{_obj_ref['Name']}Missing", data_id)
                logger.error(f"The reference object {_obj_ref['Name']} absent in input mask array!")
            else:
                obj_ref[_bound]['boundPlane'] = None
        else:
            # List of objects with left and right counterparts
            if any([_l not in labels_found for _l in _obj_label]):
                obj_missed = [cls_map[_l] for _l in _obj_label if _l not in labels_found]
                obj_ref[_bound]['boundPlane'] = -1
                if dataset_tag is not None:
                    if len(obj_missed) > 1:
                        tag_str = f"{obj_missed[0]}{obj_missed[1].capitalize()}Missing"
                    else:
                        tag_str = f"{obj_missed[0]}Missing"
                    add_tag_to_data(dataset_tag, tag_str, data_id)
                logger.error(f'The reference object {obj_missed} absent in input mask array!')
            else:
                obj_ref[_bound]['boundPlane'] = None
    if all([_v['boundPlane'] is None for _v in obj_ref.values()]):
        # As we will take vertical distance between vertebral bodies as reference for defining margins, regardless
        # whether any of vertebral bones is taken as the reference object, we try to find all present vertebrae.
        labels_vert_found = [k for k, v in cls_map.items() if 'vertebrae' in v and k in labels_found]
        mask_vert = img_3d_seg.copy()
        # Get a list of masks which do not correspond to vertebrae.
        list_mask = list()
        for i, _l in enumerate(labels_vert_found):
            if np.sum(img_3d_seg == _l) > 1000:
                list_mask.append(img_3d_seg == _l)
            else:
                labels_vert_found.pop(i)
        mask_vert[~reduce(operator.xor, list_mask)] = 0
        slab_hw = 5
        height_vol = img_3d_seg.shape[2]
        y_index = get_index_of_plane_with_largest_mask_area(mask_vert, dim='y')
        dist_margin = get_median_label_dist(labels_vert_found)
        for _bound, _obj_ref in obj_ref.items():
            _obj_name = _obj_ref['Name']
            _obj_label = _obj_ref['Label']
            if 'vert' in _obj_name and _bound == 'UB':
                # Set the lower boundary line of the final upperbound using the upper side of reference vertebra.
                line_lower, flag = get_vert_boundary_line(_obj_label, 't')
                if flag != 1:
                    # flag of 1 suggests that the upper-side of reference vertebra is cut.
                    if (_obj_label + 1) in labels_vert_found:
                        # TotalSegmentator names the vertebrae using reversed order of int labels.
                        # Use the bottom line of vertebra that lies above the reference as the upper line.
                        line_upper, _ = get_vert_boundary_line(_obj_label + 1, 'b')
                        upperbound = int(np.round(np.mean(np.vstack((line_upper, line_lower)), axis=0)[1]))
                    else:
                        line_lower, _ = get_vert_boundary_line(_obj_label, 't')
                        upperbound_ = np.mean(line_lower)
                        if upperbound_ - dist_margin < 0:
                            upperbound = 0
                            logger.warning('The reference object is very close to the upper boundary without '
                                           'enough margin!')
                        else:
                            upperbound = int(np.round(upperbound_ - dist_margin))
                    # Reset upperbound in z by indexing from image bottom. This is needed because the coordinate
                    # calculation in the nested function 'get_boundary_line' involves an upside-down flipping.
                    obj_ref[_bound]['boundPlane'] = height_vol - upperbound
                    print(f"Upperbound: {obj_ref[_bound]['boundPlane']}")
                else:
                    obj_ref[_bound]['boundPlane'] = -2
                    if dataset_tag is not None:
                        add_tag_to_data(dataset_tag, f'{_obj_name}Cropped', data_id)
                    logger.error(f'The upperbound reference object {_obj_name} {_append_data_id_str()} is cropped. '
                                 f'Please double check the data!')
            elif 'vert' in _obj_name and _bound == 'LB':
                # Set the upper boundary line of the final lower-bound using the lower side of reference vertebra.
                line_upper, flag = get_vert_boundary_line(_obj_label, 'b')
                if flag != 3:
                    if (_obj_label - 1) in labels_vert_found:
                        line_lower, _ = get_vert_boundary_line(_obj_label - 1, 't')
                        lowerbound = int(np.round(np.mean(np.vstack((line_upper, line_lower)), axis=0)[1]))
                    else:
                        line_lower, _ = get_vert_boundary_line(_obj_label, 'b')
                        lowerbound_ = np.mean(line_lower)
                        if lowerbound_ + dist_margin > height_vol:
                            lowerbound = height_vol
                            logger.warning('The reference object is very close to the lower boundary!')
                        else:
                            lowerbound = int(np.round(lowerbound_ + dist_margin))
                    # Reset lowerbound in z by indexing from image bottom.
                    obj_ref[_bound]['boundPlane'] = height_vol - lowerbound
                    print(f"Lowerbound: {obj_ref[_bound]['boundPlane']}")
                else:
                    # flag of 3 suggests that the lower-side of reference vertebra is cut.
                    obj_ref[_bound]['boundPlane'] = -2
                    if dataset_tag is not None:
                        add_tag_to_data(dataset_tag, f'{_obj_name}Cropped', data_id)
                    logger.error(f'The lowerbound reference object {_obj_name} {_append_data_id_str()} is cropped. '
                                 f'Please double check the data!')
            else:
                if isinstance(_obj_label, int):
                    bound_, coord = get_bounds_non_vert_ref_obj(img_3d_seg == _obj_label)
                else:
                    bound_, coord = get_bounds_non_vert_ref_obj(reduce(operator.xor, [img_3d_seg == _l
                                                                                          for _l in _obj_label]))
                if _bound == 'UB':
                    if height_vol - dist_margin < bound_ < height_vol:
                        obj_ref[_bound]['boundPlane'] = height_vol
                        logger.warning('The reference object is very close to the upper boundary without enough '
                                        'margin!')
                    elif bound_ == height_vol:
                        if np.sum(coord == height_vol) / np.sum(coord > bound_) > 0.25:
                            obj_ref[_bound]['boundPlane'] = -2
                            if dataset_tag is not None:
                                add_tag_to_data(dataset_tag, f'{_obj_name}Cropped', data_id)
                            logger.error(f'The upperbound reference object {_obj_name} {_append_data_id_str()} is'
                                         f' cropped. Please double check the data!')
                        logger.warning('The reference object is very close to the upper boundary without enough '
                                        'margin!')
                    else:
                        obj_ref[_bound]['boundPlane'] = int(np.round(bound_ + dist_margin))
                    print(f"Upperbound: {obj_ref[_bound]['boundPlane']}")
                else:
                    if 0 < bound_ < dist_margin:
                        obj_ref[_bound]['boundPlane'] = 0
                        logger.warning('The reference object is very close to the lower boundary without enough '
                                        'margin!')
                    elif bound_ == 0:
                        if np.sum(coord == 0) / np.sum(coord == 1) > 0.05:
                            obj_ref[_bound]['boundPlane'] = -2
                            if dataset_tag is not None:
                                add_tag_to_data(dataset_tag, f'{_obj_name}Cropped', data_id)
                            logger.error(f'The lowerbound reference object {_obj_name} {_append_data_id_str()} is '
                                         f'cropped. Please double check the data!')
                        else:
                            logger.warning('The reference object is very close to the lower boundary without enough '
                                            'margin!')
                    else:
                        obj_ref[_bound]['boundPlane'] = int(np.round(bound_ - dist_margin))
                    print(f"Lowerbound: {obj_ref[_bound]['boundPlane']}")


def prosthesis_detection_at_lower_bound(img_3d: np.ndarray, obj_ref: dict, plot_image: bool = False,
                                        voxel_size: Union[tuple, np.ndarray, list, None] = None,
                                        save_fig_name: Union[str, None] = None,
                                        img_3d_seg_ref: Union[np.ndarray, None] = None,
                                        data_id: Union[str, None] = None, dataset_tag: Union[NestedDict, None] = None):
    """

    :param img_3d: 3D image array.
    :param obj_ref: dict with keys of 'UB' and 'LB'. Check function define_volume_bounds_by_objects for more details.
    :param plot_image: bool on whether to display the image with bound lines.
    :param voxel_size: a list, numpy array or a tuple of 3 elements. It is only needed if image will be plotted.
    :param save_fig_name: save the figure without showing if a string is provided. Otherwise, show the result.
    :param img_3d_seg_ref: 3D labeled mask array or None, needed only for plotting.
    :param data_id: str or None. It is useful for logging especially if it is used for deselection of images.
    :param dataset_tag: NestedDict or None. It is useful for logging and accumulating deselected the image data and the
                        reasons.
    :return: prosthesis_detected: int, 0 or 1.
    """
    if dataset_tag is not None:
        assert data_id is not None
    lower_bound = obj_ref['LB']['boundPlane']
    z_max = img_3d.shape[2]
    if lower_bound >= 0:
        img_cor_cropped = np.flipud(np.max(img_3d[:, :, lower_bound:], axis=0).T)
        seg_labeled, seg_prosthesis = get_lb_bright_objects(img_cor_cropped)
        prosthesis_detected = 0 if np.isscalar(seg_prosthesis) else 1
        if prosthesis_detected:
            if dataset_tag is not None:
                add_tag_to_data(dataset_tag, 'prosthesisDetected', data_id, 'Warning')
    else:
        # No need to add annotation that the LB object is cropped to data_tag dict as this must already be handled.
        print_highlighted_text(f"Lower-bound object {obj_ref['LB']['Name']} is cropped, prosthesis detection is "
                               f"skipped!")
        seg_prosthesis = -1
        prosthesis_detected = -1

    if plot_image:
        if voxel_size is None:
            aspect_ratios = np.array([1, 1, 1])
        else:
            if not isinstance(voxel_size, np.ndarray):
                voxel_size = np.array(voxel_size)
            aspect_ratios = voxel_size/voxel_size[0]
        img_cor = np.flipud(np.max(img_3d, axis=0).T)
        if not np.isscalar(seg_prosthesis):
            # noinspection PyUnboundLocalVariable
            int_max = np.percentile(img_cor_cropped[~seg_labeled.astype(np.bool_)], 99.5)
            mask = np.zeros_like(img_cor)
            mask[:z_max - lower_bound:, :] = seg_prosthesis
            dict_colors = {1: (1, 1, 0)}
            fig, ax = volumeViewerMPL.show_mask_superimposed_2d_image(img_cor, mask, dict_colors,
                                                                      return_fig=True, vmax=int_max,
                                                                      aspect=aspect_ratios[2])
        else:
            seg_labeled, _ = get_lb_bright_objects(img_cor)
            if not np.isscalar(seg_labeled):
                int_max = np.percentile(img_cor[~seg_labeled.astype(np.bool_)], 99.5)
            else:
                int_max = np.percentile(img_cor, 99.5)
            fig, ax = volumeViewerMPL.get_sectional_view_image(img_3d, 'x', window_image=None, return_fig=True,
                                                               aspect=aspect_ratios[2], vmax=int_max)
        if lower_bound >= 0:
            ax.plot(range(img_cor.shape[1]), [z_max - lower_bound] * img_cor.shape[1], 'g-', alpha=0.5, linewidth=3)
        elif lower_bound == -2 and img_3d_seg_ref is None:
            ax.plot(range(img_cor.shape[1]), [z_max] * img_cor.shape[1], 'm-', alpha=0.5, linewidth=3)
        if 'UB' in obj_ref.keys():
            upper_bound = obj_ref['UB']['boundPlane']
            if upper_bound > 0:
                ax.plot(range(img_cor.shape[1]), [z_max - obj_ref['UB']['boundPlane']] * img_cor.shape[1], 'g-',
                        alpha=0.5, linewidth=3)
            elif upper_bound == -2:
                ax.plot(range(img_cor.shape[1]), [0] * img_cor.shape[1], 'r-', alpha=0.5, linewidth=3)
        if img_3d_seg_ref is not None:
            for _bound, _dict_obj_ref in obj_ref.items():
                edge_style_norm = 'c--' if _bound == 'UB' else 'b--'
                edge_style_crop = 'm--' if _bound == 'UB' else 'r--'
                if _dict_obj_ref['boundPlane'] != -1:
                    obj_label = _dict_obj_ref['Label']
                    if isinstance(obj_label, int):
                        mask = np.flipud(np.max((img_3d_seg_ref == obj_label).astype(np.uint8), axis=0).T)
                    else:
                        mask = np.flipud(np.max((reduce(operator.xor,
                                                        [img_3d_seg_ref == _l for _l in obj_label])).astype(np.uint8),
                                                axis=0).T)
                    mask = remove_segments_with_size_limit(mask, dict_size_limit={'LB': 100})
                    contours = find_contours(mask, 0.5)
                    for i, _cnt in enumerate(contours):
                        plt.plot(_cnt[:, 1], _cnt[:, 0], edge_style_norm, alpha=0.5)
                    if _dict_obj_ref['boundPlane'] == -2:
                        cnt_all = np.argwhere(mask.astype(np.bool_))
                        cnt_b = cnt_all[cnt_all[:, 0] == 0, :] if _bound == 'UB' else \
                            cnt_all[cnt_all[:, 0] == z_max - 1, :]
                        plt.plot(cnt_b[:, 1], cnt_b[:, 0], edge_style_crop, alpha=0.5, linewidth=3)
        if save_fig_name is not None:
            fig.savefig(save_fig_name, dpi=300)
        else:
            plt.show(block=True)
    return prosthesis_detected


# noinspection SpellCheckingInspection
def separate_arms_and_legs(img_3d_seg_body: np.ndarray, img_3d_seg_ref: Union[np.ndarray, None] = None,
                           cls_map_ref: Union[dict, str, None] = None, obj_ref: Union[dict, None] = None,
                           all_within_bounds: [bool, None] = None, leg_norm_dist_std: Union[float, None] = None,
                           body_cropping_detection: bool = False, plot_image: bool = False,
                           voxel_size: Union[tuple, np.ndarray, list, None] = None,
                           img_3d: Union[np.ndarray, None] = None, save_fig_name: Union[str, None] = None,
                           data_id: Union[str, None] = None, dataset_tag: Union[NestedDict, None] = None):

    """
    :param img_3d_seg_body: numpy array of segmentation result from TotalSegmentator after running task 'body'.
    :param img_3d_seg_ref: numpy array of segmentation result from TotalSegmentator which supposedly contains
                           arm-associated segments such as bones like humerus, ulna and radius.
                           If the user has installed TotalSegmentator v.2, it is recommended to not use the original
                           nnUNet_predict_image function to generate the segmentation results using model weights
                           corresponding to the task 'appendicular_bones' as it removes automatically humerus from the
                           result which however would be helpful for detecting arms.
    :param cls_map_ref: class map that is matched with segmentation in 'img_3d_seg_ref'. It can be a dict provided
                        directly, or str that corresponds to the name of the class map or None.
                        If None, it is assumed to be the class map for 'bones_tissue_test' for version 1 of
                        TotalSegmentator.
                        If version 2 is installed, the assumed class map is either 'appendicular_bones' which
                        contains ulna and radius, or 'total' which contains humeras.
    :param obj_ref: dict with keys of 'UB' and 'LB' or None. Check function define_volume_bounds_by_objects for more
                    details. If provided, all analysis will be performed after using 'UB' and 'LB' clipping planes.
    :param all_within_bounds: bool or None. Only valid if obj_ref is provided. If it is set to False, while parameter
                              body_cropping_detection is set to True and obj_ref is provided, the body cropping
                              detection will be performed within bounds specified by obj_ref.
    :param leg_norm_dist_std: std of normalized distances of upperside of leg-related segments to the upperbound of
                              image. Used only if reference free method is needed.
                              If None, 0.015 is used. The arms are identified to be segments whose normalized
                              distances are less than max(arr_z_norm) - 6 * leg_norm_dist_std. The implicit assumption
                              is that the uppersides of the arms are much closer to the upperboundary of the
    :param body_cropping_detection: bool on whether to detect if the body is cropped.
    :param plot_image: bool on whether to plot the image as control for manual inspection later.
    :param voxel_size: a list, numpy array or a tuple of 3 elements. It is only needed if image will be plotted.
    :param img_3d: 3D image array. Needed only if plot_image is set to True.
    :param save_fig_name: save the figure without showing if a string is provided. Otherwise, show the result.
    :param data_id: str or None. It is useful for logging especially if it is used for deselection of images.
    :param dataset_tag: NestedDict or None. It is useful for logging and accumulating deselected the image data and the
                        reasons.
    :return: mask_arms: dict of bool numpy array of arms that may contain keywords of 'LeftArm' or 'RightArm' or both.
    """

    def is_seg_at_bottom(buffer_dist=1):
        arr_z = np.zeros(num_ext + 1)
        for _i in range(1, num_ext + 1):
            _mask = mask_ext_labeled == _i
            arr_z[_i] = np.min(np.argwhere(_mask)[:, 2])
        return arr_z < buffer_dist

    def _detect_body_cropping(_plane: str = 'cor'):
        atol = 1
        nonlocal flag_body_cropped
        if _plane == 'cor':
            _mask_body_2d = np.flipud(np.max(mask_body_selected, axis=0).T)
        else:  # _plane == 'axi'
            _mask_body_2d = np.max(mask_body_selected, axis=2)
        _dict_bounds = get_2d_bool_mask_boundary_points(_mask_body_2d, atol)
        if isinstance(_dict_bounds, dict):
            if _plane == 'cor':
                if len(_dict_bounds[2]) > 0 or len(_dict_bounds[4]) > 0:
                    # Use coronal image for plotting if plot_image is set to True
                    flag_body_cropped = 1
            else:
                # Use axial image for plotting if plot_image is set to True
                flag_body_cropped = 2

        return _dict_bounds
    if (obj_ref is not None) and (all_within_bounds is True):
        mask_body = img_3d_seg_body[:, :, obj_ref['LB']['boundPlane']: obj_ref['UB']['boundPlane']]
    else:
        mask_body = img_3d_seg_body
    mask_trunc = mask_body == 1
    mask_ext = mask_body == 2
    # Merge extremeties with body and then remove small segments to reduce the risk that small segments of extremetries
    # are removed by mistakes.
    mask_trunc_with_ext = remove_segments_with_size_limit(np.logical_or(mask_trunc, mask_ext),
                                                             dict_size_limit={'LB': 100})
    mask_ext = mask_ext * mask_trunc_with_ext
    result = dict()
    if np.sum(mask_ext) > 0:
        mask_ext_labeled, num_ext = ndi.label(mask_ext)
        if img_3d_seg_ref is not None:
            labels_found_in_ref = np.unique(img_3d_seg_ref[img_3d_seg_ref > 0])
            if cls_map_ref is None:
                cls_map_ref = get_v_dependent_cls_map('appendicular_bones')
            elif isinstance(cls_map_ref, str):
                cls_map_ref = get_v_dependent_cls_map(cls_map_ref)
            ref_arm_ids = ['humerus', 'ulna', 'radius']
            bool_arms_bones = np.array([k in labels_found_in_ref for k, v in cls_map_ref.items() if v in ref_arm_ids])
            if len(bool_arms_bones) > 0:
                bool_arms_bones = np.insert(bool_arms_bones, 0, False)
                mask_arm_bones = bool_arms_bones[img_3d_seg_ref.reshape(-1)].reshape(img_3d_seg_ref.shape)
                mask_arms_list = list()
                for i in range(1, num_ext + 1):
                    bool_obj_sel = np.zeros(num_ext + 1).astype(np.bool_)
                    bool_obj_sel[i] = True
                    _mask_arm_candidate = bool_obj_sel[mask_ext_labeled.reshape(-1)].reshape(mask_ext_labeled.shape)
                    # noinspection PyUnboundLocalVariable
                    _mask_arms_ref = np.logical_and(_mask_arm_candidate, mask_arm_bones)
                    if np.sum(_mask_arms_ref) > 0:
                        mask_arms_list.append(_mask_arm_candidate)
                if len(mask_arms_list) > 0:
                    mask_arms = reduce(operator.xor, mask_arms_list)
                else:
                    mask_arms = -1
            else:
                mask_arms = -1
            if np.isscalar(mask_arms):
                mask_legs = mask_ext
            else:
                mask_legs = mask_ext * (~mask_arms)
        else:
            print_highlighted_text('It is not recommended to use reference free method alone using only segments of'
                                   ' body trunc and extremeties to separate arms and legs.\nUnless it has been '
                                   'verified that perlvic bones are contained in the image, it can be tricky to '
                                   'distinguish legs and arms using coordinates.')
            if leg_norm_dist_std is None:
                leg_norm_dist_std = 0.015
            bool_at_bottom = is_seg_at_bottom()
            # Any extremeties not at the bottom are considered as arms presumably.
            bool_arms = ~bool_at_bottom
            if np.sum(bool_at_bottom) == 0:
                mask_arms = mask_ext
                mask_legs = -1
            else:
                mask_ext_at_bottom = mask_ext_labeled.copy()
                mask = bool_at_bottom[mask_ext_labeled.reshape(-1)].reshape(mask_ext_labeled.shape)
                mask_ext_at_bottom[~mask] = 0
                # There is a possibility that arms are also at the bottom of the image. We need to identify the objects
                # correspond to arms.
                idx_obj_at_bottom = np.argwhere(bool_at_bottom).reshape(-1)[1:]
                arr_z_norm = np.zeros(len(idx_obj_at_bottom))
                for i, _l in enumerate(idx_obj_at_bottom):
                    mask_l = mask_ext_labeled == _l
                    arr_z_norm[i] = 1 - np.max(np.argwhere(mask_l)[:, 2]) / mask_body.shape[2]
                thre_to_detect_arm = np.max(arr_z_norm) - 6 * leg_norm_dist_std
                if thre_to_detect_arm > 0:
                    bool_arm_at_bottom = arr_z_norm < thre_to_detect_arm
                    if len(arr_z_norm[bool_arm_at_bottom]) > 0:
                        print('Objects with large deivation from the normalized distances from the upper-side of '
                              'bottom-reaching extremeties to the image upperbound are found. Arms are likely to be '
                              'detected among them!')
                        idx_arm_at_bottom = idx_obj_at_bottom[bool_arm_at_bottom]
                        bool_arms[idx_arm_at_bottom] = True
                        bool_legs = bool_at_bottom.copy()
                        bool_legs[idx_arm_at_bottom] = False
                    else:
                        bool_legs = bool_at_bottom
                    if np.sum(bool_arms) > 1:
                        mask_arms = bool_arms[mask_ext_labeled.reshape(-1)].reshape(mask_ext_labeled.shape)
                    else:
                        mask_arms = -1
                    mask_legs = bool_legs[mask_ext_labeled.reshape(-1)].reshape(mask_ext_labeled.shape)
                else:
                    raise ValueError("Without providing 'img_3d_seg_ref', it is hard to verfiy whether the detected "
                                     "long-range extremeties are arms or legs!")
    else:
        mask_arms = -1
        mask_legs = -1

    if not np.isscalar(mask_arms):
        if dataset_tag is not None:
            assert data_id is not None
            add_tag_to_data(dataset_tag, 'armDetected', data_id, 'Warning')
    if body_cropping_detection:
        flag_body_cropped = 0
        if (obj_ref is not None) and (all_within_bounds is not True):
            mask_body_selected = mask_trunc[:, :, obj_ref['LB']['boundPlane']: obj_ref['UB']['boundPlane']]
        else:
            mask_body_selected = mask_trunc.copy()
        bounds_cor = _detect_body_cropping('cor')
        bounds_axi = _detect_body_cropping('axi')
        if flag_body_cropped > 0:
            if dataset_tag is not None:
                add_tag_to_data(dataset_tag, 'bodyCropped', data_id)
    else:
        flag_body_cropped = -1
    mask_without_arms = np.logical_or(mask_trunc, mask_legs) if not np.isscalar(mask_legs) else mask_trunc
    result['arms'] = mask_arms
    result['legs'] = mask_legs
    result['bodyWithoutArms'] = mask_without_arms
    result['bodyCroppedFlag'] = flag_body_cropped
    plot_image = plot_image or isinstance(save_fig_name, str)
    if plot_image:
        assert img_3d is not None
        if voxel_size is None:
            aspect_ratios = np.array([1, 1, 1])
        else:
            if not isinstance(voxel_size, np.ndarray):
                voxel_size = np.array(voxel_size)
            aspect_ratios = voxel_size/voxel_size[0]
        if flag_body_cropped < 2:
            fig, ax = volumeViewerMPL.get_sectional_view_image(img_3d, 'x', return_fig=True, window_image=None,
                                                               aspect=aspect_ratios[2])
            mask_2d = np.flipud(np.max(mask_without_arms, axis=0).T)
        else:
            fig, ax = volumeViewerMPL.get_sectional_view_image(img_3d, 'z', return_fig=True, window_image=None,
                                                               aspect=aspect_ratios[2])
            mask_2d = np.max(mask_without_arms, axis=2)
        contours_body = find_contours(mask_2d, 0.5)
        for i, _cnt in enumerate(contours_body):
            ax.plot(_cnt[:, 1], _cnt[:, 0], 'b-.', alpha=0.5, linewidth=3)
        if flag_body_cropped == 1:
            # noinspection PyUnboundLocalVariable
            bound_left, bound_right = bounds_cor[4], bounds_cor[2]
            if len(bound_left) > 0:
                ax.plot(bound_left[:, 1], bound_left[:, 0], 'm-.', linewidth=3)
            if len(bound_right) > 0:
                ax.plot(bound_right[:, 1], bound_right[:, 0], 'm-.', linewidth=3)
        elif flag_body_cropped == 2:
            # noinspection PyUnboundLocalVariable
            for idx_bound, _bound in bounds_axi.items():
                if len(_bound) > 0:
                    ax.plot(_bound[:, 1], _bound[:, 0], 'm-.', linewidth=3)
        if save_fig_name is not None:
            plt.savefig(save_fig_name, dpi=300)
            plt.close(fig)
        else:
            plt.show(block=True)
    return result


__all__ = ['add_tag_to_data', 'get_object_area_at_indexed_plane', 'set_central_ref_plane',
           'define_volume_bounds_by_objects', 'prosthesis_detection_at_lower_bound', 'separate_arms_and_legs']

