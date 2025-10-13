import os
import re
import util
import sparse
import operator
import numpy as np
import simpleStats
import simpleGeometry
import networkx as nx
import nibabel as nib
from scipy import ndimage
from typing import List, Union
from functools import reduce
from genericImageIO import convert_nifti_to_numpy
from volViewerMPL import get_sectional_view_image, show_mask_superimposed_2d_image
from segModel import get_v_dependent_cls_map, perform_segmentation_generic, image_resample
from segModel import class_map, v_totalsegmentator, segmentation_settings, anatomy_name_standardization


def get_mask_cls_map_version_from_filename(nii_filename: str):
    str_matched = re.search('v(\d)', '_'.join(nii_filename.split('_')[2:]))
    if str_matched:
        return int(str_matched.group(1))
    else:
        return v_totalsegmentator


# noinspection SpellCheckingInspection
def get_preset_anatomy_groups():
    # Anatomies not supported in TotalSegmentator are not included here.
    anatomy_groups = {
        'bone': ['vertebrae', 'rib', 'humerus', 'scapula', 'clavicula', 'femur', 'pelvic', 'sacrum', 'patella',
                 'tibia', 'fibula', 'tarsal', 'phalange', 'ulna', 'radius', 'carpal', 'sternum', 'skull'],
        'digestive_accessory': ['gallbladder', 'liver', 'pancreas'],
        'intestine': ['duodenum', 'small_bowel', 'colon'],
        'muscle': ['iliopsoas', 'erector', 'gluteus', 'muscle'],
        'endocrine': ['adrenal_gland', 'thyroid_gland', 'pancreas'],
        'parenchyma': ['brain', 'kidney', 'spleen', 'liver', 'pancreas'],
        'vasculature': ['brachiocephalic', 'aorta', 'artery', 'vena', 'vein', 'vessel'],
        'urinary': ['kidney', 'adrenal_gland', 'urinary_bladder', 'prostate']
    }
    anatomy_groups['cardiovascular'] = ['heart', 'atrial'] + anatomy_groups['vasculature']
    anatomy_groups['musculoskeletal'] = anatomy_groups['bone'] + anatomy_groups['muscle']
    anatomy_groups['gastrointestinal'] = ['esophagus', 'stomach'] + anatomy_groups['intestine']
    anatomy_groups['digestive'] = anatomy_groups['digestive_accessory'] + anatomy_groups['gastrointestinal']
    return anatomy_groups


def get_liver_associated_anatomy_and_kwd() -> dict:
    return {'liv': 'liver', 'psv': 'portal_vein_and_splenic_vein', 'ivc': 'inferior_vena_cava'}


def rename_anatomy_for_print(name):
    if '_' in name:
        name_new = name.split('_')
        if 'left' in name_new or 'right' in name_new:
            if 'left' in name_new:
                name_new.insert(0, name_new.pop(name_new.index('left')))
            else:
                name_new.insert(0, name_new.pop(name_new.index('right')))
        return ' '.join(name_new)
    else:
        return name


def get_selected_group_dict(selected_group: Union[str, list, None] = None) -> dict:
    def _get_members(_list: list):
        for _k in _list:
            group_dict[_k] = anatomy_groups[_k]

    anatomy_groups = get_preset_anatomy_groups()
    group_dict = dict()
    if selected_group is None:
        _get_members(['cardiovascular', 'musculoskeletal'])
    elif isinstance(selected_group, str):
        group_dict[selected_group] = anatomy_groups[selected_group]
    else:
        _get_members(selected_group)
    return group_dict


def download_totalsegmentator_pretrained_weights_with_license(config_file: dict, task_name):
    import time
    import sys
    import requests
    from tqdm import tqdm
    import zipfile

    # It is important that the license number should be removed from the code prior to publication!!!
    license_number = util.decode_string(config_file['LicenseNumber'])
    config_dir = os.environ['RESULTS_FOLDER']
    assert config_dir is not None
    temp_file = os.path.join(config_dir, "tmp_download_file.zip")
    url = "http://backend.totalsegmentator.com:80/"

    # Download
    try:
        st = time.time()
        r = requests.post(url + "download_weights",
                          json={"license_number": license_number,
                                "task": task_name},
                          timeout=300,
                          stream=True)
        r.raise_for_status()  # Raise an exception for HTTP errors (4xx, 5xx)

        if r.ok:
            with open(temp_file, "wb") as f:
                total_size = int(r.headers.get('content-length', 0))
                progress_bar = tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading")
                for chunk in r.iter_content(chunk_size=8192 * 16):
                    progress_bar.update(len(chunk))
                    f.write(chunk)
                progress_bar.close()

            print("Download finished. Extracting...")
            with zipfile.ZipFile(temp_file, 'r') as zip_f:
                zip_f.extractall(config_dir)
            print(f"  downloaded in {time.time() - st:.2f}s")
        else:
            if r.json()['status'] == "invalid_license":
                print(
                    f"ERROR: Invalid license number ({license_number}). "
                    f"Please check your license number or contact support.")
                sys.exit(1)

    except Exception as e:
        raise e
    finally:
        if os.path.isfile(temp_file):
            os.remove(temp_file)


def get_default_tissue_hu_range():
    dict_hu_range = dict()
    dict_hu_range['muscle'] = {'LB': 30, 'UB': 150}
    dict_hu_range['fat'] = {'LB': -190, 'UB': -30}
    dict_hu_range['fatty_muscle'] = {'LB': -29, 'UB': 29}
    # The category of 'others' exclude lung lobes.
    dict_hu_range['others'] = {'LB': -500, 'UB': 2000}
    return dict_hu_range


def get_tissue_hierarchy_dict(class_map_name='total'):
    cls_map = class_map[class_map_name]
    list_anatomies = anatomy_name_standardization(cls_map)
    list_label = list(cls_map.keys())
    # list_type = [v.replace('hip', 'pelvic') if 'hip' in v else v.replace('autochthon', 'spinal_erectors')
    #               if 'autochthon' in v else v for k, v in cls_map_tot.items()]
    list_bones = get_preset_anatomy_groups()['bone']
    tissue_dict = util.NestedDict()
    for _anatomy in list_anatomies:
        label = list_label[list_anatomies.index(_anatomy)]
        if any([b in _anatomy for b in list_bones]):
            if '_' not in _anatomy:
                if _anatomy in ['femur', 'humerus']:
                    continue
                else:
                    tissue_dict['Bone'][_anatomy] = label
            else:
                bone_name_index = np.where(np.array([b in _anatomy for b in list_bones]))[0][0]
                tissue_dict['Bone'][list_bones[bone_name_index]][_anatomy] = label
        elif 'lung' in _anatomy:
            tissue_dict['Lung'][_anatomy] = label
        elif 'face' in _anatomy:
            continue
        else:
            if '_' in _anatomy:
                name_str_list = _anatomy.split('_')
                if name_str_list[0] in ['heart', 'gluteus']:
                    tissue_dict['Soft_tissue'][name_str_list[0]][_anatomy] = label
                elif name_str_list[0] == 'iliac':
                    tissue_dict['Soft_tissue']['iliac_vessels'][_anatomy] = label
                else:
                    if ('left' in _anatomy) or ('right' in _anatomy):
                        name_str = '_'.join(name_str_list[:len(name_str_list) - 1])
                        tissue_dict['Soft_tissue'][name_str][_anatomy] = label
                    else:
                        tissue_dict['Soft_tissue'][_anatomy] = label
            else:
                tissue_dict['Soft_tissue'][_anatomy] = label
    return tissue_dict


def split_nii_filename(nii_filename: str):
    strs_wo_ext = nii_filename.split('.')
    str_ext = '.'.join(strs_wo_ext[1:])
    strs = strs_wo_ext[0].split('_')
    # Get the segment name before the 'seg' keyword.
    idx_segment = strs.index('seg') - 1
    return strs, idx_segment, str_ext


def get_largest_component(mask: np.ndarray):
    if mask.dtype == np.bool_:
        seg_labeled, seg_num = ndimage.label(mask)
    else:
        seg_labeled = mask
        seg_num = len(np.unique(mask[mask > 0]))
    size_arr = np.zeros(seg_num)
    for i in range(seg_num):
        size_arr[i] = np.sum(seg_labeled == i + 1)
    return seg_labeled == np.argmax(size_arr) + 1


def segmentation_postfix(img_3d: np.ndarray, img_3d_seg: np.ndarray, task_name: str, mask_dict: dict,
                         selected_components: Union[List[str], None] = None, dict_hu_range: Union[dict, None] = None,
                         enforce_muscle_fat_range: bool = True, set_extra_bounds: bool = False, **kwargs):
    # Currently it is only valid for muscles and fat. For segmentation problems identified with vertebrae, dedicated
    # algorithm has to be developed later.
    # For bones and soft tissues other than lung lobes, a lower-bound of -500 in HU value is introduced.
    def modify_seg(tissue_type: str, list_indices):
        nonlocal img_3d, img_3d_seg
        range_HU = dict_hu_range[tissue_type]
        for i in list_indices:
            label = list_labels[i]
            mask = img_3d_seg == label
            mask_outlier = np.logical_or(np.ma.masked_array(img_3d, ~mask) < range_HU['LB'],
                                         np.ma.masked_array(img_3d, ~mask) > range_HU['UB'])
            mask_outlier = mask_outlier.filled(fill_value=False)
            if tissue_type == 'fatty_muscle':
                # noinspection PyTypeChecker
                _tissue_name = list_anatomies[i] + '_fatty' if i in idx_muscles \
                    else '_'.join(list_anatomies[i].split('_')[:-1] + [tissue_type])
            elif tissue_type == 'fat':
                _tissue_name = list_anatomies[i] + '_fat' if i in idx_muscles else list_anatomies[i]
            else:
                _tissue_name = list_anatomies[i]
            mask_dict[_tissue_name] = ~mask_outlier * mask
            list_anatomies_selected.append(_tissue_name)

    def process_without_modification(list_indices):
        for i in list_indices:
            label = list_labels[i]
            mask = img_3d_seg == label
            mask_dict[list_anatomies[i]] = mask
            list_anatomies_selected.append(list_anatomies[i])

    assert img_3d.shape == img_3d_seg.shape
    cls_map = get_v_dependent_cls_map(task_name, kwargs.get('full_name', True))
    list_anatomies = anatomy_name_standardization(cls_map)
    list_labels = list(cls_map.keys())
    list_labels_found = np.unique(img_3d_seg[img_3d_seg > 0])
    if selected_components is not None:
        idx_components_selected = [list_anatomies.index(_c) for _c in selected_components]
    else:
        idx_components_selected = [list_labels.index(_l) for _l in list_labels_found]
    list_anatomies_selected = list()
    if v_totalsegmentator == 2 and task_name == 'tissue_4_types':
        util.print_highlighted_text("The function for muscle fat range enforcement given by AnatomyArchive will not be"
                                    " applied as the user chooses to use the segmentation results of 'tissue_4_types'"
                                    " provided by TotalSegmentator")
        process_without_modification(idx_components_selected)
    else:
        if enforce_muscle_fat_range:
            if dict_hu_range is None:
                dict_hu_range = get_default_tissue_hu_range()
            if 'skeletal_muscle' in cls_map.values():
                muscle_keywords = ['muscle']
            else:
                muscle_keywords = get_preset_anatomy_groups()['muscle']
            idx_muscles = [i for i, _a in enumerate(list_anatomies) if any([k in _a for k in muscle_keywords])]
            idx_muscles_selected = [i for i in idx_components_selected if i in idx_muscles]
            if len(idx_muscles_selected) != 0:
                modify_seg('muscle', idx_muscles_selected)
                modify_seg('fatty_muscle', idx_muscles_selected)
                modify_seg('fat', idx_muscles_selected)
                idx_components_selected = list(set(idx_components_selected) - set(idx_muscles_selected))
            idx_fat = [i for i, _a in enumerate(list_anatomies) if 'fat' in _a]
            idx_fat_selected = [i for i in idx_components_selected if i in idx_fat]
            if len(idx_fat_selected) != 0:
                modify_seg('fat', idx_fat_selected)
                # modify_seg('fatty_muscle', idx_fat_selected)
                idx_components_selected = list(set(idx_components_selected) - set(idx_fat_selected))
    if set_extra_bounds:
        idx_lung = [i for i, _a in enumerate(list_anatomies) if 'lung' in _a and 'lobe' in _a]
        idx_lung_selected = [i for i in idx_components_selected if i in idx_lung]
        if len(idx_lung_selected) != 0:
            process_without_modification(idx_lung_selected)
            idx_components_selected = list(set(idx_components_selected) - set(idx_lung_selected))
        modify_seg('others', idx_components_selected)
    mask_dict['SelectedAnatomies'] = list_anatomies_selected


def create_vol_segment_and_coo_component(img_3d: Union[np.ndarray, nib.nifti1.Nifti1Image],
                                         img_3d_seg: Union[np.ndarray, nib.nifti1.Nifti1Image], file_name: str,
                                         data_dict: util.NestedDict, class_map_name: str = 'total',
                                         save_data_array=False, with_hierarchy: bool = True,
                                         hist_dict: Union[dict, None] = None,
                                         bin_edges: Union[None, np.ndarray] = None):
    # if save_data_array is True, dictionary structure of subversion 2 is used, otherwise subversion 1.
    import msgpack
    import msgpack_numpy as m  # Needed for serializing numpy array
    m.patch()
    if isinstance(img_3d, nib.nifti1.Nifti1Image):
        img_3d, *_ = convert_nifti_to_numpy(img_3d)
        img_3d = img_3d.astype(np.int16)
    if isinstance(img_3d, nib.nifti1.Nifti1Image):
        img_3d_seg, *_ = convert_nifti_to_numpy(img_3d_seg)
        img_3d_seg = img_3d_seg.astype(np.uint8)
    if save_data_array:
        if 'DataArray' not in data_dict.keys():
            # Bed removal will not affect anatomical segments, therefore it is OK to save the bed-removed data array
            # to dict without updating img_3d itself.
            data_dict['DataArray'] = img_3d
    labels_contained = np.unique(img_3d_seg[img_3d_seg > 0])
    if (bin_edges is None) and (hist_dict is not None):
        bin_edges = np.linspace(-1000, 1600, 261, endpoint=True)
    elif isinstance(bin_edges, np.ndarray) and (hist_dict is None):
        hist_dict = util.NestedDict()
    if with_hierarchy:
        cls_map = get_tissue_hierarchy_dict(class_map_name)
        for cat in cls_map.keys():
            tissue_names_sub_cat = list(cls_map[cat].keys())
            for t in tissue_names_sub_cat:
                if isinstance(cls_map[cat][t], dict):
                    tissue_names_obj = list(cls_map[cat][t].keys())
                    for o in tissue_names_obj:
                        if cls_map[cat][t][o] not in labels_contained:
                            continue
                        else:
                            print(f'Get {o} from tissue {t} of category {cat} with label of '
                                  f'{cls_map[cat][t][o]}')
                            mask_binary = img_3d_seg == cls_map[cat][t][o]
                            data_dict[cat][t][o]['MaskCoord'] = np.argwhere(mask_binary).astype(np.int16)
                            if not save_data_array:
                                data_dict[cat][t][o]['HU_value'] = img_3d[mask_binary].astype(np.int16)
                            if hist_dict is not None:
                                hist_dict[cat][t][o] = np.histogram(img_3d[mask_binary], bins=bin_edges,
                                                                    density=True)[0]
                else:
                    if cls_map[cat][t] not in labels_contained:
                        continue
                    else:
                        print(f'Get tissue {t} of category {cat} with label of {cls_map[cat][t]}')
                        mask_binary = img_3d_seg == cls_map[cat][t]
                        data_dict[cat][t]['MaskCoord'] = np.argwhere(mask_binary).astype(np.int16)
                        if not save_data_array:
                            data_dict[cat][t]['HU_value'] = img_3d[mask_binary].astype(np.int16)
                        if hist_dict is not None:
                            hist_dict[cat][t] = np.histogram(img_3d[mask_binary], bins=bin_edges, density=True)[0]
    else:
        cls_map = class_map[class_map_name]
        for _label, _obj in cls_map.items():
            if _label not in labels_contained:
                continue
            else:
                mask_binary = img_3d_seg == _label
                data_dict[_obj]['MaskCoord'] = np.argwhere(mask_binary).astype(np.int16)
                if not save_data_array:
                    data_dict[_obj]['HU_value'] = img_3d[mask_binary].astype(np.int16)
                if hist_dict is not None:
                    hist_dict[_obj] = np.histogram(img_3d[mask_binary], bins=bin_edges, density=True)[0]

    if file_name.split('.')[1] != 'msgpack':
        if not save_data_array:
            file_name = file_name.split('.')[0] + '_v1.msgpack'
        else:
            file_name = file_name.split('.')[0] + '_v2.msgpack'
    with open(file_name, 'wb') as fh:
        fh.write(msgpack.packb(data_dict))


def merge_and_store_vol_segments_in_nested_dict(file_in: str, class_map_name_list: List[str],
                                                coarse_resolution: Union[bool, None] = True, uniform_size=False,
                                                save_data_array=False, bin_edges: Union[None, np.ndarray] = None,
                                                return_data=False, with_hierarchy: bool = True):
    assert all([_e in list(segmentation_settings.keys()) for _e in class_map_name_list])
    img_3d = nib.load(file_in)
    data_dict = util.NestedDict()
    if uniform_size:
        # noinspection PyTypeChecker
        img_3d = image_resample(img_3d)
        # noinspection PyUnresolvedReferences
        data_dict['VoxelSize'] = img_3d.header.get_zooms()[0]
    else:
        # noinspection PyUnresolvedReferences
        data_dict['VoxelSize'] = img_3d.header.get_zooms()
    data_dict['Min'] = np.min(img_3d.get_fdata().astype(np.int16))
    data_dict['Shape'] = img_3d.shape
    # data_dict['DataArray'] = img_3d
    if isinstance(bin_edges, np.ndarray):
        hist_dict = util.NestedDict()
        return_hist_data = True
    else:
        hist_dict = None
        return_hist_data = False
    for map_name in class_map_name_list:
        if map_name in ['total', 'body']:
            if (coarse_resolution is None) or (coarse_resolution is True):
                seg_config = segmentation_settings[map_name]['coarse']
            else:
                seg_config = segmentation_settings[map_name]['fine']
        else:
            seg_config = segmentation_settings[map_name]
        img_3d_seg = perform_segmentation_generic(file_in, seg_config)
        if uniform_size:
            img_3d_seg = image_resample(img_3d_seg, remove_negative=True)
        create_vol_segment_and_coo_component(img_3d, img_3d_seg, file_in, data_dict, map_name, save_data_array,
                                             with_hierarchy, hist_dict, bin_edges)

    if return_data:
        if return_hist_data:
            return data_dict, hist_dict
        else:
            return data_dict


def convert_dict_to_multi_digraph(_dict: dict, group_dict: Union[dict, None] = None, root_node_name: str = 'root_node'):
    if group_dict is None:
        group_dict = get_selected_group_dict()
    g = nx.MultiDiGraph()
    keys = [k for k in _dict.keys() if isinstance(_dict[k], dict)]
    # Assign edge keys to connection edges from root node to the primary nodes.
    nodes_key = zip([root_node_name] * (len(keys)), keys, keys)
    g.add_edges_from(nodes_key)
    is_dict = [isinstance(_dict[k], dict) for k in keys]
    for i in range(len(keys)):
        if is_dict[i]:
            _k = list(_dict[keys[i]].keys())
            edge_names_overlap = list()
            edge_keys_overlap = list()
            for _kw in group_dict.keys():
                # Add edge keys to the edges directly decedent from the primary nodes.
                _edge_names_overlap = [e for e in _k if any(_e in e for _e in group_dict[_kw])]
                _edge_keys_overlap = [_kw] * len(_edge_names_overlap)
                edge_names_overlap += _edge_names_overlap
                edge_keys_overlap += _edge_keys_overlap
            edge_names = _k + edge_names_overlap
            edge_key_list = [keys[i]] * (len(_k)) + edge_keys_overlap
            edge_list = zip([keys[i]] * (len(edge_names)), edge_names, edge_key_list)
            # Add list consists of 3-element tuples
            g.add_edges_from(edge_list)
            _is_dict = [isinstance(_dict[keys[i]][e], dict) and 'MaskCoord' not in _dict[keys[i]][e] for e in _k]
            _kk = [_k[j] for j, x in enumerate(_is_dict) if x]
            if len(_kk) > 0:
                for _kk_ in _kk:
                    # No edge key assignment for edges remotely decedent from the primary nodes.
                    _kkk = list(_dict[keys[i]][_kk_].keys())
                    _edge_list = zip([_kk_] * (len(_kkk)), _kkk)
                    g.add_edges_from(_edge_list)
    return g


def get_color_dict_example():
    color_dict = dict()
    color_dict['node'] = {}
    color_dict['node']['Soft_tissue'] = 'orange'
    color_dict['node']['Bone'] = 'lightgray'
    color_dict['node']['Lung'] = 'lightseagreen'
    color_dict['edge'] = {}
    color_dict['edge']['Soft_tissue'] = 'goldenrod'
    color_dict['edge']['Bone'] = 'dimgray'
    color_dict['edge']['Lung'] = 'lightseagreen'
    color_dict['edge']['musculoskeletal'] = 'navy'
    color_dict['edge']['cardiovascular'] = 'darkred'
    color_dict['tip_key'] = {}
    color_dict['tip_key']['left'] = 'royalblue'
    color_dict['tip_key']['right'] = 'orangered'
    return color_dict


def plot_multi_digraph_with_pygraphviz(g, color_dict: dict, file_name: str, show_image=False):
    assert isinstance(g, nx.classes.multidigraph.MultiDiGraph)
    assert all(e in color_dict.keys() for e in ['edge', 'node'])

    def update_node_color(n, color_n, color_tip_dict: Union[dict, None] = None):
        a.get_node(n).attr.update(style='filled', color=color_n, label=rename_anatomy_for_print(n))
        if isinstance(color_tip_dict, dict):
            if any(_k in n.name for _k in color_tip_dict.keys()):
                tip_key = [_k for _k in color_tip_dict.keys() if _k in n.name][0]
                a.get_node(n).attr.update(style='filled', color=color_tip_dict[tip_key], penwidth=3, fillcolor=color_n)
        list_child = set(a.successors(n))
        for _child in list_child:
            if _child:
                update_node_color(_child, color_n, color_tip_dict)

    a = nx.nx_agraph.to_agraph(g)
    a.graph_attr.update(overlap='false', splines='curved')
    a.edge_attr.update(arrowsize=0.5)
    node_root = a.nodes()[0]
    node_colors_major = [color_dict['node'][e] for e in set(a.successors(node_root))]
    node_colors_root = (';{0:.2f}'.format(1 / len(node_colors_major)) + ':').join(node_colors_major)
    a.get_node(node_root).attr.update(style='wedged', fillcolor=node_colors_root, penwidth=0)
    for i, _node in enumerate(set(a.successors(node_root))):
        update_node_color(_node, node_colors_major[i], color_tip_dict=color_dict['tip_key']
                          if 'tip_key' in color_dict.keys() else None)
    for e in a.edges_iter(keys=True):
        if e[2] in color_dict['edge'].keys():
            if e[0] == node_root:
                (a.get_edge(e[0], e[1], key=e[2])).attr.update(color=color_dict['edge'][e[2]], penwidth=3)
            else:
                (a.get_edge(e[0], e[1], key=e[2])).attr.update(color=color_dict['edge'][e[2]])
        else:
            node_ancestor = a.predecessors(e[0])[0]
            (a.get_edge(e[0], e[1])).attr.update(color=color_dict['edge'][node_ancestor], style='dashed')

    a.draw(file_name, prog='fdp')

    if show_image:
        from PIL import Image
        img = Image.open(file_name)
        img.show()


def create_legend_for_multi_digraph(node_root, color_dict, file_name, append_str='system',
                                    add_secondary_edge=False, layout='dot'):
    import pygraphviz as pyg
    from PIL import Image, ImageOps

    a = pyg.AGraph(directed=True)
    b = a.add_subgraph(name='Legend', color='black', label='Legend', fontsize=20)
    node_colors_major = list(color_dict['node'].values())
    node_colors_root = (';{0:.2f}'.format(1 / len(node_colors_major)) + ':').join(node_colors_major)
    b.add_node(node_root, style='wedged', fillcolor=node_colors_root, penwidth=0)
    b.add_node('0', style='invis')
    b.add_edge(node_root, '0', label=' root\n branch', penwidth=3)

    for i, _k in enumerate(color_dict['node'].keys()):
        b.add_node(_k, color=color_dict['node'][_k], style='filled', label=_k.replace('_', ' '))
        b.add_node(str(i + 1), style='invis')
        b.add_edge(_k, str(i + 1), color=color_dict['edge'][_k], label=' primary\n branch',
                   fontcolor=color_dict['edge'][_k])
    num_node_invis = len(color_dict['node'].keys()) + 1

    if 'tip_key' in color_dict.keys():
        for _k in color_dict['tip_key'].keys():
            b.add_node(_k, color=color_dict['tip_key'][_k], penwidth=3)
    edges_additional = set(color_dict['edge'].keys()) - set(color_dict['node'].keys())
    for j, _e in enumerate(edges_additional):
        if j + 1 <= num_node_invis:
            b.add_node(str(num_node_invis + j), style='invis')
            b.add_edge(str(j), str(num_node_invis + j),
                       label=' ' + _e + '\n ' + append_str, color=color_dict['edge'][_e],
                       fontcolor=color_dict['edge'][_e])
            num_node_invis += 1
        else:
            b.add_node(str(2 * j), style='invis')
            b.add_node(str(2 * j + 1), style='invis')
            b.add_edge(str(2 * j), str(2 * j + 1),
                       label=' ' + _e + '\n ' + append_str, color=color_dict['edge'][_e],
                       fontcolor=color_dict['edge'][_e])
            num_node_invis += 2

    if add_secondary_edge:
        if len(color_dict['node'].keys()) > len(edges_additional):
            node_predecessor = len(color_dict['node'].keys()) - len(edges_additional) + 1
            b.add_node(str(num_node_invis + 1), style='invis')
            b.add_edge(str(node_predecessor), str(num_node_invis + 1), label=' secondary\n branch', style='dashed')
        else:
            raise Exception('Not implemented!')
    file_name_str, ext = file_name.split('.')
    file_name_new = file_name_str + '_insert.' + ext
    a.draw(file_name_new, prog=layout)
    img = Image.open(file_name_new)
    img_new = ImageOps.expand(img, border=(3, 3, 3, 3), fill='black')
    img_new.save(file_name_new)


def recover_3d_mask_array_from_coord(arr_coord, arr_shape):
    assert arr_coord.shape[1] == 3
    mask_binary = np.full(shape=arr_shape, fill_value=False)
    mask_binary[arr_coord[:, 0], arr_coord[:, 1], arr_coord[:, 2]] = True
    return mask_binary


def reconstruct_data_from_coo_components(arr_coord: np.ndarray, arr_value: np.ndarray, arr_shape: Union[list, tuple],
                                         fill_value=0) -> np.ndarray:
    coord_size = arr_coord.shape
    # Ensure that the data to be reconstructed is between 2D and 4D.
    assert (min(coord_size) in [2, 3, 4]) and (min(coord_size) == len(arr_shape))
    assert max(coord_size) == len(arr_value)
    if len(arr_shape) == 4:
        if coord_size[0] > coord_size[1]:
            assert arr_shape[-1] == len(np.unique(arr_coord[:, -1]))
        else:
            assert arr_shape[-1] == len(np.unique(arr_coord[-1, :]))
    if coord_size[0] > coord_size[1]:
        data = sparse.COO(arr_coord.T, arr_value, arr_shape, fill_value=fill_value)
    else:
        data = sparse.COO(arr_coord, arr_value, arr_shape, fill_value=fill_value)
    # noinspection PyTypeChecker
    return data.todense()


def inquire_target_in_graph(data_graph: nx.classes.multidigraph.MultiDiGraph, target_anatomy: str,
                            return_all_substring_matched_cases=False):
    """
    This method tries to find a case-insensitive match of the target anatomy among all node and edge names. If not
    found, it tries to find the case-insensitive matches by considering the input string as a substring. If many
    matches are found, the user gets the option to return them all or only the one which is the most similar to the
    target_anatomy. Or if no substring matching is found, the closet match is returned.

    :param data_graph: a MultiDiGraph object converted from data_dict. The nodes and edges defined in the graph are
                       inquired to identify the targeted anatomy.
    :param target_anatomy: a string or a substring that represents the targeted anatomy.
    :param return_all_substring_matched_cases: bool value determines if all matched cases for substring matching are
                                               outputted. This is useful if the user for instance intends to get all
                                               anatomies of the left or right components using substring 'left' or
                                               'right' as target_anatomy.
    :return:
    """

    def search_with_tolerance():
        from difflib import SequenceMatcher
        prob = [(SequenceMatcher(a=target_anatomy.lower(), b=_str.lower())).ratio() for _str in strs_inquire]
        str_closet = strs_inquire[prob.index(max(prob))]
        print(f"The most similar one to target anatomy is '{str_closet}'")
        return str_closet

    # noinspection PyCallingNonCallable
    # root_node = [_n for _n, _d in data_graph.in_degree() if _d == 0][0]

    strs_inquire = list(set([e[1] for e in data_graph.edges]).union(set([e[2] for e in data_graph.edges
                                                                         if isinstance(e[2], str)])))
    strs_inquire.sort()
    str_found = [s for s in strs_inquire if target_anatomy.lower() == s.lower()]
    if len(str_found) == 0:
        util.print_highlighted_text(f"No exact matching of string '{target_anatomy}' is found!")
        str_found = [s for s in strs_inquire if target_anatomy.lower() in s.lower()]
        if len(str_found) == 0:
            util.print_highlighted_text(f"No exact matching of substring '{target_anatomy}' is found!\n"
                                        f"Calculate the similarity to suggest the best matched one...")
            result = search_with_tolerance()
        elif len(str_found) == 1:
            print(f"Found targeted anatomy by substring matching: '{str_found[0]}'!")
            result = str_found[0]
        else:
            if return_all_substring_matched_cases:
                result = str_found
                print('Found substring-matched anatomies:\n\t' + '\n\t'.join(result))
            else:
                util.print_highlighted_text(f"More than one match of substring {target_anatomy} are found!\n"
                                            f"Calculate the similarity to suggest the best matched one...")
                result = search_with_tolerance()
    else:
        print(f"Found targeted anatomy: '{str_found[0]}'!")
        result = str_found[0]
    if isinstance(result, str):
        if result in list(data_graph.nodes):
            is_node = True
        else:
            is_node = False
    else:
        is_node = True
    return result, is_node


def reconstruct_data_from_lists_of_coo_components(list_arr_coord: List[np.ndarray], list_arr_value: List[np.ndarray],
                                                  arr_shape: Union[list, tuple], fill_value=0):
    assert [max(_coord.shape) == len(_value) for _coord, _value in zip(list_arr_coord, list_arr_value)] and \
           [min(_coord.shape) == len(arr_shape) for _coord in list_arr_coord]
    # Transpose _coord in list_arr_coord if it is NOT a tall array. It can be implemented in the opposite way so that
    # one does not need to transpose the concatenated array when reconstructing the data at the last step. If this is
    # the case, one should remember to specify the axis to be 1 when concatenating the arrays in list_arr_coord.
    list_arr_coord = [_coord if _coord.shape[0] > _coord.shape[1] else _coord.T for _coord in list_arr_coord]
    arr_coord = np.concatenate(list_arr_coord)
    arr_value = np.concatenate(list_arr_value)
    data = sparse.COO(arr_coord.T, arr_value, arr_shape, fill_value=fill_value)
    return data.todense()


def reconstruct_data_with_mask(data_arr: np.ndarray, mask_bool: np.ndarray, fill_value):
    assert mask_bool.dtype == bool
    assert data_arr.shape == mask_bool.shape
    data = np.ma.masked_array(data_arr, mask=~mask_bool)
    return data.filled(fill_value=fill_value)


def load_dict_from_msgpack_file(file_name):
    import msgpack
    import msgpack_numpy as m
    m.patch()
    with open(file_name, 'rb') as fh:
        data_byte = fh.read()
    data_dict = msgpack.unpackb(data_byte)
    return data_dict


def reconstruct_data_from_loaded_dict(data_dict, target_anatomy: str, root_node_name='root',
                                      data_graph: Union[nx.classes.multidigraph.MultiDiGraph, None] = None,
                                      return_all_substring_matched_cases=False):
    """
    Check method 'inquire_target_in_graph' for the part concerning how a string match is calculated to make sure that
    the returned match will correspond to what is expected by the user.
    :param data_dict: a nested dictionary where anatomic segments are stored with a pre-determined hierarchy,
                      which is represented by a tree graph, data_graph.
    :param target_anatomy: it can be a specific organ or tissue type, or a particular group that represents a node in
                           the graph created from the loaded dictionary.
    :param root_node_name: user-specified root node name to generate a tree graph so one can take advantage of the
                           shortest path algorithm to traverse the targeted anatomy through the loaded "data_dict".
    :param data_graph: a MultiDiGraph object converted from data_dict. To avoid repetitive regeneration when using this
                       function, it can also be provided as an input, otherwise, as a None object.
    :param return_all_substring_matched_cases: bool about whether to return all matched cases for substring-matching.
                                               Default: False.
    :return:
    """

    def _merge_masks_in_nested_dict(_d: dict, _mask_list: List[np.ndarray], _name_list: Union[List[str], None] = None):
        for k, v in _d.items():
            if not isinstance(v, dict):
                if k == 'MaskCoord':
                    _mask_list.append(recover_3d_mask_array_from_coord(v, arr_shape))
            else:
                if _name_list is not None and 'MaskCoord' in v.keys():
                    _name_list.append(k)
                _merge_masks_in_nested_dict(v, _mask_list, _name_list)
        return reduce(operator.xor, _mask_list)

    def _combine_coo_components_in_nested_dict(_d: dict, _mask_list: List[np.ndarray], _value_list: List[np.ndarray],
                                               _name_list: Union[List[str], None] = None):
        for k, v in _d.items():
            if not isinstance(v, dict):
                if k == 'MaskCoord':
                    _mask_list.append(v)
                elif k == 'HU_value':
                    _value_list.append(v)
            else:
                if _name_list is not None and 'MaskCoord' in v.keys():
                    _name_list.append(k)
                _combine_coo_components_in_nested_dict(v, _mask_list, _value_list, _name_list)

    def get_echelon_nodes():
        _nodes_parents = set([e[0] for e in data_graph.edges if e[2] == target_found])
        _nodes_echelon = list()
        for _parent in _nodes_parents:
            _nodes_children_of_parent = list(data_graph.successors(_parent))
            _children_of_parent_in_group = [e[1] for e in data_graph.edges if e[0] == _parent and
                                            e[2] == target_found]
            # Check if all direct decedent nodes of _parent node can be found in the group.
            if len(_nodes_children_of_parent) == len(_children_of_parent_in_group):
                _nodes_echelon.append(_parent)
            else:
                _nodes_echelon.extend(_children_of_parent_in_group)
        return _nodes_echelon

    def _combine_and_reconstruct_data_from_node_list(_group: List[str]):
        _list_masks = list()
        if subversion == 1:
            _list_values = list()
            for _node in _group:
                _path_nodes = nx.shortest_path(data_graph, source=root_node_name, target=_node)
                _dict_ = reduce(dict.get, _path_nodes[1::], data_dict)
                _combine_coo_components_in_nested_dict(_dict_, _list_masks, _list_values)
            _result = reconstruct_data_from_lists_of_coo_components(_list_masks, _list_values, arr_shape, fill_value)
        else:
            for _node in _group:
                _path_nodes = nx.shortest_path(data_graph, source=root_node_name, target=_node)
                _dict_ = reduce(dict.get, _path_nodes[1::], data_dict)
                _ = _merge_masks_in_nested_dict(_dict_, _list_masks)
            _mask = reduce(operator.xor, list_masks)
            _result = data_dict['DataArray'][_mask]
        return _result

    assert isinstance(data_dict, dict)
    arr_shape = data_dict['Shape']
    fill_value = data_dict['Min']
    if 'DataArray' not in data_dict.keys():
        subversion = 1
    else:
        subversion = 2
    if data_graph is None:
        data_graph = convert_dict_to_multi_digraph(data_dict, root_node_name=root_node_name)

    target_found, is_node = inquire_target_in_graph(data_graph, target_anatomy, return_all_substring_matched_cases)
    if is_node:
        if isinstance(target_found, str):
            path_nodes = nx.shortest_path(data_graph, source=root_node_name, target=target_found)
            # Skip the root node name from path_nodes as it is not included in the loaded data_dict.
            _dict = reduce(dict.get, path_nodes[1::], data_dict)
            if len(list(data_graph.successors(target_found))) == 0:
                print(f'Single anatomy {target_found} without lower hierarchy elements is to be reconstructed!')
                if subversion == 1:
                    result = reconstruct_data_from_coo_components(_dict['MaskCoord'], _dict['HU_value'],
                                                                  arr_shape, fill_value)
                else:
                    mask = recover_3d_mask_array_from_coord(_dict['MaskCoord'], arr_shape)
                    result = reconstruct_data_with_mask(data_dict['DataArray'], mask, fill_value)
            else:
                list_masks = list()
                list_names = list()
                if subversion == 1:
                    list_values = list()
                    _combine_coo_components_in_nested_dict(_dict, list_masks, list_values, list_names)
                    list_names.sort()
                    print('The following masks are merged to reconstruct anatomy group:\n\t' + '\n\t'.join(list_names))
                    result = reconstruct_data_from_lists_of_coo_components(list_masks, list_values, arr_shape,
                                                                           fill_value)
                else:
                    mask = _merge_masks_in_nested_dict(_dict, list_masks, list_names)
                    list_names.sort()
                    print('The following masks are merged to reconstruct anatomy group:\n\t' + '\n\t'.join(list_names))
                    result = reconstruct_data_with_mask(data_dict['DataArray'], mask, fill_value)
        else:
            result = _combine_and_reconstruct_data_from_node_list(target_found)
    else:
        nodes_children = list(set([e[1] for e in data_graph.edges if e[2] == target_found]))
        print(f'Masks are to be merged in order to reconstruct the group {target_found}:\n\t' +
              '\n\t'.join(nodes_children))
        nodes_echelon_in_group = get_echelon_nodes()
        result = _combine_and_reconstruct_data_from_node_list(nodes_echelon_in_group)
    return result


def set_hierarchic_bin_width(data_dict: dict, data_graph: nx.classes.multidigraph.MultiDiGraph,
                             bin_width_opt: Union[np.ndarray, None] = None,
                             predefined_subgroups: Union[List[str], None] = None) -> dict:
    """

    :param data_dict: dict of segmentations with default inbuilt hierarchy.
    :param data_graph: graph representation converted from data_dict. Consult function 'convert_dict_to_multi_digraph'.
    :param bin_width_opt: Optional bin width array.
    :param predefined_subgroups: predefined subgroups of soft tissues that are supposed to have different
                                 subgroup-specific bin width.
    :return: dict_bin_width
    """

    def round_bin_width(_bin_width):
        ratio = _bin_width / bin_width_opt
        idx = (np.abs(ratio - 1)).argmin()
        return bin_width_opt[idx]

    def set_bin_width(_data, _type_tissue, _sub_type_tissue: Union[str, None] = None):
        mask = _data > -1024
        _data = _data[mask]
        _min = _data.min()
        _max = _data.max()
        _bin_width = round_bin_width(simpleStats.get_optimal_hist_bin_width(_data, round_to_int=True))
        print(f'Bin width for {_subtype} of {_type_tissue}: {_bin_width}')
        if _type_tissue == 'Lung':
            dict_bin_width[_type_tissue] = _bin_width
        elif _type_tissue == 'Bone':
            if _type_tissue not in dict_bin_width.keys():
                dict_bin_width[_type_tissue] = _bin_width
            else:
                if _bin_width < dict_bin_width[_type_tissue]:
                    dict_bin_width[_type_tissue] = _bin_width
        else:
            if _sub_type_tissue not in dict_bin_width[_type_tissue].keys():
                dict_bin_width[_type_tissue][_sub_type_tissue] = _bin_width
            else:
                if _bin_width < dict_bin_width[_type_tissue][_sub_type_tissue]:
                    dict_bin_width[_type_tissue][_sub_type_tissue] = _bin_width

    if bin_width_opt is None:
        bin_width_opt = np.array([2, 5, 10, 20, 40, 50])

    if predefined_subgroups is None:
        predefined_subgroups = ['vasculature', 'muscle', 'gastrointestinal']
    group_dict = get_selected_group_dict(predefined_subgroups)
    tissue_types = [k for k, v in data_dict.items() if isinstance(v, dict)]
    dict_bin_width = util.NestedDict()

    for _type in tissue_types:
        if _type == 'Lung':
            # Group lung tissues together for bin width calculation
            data = reconstruct_data_from_loaded_dict(data_dict, _type, data_graph=data_graph)
            set_bin_width(data, _type)
            print(f'Bin width for {_type}: {dict_bin_width[_type]}')
        elif _type == 'Bone':
            # There are too many subtypes of bones, therefore iterate each subtype. However, all bones are set to use
            # the same smallest bin width found from all subtypes.
            for _subtype in data_dict[_type].keys():
                data = reconstruct_data_from_loaded_dict(data_dict, _subtype, data_graph=data_graph)
                set_bin_width(data, _type)
            print(f'Bin width for {_type}: {dict_bin_width[_type]}')
        else:
            # The soft tissues are very heterogeneous, therefore multiple subtype-specific bin widths are set.
            for _subtype in data_dict[_type].keys():
                data = reconstruct_data_from_loaded_dict(data_dict, _subtype, data_graph=data_graph)
                if 'fat' in _subtype and 'fatty' not in _subtype:
                    set_bin_width(data, _type, 'fat')
                elif 'fatty' in _subtype:
                    set_bin_width(data, _type, 'fatty_muscle')
                else:
                    found_subtype = 0
                    for _soft_subtype in predefined_subgroups:
                        if any([e in _subtype for e in group_dict[_soft_subtype]]):
                            set_bin_width(data, _type, _soft_subtype)
                            found_subtype = 1
                    if found_subtype == 0:
                        set_bin_width(data, _type, _subtype)
            for _subtype in dict_bin_width[_type].keys():
                print(f'Bin width for {_subtype} of {_type}: {dict_bin_width[_type][_subtype]}')
    return dict_bin_width


def set_color_labels_for_display(cls_map: dict):
    num_labels = len(cls_map)
    colors = util.set_colors(num_labels)
    dict_colors = {k: colors[i, :] for i, k in enumerate(cls_map.keys())}
    return dict_colors


def vertebrae_compression_detection_via_auto_landmarks(mask_3d: np.ndarray, y_index: Union[int, None] = None,
                                                       slab_hw: Union[int, None] = None, smooth_contour: bool = False):
    def get_vertebral_grade_type(_dists):
        height_reduction = (np.max(_dists) - np.min(_dists)) / np.max(_dists)
        if 0.2 <= np.round(height_reduction, 2) <= 0.25:
            _grade = 1
        elif np.round(height_reduction, 2) >= 0.26 and height_reduction <= 0.4:
            _grade = 2
        elif height_reduction > 0.4:
            _grade = 3
        else:
            _grade = 0
        if _grade > 0:
            if np.argmin(_dists) == 0:
                _grade_type = 'W'
            elif np.argmin(_dists) == 1:
                _grade_type = 'B'
            else:
                _grade_type = 'C'
            _grade_type += str(_grade)
        else:
            _grade_type = 0
        return _grade_type
    mask_2d = get_sectional_view_image(mask_3d, 'y', slice_idx=y_index, slab_hw=slab_hw, is_mask=True,
                                       plot_figure=False)
    cls_map = class_map['total_v2']
    cls_map_vert = {k: v for k, v in cls_map.items() if ('e_L' in v) or ('_T12' in v)}
    # dict_colors = set_color_labels_for_display(cls_map_vert)
    labels = set(np.unique(mask_2d[mask_2d > 0]))
    dict_results = dict()
    for _l in labels:
        _dict = simpleGeometry.mask_bbox_dists_with_midpoints(mask_2d, _l, smooth_contour)
        if 'BBox' in _dict.keys():
            grade = get_vertebral_grade_type(_dict['Distances'])
            dict_results[cls_map_vert[_l]] = {'BBox': _dict['BBox'], 'Grade': grade, 'Landmarks': _dict['Landmarks']}
        else:
            dict_results[cls_map_vert[_l]] = _dict
    return dict_results


def visualize_vertebrae_with_mask_and_landmarks(img_3d: np.ndarray, mask_3d: np.ndarray,
                                                slab_hw: Union[int, None] = None, smooth_contour: bool = False,
                                                title: Union[str, None] = None, apply_window: Union[int, None] = None,
                                                save_file_name: Union[str, None] = None):

    y_index = simpleStats.get_index_of_plane_with_largest_mask_area(mask_3d, dim='y')
    dict_results = vertebrae_compression_detection_via_auto_landmarks(mask_3d, y_index, slab_hw, smooth_contour)
    cls_map = class_map['total_v2']
    cls_map_vert = {k: v for k, v in cls_map.items() if ('e_L' in v) or ('_T12' in v)}
    dict_colors = set_color_labels_for_display(cls_map_vert)
    img_2d = get_sectional_view_image(img_3d, 'y', slice_idx=y_index, slab_hw=slab_hw, is_mask=False,
                                      window_image='SoftTissue', plot_figure=False)
    mask_2d = get_sectional_view_image(mask_3d, 'y', slice_idx=y_index, slab_hw=slab_hw, is_mask=True,
                                       plot_figure=False)
    if apply_window is None:
        if slab_hw is None:
            apply_window = 1
            title = f'{title}: at MA slice' if title is not None else None
        else:
            apply_window = 2
            title = f'{title}: MIP with HW {slab_hw}' if title is not None else None
    fig, ax = show_mask_superimposed_2d_image(img_2d, mask_2d, dict_colors, alpha=0.5, apply_window=apply_window,
                                              title=title, cls_map=cls_map_vert, return_fig=True)
    for obj in dict_results.keys():
        if 'BBox' in dict_results[obj].keys():
            grade = dict_results[obj]['Grade']
            points = dict_results[obj]['Landmarks']
            points_t = simpleGeometry.get_side_from_closed_curve(points, 'top')
            points_b = simpleGeometry.get_side_from_closed_curve(points, 'bottom')
            bbox = simpleGeometry.sort_curve_points_cw(dict_results[obj]['BBox'])
            # Add back the first point to connect all dots in line plot
            bbox = np.vstack((bbox, bbox[0, :]))
            ax.scatter(points_t[:, 0], points_t[:, 1], s=5, c='blue', marker='^')
            ax.scatter(points_b[:, 0], points_b[:, 1], s=5, c='blue', marker='v')
            if isinstance(grade, str):
                ax.plot(bbox[:, 0], bbox[:, 1], 'r--')
                centroid = np.mean(dict_results[obj]['BBox'], axis=0)
                ax.text(centroid[0]-5, centroid[1]+2, grade, c='r')
        else:
            util.print_highlighted_text(f'Mask for {obj} is at {list(dict_results[obj].values())[0]} border')
    if save_file_name is not None:
        fig.savefig(save_file_name)


# noinspection SpellCheckingInspection
def modify_subcomponent_labels(mask_a: np.ndarray, mask_b: np.ndarray, cls_map_names: List[str],
                               objs_to_modify: Union[list, None] = None):
    """

    :param mask_a: mask array whose labels are to be modified.
    :param mask_b: mask array containing subcomponent labels used to modify labels in mask_a.
    :param cls_map_names: list of names of the classmaps for mask_a & mask_b.
    :param objs_to_modify: names of the objects which coexist in both mask_a & mask_b. If not specified, all labels in
                           mask_b will be used to modify the corresponding labels in mask_a.
    :return:
    """
    assert mask_a.shape == mask_b.shape
    assert len(cls_map_names) == 2
    cls_map_rev_a = {v: k for k, v in class_map[cls_map_names[0]].items()}
    cls_map_rev_b = {v: k for k, v in class_map[cls_map_names[1]].items()}
    labels_a = np.unique(mask_a[mask_a > 0])
    labels_b = np.unique(mask_b[mask_b > 0])
    objs_found = [k for k, v in cls_map_rev_a.items() if v in labels_a]
    if objs_to_modify is None:
        objs_to_modify = [k for k, v in cls_map_rev_b.items() if v in labels_b]
    assert set(objs_to_modify).issubset(set(objs_found))
    mask_a_updated = mask_a.copy()
    for i, _obj in enumerate(objs_to_modify):
        _mask_a = mask_a == cls_map_rev_a[_obj]
        _mask_b = mask_b == cls_map_rev_b[_obj]
        mask_a_updated[~_mask_b * _mask_a] = 0
    return mask_a_updated



