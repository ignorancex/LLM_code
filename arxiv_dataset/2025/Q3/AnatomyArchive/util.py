import warnings
import numpy as np
import os
from typing import List, Union
import matplotlib.pyplot as plt
import json
import shutil
from functools import reduce
import operator
import re
import pandas as pd


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def encode_string(_str: str):
    import base64
    return (base64.b64encode(_str.encode('ascii'))).decode('ascii')


def decode_string(_str: str):
    import base64
    return (base64.b64decode(_str.encode('ascii'))).decode('ascii')


def read_json(json_file, print_content=False):
    with open(json_file) as j_file:
        contents = json.load(j_file)
    if print_content:
        from pprint import pprint as pretty
        pretty(contents)

    return contents


def write_json(data, json_file, sort_keys: bool = False):
    with open(json_file, 'w') as j_file:
        json.dump(data, j_file, indent=2, sort_keys=sort_keys)


def copy_file(dir_src, dir_des, dict_patient_include):
    import progressbar

    num_patient = sum([len(v) for v in dict_patient_include.values()])
    num = 0
    with progressbar.ProgressBar(max_value=num_patient) as bar:
        for cls in dict_patient_include.keys():
            dir_sub = os.listdir(os.path.join(dir_src, cls))
            for _f in dir_sub:
                if str(_f).split(' ')[0] in dict_patient_include[cls]:
                    num += 1
                    if os.path.isdir(os.path.join(dir_des, cls, str(_f).split(' ')[0])):
                        continue
                    else:
                        print(f'Copy files from {os.path.join(dir_src, cls, _f)} to {dir_des}...')
                        shutil.copytree(os.path.join(dir_src, cls, _f), os.path.join(dir_des, cls,
                                                                                     str(_f).split(' ')[0]))
                    bar.update(num)


def sort_list_by_idx_e_in_tuple(tuples_list: List[tuple], key_idx: int = 0, skip_idx_e: bool = True,
                                reverse: bool = False, return_index: bool = False):
    """
    :param tuples_list: a list of tuples of the same size (with at least two elements) to be sorted.
    :param key_idx: index of key element in tuple used for sorting.
    :param skip_idx_e: bool on whether to skip the index element for sorting, default: True.
    :param reverse: whether to reverse the order
    :param return_index: bool on whether the indices after sorting should be returned.
    :return: _list_sorted:
    """
    assert len(tuples_list[0]) >= 2

    sorted_list = sorted(tuples_list, key=lambda x: x[key_idx], reverse=reverse)
    if return_index:
        indices = [tuples_list.index(_t) for _t in sorted_list]
    if skip_idx_e:
        if len(tuples_list[0]) == 2:
            if key_idx == 0:
                sorted_list = [_t[1] for _t in sorted_list]
            else:
                sorted_list = [_t[0] for _t in sorted_list]
        else:
            sorted_list = [(*_t[:key_idx], *_t[key_idx:]) for _t in sorted_list]
    if not return_index:
        return sorted_list
    else:
        # noinspection PyUnboundLocalVariable
        return sorted_list, indices


def split_array1d_by_slice_indices(idx_slices: np.ndarray, size: int):
    # It is important that the idx_slices are already sorted!!!
    idx_array_full = np.arange(size)
    # noinspection PyUnusedLocal
    idx_list_split = [None]*(len(idx_slices)+1)
    idx_list_split[0] = idx_array_full[:idx_slices[0]]
    if len(idx_slices) > 1:
        for i in range(1, len(idx_slices)):
            idx_list_split[i] = idx_array_full[idx_slices[i-1]+1:idx_slices[i]]
    idx_list_split[-1] = idx_array_full[idx_slices[-1]+1:]
    return idx_list_split


def get_key_by_value_from_dict(_dict: dict, _value):
    return list(_dict.keys())[list(_dict.values()).index(_value)]


def remove_empty_list_in_dict(_dict):
    key_list = list(_dict.keys())
    for k in key_list:
        if len(_dict[k]) == 0:
            _dict.pop(k)
    return _dict


def initiate_nested_list(size):
    # Don't use [[]] * series_num for initialization. Otherwise, all lists will be appended with the same member.
    # noinspection PyUnusedLocal
    return [[] for i in range(size)]


def concatenate_substrings(_str: str):
    words = _str.split()
    # Initialize the result string
    result = ""
    for i in range(len(words)):
        if i == 0:
            result += words[i]
        else:
            if words[i][0].isupper():
                if result[-1].isupper():
                    result += "-" + words[i]
                else:
                    result += words[i].capitalize()
            else:
                result += words[i].capitalize()
    return result


class NestedDict(dict):
    def __getitem__(self, item):
        try:
            return dict.__getitem__(self, item)
        except KeyError:
            value = self[item] = type(self)()
            return value


def nested_dict_hierarchy_like(_dict: NestedDict, nested_type: str, **kwargs):
    from copy import deepcopy

    def create_np_array():
        if 'size' not in kwargs.keys():
            raise NameError(f"Data size must be provided, either as array shape or as a single number!")
        else:
            return np.zeros(kwargs['size'])

    def get_init_item(_d):
        if isinstance(_d, dict):
            for _k, _v in _d.items():
                if isinstance(_v, dict):
                    get_init_item(_v)
                else:
                    if nested_type == 'list':
                        _d[_k] = list()
                    elif nested_type == 'array':
                        _d[_k] = create_np_array()
                    else:
                        raise NotImplementedError("Only 'list' or 'array' are supported as input!")

    new_dict = deepcopy(_dict)
    get_init_item(new_dict)
    return new_dict


def get_path_to_key_in_nested_dict(_dict: dict, target_key: str):
    # target key name must be unique
    def _get_path(_d, _key):
        nonlocal path
        for k, v in _d.items():
            if k.lower() == _key.lower():
                path += (k, )
            else:
                if isinstance(v, dict) and (_key.lower() not in [_k.lower() for _k in _d.keys()]):
                    path += (k,)
                    _get_path(v, _key)

    path = ()
    _get_path(_dict, target_key)
    return path


def get_value_in_nested_dict_by_target_key(_dict: dict, target_key: str):
    _path = get_path_to_key_in_nested_dict(_dict, target_key)
    return reduce(operator.getitem, _path, _dict)


def get_value_in_nested_dict_by_path(_dict: dict, _path: Union[list, tuple]):
    return reduce(operator.getitem, _path, _dict)


def swap_tuple(_tuple, a, b):
    _list = list(_tuple)
    _list[b], _list[a] = _list[a], _list[b]
    return tuple(_list)


def resample_with_low_repeats(_list: list, size: int, random_seed: int = 1):
    np.random.seed(random_seed)
    num = len(_list)
    assert num > 0 and size > 0
    if num >= size:
        indices = np.random.choice(np.arange(num), size, replace=False)
    else:
        indices = np.random.choice(np.repeat(np.arange(num), int(np.ceil(size/num))), size, replace=False)
    new_list = [_list[i] for i in indices]
    return new_list


def remove_empty_nested_keys(_dict, reset_dict_to_count=False, count_ini=0):
    """
    This method is used to remove empty elements in nested dictionary upto three level.
    :param _dict: Input nested dictionary
    :param reset_dict_to_count: binary on whether to reset the dictionary as a counter
    :param count_ini: default value 0 as the initial count, valid only if 'reset_dict_to_count' is set to True.
    :return:
    """
    for _k in _dict.keys():
        if isinstance(_dict[_k], dict):
            _k_list = list(_dict[_k].keys())
            for _kk in _k_list:
                if isinstance(_dict[_k][_kk], dict):
                    _kk_list = list(_dict[_k][_kk].keys())
                    for _kkk in _kk_list:
                        if (not isinstance(_dict[_k][_kk][_kkk], int) and len(_dict[_k][_kk][_kkk]) == 0) or \
                                (isinstance(_dict[_k][_kk][_kkk], int) and _dict[_k][_kk][_kkk] == 0):
                            _dict[_k][_kk].pop(_kkk)
                        else:
                            if reset_dict_to_count:
                                _dict[_k][_kk][_kkk] = count_ini

                    if len(_dict[_k][_kk].keys()) == 0:
                        _dict[_k].pop(_kk)
                else:
                    if (not isinstance(_dict[_k][_kk], int) and len(_dict[_k][_kk]) == 0) or \
                            (isinstance(_dict[_k][_kk], int) and _dict[_k][_kk] == 0):
                        _dict[_k].pop(_kk)
                    else:
                        if reset_dict_to_count:
                            _dict[_k][_kk] = count_ini


def get_files_in_folder(file_dir, ext: Union[str, None] = None, string_in_filename: Union[str, None] = None,
                        keep_file_with_str=False):
    files = os.listdir(file_dir)
    if ext is not None:
        files = [file for file in files if file.endswith(ext)]
    if string_in_filename is not None:
        if keep_file_with_str:
            idx_to_keep = np.where(np.array([string_in_filename in i for i in files]))[0]
        else:
            idx_to_keep = np.where(np.array([string_in_filename not in i for i in files]))[0]
        files = [files[i] for i in idx_to_keep]
    return files


def clear_fig_in_memory(fig_keep=None):
    all_fig_nums = plt.get_fignums()
    for i in all_fig_nums:
        fig = plt.figure(i)
        if fig_keep is not None:
            if isinstance(fig_keep, list):
                if fig in fig_keep:
                    continue
                else:
                    fig.clear()
                    plt.close(fig)
            else:
                if fig != fig_keep:
                    fig.clear()
                    plt.close(fig)
        else:
            fig.clear()
            plt.close(fig)


def print_highlighted_text(_str):
    print("\x1b[7;33;40m" + _str + "\x1b[0m")


def date_str_match_test(_str):
    return bool(re.match('(\d{4})[/.-](\d{1,2})[/.-](\d{1,2})', _str))


def delete_variables_but(_list: list):
    for var in list(globals().keys()):
        if var not in _list and not var.startswith("__"):
            del globals()[var]


def set_colors(num_colors, cmap_str='turbo'):
    import copy
    """Assigns a random colored material for each found segment"""
    # Use deep copy to avoid modification to inbuilt color map.
    # noinspection PyUnresolvedReferences
    colors_cmp = copy.deepcopy(plt.get_cmap(cmap_str).colors)
    colors_idx = np.linspace(0, 255, num_colors).astype('int')
    color_rgb = np.array([colors_cmp[i] for i in colors_idx])
    return color_rgb


def get_col_idx_by_substr_from_dataframe(df: pd.DataFrame, substr: str):
    indices = [df.columns.get_loc(col) for col in df.columns if substr in col]
    if len(indices) > 0:
        if len(indices) == 1:
            return indices[0]
        else:
            warnings.warn(f'Multiple columns are found to contain substring {substr}: {df.columns[indices]}.')
            return indices
    else:
        warnings.warn(f'No colum names contain substring {substr}!')
        return -1


def normalized_image_to_8bit(img: np.ndarray):
    assert (np.min(img) >= 0 and np.max(img) <= 1)
    img_8bit = 255 * img
    img_8bit = img_8bit.astype(np.uint8)
    return img_8bit

