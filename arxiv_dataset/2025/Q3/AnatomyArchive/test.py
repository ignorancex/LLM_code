import os
import numpy as np
import nibabel as nib
import msgpack
import msgpack_numpy as m
import random
import util
import timeit
import progressbar
import simpleStats
import matplotlib.pyplot as plt
from typing import Union
from segManager import reconstruct_data_from_coo_components, recover_3d_mask_array_from_coord, image_resample


m.patch()


def test_file_loading(file_path, file_name):
    file_full_name = os.path.join(file_path, file_name)
    strs = file_name.split('.')[1:]
    if len(strs) > 1:
        file_ext = '.'.join(strs)
    else:
        file_ext = strs
    if file_ext == 'nii.gz':
        data = nib.load(file_full_name)
        # noinspection PyUnresolvedReferences
        _ = data.get_fdata().astype(np.int16)
    else:
        with open(file_full_name, 'rb') as fh:
            data_byte = fh.read()
        data = msgpack.unpackb(data_byte)
    return data


def test_file_saving(_data: Union[dict, nib.Nifti1Image], file_path, file_name):
    if isinstance(_data, dict):
        with open(os.path.join(file_path, file_name), 'wb') as fh:
            fh.write(msgpack.packb(_data))
    else:
        _data.to_filename(os.path.join(file_path, file_name))


def test_get_masked_obj_from_data_array(data_array: np.ndarray, mask_array: np.ndarray,
                                        selected_labels: np.ndarray | None = None):
    if selected_labels is None:
        selected_labels = np.unique(mask_array[mask_array > 0])
    data_shape = data_array.shape
    data_min = np.min(data_array)
    for _l in selected_labels:
        _mask = mask_array == _l
        _data = np.empty(data_shape)
        _data.fill(data_min)
        _data[_mask] = data_array[_mask]


def test_get_masked_obj_from_nested_dict(mask_dict: dict):
    data_min = mask_dict['Min']
    data_shape = mask_dict['Shape']
    type_tissue = [k for k, v in mask_dict.items() if isinstance(v, dict)]
    for _t in type_tissue:
        for _s in mask_dict[_t].keys():
            if 'MaskCoord' in mask_dict[_t][_s].keys():
                _data = mask_dict[_t][_s]['HU_value']
                mask_coord = mask_dict[_t][_s]['MaskCoord']
                _data_new = reconstruct_data_from_coo_components(mask_coord, _data, data_shape, data_min)
            else:
                for _o in mask_dict[_t][_s].keys():
                    _data = mask_dict[_t][_s][_o]['HU_value']
                    mask_coord = mask_dict[_t][_s][_o]['MaskCoord']
                    _data_new = reconstruct_data_from_coo_components(mask_coord, _data, data_shape,
                                                                                    data_min)


def test_get_masked_obj_from_flat_dict(mask_dict: dict):
    # data_array = mask_dict['DataArray']
    data_min = mask_dict['Min']
    data_shape = tuple(mask_dict['Shape'])
    type_tissue = [k for k, v in mask_dict.items() if isinstance(v, dict)]
    for _t in type_tissue:
        _data_new = reconstruct_data_from_coo_components(mask_dict[_t]['MaskCoord'], mask_dict[_t]['HU_value'],
                                                         data_shape, data_min)


def get_num_obj_in_dict(mask_dict: dict, is_flat: bool = True):
    type_tissue = [k for k, v in mask_dict.items() if isinstance(v, dict)]
    num_obj = 0
    if is_flat:
        for _t in type_tissue:
            if len(mask_dict[_t]['MaskCoord']) != 0:
                num_obj += 1
            else:
                mask_dict.pop(_t)
    else:
        for _t in type_tissue:
            for _s in mask_dict[_t].keys():
                if 'MaskCoord' in mask_dict[_t][_s].keys():
                    if len(mask_dict[_t][_s]['MaskCoord']) != 0:
                        num_obj += 1
                    else:
                        mask_dict[_t].pop(_s)
                else:
                    for _o in mask_dict[_t][_s].keys():
                        if len(mask_dict[_t][_s][_o]['MaskCoord']) != 0:
                            num_obj += 1
                        else:
                            mask_dict[_t][_s].pop(_o)
    return num_obj


def prepare_data_for_test(data_dir, list_labels: list, num_selected=50):
    files = util.get_files_in_folder(data_dir, 'nii.gz', 'total', True)
    ids_candidate = list()
    ids_excluded = list()
    for f in files:
        # noinspection PyUnresolvedReferences
        mask = nib.load(os.path.join(data_dir, f)).get_fdata().astype(np.uint8)
        labels = np.unique(mask[mask > 0])
        if all([_l in labels for _l in list_labels]):
            ids_candidate.append(f.split('_')[1])
        else:
            ids_excluded.append(f.split('_')[1])
    random.seed(10)
    ids_selected = random.sample(ids_candidate, num_selected)
    return ids_selected, ids_excluded


# noinspection SpellCheckingInspection
def test_radiomic_feature_robustness(file_params: str, data_dir: str, ids_selected: list, selected_label: int,
                                     target_param: str, target_range: list, dict_results: util.NestedDict,
                                     include_default=True, voxel_norm=False, do_ttest=False,
                                     plot_result=False, save_stats_path: Union[str, None] = None, **kwargs):
    """
    :param file_params: full file name for parameter configuration that specifies the non-tissue specific settings;
    :param data_dir: data diretory where Nifti images are stored;
    :param ids_selected: selected list of patient IDs (the image files are named after these IDs);
    :param selected_label: integer label that represent a specific anatomy;
    :param target_param: 'voxel_size' or 'kernel_radius'
    :param target_range: list of target variable values to be tested;
    :param dict_results: a nested dict, and an empty one can be initialized by using util.NestedDict();
    :param include_default: bool on whether to include default voxel size of stored nifti images or kernel radius
                            specified in file_params for this test;
    :param voxel_norm: bool on whether to include modification of the original voxel-based features as defined
                       in function get_voxel_based_radiomic_features;
    :param do_ttest: bool on whether to include t-test between the original and standardized features extracted at
                     different target values. It should be noted that Welch's t-test may be automatically chosen
                     depending on whether the variances are found compariable or not using incorporated variance test.
                     For more information, please refer to the simpleStats to check all relevant method documentations;
    :param plot_result: bool on whether to plot the results, enabled only if 'do_ttest' is set to True;
    :param save_stats_path: file path to save the intermediate OCCC values for robustness evaluation as well as the
                            result plots if the plots are made;
    :param kwargs: supported parameters include:
                   normal_var_sens, bool; paired: bool; alternative: str; return_f_test, bool; ttest_low_normal, bool;
                   For more details, please refer to function simpleStats.ttest_with_auto_checks.
    :return:
    """
    import featureAnalysis as fa
    from totalsegmentator.map_to_binary import class_map
    import SimpleITK as sitk
    from sklearn.preprocessing import StandardScaler

    # noinspection SpellCheckingInspection
    def get_resized_feature_map(data_shape: Union[tuple, None] = None, _target_voxel_size: Union[float, None] = None,
                                _target_kernel_radius: Union[float, None] = None):
        _dict_fmap = util.NestedDict()
        if _target_voxel_size is None:
            if _target_kernel_radius is None:
                fa.get_voxel_based_radiomic_features(extractor, file_image, file_mask, [selected_label], _dict_fmap,
                                                     voxel_norm=voxel_norm)
            else:
                fa.get_voxel_based_radiomic_features(extractor, file_image, file_mask, [selected_label], _dict_fmap,
                                                     voxel_norm=voxel_norm, kernel_radius=_target_kernel_radius)
        else:
            fa.get_voxel_based_radiomic_features(extractor, file_image, file_mask, [selected_label], _dict_fmap,
                                                 voxel_norm=voxel_norm, target_voxel_size=_target_voxel_size)

        for _fc in features_enabled.keys():
            for _fname in features_enabled[_fc]:
                if _target_voxel_size is None or _dict_fmap['VoxelSize'] == voxel_size:
                    _fmap = _dict_fmap[obj_str][_fc][_fname]
                else:
                    mask = recover_3d_mask_array_from_coord(_dict_fmap[obj_str]['MaskCoord'], _dict_fmap['Shape'])
                    _fmap = reconstruct_data_from_coo_components(_dict_fmap[obj_str]['MaskCoord'],
                                                                 _dict_fmap[obj_str][_fc][_fname],
                                                                 data_shape)
                    _fmap = image_resample(_fmap, _dict_fmap['VoxelSize'], voxel_size)[mask]
                dict_feature_maps[obj_str][_fc][_fname].append(_fmap)

    assert target_param in ['voxel_size', 'kernel_radius']
    cls_map = class_map['total']
    obj_str = cls_map[selected_label]
    files = util.get_files_in_folder(data_dir, 'nii.gz', 'total', True)
    files_selected = list()
    num_data = len(ids_selected)
    for i in range(num_data):
        files_selected.append([_f for _f in files if ids_selected[i] in _f][0])
    ids_pstr = [_f.split('_')[0] for _f in files_selected]
    extractor = fa.get_feature_extractor(file_params)
    features_enabled = extractor.enabledFeatures
    type_modes = ['original', 'standardized']
    dict_feature_hierarchy = util.NestedDict()
    for fc in features_enabled.keys():
        for fname in features_enabled[fc]:
            dict_feature_hierarchy[obj_str][fc][fname] = 0
            for t in type_modes:
                dict_results[obj_str][fc][fname][t] = np.zeros(num_data)
    ks = extractor.settings['kernelRadius']
    with progressbar.ProgressBar(max_value=num_data) as bar:
        for i in range(num_data):
            bar.update(i)
            print(f'---Process data for patient ID of {ids_selected[i]} with index {i}---')
            dict_feature_maps = util.nested_dict_hierarchy_like(dict_feature_hierarchy, 'list')
            file_image = os.path.join(data_dir, f'{ids_pstr[i]}_{ids_selected[i]}.nii.gz')
            file_mask = os.path.join(data_dir, f'{ids_pstr[i]}_{ids_selected[i]}_total_seg.nii.gz')
            image = sitk.ReadImage(file_image)
            voxel_size = image.GetSpacing()
            if include_default:
                # Passing extractor to function 'get_voxel_based_radiomic_features' might modify the default kernel
                # radius. Instead of using deep copy, it is easier to set the value back to the default.
                if extractor.settings['kernelRadius'] != ks:
                    extractor.settings['kernelRadius'] = ks
                print(f"Retrieve feature maps generated with default kernel radius "
                      f"{extractor.settings['kernelRadius']} and voxel size {voxel_size}")
                get_resized_feature_map()
            if target_param == 'voxel_size':
                for _i, _value in enumerate(target_range):
                    print(f'Retrieve feature maps generated with voxel size of {_value}')
                    get_resized_feature_map(data_shape=image.GetSize(), _target_voxel_size=_value)
            else:
                for _i, _value in enumerate(target_range):
                    print(f'Retrieve feature maps generated with kernel radius of {_value}')
                    get_resized_feature_map(_target_kernel_radius=_value)
            print(f'---Save extracted features for ID of {ids_selected[i]} with index {i}---')
            test_file_saving(dict_feature_maps, data_dir, f'{ids_pstr[i]}_{ids_selected[i]}_extractedFeatures.msgpack')
            for fc in features_enabled.keys():
                for fname in features_enabled[fc]:
                    # Stacking extracted and resized feature maps vertically produces an array with size of k*n, where
                    # k represents the number of repeated measures of feature maps at different voxel sizes/kernel
                    # radii, n reprents the number of voxel elements in VOI.
                    # To fit the usage requirement for icc or ccc functions, the resulted array has to be transposed.
                    fmaps = np.vstack(dict_feature_maps[obj_str][fc][fname]).T
                    dict_results[obj_str][fc][fname]['original'][i] = simpleStats.get_occc(fmaps)
                    scaler = StandardScaler()
                    # For standardizing the feature at each condition, each feature has to be reshaped to a column
                    # vector, and horizontally stacked. This is different from the originally extracted feature array,
                    # in order to get the same final data shape for calculating the OCCC value.
                    fmaps_scaled = np.hstack([scaler.fit_transform(_e.reshape(-1, 1)) for _e in
                                                   dict_feature_maps[obj_str][fc][fname]])
                    dict_results[obj_str][fc][fname]['standardized'][i] = simpleStats.get_occc(fmaps_scaled)
    if save_stats_path is not None:
        test_file_saving(dict_results, save_stats_path,
                         f"FeatureRobustnessTestStats{''.join([e.capitalize() for e in target_param.split('_')])}."
                         f"msgpack")
    if do_ttest:
        return_f_test = kwargs.get('return_f_test', True)
        dict_stats = util.NestedDict()
        for _k in dict_results[obj_str].keys():
            simpleStats.compute_data_stats_in_dict(dict_results[obj_str], dict_stats[obj_str], _k, type_modes,
                                                   **kwargs)
        test_file_saving(dict_results, save_stats_path,
                         f"FeatureRobustnessTestComparison{''.join([e.capitalize() for e in target_param.split('_')])}."
                         f"msgpack")
        if plot_result:
            for _k in dict_results[obj_str].keys():
                fig_box = simpleStats.boxplot_stats_in_dict(dict_results[obj_str], dict_stats[obj_str], _k, type_modes)
                # list_fnames = list(dict_results[obj_str][_k].keys())
                if return_f_test:
                    fig_sc = simpleStats.plot_var_median_ratio_in_dict(dict_stats[obj_str], _k)
                    if save_stats_path is not None:
                        fig_sc.savefig(os.path.join(save_stats_path, f'FeatureRobustnessComparisonScatter.png'))

                    else:
                        fig_sc.show()
                if save_stats_path is not None:
                    fig_box.savefig(os.path.join(save_stats_path, f'FeatureRobustnessComparisonBoxplot.png'))
                else:
                    fig_box.show()


# noinspection SpellCheckingInspection
def eval_feature_robustness_with_pool(data_dir, ids_selected, n_conditions: int, n_components: int,
                                      selected_feature: Union[str, list],
                                      dict_results: Union[util.NestedDict, dict, None] = None,
                                      plot_result: bool = False, save_stats_path: Union[str, None] = None):
    from sklearn.preprocessing import StandardScaler
    from itertools import combinations

    def init_dict(_dict: dict):
        if not isinstance(_dict, util.NestedDict):
            for _m in post_processing:
                dict_results[_m] = np.zeros(num_data)
        else:
            for _f_name in selected_feature:
                for _m in post_processing:
                    dict_results[_f_name][_m] = np.zeros(num_data)

    def concatenate_arr_in_list_by_indices(_list, _list_idx: Union[list, tuple]):
        arr = np.vstack([_e for _i, _e in enumerate(_list) if _i in _list_idx]).T
        return arr

    def standardize_feature_in_list(_list):
        _scaler = StandardScaler()
        _list = [_scaler.fit_transform(_e.reshape(-1, 1)).ravel() for _e in _list]
        return _list

    # noinspection SpellCheckingInspection
    def get_avg_pooled_feature(_i, _list_fmaps, _f_name: Union[str, None] = None):
        # The first local standardization process
        _list_fmaps_std = standardize_feature_in_list(_list_fmaps)
        _list_fmaps_avg = list()
        scaler = StandardScaler()
        for _comb in list(comb):
            fmaps = concatenate_arr_in_list_by_indices(_list_fmaps_std, _comb)
            # Apply standardization on randomly booled a subset from the list of features extracted under different
            # conditions
            fmaps_scaled = scaler.fit_transform(fmaps)
            _list_fmaps_avg.append(np.mean(fmaps_scaled, axis=1))
        if _f_name is None:
            dict_results['original'][_i] = simpleStats.get_occc(np.vstack(_list_fmaps).T)
            dict_results['SubsetAveragePool'][_i] = simpleStats.get_occc(np.vstack(_list_fmaps_avg).T)
        else:
            dict_results[_f_name]['original'][_i] = simpleStats.get_occc(np.vstack(_list_fmaps).T)
            dict_results[_f_name]['SubsetAveragePool'][_i] = simpleStats.get_occc(np.vstack(_list_fmaps_avg).T)

    assert n_conditions > n_components
    # It takes the save feature maps generated during function 'test_radiomic_feature_robustness'.
    files = util.get_files_in_folder(data_dir, 'msgpack', 'extracted', True)
    files_selected = list()
    num_data = len(ids_selected)
    for i in range(num_data):
        files_selected.append([_f for _f in files if ids_selected[i] in _f][0])
    post_processing = ['original', 'SubsetAveragePool']
    if dict_results is None:
        if isinstance(selected_feature, str):
            dict_results = dict()
        else:
            dict_results = util.NestedDict()
    init_dict(dict_results)
    comb = list(combinations(np.arange(n_conditions), n_components))
    with progressbar.ProgressBar(max_value=num_data) as bar:
        for i, _file in enumerate(files_selected):
            bar.update(i)
            dict_fmaps = test_file_loading(data_dir, _file)
            if isinstance(selected_feature, str):
                list_fmaps = util.get_value_in_nested_dict_by_target_key(dict_fmaps, selected_feature)
                get_avg_pooled_feature(i, list_fmaps)
            else:
                for _fname in selected_feature:
                    list_fmaps = util.get_value_in_nested_dict_by_target_key(dict_fmaps, _fname)
                    get_avg_pooled_feature(i, list_fmaps, _fname)

    if save_stats_path is not None:
        test_file_saving(dict_results, save_stats_path, f'RobustnessTestWithDRT_{n_components}Comp.msgpack')
    if plot_result:
        list_strs = '{' + str(n_components) + '/' + str(n_conditions) + '}'
        labels = [f'${_str}_{list_strs}$' if 'Pool' in _str else _str for _str in post_processing]
        if isinstance(selected_feature, str):
            fig = simpleStats.boxplot_with_sig_bar(dict_results, labels)
        else:
            dict_stats = dict()
            simpleStats.compute_data_stats_in_dict(dict_results, dict_stats, None, post_processing)
            fig = simpleStats.boxplot_stats_in_dict(dict_results, dict_stats, None, post_processing, labels=labels)
        if save_stats_path is None:
            fig.show()
        else:
            fig.savefig(os.path.join(save_stats_path, f'FeatureRobustnessComparisonBoxplotSubsetPool.png'))


# noinspection SpellCheckingInspection
def get_data_and_test_time_efficiency(data_dir, data_tmp_path, ids_selected: Union[list, None] = None,
                                      list_labels: Union[list, None] = None, num_selected=50, nii_str='total_seg'):
    # Get all coo files in dir
    files_coo = util.get_files_in_folder(data_dir, 'msgpack', 'extracted', False)
    if 'v1' in files_coo[0]:
        is_flat = True
    else:
        is_flat = False

    if ids_selected is None:
        assert list_labels is not None
        ids_selected = prepare_data_for_test(data_dir, list_labels, num_selected)
    else:
        num_selected = len(ids_selected)
    assert num_selected <= len(files_coo)

    # Get selected coo files
    files_coo_selected = list()
    for i in range(num_selected):
        files_coo_selected.append([_f for _f in files_coo if ids_selected[i] in _f][0])
    ids_pstr = [_f.split('_')[0] for _f in files_coo_selected]
    sizes_arr = list()
    list_formats = ['Nifti', 'Msgpack']
    list_test_types = ['load data to memory', 'save data to file', 'reconstruct an object']
    t_arr = np.zeros((num_selected, len(list_test_types)*len(list_formats)))
    for i in range(num_selected):
        list_timer = list()
        print(f'---Test on data with index of {i} and ID of {ids_selected[i]}---')
        # Test data loading...
        list_timer.append(timeit.Timer(lambda: test_file_loading(data_dir, f'{ids_pstr[i]}_{ids_selected[i]}.nii.gz')))
        list_timer.append(timeit.Timer(lambda: test_file_loading(data_dir, files_coo_selected[i])))
        # Test data writing... But we need to load the data first before writing to a file
        img_nii = test_file_loading(data_dir, f'{ids_pstr[i]}_{ids_selected[i]}.nii.gz')
        list_timer.append(timeit.Timer(lambda: test_file_saving(img_nii, data_tmp_path, 'tmp.nii.gz')))
        data_dict = test_file_loading(data_dir, files_coo_selected[i])
        list_timer.append(timeit.Timer(lambda: test_file_saving(data_dict, data_tmp_path, 'tmp.msgpack')))
        sizes_arr.append(np.prod(data_dict['Shape'])/1e8)
        # Test masked object reconstruction
        # From array
        data_array = img_nii.get_fdata().astype(np.int16)
        # For nii image, seg file has to be loaded as well
        img_nii_seg = test_file_loading(data_dir, f'{ids_pstr[i]}_{ids_selected[i]}_{nii_str}.nii.gz')
        mask_array = img_nii_seg.get_fdata().astype(np.uint8)
        selected_labels = np.unique(mask_array[mask_array > 0])
        if len(selected_labels) > 10:
            random.seed(10)
            selected_labels = random.sample(list(selected_labels), 10)
        num_obj_in_array = len(selected_labels)
        list_timer.append(timeit.Timer(lambda: test_get_masked_obj_from_data_array(data_array, mask_array,
                                                                                   selected_labels)))
        # From dict
        num_obj_in_dict = get_num_obj_in_dict(data_dict, is_flat)
        if is_flat:
            list_timer.append(timeit.Timer(lambda: test_get_masked_obj_from_flat_dict(data_dict)))
        else:
            list_timer.append(timeit.Timer(lambda: test_get_masked_obj_from_nested_dict(data_dict)))
        for j in range(len(list_formats)):
            for k in range(len(list_test_types)):
                t_idx = j + 2*k
                t = list_timer[t_idx]
                if t_idx < 4:
                    t_arr[i, t_idx] = t.timeit(3) / 3
                elif t_idx == 4:
                    t_arr[i, t_idx] = t.timeit(3) / 3 / num_obj_in_array
                else:
                    t_arr[i, t_idx] = t.timeit(3) / 3 / num_obj_in_dict
                print(f'\tTest {list_test_types[k]} for format {list_formats[j]} takes {t_arr[i, t_idx]:.3f}s.')

    list_colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][:3]
    list_markers = ['o', '*']
    fig1, ax = plt.subplots(1, 1, layout='constrained')
    for j in range(len(list_formats)):
        for k in range(len(list_test_types)):
            ax.scatter(sizes_arr, t_arr[:, j+2*k], marker=list_markers[j], c=list_colors[k],
                        label=list_formats[j] + ': ' + list_test_types[k])
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)
    ax.set_xlabel(r'# Image array voxels ($\times 10^{8}$)')
    ax.set_ylabel('Time (s)')
    fig1.savefig(os.path.join(data_tmp_path, 'TimeCostComparison_a.png'))

    fig2, ax = plt.subplots(1, 1, layout='constrained')
    t_fold = t_arr[:, ::2]/t_arr[:, 1::2]
    plots = ax.violinplot(t_fold, showmeans=False, showmedians=True, showextrema=False)
    for _pc, _c in zip(plots['bodies'], list_colors):
        _pc.set_facecolor(_c)
    plots['cmedians'].set_colors(list_colors)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(list_test_types, rotation=45, ha='right')
    ax.set_ylabel(r'$\frac{t_{Nifti}}{t_{Msgpack}}$', fontsize=15)
    fig2.savefig(os.path.join(data_tmp_path, 'TimeCostComparison_b.png'))
    return t_arr





