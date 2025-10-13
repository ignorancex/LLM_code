import os
import re
import util
import pydicom
import numpy as np
import nibabel as nib
from typing import Union, List


def get_slices_from_file_path(dicom_path):
    files = []
    for root, dirs, file_names in os.walk(dicom_path):
        for file in file_names:
            try:
                files.append(pydicom.dcmread(os.path.join(root, file)))
            except pydicom.errors.InvalidDicomError:
                print('\x1b[7;33;40m' + 'Non-DICOM file detected with file name of {}'.format(
                    os.path.join(root, file)) + '\x1b[0m')
    # noinspection PyUnresolvedReferences
    files = [file for file in files if not isinstance(file, pydicom.dicomdir.DicomDir)]
    # skip files with no SliceLocation (e.g. scout views)
    slices = []
    skip_count = 0
    patient_id = files[0].PatientID
    for i, f in enumerate(files):
        if hasattr(f, 'SliceLocation'):
            slices.append(f)
            if i > 0:
                assert (f.PatientID == patient_id)
        else:
            skip_count = skip_count + 1
    if skip_count != 0:
        print('Total N0. of valid slices with {} skipped due to unspecified locations {} for patient {}.'
              .format(len(slices), skip_count, patient_id))
    scan_direction, reverse = get_scan_direction(slices[0])
    slices = sorted(slices, key=lambda sl: sl.ImagePositionPatient[scan_direction], reverse=reverse)
    return slices, patient_id


def get_scan_direction(dicom_slice: pydicom.dataset.FileDataset, return_str: bool = False):
    img_orientation = np.array(dicom_slice.ImageOrientationPatient)
    plane = np.cross(img_orientation[:3], img_orientation[3:])
    flag = np.argwhere(np.abs(plane))[0][0]
    if plane[flag] == -1:
        reverse = True
    else:
        reverse = False
    if return_str:
        dict_flag = {0: 'Coronal', 1: 'Sagittal', 2: 'Transverse'}
        flag = dict_flag[flag]
    return flag, reverse


def sort_slices_by_position(slices, return_direction: bool = False, return_index: bool = False):
    scan_direction, reverse = get_scan_direction(slices[0])
    slices_sorted = sorted(slices, key=lambda _s: _s.ImagePositionPatient[scan_direction], reverse=reverse)
    if (not return_direction) and (not return_index):
        return slices_sorted
    else:
        if return_index:
            indices = [slices_sorted.index(_t) for _t in slices]
            if (not return_direction) and return_index:
                return slices_sorted, indices
            else:
                return slices_sorted, scan_direction, indices
        else:
            return slices_sorted, scan_direction


def check_and_split_slices_by_scan_settings(slices: List[pydicom.dataset.FileDataset], return_index: bool = False):
    orient_list = [tuple(_s.ImageOrientationPatient) for _s in slices]
    orient_arr = np.empty(len(orient_list), dtype=object)
    orient_arr[:] = orient_list
    orient_unq, orient_idx = np.unique(orient_arr, return_inverse=True)
    if len(orient_unq) > 1:
        flag = 1
        list_slices = util.initiate_nested_list(len(orient_unq))
        if return_index:
            list_indices = util.initiate_nested_list(len(orient_unq))
        for i in range(len(orient_unq)):
            _idx_orient_unq = np.argwhere(orient_idx == i).flatten()
            _slices = [slices[_e] for _e in _idx_orient_unq]
            if len(_slices) > 1:
                if not return_index:
                    list_slices[i] = sort_slices_by_position(_slices)
                else:
                    _slices_sorted, _indices = sort_slices_by_position(_slices, return_index=True)
                    list_slices[i] = _slices_sorted
                    # noinspection PyUnboundLocalVariable
                    list_indices[i] = _indices
            else:
                # _indices to be added for return
                list_slices[i] = _slices
        if not return_index:
            return list_slices, flag
        else:
            return list_slices, flag, list_indices
    else:
        if not return_index:
            slices, scan_dir = sort_slices_by_position(slices, return_direction=True)
        else:
            slices, scan_dir, indices = sort_slices_by_position(slices, return_direction=True, return_index=True)
        pos_arr = np.array([_s.ImagePositionPatient for _s in slices])
        pos_arr_ref = pos_arr[:, np.roll(np.arange(3), 1)[scan_dir]]
        pos_unq, pos_idx = np.unique(pos_arr_ref, return_inverse=True)
        if len(pos_unq) > 1:
            flag = 2
            list_slices = util.initiate_nested_list(len(pos_unq))
            if return_index:
                list_indices = util.initiate_nested_list(len(pos_unq))
            for i in range(len(pos_unq)):
                _idx_pos_unq = np.argwhere(pos_idx == i).flatten()
                list_slices[i] = [slices[_i] for _i in _idx_pos_unq]
                if return_index:
                    # noinspection PyUnboundLocalVariable
                    list_indices[i] = [indices[_i] for _i in _idx_pos_unq]
            if not return_index:
                return list_slices, flag
            else:
                return list_slices, flag, list_indices
        else:
            flag = 0
            if not return_index:
                return slices, flag
            else:
                # noinspection PyUnboundLocalVariable
                return slices, flag, indices


def parse_dicom_dir_and_convert_dicom2nii(dicom_dir_fpath: str, dict_series: Union[util.NestedDict, None] = None,
                                          save_dir: Union[str, None] = None, get_slices: bool = False,
                                          slice_specs: Union[util.NestedDict, None] = None,
                                          return_dicom_dir: bool = False):
    from dicom2nifti.convert_generic import dicom_to_nifti

    def convert_slices_to_nifti_image(_slices_, _series_descr: str, _nii_file_name: Union[str, None] = None):
        try:
            img_nii = dicom_to_nifti(_slices_, _nii_file_name)['NII']
        except Exception:
            util.print_highlighted_text(f"Error in converting slices to NifTi image for series: "
                                        f"{_series_descr}")
            return -1
        else:
            return img_nii

    def validate_slices(_slices, idx_internal: Union[int, None] = None):
        is_enhanced = True if hasattr(_slices[0], 'ContrastBolusAgent') and \
                              _slices[0].ContrastBolusAgent != 'No' else False
        if is_enhanced:
            time_acq = min([_sl.timestamp for _sl in _slices])
            times_acq.append(time_acq)
            if idx_internal is None:
                dict_series[patient_id][study_id]['Exams'][i]['CE'] = 1
                dict_series[patient_id][study_id]['Exams'][i]['DeltaTime'] = time_acq
            else:
                dict_series[patient_id][study_id]['Exams'][i][idx_internal]['CE'] = 1
                dict_series[patient_id][study_id]['Exams'][i][idx_internal]['DeltaTime'] = time_acq
        else:
            if hasattr(_slices[0], 'ContrastBolusAgent'):
                print(_slices[0].ContrastBolusAgent)
        if idx_internal is None:
            if get_slices:
                dict_series[patient_id][study_id]['Exams'][i]['Slices'] = _slices
        else:
            if get_slices:
                dict_series[patient_id][study_id]['Exams'][i][idx_internal]['Slices'] = _slices
        if idx_internal is None:
            series_descr = f'{study_id}_{i}_{study_descr}'
        else:
            series_descr = f'{study_id}_{i}_{idx_internal}_{study_descr}'
        if len(_slices) <= 3:
            print(f'Scout image: {series_descr}')
        else:
            if save_dir is not None:
                if not os.path.exists(os.path.join(save_dir, patient_id, study_id)):
                    os.makedirs(os.path.join(save_dir, patient_id, study_id))
                if is_enhanced:
                    nii_file_name = os.path.join(save_dir, patient_id, study_id,
                                                 f'{patient_id}_{series_descr}_CE.nii.gz')
                else:
                    nii_file_name = os.path.join(save_dir, patient_id, study_id,
                                                 f'{patient_id}_{series_descr}.nii.gz')
                if not os.path.isfile(nii_file_name):
                    print(f'Save image: {nii_file_name}')
                    img_nii = convert_slices_to_nifti_image(_slices, series_descr, nii_file_name)
                else:
                    img_nii = nib.load(nii_file_name)
            else:
                img_nii = convert_slices_to_nifti_image(_slices, series_descr, None)
            # One may want to add tags about the assigned phases in slice_specs.
            # But this is phase assignment is not done yet.
            if slice_specs is not None:
                if idx_internal is None:
                    slice_specs[patient_id][study_id][i] = [files_dir[_i] for _i in list_indices]
                else:
                    slice_specs[patient_id][study_id][i][idx_internal] = [files_dir[_i] for _i in
                                                                          list_indices[idx_internal]]
            if (not get_slices) and (save_dir is not None):
                if isinstance(img_nii, nib.Nifti1Image):
                    if idx_internal is None:
                        dict_series[patient_id][study_id]['Exams'][i]['NII'] = img_nii.get_filename()
                    else:
                        dict_series[patient_id][study_id]['Exams'][i][idx_internal]['NII'] = img_nii.get_filename()
                else:
                    util.print_highlighted_text(f"Failure occurred when processing {series_descr}, "
                                                f"further analysis will be skipped!")

    dicom_dir = pydicom.dcmread(dicom_dir_fpath)
    if dict_series is None:
        dict_series = util.NestedDict()
    for _record in dicom_dir.patient_records:
        patient_id = _record.PatientID
        if hasattr(_record, 'PatientSex') and _record.PatientSex in ['M', 'F']:
            if 'Gender' not in dict_series[patient_id].keys():
                dict_series[patient_id]['Gender'] = _record.PatientSex
        for _exam in _record.children:
            study_id = f'{_exam.StudyDate}_{_exam.StudyID}'
            study_descr = util.concatenate_substrings(_exam.StudyDescription)
            times_acq = list()
            for i,  _series in enumerate(_exam.children):
                info_slices = _series.children
                slices = list()
                if slice_specs is not None:
                    files_dir = list()
                sub_folder_name = os.path.join(os.path.dirname(dicom_dir_fpath),
                                               os.sep.join(info_slices[0].ReferencedFileID[:-1]))
                print(f"Process {patient_id} on {study_id} from {sub_folder_name}")
                for _s in info_slices:
                    file_name = os.path.join(os.path.dirname(dicom_dir_fpath), os.sep.join(_s.ReferencedFileID))
                    try:
                        file = pydicom.dcmread(file_name)
                    except pydicom.errors.InvalidDicomError:
                        util.print_highlighted_text('Non-DICOM file detected with file name of {}'.format(file_name))
                    else:
                        slices.append(file)
                        if slice_specs is not None:
                            # noinspection PyUnboundLocalVariable
                            files_dir.append(file_name)
                if 'PatientAge' not in dict_series[patient_id][study_id].keys():
                    if hasattr(slices[0], 'PatientAge'):
                        find_age = re.search('(\d+)Y', slices[0].PatientAge)
                        age = int(find_age.group(1)) if find_age else 'NA'
                    else:
                        age = 'NA'
                    dict_series[patient_id][study_id]['PatientAge'] = age
                if hasattr(slices[0], 'ImageOrientationPatient') and len(slices) > 1:
                    if slice_specs is None:
                        list_slices, flag_split = check_and_split_slices_by_scan_settings(slices)
                    else:
                        list_slices, flag_split, list_indices = check_and_split_slices_by_scan_settings(slices,
                                                                                                        return_index=True)
                    if flag_split == 0:
                        validate_slices(list_slices)
                    else:
                        print(f'Multiple series found: {study_id}_{i}_{study_descr}')
                        for j in range(len(list_slices)):
                            validate_slices(list_slices[j], j)
                else:
                    continue
            if len(times_acq) > 1:
                time_start = min(times_acq)
                for _k, _v in dict_series[patient_id][study_id]['Exams'].items():
                    if isinstance(list(_v.keys())[0], str):
                        if 'CE' in _v.keys() and _v['CE'] == 1:
                            _v['DeltaTime'] -= time_start
                    else:
                        for _kk, _vv in _v.items():
                            if 'CE' in _vv.keys() and _vv['CE'] == 1:
                                _vv['DeltaTime'] -= time_start
    if return_dicom_dir:
        return dict_series, dicom_dir
    else:
        return dict_series

