import os
import vtk
import platform
import subprocess
import numpy as np
import vtkmodules
import nibabel as nib
from pathlib import Path
from typing import Union
# noinspection SpellCheckingInspection
import SimpleITK as sitk
from volViewerVispy import volume_viewer
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from nibabel.orientations import apply_orientation, io_orientation, inv_ornt_aff, axcodes2ornt, ornt_transform
from totalsegmentator.dicom_io import command_exists, download_dcm2niix


def get_spacing_from_affine(affine: np.ndarray):
    return np.linalg.norm(affine[:3, :3], axis=0)


def get_origin_from_affine(affine: np.ndarray):
    return affine[:3, 3]


def get_direction_from_affine(affine: np.ndarray):
    return affine[:3, :3] / get_spacing_from_affine(affine)


# noinspection SpellCheckingInspection
def get_direction_from_axcodes(axcodes: tuple = ('P', 'L', 'S')):
    ort = axcodes2ornt(axcodes)
    # Add 0.0 to avoid -0.0 in array
    return (np.eye(3)[ort[:, 0]] * ort[:, 1][:, np.newaxis]) + 0.0


def unpack_key_metadata_from_dict(meta_data: dict):
    spacing = meta_data.get('Spacing')
    origin = meta_data.get('Origin')
    direction = meta_data.get('Direction', get_direction_from_axcodes())
    return spacing, origin, direction


def get_affine_from_dict_metadata(meta_data: dict):
    spacing, origin, direction = unpack_key_metadata_from_dict(meta_data)
    # Scale direction matrix by spacing
    scaled_direction = direction * spacing[np.newaxis, :]
    # Construct 4x4 affine matrix
    affine = np.eye(4)
    affine[:3, :3] = scaled_direction
    affine[:3, 3] = origin
    return affine


def get_key_image_metadata(meta_data: Union[dict, np.ndarray]):
    if isinstance(meta_data, dict):
        spacing, origin, direction = unpack_key_metadata_from_dict(meta_data)
    else:
        # If numpy array, it should be a 4*4 affine matrix
        assert meta_data.shape == (4, 4)
        spacing = get_spacing_from_affine(meta_data)
        origin = get_origin_from_affine(meta_data)
        direction = get_direction_from_affine(meta_data)
    return spacing, origin, direction


# noinspection SpellCheckingInspection
def convert_numpy_to_image_with_metadata(data_array: np.ndarray, meta_data: Union[dict, np.ndarray], img_type: str):
    img_type = img_type.lower()
    if img_type == 'nifti':
        affine = meta_data if isinstance(meta_data, np.ndarray) else get_affine_from_dict_metadata(meta_data)
        image = nib.Nifti1Image(data_array, affine)
    else:
        spacing, origin, direction = get_key_image_metadata(meta_data)
        if img_type == 'sitk':
            # Orientation convention for sitk image is different.
            data_array = np.transpose(data_array, (2, 1, 0))
            image = sitk.GetImageFromArray(data_array)
            image.SetSpacing(spacing)
            image.SetOrigin(origin)
            image.SetDirection(direction.T.flatten())
        elif img_type == 'vtk':
            image = vtk.vtkImageData()
            image.SetOrigin(origin)  # type: ignore
            image.SetSpacing(spacing)  # type: ignore
            image.SetDimensions(data_array.shape)  # type: ignore
            mat_vtk = vtk.vtkMatrix3x3()
            mat_vtk.DeepCopy(direction.ravel())
            image.SetDirectionMatrix(mat_vtk)
            image.GetPointData().SetScalars(numpy_to_vtk(data_array.reshape(-1, order='F'), deep=True))
        else:
            raise ValueError(f'The output image type : {img_type} not supported!')
    return image


def read_nifti_as_vtk_image(nifti_file_name):
    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(nifti_file_name)
    reader.Update(0)
    image = reader.GetOutput()
    return image


def convert_nifti_to_vtk_image(img_nii: nib.nifti1.Nifti1Image):
    array, affine, spacing = convert_nifti_to_numpy(img_nii)
    return convert_numpy_to_image_with_metadata(array, affine, 'vtk')


# noinspection SpellCheckingInspection
def convert_sitk_to_vtk_image(img_sitk: sitk.Image):
    array, affine, spacing = convert_sitk_image_to_numpy(img_sitk)
    return convert_numpy_to_image_with_metadata(array, affine, 'vtk')


def check_image_array_dtype(image_arr: np.ndarray):
    if 0 <= np.min(image_arr) and np.max(image_arr) <= 1:
        raise ValueError("The provided image data array should not be normalized!")
    elif np.min(image_arr) < 0 or np.max(image_arr) > 255:
        dtype = np.int16
    else:
        dtype = np.uint8
    return dtype


# noinspection SpellCheckingInspection
def image_reorientation(image: Union[np.ndarray, nib.Nifti1Image], affine: Union[np.ndarray, None] = None,
                        target_axcodes: tuple = ('P', 'L', 'S'), return_numpy: bool = False):
    if isinstance(image, np.ndarray):
        assert affine is not None
    else:
        affine = image.affine
        image = np.asanyarray(image.dataobj)
    if 0 <= np.min(image) and np.max(image) <= 1:
        raise ValueError("The provided image data array should not be normalized!")
    elif np.min(image) < 0 or np.max(image) > 255:
        dtype = np.int16
    else:
        dtype = np.uint8
    if nib.aff2axcodes(affine) != target_axcodes:
        ornt_new = io_orientation(affine) if target_axcodes == ('R', 'A', 'S') else \
            ornt_transform(io_orientation(affine), axcodes2ornt(target_axcodes))
        image_new = apply_orientation(image, ornt_new).astype(dtype)
        affine_new = affine.dot(inv_ornt_aff(ornt_new, image.shape))
    else:
        print('Target orientation is the same as the original. No reorientation is needed.')
        image_new = image
        affine_new = affine
    if not return_numpy:
        return nib.Nifti1Image(image_new, affine_new)
    else:
        spacing = get_spacing_from_affine(affine_new)
        return image_new, affine_new, tuple(spacing)


def convert_nifti_to_numpy(image: nib.Nifti1Image, show_image=False):
    data_array, affine, spacing = image_reorientation(image, return_numpy=True)
    if show_image:
        volume_viewer(data_array, np.array(spacing)/spacing[0])
    return data_array, affine, spacing


def get_metadata_from_image(image: Union[sitk.Image, vtkmodules.vtkCommonDataModel.vtkImageData],
                            return_affine: bool = False):
    meta_data = dict()
    if isinstance(image, sitk.Image):
        meta_data['Spacing'] = np.array(image.GetSpacing())
        meta_data['Origin'] = np.array(image.GetOrigin())
        meta_data['Direction'] = np.array(image.GetDirection()).reshape(3, 3).T
    elif isinstance(image, vtkmodules.vtkCommonDataModel.vtkImageData):
        meta_data['Spacing'] = np.array(image.GetSpacing())
        meta_data['Origin'] = np.array(image.GetOrigin())
        meta_data['Direction'] = np.array(image.GetDirectionMatrix().GetData()).reshape(3, 3)
    if return_affine:
        return get_affine_from_dict_metadata(meta_data)
    else:
        return meta_data


# noinspection SpellCheckingInspection
def convert_sitk_image_to_numpy(img_sitk: sitk.Image, show_image=False):
    # SITK has an axis convention of Z * Y * X, therefore the x and z axis has to be swapped.
    data_array = np.transpose(sitk.GetArrayFromImage(img_sitk), (2, 1, 0))
    # Use nifiti image affine matrix to determine image orientation, as this is easier for debugging.
    affine = get_metadata_from_image(img_sitk, True)
    # noinspection PyUnresolvedReferences
    data_array, affine, spacing = image_reorientation(data_array, affine=affine, return_numpy=True)
    if show_image:
        volume_viewer(data_array, np.array(spacing)/spacing[0])
    return data_array, affine, spacing


def convert_vtk_image_to_numpy(img_vtk: vtkmodules.vtkCommonDataModel.vtkImageData,
                               points_label: Union[None, int] = None):
    x_size, y_size, z_size = img_vtk.GetDimensions()
    # vtk implements axes different from numpy array convention
    data_array = np.swapaxes(vtk_to_numpy(img_vtk.GetPointData().GetScalars()).reshape((z_size, x_size, y_size)), 0, 2)
    meta_data = get_metadata_from_image(img_vtk)
    assert img_vtk.GetDimensions() == data_array.shape
    if points_label is not None:
        assert isinstance(points_label, int)
        idx = np.column_stack(np.where(data_array == points_label))
        # Multiply the voxel size to make the coordinates of uniform size.
        # rotation_matrix = Rotation.from_euler('z', 180, degrees=True)
        meta_data['Indices'] = idx
        meta_data['Points'] = idx * meta_data['Spacing']
        meta_data['Offset'] = np.min(meta_data['Points'], axis=0)
    return data_array, meta_data


# noinspection SpellCheckingInspection
def dicom_to_nifti(input_path, output_path: Union[str, Path]):
    """
    input_path: a directory of dicom slices
    output_path: a nifti file path
    """
    if isinstance(output_path, str):
        output_path = Path(output_path)

    config_dir = Path(os.environ["TOTALSEG_WEIGHTS_PATH"]) / "nnUNet" 
    if command_exists("dcm2niix"):
        dcm2niix = "dcm2niix"
    else:
        if platform.system() == "Windows":
            dcm2niix = config_dir / "dcm2niix.exe"
        else:
            dcm2niix = config_dir / "dcm2niix"
        if not dcm2niix.exists():
            download_dcm2niix()
    output_file_no_ext = output_path.name.split('.')[0]
    subprocess.call(f'"{dcm2niix}" -o "{output_path.parent}" -z y -f "{output_file_no_ext}" "{input_path}"')
    os.remove(os.path.join(output_path.parent, output_file_no_ext + ".json"))

