import warnings
import numpy as np
import vtk
import vtkmodules
from typing import List, Union
from util import initiate_nested_list
from segManager import recover_3d_mask_array_from_coord
from vtkmodules.util.numpy_support import vtk_to_numpy, numpy_to_vtk
from genericImageIO import convert_numpy_to_image_with_metadata, get_metadata_from_image


def get_2d_mip_from_3d_vtk_image(img: vtkmodules.vtkCommonDataModel.vtkImageData):

    mat_axial = vtk.vtkMatrix4x4()
    center = img.GetCenter()
    mat_axial.DeepCopy((1, 0, 0, center[0], 0, 1, 0, center[1], 0, 0, 1, center[2], 0, 0, 0, 1))

    filter_mip = vtk.vtkImageReslice()
    filter_mip.SetInputData(img)
    filter_mip.SetOutputDimensionality(2)
    filter_mip.SetResliceAxes(mat_axial)
    filter_mip.SetSlabModeToMax()
    filter_mip.SetSlabNumberOfSlices(1)
    filter_mip.Update(0)
    return filter_mip.GetOutput()


def opening_with_vtk_image_2d(img_2d: vtkmodules.vtkCommonDataModel.vtkImageData, struct_size=(15, 15)):
    filter_open = vtk.vtkImageOpenClose3D()
    filter_open.SetInputData(img_2d)
    filter_open.SetKernelSize(struct_size[0], struct_size[1], 1)  # Setting z_size to 1 to get a 2D filter
    filter_open.SetOpenValue(1)  # Define the opening operation to act on mask image with value of 1.
    filter_open.SetCloseValue(0)
    filter_open.Update(0)
    return filter_open.GetOutput()


def fill_holes_in_vtk_image_2d(img_2d: vtkmodules.vtkCommonDataModel.vtkImageData, struct_size=(50, 50)):
    filter_close = vtk.vtkImageOpenClose3D()
    filter_close.SetInputData(img_2d)
    filter_close.SetKernelSize(struct_size[0], struct_size[1], 1)
    filter_close.SetCloseValue(1)
    filter_close.Update(0)
    return filter_close.GetOutput()


def convert_vtk_image_to_binary_with_threshold(img: vtkmodules.vtkCommonDataModel.vtkImageData, threshold=-200):
    filter_mask = vtk.vtkImageThreshold()
    filter_mask.SetInputData(img)
    filter_mask.ThresholdByUpper(threshold)
    filter_mask.SetInValue(1)
    filter_mask.SetOutValue(0)
    filter_mask.ReplaceInOn()
    filter_mask.ReplaceOutOn()
    filter_mask.SetOutputScalarTypeToUnsignedChar()  # Required to use it as a mask
    filter_mask.Update(0)
    return filter_mask.GetOutput()


def expand_to_3d_vtk_image(img_2d: vtkmodules.vtkCommonDataModel.vtkImageData, z_size):

    dims = img_2d.GetDimensions()
    # Convert vtk image to numpy array
    img_arr_2d = (vtk_to_numpy(img_2d.GetPointData().GetScalars())).reshape(dims)
    img_arr_3d = np.rollaxis(np.repeat(img_arr_2d[:, :, np.newaxis], repeats=z_size, axis=2), 2)
    # Create 3D image data
    img_3d = vtk.vtkImageData()
    img_3d.SetDimensions(dims[0], dims[1], z_size)
    if img_2d.GetScalarType() == 3:
        img_3d.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
        vtk_arr_3d = numpy_to_vtk(img_arr_3d.ravel(), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
    else:
        img_3d.AllocateScalars(vtk.VTK_SHORT, 1)
        vtk_arr_3d = numpy_to_vtk(img_arr_3d.ravel(), deep=True, array_type=vtk.VTK_SHORT)
    img_3d.GetPointData().SetScalars(vtk_arr_3d)
    return img_3d


def filter_vtk_image_with_mask(img_vtk: vtkmodules.vtkCommonDataModel.vtkImageData,
                               mask: vtkmodules.vtkCommonDataModel.vtkImageData,
                               fill_value: Union[int, float, np.int_, np.float_, None] = None, reverse=False) \
                               -> vtkmodules.vtkCommonDataModel.vtkImageData:

    mask_filter = vtk.vtkImageMask()
    mask_filter.SetImageInputData(img_vtk)
    mask_filter.SetMaskInputData(mask)
    if reverse:
        # Use the inverse of the original mask
        mask_filter.NotMaskOn()
    if fill_value is None:
        fill_value = vtk_to_numpy(img_vtk.GetPointData().GetScalars()).min()
    # noinspection PyArgumentList
    mask_filter.SetMaskedOutputValue(float(fill_value))
    mask_filter.Update(0)
    return mask_filter.GetOutput()


def remove_bed_from_vtk_image(img: vtkmodules.vtkCommonDataModel.vtkImageData, threshold=-200, size_open=(15, 15),
                              size_close=(50, 50)) -> vtkmodules.vtkCommonDataModel.vtkImageData:
    # Generate 2D MIP image from 3D img_vtk
    img_2d_mip = get_2d_mip_from_3d_vtk_image(img)

    # Generate 2D mask based on the MIP image. A mask is initialized by using a global threshold, and then an opening
    # operation, which removes the bed, is applied to it.
    img_2d_mask = convert_vtk_image_to_binary_with_threshold(img_2d_mip, threshold)
    img_2d_mask = opening_with_vtk_image_2d(img_2d_mask, size_open)
    img_2d_mask = fill_holes_in_vtk_image_2d(img_2d_mask, size_close)
    # Expand the 2D mask to the z dimension
    img_3d_mask = expand_to_3d_vtk_image(img_2d_mask, img.GetDimensions()[2])
    # noinspection PyArgumentList
    img_3d_mask.SetSpacing(img.GetSpacing())
    img_filtered = filter_vtk_image_with_mask(img, img_3d_mask)
    return img_filtered


def convert_mask_to_vtk_image(mask: np.ndarray, meta_data: Union[dict, np.ndarray],
                              array_shape: Union[tuple, None] = None):
    if min(mask.shape) == 3:
        # The input array is an array of mask coordinates. Convert the coordinates back to bool array.
        if array_shape is None:
            raise Exception("For input of mask coordinate array, the 'array_shape' parameter must be provided in order "
                            "to reconstruct the boolean mask!")
        _mask = recover_3d_mask_array_from_coord(mask, array_shape)
    if mask.dtype != np.bool_:
        warnings.warn(f'Input mask array is of type {mask.dtype}. It is internally converted to boolean before '
                      f'conversion to vtk image.')
        mask = mask.astype(np.bool_)
    # Convert mask numpy array to vtk image type.
    mask = convert_numpy_to_image_with_metadata(mask.astype(np.uint8), meta_data, 'vtk')
    return mask


def merge_list_masks_and_convert_to_vtk_image(list_masks: List[np.ndarray], meta_data: Union[dict, np.ndarray],
                                              array_shape: Union[tuple, None] = None):
    from functools import reduce
    import operator

    if min(list_masks[0].shape) == 3:
        # Arrays in list are mask coordinates. Stack them together and take the unique rows to form a
        # new coordinate stack.
        if array_shape is None:
            raise Exception("For input of mask coordinate arrays, the 'array_shape' parameter must be provided in "
                            "order to reconstruct the final merged boolean mask!")
        mask = np.unique(np.vstack(list_masks), axis=0)
    else:
        if not all([_m.dtype == np.bool_ for _m in list_masks]):
            list_masks = [_m.astype(np.bool_) for _m in list_masks]
        mask = reduce(operator.xor, list_masks)
    mask = convert_mask_to_vtk_image(mask, meta_data, array_shape)
    return mask


def get_rendering_preset_from_xml(name_preset: str, xml_file_name: str='presets.xml'):
    def to_nested_list(_list, n):
        return [_list[i:i + n] for i in range(0, len(_list), n)]

    from xml.etree import ElementTree
    root = (ElementTree.parse(xml_file_name)).getroot()
    preset_dict = root[[e.attrib['name'] == name_preset for e in root].index(True)].attrib
    attrs = ['ambient', 'colorTransfer', 'diffuse', 'effectiveRange', 'gradientOpacity', 'scalarOpacity',
                  'specular', 'specularPower', 'shade']
    list_keys = list(preset_dict.keys())
    for k in list_keys:
        if k not in attrs:
            preset_dict.pop(k)
        else:
            if ' ' in preset_dict[k].strip():
                _value = preset_dict[k].strip().split(' ')
                if '.' in preset_dict[k]:
                    _value = [float(_v) for _v in _value]
                else:
                    _value = [int(_v) for _v in _value]
                # In current implementation, effective range has no impact on the rendering result.
                if k != 'effectiveRange':
                    assert int(_value[0]) == len(_value[1:])
                    if 'color' in k:
                        assert len(_value[1:]) % 4 == 0
                        _value = to_nested_list(_value[1:], 4)
                    else:
                        assert len(_value[1:]) % 2 == 0
                        _value = to_nested_list(_value[1:], 2)
            else:
                if '.' in preset_dict[k]:
                    _value = float(preset_dict[k])
                else:
                    _value = int(preset_dict[k])
            # noinspection PyTypeChecker
            preset_dict[k] = _value
    return preset_dict


def create_vtk_volume_property_from_dict(_dict, spacing: Union[tuple, None] = None):
    prop = vtk.vtkVolumeProperty()
    # prop.SetIndependentComponents(True)
    prop.SetSpecularPower(0, _dict['specularPower'])
    prop.SetSpecular(0, _dict['specular'])
    prop.SetAmbient(0, _dict['ambient'])
    prop.SetDiffuse(0, _dict['diffuse'])

    if _dict['shade'] == 1:
        prop.ShadeOn(0)

    gradient_opacity = vtk.vtkPiecewiseFunction()
    for item in _dict['gradientOpacity']:
        x, y = item
        gradient_opacity.AddPoint(x, y)
    prop.SetGradientOpacity(0, gradient_opacity)

    scalar_opacity = vtk.vtkPiecewiseFunction()
    for item in _dict['scalarOpacity']:
        x, y = item
        scalar_opacity.AddPoint(x, y)
    prop.SetScalarOpacity(0, scalar_opacity)

    color_transfer_func = vtk.vtkColorTransferFunction()
    for item in _dict['colorTransfer']:
        x, r, g, b = item
        color_transfer_func.AddRGBPoint(x, r, g, b)

    prop.SetColor(0, color_transfer_func)
    prop.SetInterpolationTypeToLinear()
    if spacing is not None:
        prop.SetScalarOpacityUnitDistance(0, min(spacing))
    return prop


# noinspection SpellCheckingInspection
def create_actor_for_volume_rendering_with_preset(img_vtk: vtkmodules.vtkCommonDataModel.vtkImageData,
                                                  preset_dict: dict, render_mode: int = 2, with_jitter: bool = False,
                                                  blending_coeff: float = 1.0):
    """

    :param img_vtk: vtkImageData type;
    :param preset_dict: dict that specifies the rendering settings, either loaded from xml file or manually defined;
    :param render_mode: supported rendering modes are:
                        0: default mode, determined automatically based on rendering parameters and hardware support;
                        1: CPU-based ray cast render mode;
                        2: GPU-based ray cast render mode;
                        3: Intel CPU-based OSPRay render mode;
                        4: an industry standard, Khronos ANARI cross-platform 3D rendering engine.
    :param with_jitter: bool, valid only if 2 is chosen for render_mode, default False.
                        If set to True, slight pertubations to ray transveral direction will be added when rendering.
    :param blending_coeff: float number between 0 and 2.
                           O: no scattered rays will be cast;
                           2: the shader will only use volumetric scattering model;
                           Otherwise: the shader blends the two ray casting models with the provided coefficient.
    :return:
    """

    prop = create_vtk_volume_property_from_dict(preset_dict, img_vtk.GetSpacing())
    if render_mode == 2:
        mapper = vtk.vtkGPUVolumeRayCastMapper()
        if with_jitter:
            mapper.UseJitteringOn()
    else:
        mapper = vtk.vtkSmartVolumeMapper()
        mapper.SetRequestedRenderMode(render_mode)
    mapper.SetVolumetricScatteringBlending(blending_coeff)
    mapper.SetInputData(img_vtk)

    actor = vtk.vtkVolume()
    # noinspection PyTypeChecker
    actor.SetMapper(mapper)
    actor.SetProperty(prop)
    actor.Update()
    return actor


def save_render_window_to_png_image(window: vtk.vtkRenderWindow, save_fig_name: str, scale_window: int = 4):
    filter_screenshot = vtk.vtkWindowToImageFilter()
    filter_screenshot.SetInput(window)
    filter_screenshot.SetInputBufferTypeToRGBA()
    filter_screenshot.SetScale(scale_window, scale_window)
    filter_screenshot.Update(0)
    writer_image = vtk.vtkPNGWriter()
    writer_image.SetFileName(save_fig_name)
    writer_image.SetInputData(filter_screenshot.GetOutput())
    writer_image.Write()


def volume_rendering_with_actor(actor: Union[vtkmodules.vtkRenderingCore.vtkProp3D, list],
                                window_name: str = 'VolumeViewer', yaw: Union[int, None] = None,
                                pitch: Union[int, None] = None, save_fig_name: Union[str, None] = None,
                                scale_window: Union[int, None] = 4):
    from vtkmodules.vtkCommonColor import vtkNamedColors
    win_output = vtk.vtkOutputWindow()
    win_output.GlobalWarningDisplayOff()
    renderer = vtk.vtkRenderer()
    if isinstance(actor, vtkmodules.vtkRenderingCore.vtkProp3D):
        # vtkVolume is a subclass of vtkProp3D.
        renderer.AddActor(actor)
    else:
        assert all([isinstance(e, vtkmodules.vtkRenderingCore.vtkProp3D) for e in actor])
        for _actor in actor:
            renderer.AddActor(_actor)
    colors = vtkNamedColors()
    # noinspection PyArgumentList
    renderer.SetBackground(colors.GetColor3d('Black'))
    win = vtk.vtkRenderWindow()
    win.AddRenderer(renderer)
    win.SetWindowName(window_name)
    # It is observed that the -90 degrees of rotation is needed to ensure upright orientation.
    renderer.GetActiveCamera().Elevation(-90)
    # Reset camera brings it to the center, otherwise the object will likely be out of scene.
    renderer.ResetCamera()
    # Please be noted that every time one rotate the object, one should reset the camera to bring the object back to
    # the center.
    if yaw is not None:
        renderer.GetActiveCamera().Yaw(yaw)
        renderer.ResetCamera()
    if pitch is not None:
        renderer.GetActiveCamera().Pitch(pitch)
        renderer.ResetCamera()
    # noinspection PyArgumentList
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(win)
    interactor.Initialize()
    win.Render()
    if save_fig_name is not None:
        save_render_window_to_png_image(win, save_fig_name, scale_window)
    interactor.Start()


def prepare_masked_volume_for_rendering(img_vtk: vtkmodules.vtkCommonDataModel.vtkImageData,
                                        mask: Union[np.ndarray, vtkmodules.vtkCommonDataModel.vtkImageData],
                                        reverse: Union[bool, None] = None) \
                                        -> vtkmodules.vtkCommonDataModel.vtkImageData:
    if isinstance(mask, np.ndarray):
        meta_data = get_metadata_from_image(img_vtk)
        mask = convert_mask_to_vtk_image(mask, meta_data)
    if reverse is not None:
        img_vtk = filter_vtk_image_with_mask(img_vtk, mask, reverse=reverse)
    else:
        img_vtk = filter_vtk_image_with_mask(img_vtk, mask)
    return img_vtk


def create_rendering_for_volume_with_preset(img_vtk: vtkmodules.vtkCommonDataModel.vtkImageData,
                                            preset_dict: dict, with_jitter: bool = False,
                                            mask: Union[np.ndarray, None] = None, reverse: Union[bool, None] = None,
                                            smooth: bool = False, **kwargs):
    if mask is not None:
        img_vtk = prepare_masked_volume_for_rendering(img_vtk, mask, reverse=reverse)
    if smooth:
        std = 1.0 if 'std' not in kwargs.keys() else kwargs['std']
        radius = 1.0 if 'radius' not in kwargs.keys() else kwargs['radius']
        img_vtk = gaussian_smooth_vtk_image(img_vtk, std, radius)
    actor = create_actor_for_volume_rendering_with_preset(img_vtk, preset_dict, with_jitter=with_jitter)
    save_fig_name = None if 'save_fig_name' not in kwargs.keys() else kwargs['save_fig_name']
    # noinspection PyTypeChecker
    volume_rendering_with_actor(actor, save_fig_name=save_fig_name)


def create_composite_rendering_with_mask_associated_presets(img_vtk: vtkmodules.vtkCommonDataModel.vtkImageData,
                                                            mask: Union[List[np.ndarray], np.ndarray],
                                                            preset_fg: Union[List[dict], dict],
                                                            preset_bg: Union[dict, None] = None,
                                                            with_jitter: bool = False, smooth: bool = False,
                                                            **kwargs):

    flag_merge_masks = False
    meta_data = get_metadata_from_image(img_vtk)
    array_shape = img_vtk.GetDimensions()
    if isinstance(mask, np.ndarray):
        # Array shape information might not be needed, if the 'mask' is not an array of mask coordinates.
        mask = convert_mask_to_vtk_image(mask, meta_data, array_shape)
    else:
        # mask with list type
        if isinstance(preset_fg, list):
            try:
                assert len(preset_fg) == len(mask) and isinstance(preset_fg[0], dict)
            except AssertionError:
                raise Exception("The length of 'preset_fg' should be the same as length of 'mask' and its elements "
                                "should be of dict type.")
            else:
                flag_merge_masks = True
        else:  # preset_fg is a dict
            assert len(mask) >= 1
            if len(mask) > 1:
                warnings.warn(f'There is only one foreground rendering dictionary provided, but {len(mask)} masks '
                              f'provided, all segmented objects will be rendered using the same setting!')
                flag_merge_masks = True
            else:
                mask = convert_mask_to_vtk_image(mask[0], meta_data, array_shape)

    if flag_merge_masks:
        mask_merged = merge_list_masks_and_convert_to_vtk_image(mask, meta_data, array_shape)
    fill_value = vtk_to_numpy(img_vtk.GetPointData().GetScalars()).min()

    smooth_params = None if 'smooth_params' not in kwargs.keys() else kwargs['smooth_params']
    if smooth_params is not None:
        try:
            assert isinstance(smooth_params, dict)
        except AssertionError:
            raise Exception("The provided 'smooth_params' has to be a dict type!")
    if smooth:
        if smooth_params is None:
            # The same smoothing parameters will be applied to all structures through smoothing the original input
            # img_vtk.
            print("No 'smooth_params' provided. Default std value of 1.0 and radius of 1.0 are used to smooth the "
                  "input img_vtk, so all structures will get the same smoothing effect.")
            std = 1.0
            radius = 1.0
            img_vtk = gaussian_smooth_vtk_image(img_vtk, std, radius)

    if preset_bg is not None:
        if flag_merge_masks:
            # noinspection PyUnboundLocalVariable
            vol_bg = filter_vtk_image_with_mask(img_vtk, mask_merged, fill_value, reverse=True)
        else:
            vol_bg = filter_vtk_image_with_mask(img_vtk, mask, fill_value, reverse=True)
        if smooth and smooth_params is not None:
            if 'background' not in smooth_params.keys():
                std = smooth_params.get('std', 1.0)
                radius = smooth_params.get('radius', 1.0)
            else:
                std = smooth_params['background'].get('std', 1.0)
                radius = smooth_params['background'].get('radius', 1.0)
            vol_bg = gaussian_smooth_vtk_image(vol_bg, std, radius)
        actor_bg = create_actor_for_volume_rendering_with_preset(vol_bg, preset_bg, with_jitter=with_jitter)

    save_fig_name = None if 'save_fig_name' not in kwargs.keys() else kwargs['save_fig_name']
    if isinstance(preset_fg, dict):
        if flag_merge_masks:
            vol_fg = filter_vtk_image_with_mask(img_vtk, mask_merged, fill_value)
        else:
            vol_fg = filter_vtk_image_with_mask(img_vtk, mask, fill_value)
        if smooth and smooth_params is not None:
            if 'foreground' not in smooth_params.keys():
                std = smooth_params.get('std', 1.0)
                radius = smooth_params.get('radius', 1.0)
            else:
                std = smooth_params['foreground'].get('std', 1.0)
                radius = smooth_params['foreground'].get('radius', 1.0)
            vol_fg = gaussian_smooth_vtk_image(vol_fg, std, radius)
        actor_fg = create_actor_for_volume_rendering_with_preset(vol_fg, preset_fg, with_jitter=with_jitter)
        if preset_bg is not None:
            # noinspection PyUnboundLocalVariable
            volume_rendering_with_actor([actor_bg, actor_fg], save_fig_name=save_fig_name)
        else:
            # noinspection PyTypeChecker
            volume_rendering_with_actor(actor_fg, save_fig_name=save_fig_name)
    else:
        actor_fg = list()
        list_params = None
        if smooth and smooth_params is not None:
            if 'foreground' not in smooth_params.keys():
                std = smooth_params.get('std', 1.0)
                radius = smooth_params.get('radius', 1.0)
                # All foreground objects will be smoothed using the same 'std' and 'radius' parameters.
                list_params = [[std] * len(preset_fg), [radius] * len(preset_fg)]
            else:
                # Individualized smooth parameters will be applied to each foreground object.
                list_params = initiate_nested_list(2)
                smooth_params_fg = smooth_params['foreground']
                try:
                    assert isinstance(smooth_params_fg, list) and isinstance(smooth_params_fg[0], dict) and \
                           len(smooth_params_fg) == len(preset_fg)
                except AssertionError:
                    raise Exception("The provided foreground smooth parameters need to be a list of dict and make sure"
                                    " that its length is the same as the list of foreground rendering settings!")
                for i, _params in enumerate(smooth_params_fg):
                    list_params[0].append(_params.get('std', 1.0))
                    list_params[1].append(_params.get('radius', 1.0))

        for i, _dict in enumerate(preset_fg):
            _msk = convert_mask_to_vtk_image(mask[i], meta_data, array_shape)
            _vol_fg = filter_vtk_image_with_mask(img_vtk, _msk, fill_value)
            if list_params is not None:
                _vol_fg = gaussian_smooth_vtk_image(_vol_fg, list_params[0][i], list_params[1][i])
            actor_fg.append(create_actor_for_volume_rendering_with_preset(_vol_fg, _dict, with_jitter=with_jitter))
        if preset_bg is not None:
            # noinspection PyUnboundLocalVariable
            volume_rendering_with_actor([actor_bg] + actor_fg, save_fig_name=save_fig_name)
        else:
            volume_rendering_with_actor(actor_fg, save_fig_name=save_fig_name)


def multi_volume_rendering_with_presets(list_image_files: List[str], spacing: tuple, list_preset_dict: List[dict],
                                        with_jitter=False, save_fig_name: Union[str, None] = None):
    from vtkmodules.vtkCommonColor import vtkNamedColors

    mapper_multi = vtk.vtkGPUVolumeRayCastMapper()
    if with_jitter:
        mapper_multi.UseJitteringOn()
    actor_multi = vtk.vtkMultiVolume()
    actor_multi.SetMapper(mapper_multi)
    for i, _file in enumerate(list_image_files):
        _reader = vtk.vtkNIFTIImageReader()
        _reader.SetFileName(_file)
        _reader.Update(0)
        _prop = create_vtk_volume_property_from_dict(list_preset_dict[i], spacing)
        _actor = vtk.vtkVolume()
        _actor.SetProperty(_prop)
        mapper_multi.SetInputConnection(i, _reader.GetOutputPort(0))
        actor_multi.SetVolume(_actor, i)

    renderer = vtk.vtkRenderer()
    colors = vtkNamedColors()
    # noinspection PyArgumentList
    renderer.SetBackground(colors.GetColor3d('Black'))
    renderer.GetActiveCamera().Elevation(-90)
    # noinspection PyTypeChecker
    renderer.AddVolume(actor_multi)
    renderer.ResetCamera()
    # noinspection PyTypeChecker
    win = vtk.vtkRenderWindow()
    win.AddRenderer(renderer)
    # noinspection PyArgumentList
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(win)
    interactor.Initialize()
    win.Render()
    if save_fig_name is not None:
        save_render_window_to_png_image(win, save_fig_name)
    interactor.Start()


def gaussian_smooth_vtk_image(img_vtk: vtkmodules.vtkCommonDataModel.vtkImageData, std=2.0, radius=1.0):
    filter_gauss = vtk.vtkImageGaussianSmooth()
    filter_gauss.SetInputData(img_vtk)
    filter_gauss.SetDimensionality(3)
    filter_gauss.SetStandardDeviation(std)
    filter_gauss.SetRadiusFactor(radius)
    filter_gauss.Update(0)
    return filter_gauss.GetOutput()


def resample_vtk_image(img_vtk: vtkmodules.vtkCommonDataModel.vtkImageData, spacing=0.5):
    filter_resample = vtk.vtkImageResample()
    filter_resample.SetDimensionality(3)
    filter_resample.SetInputData(img_vtk)
    filter_resample.SetOutputSpacing(spacing, spacing, spacing)
    filter_resample.Update(0)
    return filter_resample.GetOutput()


