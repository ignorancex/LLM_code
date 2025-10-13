import matplotlib as mpl
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import cv2 as cv
import bodyPartWindow
import numpy as np
import copy
import simpleStats
from matplotlib.widgets import Slider
from typing import Union
import functools


px = 1 / plt.rcParams['figure.dpi']
mpl.use('QtAgg')


def dicom_volume_viewer_with_sliders(img_3d, aspect_ratios=(1, 1, 1), cmap='gray'):
    img_shape = img_3d.shape
    aspect_ratio_z = round(aspect_ratios[2] * img_shape[2] / img_shape[0], 2)

    fig, axs = plt.subplots(2, 2, figsize=(960 * px, 768 * px),
                            gridspec_kw={'width_ratios': [aspect_ratio_z, 1],
                                         'height_ratios': [aspect_ratio_z, 1]})
    fig.delaxes(axs[1, 1])
    slider_width = 0.25-(aspect_ratio_z-1)*0.05
    if aspect_ratio_z < 2.5:
        slider_x_position_x = 0.6 + (aspect_ratio_z-1)*0.1
    elif 2.5 <= aspect_ratio_z < 4:
        slider_x_position_x = 0.75 + (aspect_ratio_z-2.5)*0.0333
    elif aspect_ratio_z >= 4:
        slider_x_position_x = 0.8 + (aspect_ratio_z-4)*0.01
    # noinspection PyUnboundLocalVariable
    # noinspection PyTypeChecker
    slider_x = Slider(
        ax=fig.add_axes([slider_x_position_x, 0.9, slider_width, 0.03]),  #
        label='x',
        valmin=0,
        valmax=img_shape[0] - 1,
        valinit=img_shape[0] // 2,
        valstep=1
    )
    # noinspection PyTypeChecker
    slider_y = Slider(
        ax=fig.add_axes([0.48-slider_width*1.2, 0.05, slider_width, 0.03]),
        label='y',
        valmin=0,
        valmax=img_shape[1] - 1,
        valinit=img_shape[1] // 2,
        valstep=1
    )
    # noinspection PyTypeChecker
    slider_z = Slider(
        ax=fig.add_axes([0.48-slider_width*1.2, 0.9, slider_width, 0.03]),
        label='z',
        valmin=0,
        valmax=img_shape[2] - 1,
        valinit=img_shape[2] // 2,
        valstep=1
    )

    def viewer_3d(x=img_shape[0] // 2, y=img_shape[1] // 2, z=img_shape[2] // 2):
        # For viewing 3 orthogonal planes in subplots for a 3D data array.
        im_ax = axs[0, 0].imshow(img_3d[:, :, int(z)], cmap=cmap)
        axs[0, 0].set_aspect(aspect_ratios[0])
        axs[0, 0].set_axis_off()

        im_sag = axs[1, 0].imshow(img_3d[:, int(y), :], cmap=cmap)
        axs[1, 0].set_aspect(aspect_ratios[1]/aspect_ratios[2])
        axs[1, 0].set_axis_off()

        im_cor = axs[0, 1].imshow(np.flipud(img_3d[int(x), :, :].T), cmap=cmap)
        axs[0, 1].set_aspect(aspect_ratios[2]/aspect_ratios[1])
        axs[0, 1].set_axis_off()
        plt.show()
        return im_ax, im_sag, im_cor

    def update(val):
        im_ax, im_sag, im_cor = viewer_3d(x=slider_x.val, y=slider_y.val, z=slider_z.val)
        im_cor.set_data(np.flipud(img_3d[int(slider_x.val), :, :].T))
        im_sag.set_data(img_3d[:, int(slider_y.val), :])
        im_ax.set_data(img_3d[:, :, int(slider_z.val)])
        fig.canvas.draw_idle()

    slider_x.on_changed(update)
    slider_y.on_changed(update)
    slider_z.on_changed(update)

    viewer_3d()


def axial_view_with_sagittal_plane_visualiser(img_3d, aspect_ratios=(1, 1, 1), cmap='gray'):
    img_shape = img_3d.shape
    fig, axs = plt.subplots(1, 2, figsize=(960 * px, 768 * px))
    # noinspection PyUnboundLocalVariable
    # noinspection PyTypeChecker
    slider_z = Slider(
        ax=fig.add_axes([0.15, 0.9, 0.28, 0.03]),  #
        label='Z position',
        valmin=0,
        valmax=img_shape[2] - 1,
        valinit=img_shape[2] // 2,
        valstep=1
    )

    def paired_view(z=img_shape[2] // 2):
        im_ax = axs[0].imshow(img_3d[:, :, int(round(z))], cmap=cmap)
        axs[0].set_aspect(aspect_ratios[0])
        axs[0].set_axis_off()
        im_sag = axs[1].imshow(np.flipud(np.squeeze(img_3d[:, img_3d.shape[1]//2, :]).T), cmap=cmap)
        axs[1].set_aspect(aspect_ratios[2])
        axs[1].set_axis_off()
        axs[1].plot(range(img_3d.shape[1]), [img_shape[2] - z] * img_3d.shape[1], 'r-', alpha=0.5)
        plt.show()
        return im_ax, im_sag

    def update(val):
        axs[1].lines[-1].remove()
        im_ax, im_sag = paired_view(z=slider_z.val)
        im_ax.set_data(img_3d[:, :, int(round(slider_z.val))])
        im_sag.set_data(np.flipud(np.squeeze(img_3d[:, img_3d.shape[1]//2, :]).T))
        fig.canvas.draw_idle()

    slider_z.on_changed(update)
    paired_view()


def paired_axial_and_coronal_view_with_mask(img_3d: np.ndarray, mask_3d: np.ndarray, z: Union[int, None] = None,
                                            cls_map_selected: Union[dict, None] = None, aspect_ratios=(1, 1, 1),
                                            cmap='turbo', alpha=0.8, title_str: Union[str, None] = None,
                                            draw_z_as_line: bool = False, save_name: Union[str, None] = None):
    assert img_3d.shape == mask_3d.shape
    img_shape = img_3d.shape
    aspect_ratio_z = round(aspect_ratios[2] * img_shape[2] / img_shape[0], 2)
    if z is None:
        z = simpleStats.get_index_of_plane_with_largest_mask_area(mask_3d, 'z', cls_map_selected, aspect_ratios)
    if mask_3d.dtype != np.bool_:
        labels = np.unique(mask_3d[mask_3d != 0])
        if isinstance(cls_map_selected, dict):
            label_selected = list(cls_map_selected.keys())
            assert all([x in labels for x in label_selected])
            _mask_selected = functools.reduce(np.logical_or, [mask_3d == x for x in label_selected])
            # noinspection PyUnresolvedReferences
            mask_3d[~_mask_selected] = 0
            min_label_color_norm = min(label_selected)-1
        else:
            min_label_color_norm = 0.05
    else:
        mask_3d.astype(np.int8)
        min_label_color_norm = 0.05
    cmap = plt.get_cmap(cmap).copy()
    cmap.set_under(color='black')
    fig, axs = plt.subplots(1, 2, figsize=(960 * px, 768 * px),
                            gridspec_kw={'width_ratios': [aspect_ratio_z, 1], 'height_ratios': [1]})
    if title_str is not None:
        fig.suptitle(title_str, y=0.825, fontweight='bold')
    axs[0].imshow(np.fliplr(img_3d[:, :, z]), cmap='gray', aspect=aspect_ratios[0])
    axs[0].imshow(np.fliplr(mask_3d[:, :, z]), interpolation='none', cmap=cmap, alpha=alpha,
                              aspect=aspect_ratios[0], vmin=min_label_color_norm)
    axs[0].set_title(f'Axial plane')
    axs[0].set_axis_off()
    img_cor = np.flipud(np.max(img_3d, axis=0).T)
    axs[1].imshow(img_cor, cmap='gray', aspect=aspect_ratios[2], vmax=np.max(img_3d[:, :, z]))
    im_cor_seg = axs[1].imshow(np.flipud(np.max(mask_3d, axis=0).T), interpolation='none', cmap=cmap, alpha=alpha,
                               aspect=aspect_ratios[2], vmin=min_label_color_norm)
    if draw_z_as_line:
        axs[1].plot(range(img_cor.shape[1]), [img_cor.shape[0] - z] * img_cor.shape[1], 'g-')
    axs[1].set_title(f'Coronal plane')
    axs[1].set_axis_off()
    # if isinstance(cls_map_selected, dict):
    #     # noinspection PyUnboundLocalVariable
    #     colors = [im_cor_seg.cmap(im_cor_seg.norm(v)) for v in label_selected]
    #     from matplotlib import patches
    #     _patches = [patches.Patch(color=colors[i], label=cls_map_selected[label_selected[i]])
    #                 for i in range(len(label_selected))]
    #     plt.legend(handles=_patches, bbox_to_anchor=(1.05, 1.35))
    if save_name is None:
        plt.show(block=True)
    else:
        plt.savefig(save_name)


def view_all_window_images(img_2d: np.ndarray, dicom_attributes, require_recompute=False, import_window_ref=False):
    # The current implementation is done for already scaled DICOM image using slope and intercept attributes.
    # For future update: merge the method for 'get_and_view_window_fused_image' to 'view_window_image'.
    body_window_dict = bodyPartWindow.get_window_dict(dicom_attributes, import_window_ref)
    num_win = len(list(body_window_dict))
    if num_win <= 2:
        fig_win, axs_win = plt.subplots(1, num_win, figsize=(768 * px * num_win, 768 * px), layout='constrained')
    else:
        fig_win, axs_win = plt.subplots(2, int(np.ceil(num_win / 2)),
                                        figsize=(768 * px * np.ceil(num_win / 2) / 2, 768 * px),
                                        layout='constrained')
        if num_win % 2 != 0:
            fig_win.delaxes(axs_win[1, -1])

    for i in range(num_win):
        k = list(body_window_dict)[i]
        v_min, v_max = bodyPartWindow.get_window_min_max_for_display(body_window_dict[k])
        text = k + ' level: ' + str(body_window_dict[k][1]) + '; width: ' + str(body_window_dict[k][0])
        if num_win <= 2:
            if require_recompute is True:
                axs_win[i].imshow(bodyPartWindow.get_window_image(img_2d, body_window_dict[k]), cmap='gray')
            else:
                axs_win[i].imshow(img_2d, vmin=v_min, vmax=v_max, cmap='gray')
            axs_win[i].set_title(text, fontdict={'fontsize': 12})
            axs_win[i].set_axis_off()
        else:
            if require_recompute is True:
                axs_win[i // int(np.ceil(num_win / 2)), i % int(np.ceil(num_win / 2))].imshow(
                    bodyPartWindow.get_window_image(img_2d, body_window_dict[k]), cmap='gray')
            else:
                axs_win[i // int(np.ceil(num_win / 2)), i % int(np.ceil(num_win / 2))].imshow(
                    img_2d, vmin=v_min, vmax=v_max, cmap='gray')
            axs_win[i // int(np.ceil(num_win / 2)), i % int(np.ceil(num_win / 2))].set_title(text,
                                                                                             fontdict={'fontsize': 12})
            axs_win[i // int(np.ceil(num_win / 2)), i % int(np.ceil(num_win / 2))].set_axis_off()
    plt.show(block=True)


def get_sectional_view_image(img_3d: np.ndarray, dim: str, slice_idx: Union[int, None] = None,
                             slab_hw: Union[int, None] = None, is_mask: bool = False,
                             window_image: Union[str, None] = None, plot_figure: bool = True,
                             return_fig: bool = False, **kwargs):

    def _get_2d_image_with_slice_idx():
        if slab_hw is None:
            if idx_axis == 0:
                _img_2d = np.flipud(img_3d[slice_idx, :, :].T)
            elif idx_axis == 1:
                _img_2d = np.flipud(img_3d[:, slice_idx, :].T)
            else:
                _img_2d = img_3d[:, :, slice_idx]
        else:
            slice_idx_0 = slice_idx - slab_hw if slice_idx - slab_hw >= 0 else max([slice_idx - slab_hw, 0])
            slice_idx_1 = slice_idx + slab_hw if slice_idx + slab_hw < size_limit else size_limit - 1
            if idx_axis == 0:
                _img_2d = np.flipud(np.max(img_3d[slice_idx_0:slice_idx_1, :, :], axis=idx_axis).T)
            elif idx_axis == 1:
                _img_2d = np.flipud(np.max(img_3d[:, slice_idx_0:slice_idx_1, :], axis=idx_axis).T)
            else:
                _img_2d = np.max(img_3d[:, :, slice_idx_0:slice_idx_1], axis=idx_axis)
        return _img_2d

    def _get_modified_2d_image(_img_3d, mod: str = 'sum'):
        assert mod in ['sum', 'max']
        func = eval(f'np.{mod}')
        if idx_axis < 2:
            _img_2d = np.flipud(func(img_3d, axis=idx_axis).T)
        else:
            _img_2d = func(img_3d, axis=idx_axis)
        return _img_2d

    list_dim = ['x', 'y', 'z']
    assert dim in list_dim
    idx_axis = list_dim.index(dim)
    size_limit = img_3d.shape[idx_axis]
    if slice_idx is None:
        if is_mask:
            slice_idx = simpleStats.get_index_of_plane_with_largest_mask_area(img_3d, dim=dim)
            img_2d = _get_2d_image_with_slice_idx()
        else:
            if isinstance(window_image, str):
                dict_window = bodyPartWindow.triple_window_dict[window_image.split('_')[0]
                                                                if '_' in window_image else window_image]
                img_3d = bodyPartWindow.get_window_image(img_3d, dict_window)
            img_2d = _get_modified_2d_image('max' if '_' not in window_image else window_image.split('_')[1])
            # dict_window = bodyPartWindow.triple_window_dict['SoftTissue']
            # img_3d_masked_win = bodyPartWindow.get_window_image(img_3d, dict_window)
            # img_3d_masked_win[~mask] = 0
            # img_2d_sum = _get_modified_2d_image(img_3d_masked_win, 'sum')
            # img_3d_masked = img_3d.copy()
            # img_3d_masked[~mask] = 0
            # img_2d_max = _get_modified_2d_image(img_3d_masked, 'max')
            # img_2d = img_2d_sum / np.max(img_2d_sum) * 0.5 + img_2d_max / np.max(img_2d_max) * 0.5
    else:
        img_2d = _get_2d_image_with_slice_idx()

    if plot_figure:
        vmax = kwargs.get('vmax', np.percentile(img_2d, 99.5))
        aspect = kwargs.get('aspect', 1.0)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(img_2d, aspect=aspect, cmap='gray', vmax=vmax)
        ax.axis('off')
        if return_fig:
            return fig, ax
        else:
            plt.show()
    else:
        return img_2d


def create_masked_color_for_2d_display(mask_2d: np.ndarray, dict_colors: dict, alpha: float = 1.0):
    if mask_2d.dtype != np.uint8:
        mask_2d = mask_2d.astype(np.uint8)
    labels = set(np.unique(mask_2d[mask_2d > 0]))
    labels_color = set(dict_colors.keys())
    assert labels.issubset(labels_color)
    color_arrays = np.zeros(tuple(list(mask_2d.shape) + [4]))
    for _l in labels:
        mask = mask_2d == _l
        color = np.insert(dict_colors[_l], 3, alpha)
        for i in range(4):
            color_arrays[mask, i] = color[i]
    return color_arrays


# noinspection SpellCheckingInspection
def show_mask_superimposed_2d_image(img_2d: np.ndarray, mask_2d: np.ndarray, dict_colors: dict,
                                    alpha: float = 1.0, apply_window: int = 0,
                                    title: Union[str, None] = None, cls_map: Union[dict, None] = None,
                                    return_fig: bool = False, save_file_name: Union[str, None] = None,
                                    **kwargs):

    if mask_2d.dtype != np.uint8:
        mask_2d = mask_2d.astype(np.uint8)
    color_arrays = create_masked_color_for_2d_display(mask_2d, dict_colors, alpha)
    fig, ax = plt.subplots(figsize=(8, 6))
    if apply_window > 0:
        if apply_window == 1:
            dict_window = bodyPartWindow.triple_window_dict['SoftTissue']
            img_2d = bodyPartWindow.get_window_image(img_2d, dict_window)
        else:
            # 3 window fusion
            dict_window = copy.deepcopy(bodyPartWindow.triple_window_dict)
            if apply_window == 2:
                dict_window.pop('Lung')
            img_list = get_window_view_image_list(img_2d, dict_window)
            img_2d = get_window_fused_image(img_list, dict_window)
    vmax = kwargs.get('vmax', np.percentile(img_2d, 99.5))
    aspect = kwargs.get('aspect', 1.0)
    ax.imshow(img_2d, cmap='gray', vmax=vmax, aspect=aspect)
    ax.imshow(color_arrays, aspect=aspect)
    ax.set_axis_off()
    if title is not None:
        ax.set_title(title)
    if cls_map is not None:
        assert set(dict_colors.keys()) == set(cls_map.keys())
        legend_elements = list()
        labels_found = np.unique(mask_2d[mask_2d > 0])
        for _l in labels_found:
            legend_elements.append(Patch(color=dict_colors[_l], label=cls_map[_l], alpha=alpha))
        ax.legend(handles=legend_elements, loc='upper right')
    if return_fig:
        return fig, ax
    else:
        if save_file_name is None:
            plt.show(block=True)
        else:
            fig.savefig(save_file_name, dpi=300)


def get_window_view_image_list(img_2d: np.ndarray, dict_body_window: dict):
    num_win = len(list(dict_body_window))
    assert 2 <= num_win < 4
    img_list = []
    for i in range(num_win):
        k = list(dict_body_window)[i]
        img_list.append(bodyPartWindow.get_window_image(img_2d, dict_body_window[k]))
    return img_list


# noinspection SpellCheckingInspection
def get_window_fused_image(list_img_2d: list, dict_body_window: dict, fusion_method='Mertens',
                           dicom_rescale_attr=None):

    # The 'Debevec' method requires intercept stored in 'dicom_rescale_attr' to generate false exposure time.
    # slope = dicom_rescale_attr[0] and intercept = dicom_rescale_attr[1]
    assert fusion_method.lower() in ['mertens', 'debevec']
    num_win = len(dict_body_window)
    if fusion_method.lower() == 'mertens':
        # Perform window fusion using Mertens
        fuse_mertens = cv.createMergeMertens()
        img_fused = fuse_mertens.process(list_img_2d)
        img_fused = np.clip(img_fused * 255, 0, 255).astype('uint8')
    else:
        fuse_deb = cv.createMergeDebevec()
        cal_deb = cv.createCalibrateDebevec()
        exposure_times = []
        for i in range(num_win):
            k = list(dict_body_window)[i]
            ex_time = (-int(dicom_rescale_attr[1]) / (dict_body_window[k][1] - int(dicom_rescale_attr[1]))) ** 2
            exposure_times.append(ex_time)

        exposure_times = np.array(exposure_times, dtype=np.float32).reshape(1, -1)
        crf_deb = cal_deb.process(list_img_2d, exposure_times.copy())
        img_fused = fuse_deb.process(list_img_2d, times=exposure_times.copy(), response=crf_deb.copy())
    return img_fused


def get_window_view_images_for_plot(img_2d: np.ndarray, dict_body_window: dict, fusion_method='Mertens',
                                    dicom_rescale_attr=None):
    img_list = get_window_view_image_list(img_2d, dict_body_window)
    img_fused = get_window_fused_image(img_list, dict_body_window, fusion_method, dicom_rescale_attr)
    img_list.append(img_fused)
    plot_window_view_images(img_list, dict_body_window, fusion_method)


def get_window_view_images_v2(img_2d: np.ndarray, dict_body_window: dict):
    _window_dict = copy.deepcopy(dict_body_window)
    fuse_mertens = cv.createMergeMertens()
    window_names = list(_window_dict.keys())

    if any(['bone' in k.lower() for k in window_names]):
        img_list = []
        bone_window_index = np.where(np.array(['bone' in k.lower() for k in window_names]))[0][0]
        bone_window_name = window_names[bone_window_index]
        img_bone_window = bodyPartWindow.get_window_image(img_2d, _window_dict[bone_window_name])
        _window_dict.pop(bone_window_name)
        img_tmp_list = []
        for w in _window_dict.keys():
            img = bodyPartWindow.get_window_image(img_2d, _window_dict[w])
            img_list.append(img)
            img_tmp_list.append(np.clip(fuse_mertens.process([img_bone_window, img]) * 255, 0, 255).astype('uint8'))
        img_list.append(np.clip(fuse_mertens.process(img_tmp_list) * 255, 0, 255).astype('uint8'))
        img_list.insert(bone_window_index, img_bone_window)
    else:
        assert 2 <= len(dict_body_window) < 4
        img_list = []
        for k in dict_body_window.keys():
            img_list.append(bodyPartWindow.get_window_image(img_2d, dict_body_window[k]))
    plot_window_view_images(img_list, dict_body_window, 'Mertens')


def plot_window_view_images(list_img_2d: list, dict_body_window, fusion_method):
    fig_win, axs_win = plt.subplots(2, 2, figsize=(768 * px, 768 * px), layout='constrained')
    for i in range(len(list_img_2d)):
        if i < len(list_img_2d) - 1:
            k = list(dict_body_window)[i]
            text = str(k) + ' level: ' + str(dict_body_window[k][1]) + '; width: ' + str(dict_body_window[k][0])
        else:
            text = 'Image fusion using ' + fusion_method + ' method'
        axs_win[i // 2, i % 2].imshow(list_img_2d[i], cmap='gray')
        axs_win[i // 2, i % 2].set_title(text, fontdict={'fontsize': 12})
        axs_win[i // 2, i % 2].set_axis_off()
    if len(list_img_2d) == 3:
        fig_win.delaxes(axs_win[1, 1])
    plt.show(block=True)

