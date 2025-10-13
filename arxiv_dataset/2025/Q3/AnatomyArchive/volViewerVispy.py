import numpy as np
from vispy import scene, app
from vispy.color import Colormap
from vispy.visuals.transforms import STTransform  # , MatrixTransform
from vispy.scene.visuals import Text
from vispy.visuals.filters import Alpha  # , ShadingFilter, WireframeFilter
# from vispy.scene.visuals import Mesh


app.use_app('pyqt5')
cm = Colormap(['r', 'g', 'b'])


def get_32bit_data(vol_data):
    if vol_data.dtype == np.int64:
        vol_data = vol_data.astype(np.int32)
    else:
        vol_data = vol_data.astype(np.float32)
    return vol_data


def resize_3d_image_tensor(vol_data: np.ndarray, aspect_ratios=(1, 1, 1)):
    if aspect_ratios[0] > 1:
        aspect_ratios = np.array(aspect_ratios)/aspect_ratios[0]
    if not all([np.isclose(x, 1.0, atol=0.01) for x in aspect_ratios]):
        img_shape = np.array(vol_data.shape)
        target_shape = tuple([int(np.floor(x * y)) for x, y in zip(img_shape, aspect_ratios)])
        from torch.nn import Upsample
        from torch import from_numpy, squeeze
        transform = Upsample(size=target_shape, mode='trilinear')
        # pytorch needs extra dimensions of the numpy array number and channel number
        vol_tensor = from_numpy(get_32bit_data(vol_data)).view([1, 1] + list(img_shape))
        dtype = np.int16 if vol_data.dtype != np.uint8 else np.uint8
        return squeeze(transform(vol_tensor)).numpy().astype(dtype)
    else:
        return get_32bit_data(vol_data)


def volume_viewer(vol_data: np.ndarray, aspect_ratios=(1, 1, 1), label_coord=None, cut_planes=None, **kwargs):
    # aspect_ratios here concern only the voxel sizes not the actual size aspect ratios taking into account the number
    # of elements in each dimension
    if 'alpha' in kwargs.keys():
        alpha_vol = kwargs.get('alpha')
        kwargs.pop('alpha')
    else:
        alpha_vol = None
    vol_data = resize_3d_image_tensor(vol_data, aspect_ratios)
    # Prepare canvas
    canvas = scene.SceneCanvas(keys='interactive', size=(800, 600), show=True)

    # canvas.measure_fps()

    @canvas.events.mouse_move.connect
    def on_mouse_move(event):
        if event.button == 1 and event.is_dragging:
            output_txt.text = ''
            axis.transform.reset()
            axis.transform.rotate(cam.roll, (0, 0, 1))
            axis.transform.rotate(cam.elevation, (1, 0, 0))
            axis.transform.rotate(cam.azimuth, (0, 1, 0))
            axis.transform.scale((50, 50, 0.001))
            axis.transform.translate((50., 50.))
            axis.update()
            output_txt.text = 'View - ele: {}; azi: {}; roll: {}'.format(cam.elevation, cam.azimuth, cam.roll)
        else:
            output_txt.text = ''

    # Set up a view box to display the image with interactive pan/zoom
    view = canvas.central_widget.add_view()

    # Create the volume visuals, only one is visible
    volume_data = scene.visuals.Volume(vol_data, parent=view.scene, threshold=0.225, raycasting_mode='volume',
                                       **kwargs)
    if alpha_vol is not None:
        volume_data.attach(Alpha(alpha_vol))
    #
    # volume_data.transform = scene.STTransform(translate=(0, 0, 0), scale=(aspect_ratios[2], aspect_ratios[0],
    #                                                                       aspect_ratios[1]))

    # Create a turntable camera
    fov = 60.
    # Implement axis connection with cam
    cam = scene.cameras.TurntableCamera(parent=view.scene, fov=fov, elevation=0, azimuth=90, roll=90)

    if label_coord is not None:
        scatter = scene.visuals.Markers()
        scatter.scaling = True
        c = cm[np.linspace(0, 1, label_coord.shape[0])]
        # Rescale the label coordinates to match the volume which has been rescaled according to aspect_ratios.
        label_coord = np.array([label_coord[:, i] * aspect_ratios[i] for i in range(label_coord.shape[0])]).transpose()
        print('Label coordinates after aspect ratio adjustment are {}'.format(label_coord.tolist()))
        symbols = ['x', '*', 'o']
        # Reorder the coordinates to match the coordinate system in Vispy.
        scatter.set_data(label_coord[:, [2, 0, 1]], edge_width=0, face_color=c, size=10, symbol=symbols)
        view.add(scatter)

    if cut_planes is not None:
        plane_normal_arr = np.repeat(np.array([0, 0, 1]).reshape(1, -1), repeats=2, axis=0)
        plane_pos_arr = np.zeros((2, 3))
        plane_pos_arr[:, 2] = cut_planes
        for i in range(2):
            # Create the volume visual for plane rendering to visualize cut planes
            plane = scene.visuals.Volume(
                vol_data,
                parent=view.scene,
                raycasting_mode='plane',
                plane_thickness=3.0,
                plane_position=plane_pos_arr[i, :],
                plane_normal=plane_normal_arr[i, :],
                **kwargs
            )
            print('Cut plane position: {}'.format(plane_pos_arr[i, :]))
            plane.attach(Alpha(0.5))

    view.camera = cam  # Select turntable at first
    output_txt = Text('', parent=canvas.scene, color='white')
    output_txt.font_size = 12
    output_txt.pos = canvas.size[0] * 0.8, canvas.size[1] * 0.8
    # Create an XYZAxis visual
    axis = scene.visuals.XYZAxis(parent=view)
    s = STTransform(translate=(50, 50), scale=(50, 50, 50, 1))
    affine = s.as_matrix()
    axis.transform = affine
    app.run()


def view_multi_volumes(i1, i2, aspect_ratios=(1, 1, 1)):
    i1 = resize_3d_image_tensor(get_32bit_data(i1), aspect_ratios)
    i2 = resize_3d_image_tensor(get_32bit_data(i2), aspect_ratios)
    # noinspection PyTypeChecker
    canvas = scene.SceneCanvas(keys='interactive', bgcolor='black')
    # canvas.measure_fps()
    view = canvas.central_widget.add_view()
    vol1 = scene.visuals.Volume(i1, cmap='Greens', method='mip', raycasting_mode='volume', parent=view.scene)
    vol2 = scene.visuals.Volume(i2, cmap='Blues', method='mip', raycasting_mode='volume', parent=view.scene)

    vol1.set_gl_state(preset='additive')
    vol1.opacity = 0.5
    vol2.set_gl_state(preset='additive')
    vol2.opacity = 0.5

    fov = 60
    cam = scene.cameras.TurntableCamera(elevation=0, azimuth=90, roll=90, parent=view.scene, fov=fov)
    view.camera = cam
    canvas.show()
    canvas.app.run()
