# Copyright (c) OpenMMLab. All rights reserved.
# Modified by Nicola Marinello, 2025
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from mmengine.dist import master_only
from mmengine.structures import PixelData
from mmengine.visualization import Visualizer

from offsetocc.registry import VISUALIZERS


@VISUALIZERS.register_module()
class OccLocalVisualizer(Visualizer):
    """Local Visualizer.

    Args:
        name (str): Name of the instance. Defaults to 'visualizer'.
        image (np.ndarray, optional): the origin image to draw. The format
            should be RGB. Defaults to None.
        vis_backends (list, optional): Visual backend config list.
            Defaults to None.
        save_dir (str, optional): Save file dir for all storage backends.
            If it is None, the backend storage will not save any data.
        classes (list, optional): Input classes for result rendering, as the
            prediction of segmentation model is a segment map with label
            indices, `classes` is a list which includes items responding to the
            label indices. If classes is not defined, visualizer will take
            `cityscapes` classes by default. Defaults to None.
        palette (list, optional): Input palette for result rendering, which is
            a list of color palette responding to the classes. Defaults to None.
        dataset_name (str, optional): `Dataset name or alias <https://github.com/open-mmlab/mmsegmentation/blob/main/mmseg/utils/class_names.py#L302-L317>`_
            visulizer will use the meta information of the dataset i.e. classes
            and palette, but the `classes` and `palette` have higher priority.
            Defaults to None.
        alpha (int, float): The transparency of segmentation mask.
                Defaults to 0.8.

    Examples:
        >>> import numpy as np
        >>> import torch
        >>> from mmengine.structures import PixelData
        >>> from mmseg.data import SegDataSample
        >>> from mmseg.engine.visualization import SegLocalVisualizer

        >>> seg_local_visualizer = SegLocalVisualizer()
        >>> image = np.random.randint(0, 256,
        ...                     size=(10, 12, 3)).astype('uint8')
        >>> gt_sem_seg_data = dict(data=torch.randint(0, 2, (1, 10, 12)))
        >>> gt_sem_seg = PixelData(**gt_sem_seg_data)
        >>> gt_seg_data_sample = SegDataSample()
        >>> gt_seg_data_sample.gt_sem_seg = gt_sem_seg
        >>> seg_local_visualizer.dataset_meta = dict(
        >>>     classes=('background', 'foreground'),
        >>>     palette=[[120, 120, 120], [6, 230, 230]])
        >>> seg_local_visualizer.add_datasample('visualizer_example',
        ...                         image, gt_seg_data_sample)
        >>> seg_local_visualizer.add_datasample(
        ...                        'visualizer_example', image,
        ...                         gt_seg_data_sample, show=True)
    """ # noqa

    def __init__(self,
                 name: str = 'visualizer',
                 image: Optional[np.ndarray] = None,
                 vis_backends: Optional[Dict] = None,
                 save_dir: Optional[str] = None,
                 **kwargs):
        super().__init__(name, image, vis_backends, save_dir, **kwargs)

    def _draw_occ_map(self,
                      occ_map: np.array,
                      palette: np.array,
                      empty_class: int = 0,
                      x_pixels=720,
                      y_pixels=360,
                      azimuth_deg=20,
                      elevation_deg=35,
                      cam_dist=75,
                      center_offset=[0.0, 0.0, 0.0],
                      fovy=60.0,
                      render_axis=False
                      ) -> np.ndarray:

        """Create a rendered image of a voxelgrid.
            By default the camera is looking at the center of the voxel space. Changing the azimuth and elevation of the camera will make the camera spin around the voxelgrid center.

            Args:
                voxel_grid (np.array): A 3D numpy array (X, Y, Z) containing for each voxel an integer as class. The background class should be 0.
                x_pixels (int): Width of rendered image. Default: '720'
                y_pixels (int): Height of rendered image. Default: '360'
                azimuth_deg (int): Azimuth of camera in degrees. 0 degrees makes the camera look in the forward direction of the car. Default: '20'
                elevation_deg (int): Elevation of camera in degrees. Default: '35'
                cam_dist (int): Camera distance from the center of the voxelspace, expressed in number of voxel cubes. Default: '75'
                center_offset (list<float>): Offset vector (x,y,z) from the voxelgrid center to which the camera is looking. Default: '[0, 0, 0]'
                fovy (float): Vertical field of view of the camera. Default: '60'
                color_map (np.array): Array (C x 3) where each class is represented by a RGB value.
                render_axis (bool): If 'True', colored arrows towards x, y, z are rendered in the center of the pointcloud. Default: 'False'

            Returns:
                rendered_image (np.array): Image array [H, W, C] with values between 0 and 1.
            """
        # Calculate viewing center
        voxel_shape = occ_map.shape
        lookat = [center_offset[0] + voxel_shape[0] / 2, center_offset[1] + voxel_shape[1] / 2,
                  center_offset[2] + voxel_shape[2] / 2]


        # Calculate camera position based on azimuth, elevation, distance and center_offset
        azimuth = np.deg2rad(azimuth_deg + 180)
        elevation = np.deg2rad(elevation_deg)
        cam_pos = [
            np.cos(azimuth) * np.cos(elevation) * cam_dist,
            np.sin(azimuth) * np.cos(elevation) * cam_dist,
            np.sin(elevation) * cam_dist
        ]
        cam_pos = [cam_pos[0] + lookat[0], cam_pos[1] + lookat[1], cam_pos[2] + lookat[2]]


        # Create sparse pointcloud from voxelgrid by removing background
        mask = (occ_map != empty_class)
        indices = mask.nonzero()
        xx = indices[0]
        yy = indices[1]
        zz = indices[2]
        points = np.stack([xx, yy, zz]).T

        # Map colors for each point of pointcloud
        colors = palette[occ_map[mask].astype(int)]
        # Make odd voxels a bit darker to better visualise voxel size
        darker_mask = np.zeros(voxel_shape, dtype=bool)
        darker_mask[::2, ::2, ::2] = 1
        darker_mask[1::2, 1::2, ::2] = 1
        darker_mask[1::2, ::2, 1::2] = 1
        darker_mask[::2, 1::2, 1::2] = 1
        darker_mask = darker_mask[mask]
        colors[darker_mask] *= 0.9

        # Initialize a point cloud object
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # Create a voxel grid from the point cloud with a voxel_size of 1
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=1)

        # Create Renderer
        up = [0.0, 0.0, 1.0]
        aspect_ratio = x_pixels / y_pixels
        near_plane = 0.1
        far_plane = 500.0
        fov_type = o3d.visualization.rendering.Camera.FovType.Vertical
        renderer = o3d.visualization.rendering.OffscreenRenderer(x_pixels, y_pixels)
        renderer.scene.set_background(np.array([0.2, 0.2, 0.2, 1]))
        renderer.scene.set_lighting(renderer.scene.SOFT_SHADOWS, np.array([[-50], [0], [0.0]]))
        renderer.scene.camera.set_projection(fovy, aspect_ratio, near_plane, far_plane, fov_type)
        renderer.scene.camera.look_at(lookat, cam_pos, up)

        # Create Maerial
        white = o3d.visualization.rendering.MaterialRecord()
        white.base_color = [0.9, 0.9, 0.9, 0.5]
        white.base_reflectance = 0.9
        white.transmission = 0.7
        white.base_roughness = 0.6
        white.base_metallic = 1.0
        white.shader = "defaultLit"

        # Add axis to render
        if render_axis:
            origin_arrows = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=15.0,
                origin=np.array([voxel_shape[0] / 2, voxel_shape[1] / 2, voxel_shape[2] / 2])
            )
            renderer.scene.add_geometry("mesh", origin_arrows, white)

        # Add voxel_grid to render
        renderer.scene.add_geometry("voxel_grid", voxel_grid, white)

        # Render the voxel grid
        image = renderer.render_to_image()
        return np.asarray(image) / 255


    def _draw_example(self,
                      camera_images: List[np.ndarray],
                      gt_occ_map: np.array,
                      pred_occ_map: np.array,
                      mask_camera: np.array,
                      classes: Optional[List],
                      palette: List,
                      empty_class: int,
                      camera_grid: Optional[List],
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Draw semantic seg of GT or prediction.

        Args:
            image (np.ndarray): The image to draw.
            sem_seg (:obj:`PixelData`): Data structure for pixel-level
                annotations or predictions.
            classes (list, optional): Input classes for result rendering, as
                the prediction of segmentation model is a segment map with
                label indices, `classes` is a list which includes items
                responding to the label indices. If classes is not defined,
                visualizer will take `cityscapes` classes by default.
                Defaults to None.
            palette (list, optional): Input palette for result rendering, which
                is a list of color palette responding to the classes.
                Defaults to None.

        Returns:
            np.ndarray: the drawn image which channel is RGB.
        """

        # concatenate camera images
        rows = []
        for i in range(len(camera_grid)):
            row = []
            for j in range(len(camera_grid[i])):
                row.append(camera_images[camera_grid[i][j]])
            rows.append(np.concatenate(row, axis=1))
        cameras_image = np.concatenate(rows, axis=0)

        palette = np.array(palette) / 255

        # gt_occ_map[~mask_camera] = empty_class
        # pred_occ_map[~mask_camera] = empty_class

        gt_occ_map_rendered = self._draw_occ_map(gt_occ_map, palette, empty_class, render_axis=True)
        pred_occ_map_rendered = self._draw_occ_map(pred_occ_map, palette, empty_class, render_axis=True)

        # create matplotlib figure with 3 subplots: camera images, gt occ map, pred occ map
        fig = plt.figure(figsize=(20, 10))
        gs = fig.add_gridspec(3, 1)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(cameras_image)
        ax1.axis('off')
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.imshow(gt_occ_map_rendered)
        ax2.axis('off')
        ax3 = fig.add_subplot(gs[2, 0])
        ax3.imshow(pred_occ_map_rendered)
        ax3.axis('off')
        fig.tight_layout(pad=0.0)
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        return image, cameras_image, gt_occ_map_rendered, pred_occ_map_rendered

    @master_only
    def add_datasample(
            self,
            name: str,
            camera_images: List[np.ndarray],
            gt_occ_map: np.ndarray = None,
            pred_occ_map: np.ndarray = None,
            mask_camera: np.ndarray = None,
            step: int = 0) -> None:
        """Draw datasample and save to all backends.

        Args:
            name (str): The image identifier.
            image (np.ndarray): The image to draw.
            gt_sample (:obj:`SegDataSample`, optional): GT SegDataSample.
                Defaults to None.
            pred_sample (:obj:`SegDataSample`, optional): Prediction
                SegDataSample. Defaults to None.
            step (int): Global step value to record. Defaults to 0.
        """
        classes = self.dataset_meta.get('occ_class_names', None)
        palette = self.dataset_meta.get('occ_palette', None)
        empty_class = self.dataset_meta.get('empty_class', None)
        camera_grid = self.dataset_meta.get('camera_display_grid', None)

        img_to_display, cameras_image, gt_occ_map_rendered, pred_occ_map_rendered  = self._draw_example(camera_images,
                                                                                                        gt_occ_map,
                                                                                                        pred_occ_map,
                                                                                                        mask_camera,
                                                                                                        classes,
                                                                                                        palette,
                                                                                                        empty_class,
                                                                                                        camera_grid
                                                                                                        )

        # paper images
        # path to save high resolution images (do not log them to wandb)
        # path = self._vis_backends['WandbVisBackend']._save_dir
        # create folder inside path called high_res_images
        # path = path + '/high_res_images'
        # os.makedirs(path, exist_ok=True)
        # save camera_images image with cv2
        # cv2.imwrite(f'{path}/{name}_camera_images.png', cv2.cvtColor(cameras_image, cv2.COLOR_RGB2BGR))
        # save gt_occ_map_rendered image with cv2
        # cv2.imwrite(f'{path}/{name}_gt_occ_map.png', cv2.cvtColor((gt_occ_map_rendered * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        # save pred_occ_map_rendered image with cv2
        # cv2.imwrite(f'{path}/{name}_pred_occ_map.png', cv2.cvtColor((pred_occ_map_rendered * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

        # draw gt and pred occupancy maps
        self.add_image(name, img_to_display, step=step)
