from typing import Dict

import numpy as np
import torch
from mmcv.transforms.base import BaseTransform
from mmdet3d.datasets.transforms import LoadAnnotations3D
from mmdet3d.structures.bbox_3d.utils import get_lidar2img

from offsetocc.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadOccGTFromFile(BaseTransform):
    """Load occ gt from file.
    """

    def transform(self, results: dict) -> dict:
        occ_gt_path = results['occupancy']['occupancy_path']

        occ_labels = np.load(occ_gt_path)
        semantics = occ_labels['semantics']
        mask_lidar = occ_labels['mask_lidar']
        mask_camera = occ_labels['mask_camera']

        results['gt_occ_map'] = semantics
        results['gt_occ_map_mask_lidar'] = mask_lidar
        results['gt_occ_map_mask_camera'] = mask_camera

        return results

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        return "{} (data_root={}')".format(
            self.__class__.__name__, self.data_root)


@TRANSFORMS.register_module()
class LoadOccGTFromFileAux(BaseTransform):
    """Load aux occ gt from file.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def transform(self, results: dict) -> dict:

        aux_occ_gt_path = results['occupancy']['aux_occupancy_path']
        aux_occ_labels = np.load(aux_occ_gt_path)

        results['gt_occ_map_visible_obj_occ_grid'] = aux_occ_labels['gt_occ_map_visible_obj_occ_grid']
        results['gt_occ_map_visible_obj_occ_grid_start_idx'] = aux_occ_labels['gt_occ_map_visible_obj_occ_grid_start_idx']
        results['gt_occ_map_visible_obj_idx'] = aux_occ_labels['gt_occ_map_visible_obj_idx']

        return results

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        return "{} (data_root={}')".format(
            self.__class__.__name__, self.data_root)


@TRANSFORMS.register_module()
class LoadLidar2Img(BaseTransform):
    """Load occ gt from file.
    """

    def transform(self, results: dict) -> dict:

        l2i = list()
        for i in range(len(results['cam2img'])):
            c2i = torch.tensor(results['cam2img'][i]).double()
            l2c = torch.tensor(results['lidar2cam'][i]).double()
            l2i.append(get_lidar2img(c2i, l2c).float().tolist())
        results['lidar2img'] = l2i

        return results

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        return "{} (data_root={}')".format(
            self.__class__.__name__, self.data_root)


@TRANSFORMS.register_module()
class Lidar2EgoBbox(BaseTransform):
    """Move bboxes from lidar to ego coordinates.
    """

    def transform(self, results: Dict) -> dict:
        lidar2ego = np.array(results['lidar_points']['lidar2ego'])
        rot, tr = lidar2ego[:3, :3], lidar2ego[:3, 3]
        # gt_bboxes_3d will remain of type LiDARInstance3DBoxes, but they will be in ego coordinates
        results['gt_bboxes_3d'].rotate(np.linalg.inv(rot))
        results['gt_bboxes_3d'].translate(tr)
        return results

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        return "{} (data_root={}')".format(
            self.__class__.__name__, self.data_root)


@TRANSFORMS.register_module()
class PointCloudLidar2Ego(BaseTransform):
    """Move point cloud from lidar to ego coordinates.
    """

    def transform(self, results: Dict) -> dict:
        lidar2ego = np.array(results['lidar_points']['lidar2ego'])
        rot, tr = lidar2ego[:3, :3], lidar2ego[:3, 3]
        results['points'].rotate(np.linalg.inv(rot))
        results['points'].translate(tr)
        return results

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        return "{} (data_root={}')".format(
            self.__class__.__name__, self.data_root)


@TRANSFORMS.register_module()
class MapBboxLabelsToOcc(BaseTransform):
    """
        Remove bboxes labeled as -1.
        Map bbox labels to occ labels.
    """

    def __init__(self, map_to_occ_label: Dict[int, int]):
        self.map_to_occ_label = map_to_occ_label

    def transform(self, results: Dict) -> dict:

        bboxes_nonignore = results['gt_labels_3d'] != -1
        results['gt_bboxes_3d'].tensor = results['gt_bboxes_3d'].tensor[bboxes_nonignore]

        # map bbox labels to occ labels
        results['gt_labels_3d'] = np.array([self.map_to_occ_label[label] for label in results['gt_labels_3d'] if label != -1])
        results_instances = []
        for instance in results['instances']:
            if instance['bbox_label'] != -1:
                instance['bbox_label'] = self.map_to_occ_label[instance['bbox_label']]
                results_instances.append(instance)
        results['instances'] = results_instances
        return results

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        return "{} (data_root={}')".format(
            self.__class__.__name__, self.data_root)


# override MMDET3D LoadAnnotations3D class to solve the following issue
# https://github.com/open-mmlab/mmdetection3d/issues/2874
@TRANSFORMS.register_module()
class MyLoadAnnotations3D(LoadAnnotations3D):

    def _load_panoptic_3d(self, results: dict) -> dict:
        # """Private function to load 3D panoptic segmentation annotations.
        #
        # Args:
        #     results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.
        #
        # Returns:
        #     dict: The dict containing the panoptic segmentation annotations.
        # """
        # pts_panoptic_mask_path = results['pts_panoptic_mask_path']
        #
        # try:
        #     mask_bytes = get(
        #         pts_panoptic_mask_path, backend_args=self.backend_args)
        #     # add .copy() to fix read-only bug
        #     pts_panoptic_mask = np.frombuffer(
        #         mask_bytes, dtype=self.seg_3d_dtype).copy()
        # except ConnectionError:
        #     mmengine.check_file_exist(pts_panoptic_mask_path)
        #     pts_panoptic_mask = np.fromfile(
        #         pts_panoptic_mask_path, dtype=np.int64)
        #
        # if self.dataset_type == 'semantickitti':
        #     pts_semantic_mask = pts_panoptic_mask.astype(np.int64)
        #     pts_semantic_mask = pts_semantic_mask % self.seg_offset
        # elif self.dataset_type == 'nuscenes':
        #     # not sure this is how it should be done (comment on github issue does not explain it)
        #     pts_semantic_mask = pts_panoptic_mask // self.seg_offset
        #
        # results['pts_semantic_mask'] = pts_semantic_mask
        #
        # # We can directly take panoptic labels as instance ids.
        # pts_instance_mask = pts_panoptic_mask.astype(np.int64)
        # results['pts_instance_mask'] = pts_instance_mask
        #
        # # 'eval_ann_info' will be passed to evaluator
        # if 'eval_ann_info' in results:
        #     results['eval_ann_info']['pts_semantic_mask'] = pts_semantic_mask
        #     results['eval_ann_info']['pts_instance_mask'] = pts_instance_mask
        # return results

        if self.dataset_type != 'nuscenes':
            raise NotImplementedError('Only support nuscenes dataset for now.')

        pts_panoptic_mask_path = results['pts_panoptic_mask_path']
        pts_instance_mask = np.load(pts_panoptic_mask_path)['data'].astype(np.int32)
        results['pts_instance_mask'] = pts_instance_mask

        if 'eval_ann_info' in results:
            results['eval_ann_info']['pts_instance_mask'] = pts_instance_mask


        return results

@TRANSFORMS.register_module()
class SegFine2CoarseNuscMapping(BaseTransform):
    """Map fine segmentation to coarse segmentation in nuscenes dataset.
    """

    def __init__(self, mapping: Dict[int, int]):
        self.mapping = mapping

    def transform(self, results: Dict) -> dict:
        results['pts_semantic_mask'] = np.vectorize(self.mapping.__getitem__)(results['pts_semantic_mask'])
        return results

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        return "{} (data_root={}')".format(
            self.__class__.__name__, self.data_root)

@TRANSFORMS.register_module()
class PanoSegFine2CoarseNuscMapping(BaseTransform):
    """Map fine panoptic segmentation annotations to coarse panoptic segmentation annotations in nuscenes dataset.
    """

    def __init__(self, mapping: Dict[int, int]):
        self.mapping = mapping

    def transform(self, results: Dict) -> dict:
        pts_instance_mask = results['pts_instance_mask']
        pts_instance_mask_finecat = pts_instance_mask // 1000
        pts_instance_mask_coarsecat = np.vectorize(self.mapping.__getitem__)(pts_instance_mask_finecat)
        pts_instance_mask_ids = pts_instance_mask % 1000
        pts_instance_mask = pts_instance_mask_coarsecat * 1000 + pts_instance_mask_ids
        results['pts_instance_mask'] = pts_instance_mask
        return results

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        return "{} (data_root={}')".format(
            self.__class__.__name__, self.data_root)