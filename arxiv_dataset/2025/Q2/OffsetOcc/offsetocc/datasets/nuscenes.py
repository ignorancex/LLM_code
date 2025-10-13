from os import path as osp
from typing import List, Union

from mmdet3d.datasets import NuScenesDataset as MMDet3DNuScenesDataset

from offsetocc.registry import DATASETS
import copy


@DATASETS.register_module()
class NuScenesDataset(MMDet3DNuScenesDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # def parse_ann_info(self, info: dict) -> dict:
    #     pass

    def parse_data_info(self, info: dict) -> Union[List[dict], dict]:
        data_info = super().parse_data_info(info)

        occ_prefix = self.data_prefix.get('occ', '')
        data_info['occupancy']['occupancy_path'] = osp.join(occ_prefix, data_info['occupancy']['occupancy_path'])
        data_info['occupancy']['aux_occupancy_path'] = osp.join(occ_prefix, data_info['occupancy']['aux_occupancy_path'])

        if 'pts_semantic_mask_path' in info:
            data_info['pts_semantic_mask_path'] = \
                osp.join(self.data_prefix.get('pts_semantic_mask', ''),
                         info['pts_semantic_mask_path'])

        if 'pts_panoptic_mask_path' in info:
            data_info['pts_panoptic_mask_path'] = \
                osp.join(self.data_prefix.get('pts_panoptic_mask', ''),
                         info['pts_panoptic_mask_path'])

        return data_info

    def prepare_data(self, index: int) -> Union[dict, None]:
        """Data preparation for both training and testing stage.

        Called by `__getitem__`  of dataset.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict or None: Data dict of the corresponding index.
        """
        ori_input_dict = self.get_data_info(index)

        # deepcopy here to avoid inplace modification in pipeline.
        input_dict = copy.deepcopy(ori_input_dict)

        # TODO: check if it is ever triggered like that
        # pre-pipline return None to random another in `__getitem__`
        if not self.test_mode and self.filter_empty_gt:
            if len(input_dict['occupancy']['occupancy_path']) == 0:
                return None

        example = self.pipeline(input_dict)

        # TODO: to be modified for occupancy case
        # if not self.test_mode and self.filter_empty_gt:
        #     # after pipeline drop the example with empty annotations
        #     # return None to random another in `__getitem__`
        #     if example is None or len(
        #             example['data_samples'].gt_instances_3d.labels_3d) == 0:
        #         return None

        # if self.show_ins_var:
        #     if 'ann_info' in ori_input_dict:
        #         self._show_ins_var(
        #             ori_input_dict['ann_info']['gt_labels_3d'],
        #             example['data_samples'].gt_instances_3d.labels_3d)
        #     else:
        #         print_log(
        #             "'ann_info' is not in the input dict. It's probably that "
        #             'the data is not in training mode',
        #             'current',
        #             level=30)


        return example

    def _filter_with_mask(self, ann_info: dict) -> dict:

        """
        By default, results['instances'] contains all bbox annotations, even not valid ones.
        While results['gt_bboxes_3d'] contains only, so called, valid annotations.

        Valid annotations are defined as those with num_lidar_pts + num_radar_pts > 0 (see nuscenes_converter.py)

        We want to override this method to maintain all bbox annotations, even if they are not valid.

        We still permit the use of flag use_valid_flag to filter out invalid annotations.

        """

        if not self.use_valid_flag:
            return ann_info
        else:
            filtered_annotations = {}
            filter_mask = ann_info['bbox_3d_isvalid']
        for key in ann_info.keys():
            if key != 'instances':
                filtered_annotations[key] = (ann_info[key][filter_mask])
            else:
                filtered_annotations[key] = ann_info[key]
        return filtered_annotations

