# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.parallel import is_module_wrapper
from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class UpdateDatasetHook(Hook):
    """Set runner's epoch information to the model."""

    def after_train_iter(self, runner):
        dataset = runner.data_loader.dataset.dataset
        img_loss = runner.model.module.roi_head.img_loss
        # print(len(img_loss))
        for item in img_loss:
            dataset.loss_dict.add_entry(item['loss_cls'], item['idxs'][0])
            dataset.loss_dict.add_entry(item['loss_cls'], item['idxs'][1])
        dataset.loss_dict.get_idx()