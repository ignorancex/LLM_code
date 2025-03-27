from ..builder import DETECTORS
from .two_stage import TwoStageDetector


@DETECTORS.register_module()
class FasterRCNN(TwoStageDetector):

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 condense_roi_head=None,
                 neck=None,
                 pretrained=None,
                 reload_pretrained=False):
        super(FasterRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            condense_roi_head=condense_roi_head,
            pretrained=pretrained, 
            reload_pretrained=reload_pretrained)
