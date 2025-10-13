from typing import Dict, List
from typing import Union

from mmengine.model import BaseModel
from torch import Tensor

from offsetocc.registry import MODELS
from offsetocc.structures import OccDataSample
from offsetocc.utils import ForwardResults, SampleList, OptSampleList, OptConfigType, ConfigDict


@MODELS.register_module()
class OffsetOcc(BaseModel):
    def __init__(self,
                 backbone: ConfigDict,
                 img_neck: ConfigDict,
                 transformer: ConfigDict,
                 head: ConfigDict,
                 data_preprocessor: OptConfigType = None,
                 frozen_layers: List[str] = None) -> None:

        super().__init__(data_preprocessor=data_preprocessor)

        self.backbone = MODELS.build(backbone)
        self.img_neck = MODELS.build(img_neck)
        self.transformer = MODELS.build(transformer)
        self.head = MODELS.build(head)

        self.frozen_layers = frozen_layers

        self._freeze_layers()

    def forward(self,
                inputs: Union[dict, List[dict]],
                data_samples: OptSampleList = None,
                mode: str = 'tensor') -> ForwardResults:
        """The unified entry for a forward process in both training and test.

        The method should accept three modes: "tensor", "predict" and "loss":

        - "tensor": Forward the whole network and return tensor or tuple of
        tensor without any post-processing, same as a common nn.Module.
        - "predict": Forward and return the predictions, which are fully
        processed to a list of :obj:`DetDataSample`.
        - "loss": Forward and return a dict of losses according to the given
        inputs and data samples.

        Note that this method doesn't handle either back propagation or
        parameter update, which are supposed to be done in :meth:`train_step`.

        Args:
            inputs  (dict | list[dict]): When it is a list[dict], the
                outer list indicate the test time augmentation. Each
                dict contains batch inputs
                which include 'imgs' keys.
            data_samples (list[:obj:`OccDataSample`],
                list[list[:obj:`OccDataSample`]], optional): The
                annotation data of every samples. When it is a list[list], the
                outer list indicate the test time augmentation, and the
                inter list indicate the batch. Otherwise, the list simply
                indicate the batch. Defaults to None.
            mode (str): Return what kind of value. Defaults to 'tensor'.

        Returns:
            The return type depends on ``mode``.

            - If ``mode="tensor"``, return a tensor or a tuple of tensor.
            - If ``mode="predict"``, return a list of :obj:`DetDataSample`.
            - If ``mode="loss"``, return a dict of tensor.
        """
        if mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        # elif mode == 'tensor':
        #     return self._forward(inputs, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')

    def loss(self, batch_inputs_dict: Dict[List, Tensor],
             batch_data_samples: List[OccDataSample]) -> Union[dict, tuple]:
        """Calculate losses from a batch of inputs and data samples."""

        batch_input_metas = [item.metainfo for item in batch_data_samples]
        feats_3d = self.extract_feat(batch_inputs_dict, batch_input_metas)
        return self.head.loss(feats_3d, batch_data_samples)

    def predict(self, batch_inputs_dict: Dict[List, Tensor],
             batch_data_samples: List[OccDataSample]) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing."""

        batch_input_metas = [item.metainfo for item in batch_data_samples]
        feats_3d = self.extract_feat(batch_inputs_dict, batch_input_metas)
        return self.head.predict(feats_3d, batch_data_samples)

    def extract_img_feat(self, img: Tensor,
                         batch_input_metas: List[dict]) -> List[Tensor]:
        """Extract features from images.

        Args:
            img (tensor): Batched multi-view image tensor with
                shape (B, N, C, H, W).
            batch_input_metas (list[dict]): Meta information of multiple inputs
                in a batch.

        Returns:
             list[tensor]: multi-level image features.
        """

        B = img.size(0)
        if img is not None:
            input_shape = img.shape[-2:]  # bs nchw
            # update real input shape of each single img
            for img_meta in batch_input_metas:
                img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_(0)
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.view(B * N, C, H, W)
            img_feats = self.backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None

        img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    def extract_feat(self, batch_inputs: Dict,
                     batch_input_metas: List[dict]) -> List[Tensor]:
        """Extract features from images."""

        imgs = batch_inputs.get('imgs', None)
        img_feats = self.extract_img_feat(imgs, batch_input_metas)
        feats_3d = self.transformer(img_feats, batch_input_metas=batch_input_metas)

        return feats_3d

    def _freeze_layers(self) -> None:
        """Freeze layers."""
        if self.frozen_layers:
            for name, module in self.named_modules():
                if name in self.frozen_layers:
                    for param in module.parameters():
                        param.requires_grad = False

    def train(self, mode: bool = True) -> None:
        """Convert the model into training mode while keep layers freezed."""
        super().train(mode)
        self._freeze_layers()