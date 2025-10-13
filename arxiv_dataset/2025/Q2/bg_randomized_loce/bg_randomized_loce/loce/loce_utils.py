import logging
from typing import TYPE_CHECKING, Tuple, Optional, Union, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from ..hooks import Propagator, PropagatorTorchClassifier

if TYPE_CHECKING:
    from transformers.image_processing_utils import BaseImageProcessor




YOLO5_LAYERS = ('4.cv3.conv',
                '5.conv',
                '6.cv3.conv',
                '7.conv',
                '8.cv3.conv',
                '9.cv2.conv',
                '10.conv',
                '12',
                '14.conv',
                '16',
                '17.cv3.conv',
                '18.conv',
                '19',
                '20.cv3.conv',
                '21.conv',
                '22',
                '23.cv3.conv')

SSD_LAYERS = ('backbone.features.19',
              'backbone.features.21',
              'backbone.extra.0.1',
              'backbone.extra.0.3',
              'backbone.extra.0.5',
              'backbone.extra.1.0',
              'backbone.features',
              'backbone.extra.0',
              'backbone.extra.1',
              'backbone.extra.2',
              'backbone.extra.3',
              'backbone.extra.4')


MOBILENET_LAYERS = ('features.9',
                    'features.10',
                    'features.11',
                    'features.12',
                    'features.13',
                    'features.14',
                    'features.15')


EFFICIENTNET_LAYERS = ('features.4.2',
                       'features.5.0',
                       'features.5.1',
                       'features.5.2',
                       'features.6.0',
                       'features.6.1',
                       'features.6.2',
                       'features.7.0')

EFFICIENTNETV2_LAYERS = ('features.3',
                         'features.4.1',
                         'features.4.5',
                         'features.5.3',
                         'features.6.0',
                         'features.6.4',
                         'features.6.9',
                         'features.6.14')

SQUEEZENET_LAYERS = ('features.6.expand3x3',
                     'features.7.expand3x3',
                     'features.9.expand3x3',
                     'features.10.expand3x3',
                     'features.11.expand3x3',
                     'features.12.expand3x3')

VGG16_LAYERS = (
    'features.7',  # before the second MaxPool
    'features.21', # before the 4th MaxPool
    'features.28', # last conv layer (and before last MaxPool)
)


# B: batch
# H*W - 1d feature dimension size obtained from 2d feature maps
# C / d: channels / reduced channels after feature projection
# N - decoder queries dimension

DETR_LAYERS = (
    # output layer dimensions: [B, C, H, W] - regular CNN backbone layers
    # "model.backbone.conv_encoder.model.layer1",
    # "model.backbone.conv_encoder.model.layer2",
    "model.backbone.conv_encoder.model.layer3",
    "model.backbone.conv_encoder.model.layer4",

    # output layer dimensions: [B, d, H, W] - reduced dimensions / channels
    "model.input_projection",

    # output layer dimensions: [B, H*W, d] - encoder dimensionality, mind the axis swap
    "model.encoder.layers.0",
    "model.encoder.layers.1",
    "model.encoder.layers.2",
    "model.encoder.layers.3",
    "model.encoder.layers.4",
    "model.encoder.layers.5",

    # output layer dimensions: [B, N, d] - decoder is unexplainable with LoCEs because there is no reference for segmentation approximation
    # in forward pass use output_attentions=True and visualize decoder cross-attention instead
    # tutorial here: https://github.com/NielsRogge/Transformers-Tutorials/blob/master/DETR/DETR_minimal_example.ipynb
    # "model.decoder.layers.0",
    # "model.decoder.layers.1",
    # "model.decoder.layers.2",
    # "model.decoder.layers.3",
    # "model.decoder.layers.4",
    # "model.decoder.layers.5",
)

SWIN_LAYERS = (
    # conv patches
    "features.0",

    # transformer blocks (odd) with PatchMerging (even) inbetween
    "features.1",
    "features.3",
    "features.5",
    "features.7"
)

VIT_LAYERS = (
    # conv patches
    "conv_proj",

    # layers
    "encoder.layers.encoder_layer_1",
    "encoder.layers.encoder_layer_2",
    "encoder.layers.encoder_layer_3",
    "encoder.layers.encoder_layer_4",
    "encoder.layers.encoder_layer_5",
    "encoder.layers.encoder_layer_6",
    "encoder.layers.encoder_layer_7",
    "encoder.layers.encoder_layer_8",
    "encoder.layers.encoder_layer_9",
    "encoder.layers.encoder_layer_10",
    "encoder.layers.encoder_layer_11",
)

EPSILON = 0.000001

def blend_imgs(img1: Image, img2: Image, alpha: float = 0.5) -> Image:
    """
    Blend two Image instances: aplha * img1 + (1 - alpha) * img2

    Args:
        img1 (Image): image 1
        img2 (Image): image 2

    Kwargs:
        alpha (np.ndarray = 0.5): alpha for blending

    Returns:
        (Image) blended image
    """
    img1 = img1.convert("RGB")
    img2 = img2.convert("RGB")
    img2 = img2.resize(img1.size)
    return Image.blend(img1, img2, alpha)


def get_colored_mask(mask: np.ndarray, color_channels: list[int] = (1,), mask_value_multiplier: int = 1) -> Image:
    """
    Expand greyscale mask to RGB dimensions.

    Args:
        mask (np.ndarray): greyscale mask

    Kwargs:
        color_channels (list[int] = [1]): channels to fill with mask values, values of other other channels stay equal to 0. default - green mask
        mask_value_multiplier (int = 1): final value multiplier, use 255 if original mask was boolean, otherwise - 1

    Returns:
        (Image) RGB mask
    """
    rgb_img = np.zeros((*mask.shape, 3), dtype=mask.dtype)
    for c in color_channels:
        rgb_img[:, :, c] = mask
    return Image.fromarray(rgb_img.astype(np.uint8) * mask_value_multiplier)


def get_colored_mask_alt(mask: np.ndarray, color_channels: list[int] = (1,), mask_value_multipliers: Sequence[int] = (1, 1, 1)) -> Image:
    """
    Expand greyscale mask to RGB dimensions.

    Args:
        mask (np.ndarray): greyscale mask

    Kwargs:
        color_channels (list[int] = [1]): channels to fill with mask values, values of other other channels stay equal to 0. default - green mask
        mask_value_multiplier (int = 1): final value multiplier, use 255 if original mask was boolean, otherwise - 1

    Returns:
        (Image) RGB mask
    """
    rgb_img = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for c in color_channels:
        rgb_img[:, :, c] = mask.astype(np.uint8) * mask_value_multipliers[c]
    return Image.fromarray(rgb_img)


def combine_masks(masks: list[np.ndarray]) -> np.ndarray:
    """
    Combine (and rescale) masks to get an averaged mask.

    Args:
        masks (list[np.ndarray]): list of masks (may have different sizes) to combine

    Returns:
        combined_mask (np.ndarray) rescaled and combined mask
    """
    largest_mask_id = np.argmax([m.size for m in masks])

    img_masks = [Image.fromarray(m) for m in masks]
    new_size = img_masks[largest_mask_id].size
    resized_masks = [np.array(i.resize(new_size)) for i in img_masks]

    combined_mask = np.array(resized_masks).mean(axis=0).astype(np.uint8)

    return combined_mask


def binary_to_uint8_image(arr: np.ndarray) -> np.ndarray:
    """
    Convert binary np.ndarray to np.uint8 image array

    Args:
        arr (np.ndarray): array to convert

    Returns:
        (np.ndarray) np.uint8 image array
    """
    return (arr * 255).astype(np.uint8)


def downscale_numpy_img(img: np.ndarray,
                        downscale_factor: float = 5.0
                        ) -> np.ndarray:
    """
    Downscale numpy image array

    Args:
        img (np.ndarray): image

    Kwargs:
        downscale_factor (float = 5.0): downscale factor

    Returns:
        (np.ndarray) downscaled image
    """
    img = Image.fromarray(img)
    return np.array(img.resize(tuple(int(c / downscale_factor) for c in img.size)))


def plot_binary_mask(mask: np.ndarray,
                     downscale_factor: float = 5.0
                     ) -> None:
    """
    Plot (downscaled) binary mask

    Args:
        mask (np.ndarray): mask

    Kwargs:
        downscale_factor (float = 5.0): downscale factor
    """
    img_arr = downscale_numpy_img(mask, downscale_factor)
    img = Image.fromarray(img_arr)
    img.show()


def loce_stats(loce: np.ndarray) -> None:
    """
    Print LoCE stats (mean, var, sparsity)

    Args:
        loce (np.ndarray): LoCE
    """
    print('\tmean:', loce.mean())
    print('\tvar:', loce.var())
    print(f'\tsparsity: {(loce == 0).sum()}/{len(loce)}', )


def plot_projection(loce: np.ndarray,
                    acts: np.ndarray,
                    proj_name: str = None
                    ) -> None:
    """
    Plot projection of LoCE and activations

    Args:
        loce (np.ndarray): LoCE
        acts (np.ndarray): activations

    Kwargs:
        proj_name: projection name to print
    """
    if proj_name is not None:
        print(proj_name)
    loce_stats(loce)

    projecion_uint8 = get_projection(loce, acts)
    plot_binary_mask(projecion_uint8, 0.1)


def get_projection(loce: np.ndarray,
                   acts: np.ndarray,
                   downscale_factor: float = None
                   ) -> np.ndarray:
    """
    Get projection of LoCE and activations

    Args:
        loce (np.ndarray): LoCE
        acts (np.ndarray): activations

    Kwargs:
        downscale_factor (float = None): downscale factor

    Returns:
        (np.ndarray) np.uint8 image array
    """
    def sigmoid(z):
        z = np.clip(z, -10., 10.) # to avoid overflow in np.exp()
        return 1/(1 + np.exp(-z))

    loce3d = np.expand_dims(loce, axis=[1, 2])
    projecion = (acts * loce3d).sum(axis=0)
    projecion = sigmoid(projecion)  # normalize_0_to_1(projecion)

    if downscale_factor is not None:
        projecion = downscale_numpy_img(projecion, downscale_factor)

    projecion_uint8 = (projecion * 255).astype(np.uint8)
    return projecion_uint8


def get_rgb_binary_mask(mask: np.ndarray, target_size: Tuple[int, int] = None) -> np.ndarray:

    def rescale_to_range(data, new_min, new_max):
        old_min = np.min(data)
        old_max = np.max(data)

        rescaled_data = ((data - old_min) / (old_max - old_min +
                         EPSILON)) * (new_max - new_min) + new_min
        return rescaled_data

    img = Image.fromarray(mask.astype(np.float32))
    if target_size:
        img = img.resize(target_size)

    img_np = np.array(img)

    img_np = rescale_to_range(img_np, 0, 1)

    # apply colormap
    cmap = plt.get_cmap('bwr')
    img_rgba = cmap(img_np)
    # rgba to rgb
    img_rgb = (img_rgba[:, :, :3] * 255).astype(np.uint8)

    return img_rgb


def standardize_propagator(builder):
    """Decorator for propagator builders.
    The propagator builders sometimes return a tuple of (propagator, processor), but sometimes only the propagator,
    if the processor is None. Make sure they always produce the tuple format."""

    def standardized_builder(prop_layers: list[str], device: Union[str, torch.device]) -> tuple[Propagator, 'BaseImageProcessor']:
        prop = builder(prop_layers, device=device)  # could be tuple of (propagator, processor)
        proc = None
        if isinstance(prop, tuple):
            prop, proc = prop
        return prop, proc

    return standardized_builder


def yolo5_propagator_builder(layers: list[str] = YOLO5_LAYERS, device=None):
    from ..hooks import PropagatorUltralyticsYOLOv5Old
    device = device or torch.get_default_device()
    yolo5 = torch.hub.load('ultralytics/yolov5',
                           'yolov5s', skip_validation=True, verbose=False).to(device)

    yolo5_prop = PropagatorUltralyticsYOLOv5Old(yolo5, layers, device=device)
    logger = logging.getLogger("yolov5")
    logger.setLevel(logging.ERROR)
    return yolo5_prop


def ssd_propagator_builder(layers: list[str] = SSD_LAYERS):
    from ..hooks import PropagatorTorchSSD
    from torchvision.models.detection.ssd import ssd300_vgg16, SSD300_VGG16_Weights
    ssd = ssd300_vgg16(weights=SSD300_VGG16_Weights)

    ssd_prop = PropagatorTorchSSD(ssd, layers)

    return ssd_prop


def mobilenet_propagator_builder(layers: list[str] = MOBILENET_LAYERS, device=None):
    from torchvision.models.mobilenet import mobilenet_v3_large, MobileNet_V3_Large_Weights
    device = device or torch.get_default_device()
    mobilenet = mobilenet_v3_large(
        weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1).to(device)

    mobilenet_prop = PropagatorTorchClassifier(mobilenet, layers, device=device)

    return mobilenet_prop


def efficientnet_propagator_builder(layers: list[str] = EFFICIENTNET_LAYERS, device=None):
    from torchvision.models.efficientnet import efficientnet_b0, EfficientNet_B0_Weights
    device = device or torch.get_default_device()
    efficientnet = efficientnet_b0(
        weights=EfficientNet_B0_Weights.IMAGENET1K_V1).to(device)

    efficientnet_prop = PropagatorTorchClassifier(efficientnet, layers, device=device)

    return efficientnet_prop


def efficientnetv2_propagator_builder(layers: list[str] = EFFICIENTNETV2_LAYERS):
    from torchvision.models.efficientnet import efficientnet_v2_s, EfficientNet_V2_S_Weights
    efficientnet = efficientnet_v2_s(
        weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)

    efficientnet_prop = PropagatorTorchClassifier(efficientnet, layers)

    return efficientnet_prop


def squeezenet_propagator_builder(layers: list[str] = SQUEEZENET_LAYERS):
    from torchvision.models.squeezenet import squeezenet1_1, SqueezeNet1_1_Weights
    squeezenet = squeezenet1_1(weights=SqueezeNet1_1_Weights.IMAGENET1K_V1)

    squeezenet_prop = PropagatorTorchClassifier(squeezenet, layers)

    return squeezenet_prop


def detr_propagator_builder(layers: list[str] = DETR_LAYERS, device = None):
    from ..hooks import PropagatorHuggingFaceDETR
    from transformers import DetrImageProcessor, DetrForObjectDetection
    device = device or torch.get_default_device()

    # disable annoying logging messages
    import logging
    logging.getLogger('transformers.modeling_utils').setLevel(logging.ERROR)
    logging.getLogger('timm.models._builder').setLevel(logging.ERROR)
    logging.getLogger('timm.models._hub').setLevel(logging.ERROR)

    detr_processor = DetrImageProcessor.from_pretrained(
        "facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").to(device)

    detr_prop = PropagatorHuggingFaceDETR(model, layers, device=device)

    return detr_prop, detr_processor


def swin_propagator_builder(layers: list[str] = SWIN_LAYERS, device = None):
    from torchvision.models.swin_transformer import swin_t, Swin_T_Weights
    device = device or torch.get_default_device()
    swin = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1).to(device)

    swin_prop = PropagatorTorchClassifier(swin, layers, device=device)

    return swin_prop


def vit_propagator_builder(layers: list[str] = VIT_LAYERS, device=None):
    from torchvision.models.vision_transformer import vit_b_16, ViT_B_16_Weights
    device = device or torch.get_default_device()
    vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1).to(device)

    vit_prop = PropagatorTorchClassifier(vit, layers, device=device)

    return vit_prop


def vgg_propagator_builder(layers: list[str] = VGG16_LAYERS, device=None):
    from torchvision.models.vgg import vgg16, VGG16_Weights
    device = device or torch.get_default_device()
    vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).to(device)

    vgg_prop = PropagatorTorchClassifier(vgg, layers, device=device)

    return vgg_prop


class LoCEActivationsTensorExtractor:

    def __init__(self,
                 propagator: Propagator,
                 propagator_tag: str,
                 processor: 'BaseImageProcessor' = None
                 ) -> None:
        
        """
        propagator (Propagator): propagator instance
        propagator_tag (str): propogatr tag
        processor (BaseImageProcessor): HuggingFace image processor (if model requires)
        """
        self.propagator: Propagator = propagator
        self.propagator_tag = propagator_tag
        self.processor = processor

    def get_bchw_acts_preds_dict(self,
                                 image_pil: Image.Image,
                                 get_predictions: bool = True,
                                 ) -> Tuple[dict[str, torch.Tensor], torch.Tensor]:
        """
        Get activations and predictions for image

        Args:
            image_pil (Image.Image): image instance
            get_predictions (bool): whether to do a second evaluation of the model to get the predictions

        Returns:
            acts (dict[str, Tensor]) dictionary with Tensor[B,C,H,W] activations
            preds (Tensor) bbox predictions Tensor[N,6]
        """
        # custom method, HuggingFace model, a bit more complex
        if self.propagator_tag == 'detr':
            return self._get_acts_preds_dict_detr(image_pil, get_predictions=get_predictions)
        
        elif self.propagator_tag == 'vit':
            # append layer to get shape from, if not added
            # TODO: unfortunately it will increase memory use for activations in this layer. need to rework later
            aux_layer_added_flag = False
            if 'conv_proj' not in self.propagator.layers:
                self.propagator.set_layers(['conv_proj'] + self.propagator.layers)
                aux_layer_added_flag = True

            # input size is strictly (224, 224)
            image_pil = image_pil.resize((224, 224))
            acts, preds = self._get_acts_preds_dict(image_pil, get_predictions=get_predictions)

            conv_proj_shape = acts['conv_proj'].shape

            if aux_layer_added_flag:
                self.propagator.set_layers(self.propagator.layers[1:])
                acts.pop('conv_proj')

            for l, a in acts.items():
                # 'conv_proj' is CNN layer, other layers need to be processed
                if l != 'conv_proj':
                    a = a[:, 1:, :] # remove learnable class embedding (0th element) [B, H*W+1, C] -> [B, H*W, C]
                    a = a.permute(0, 2, 1) # swap axes [B, H*W, C] -> [B, C, H*W]
                    a = a.view(conv_proj_shape) # convert to 2d [B, C, H*W] -> [B, C, H, W]
                acts[l] = a

            return acts, preds

        elif self.propagator_tag == 'swin':
            acts, preds = self._get_acts_preds_dict(image_pil, get_predictions=get_predictions)
            acts = {l: a.permute(0, 3, 1, 2) for l, a in acts.items()}
            return acts, preds
        
        # other CNN-models, no need to permute or transform dimensions
        else:
            return self._get_acts_preds_dict(image_pil, get_predictions=get_predictions)


    def _get_acts_preds_dict(self,
                             image_pil: Image.Image,
                             get_predictions: bool = True,
                             ) -> Tuple[dict[str, torch.Tensor], Optional[torch.Tensor]]:
        """
        Get activations and predictions for image

        Args:
            image_pil (Image.Image): image instance

        Returns:
            acts (dict[str, Tensor]) dictionary with activations
            preds (Tensor) bbox predictions Tensor[N,6]
        """
        img_np = np.moveaxis(np.array(image_pil).astype(np.float32) / 255., [2], [0])

        img_tensor = torch.from_numpy(img_np).unsqueeze(0)

        acts: dict[str, torch.Tensor] = self.propagator.get_activations(img_tensor)
        preds: Optional[torch.Tensor] = None
        if get_predictions:
            preds = self.propagator.get_predictions(img_tensor)
        return acts, preds


    def _get_acts_preds_dict_detr(self,
                                  image_pil: Image.Image,
                                  get_predictions: bool = True,
                                  ) -> Tuple[dict[str, torch.Tensor], Optional[list[torch.Tensor]]]:
        """
        Get activations and predictions for image

        Args:
            image_pil (Image.Image): image instance
            get_predictions (bool): whether to do a second evaluation of the model to get the predictions

        Returns:
            acts (dict[str, Tensor]) dictionary with activations
            preds (Tensor) bbox predictions Tensor[N,6]
        """

        encoding = self.processor(image_pil, return_tensors="pt")

        # append layer to get shape from, if not added
        # TODO: unfortunately it will increase memory use for activations in this layer. need to rework later
        aux_layer_added_flag = False
        if 'model.input_projection' not in self.propagator.layers:
            self.propagator.set_layers(['model.input_projection'] + self.propagator.layers)
            aux_layer_added_flag = True

        acts = self.propagator.get_activations(encoding)

        # remembering CNN shape
        input_proj_shape = acts['model.input_projection'].shape

        if aux_layer_added_flag:
            self.propagator.set_layers(self.propagator.layers[1:])
            acts.pop('model.input_projection')

        activations_dict = {}

        # additional processing of activations
        for layer, act_item in acts.items():

            # ensure that representation is a tensor
            if not isinstance(act_item, torch.Tensor):
                act_item = act_item[0]

            # detach
            activations_dict[layer] = act_item.detach().cpu()

        # convert from transformer to CNN dimensions: [B, H*W, d] -> [B, d, H, W]
        for layer, act_item in activations_dict.items():
            # B: batch
            # H*W - 1d feature dimension size obtained from 2d feature maps
            # C / d: channels / reduced channels after feature projection
            if len(act_item.shape) == 3:
                # [B, H*W, d] -> [B, d, H*W]
                act_item_permuted = act_item.permute(0, 2, 1)
                # [B, d, H*W] -> [B, d, H, W]
                activations_dict[layer] = act_item_permuted.view(input_proj_shape)

        # get and process predictions
        predictions_list = None
        if get_predictions:
            preds = self.propagator.get_predictions(encoding)
            processed_preds = self.processor.post_process_object_detection(preds, target_sizes=[image_pil.size], threshold=0.5)

            # additional processing of predictions to fit (N, 6) shape: box, confidence, label
            predictions_list = []
            for pred_dict in processed_preds:
                prediction = torch.cat((pred_dict['boxes'], pred_dict['scores'].unsqueeze(1), pred_dict['labels'].unsqueeze(1)), dim=1)
                predictions_list.append(prediction.detach().cpu())

        return activations_dict, predictions_list

