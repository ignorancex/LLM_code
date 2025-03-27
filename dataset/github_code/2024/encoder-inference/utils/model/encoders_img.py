from typing import Union, Callable, Iterable
from PIL import Image
import numpy as np, timm, torch
from transformers import AutoImageProcessor, ResNetForImageClassification
from transformers import CLIPImageProcessor, CLIPVisionModel


class _base_rn:
    def __init__(self, model: Callable, trans: Callable, embed_dims: int, device: str):
        self.model = model
        self.trans = trans
        self.embed_dims = embed_dims
        self.device = device

    @torch.inference_mode()
    def __call__(self, x):
        return self.model(self.trans(x).to(self.device))


def _build_timm(name: str, embed_dims: int, device: str):
    """ ref: https://huggingface.co/timm/resnet50.a1_in1k
    """
    model = timm.create_model(name, pretrained=True, num_classes=0)
    model = model.to(device)
    model = model.eval()

    data_config = timm.data.resolve_model_data_config(model)
    trans = timm.data.create_transform(**data_config, is_training=False)

    def repacked_trans(x: Union[np.ndarray, Image.Image, torch.Tensor]):
        if isinstance(x, torch.Tensor):
            x = (x * 255).type(torch.uint8).cpu().numpy()

        if isinstance(x, np.ndarray):
            if len(x.shape) == 3:
                tensor_x = [trans(Image.fromarray(x)),]
            else:
                tensor_x = [trans(Image.fromarray(tx)) for tx in x]

        elif isinstance(x, Image.Image):
            tensor_x = [trans(x),]

        else:
            tensor_x = [trans(tx) for tx in x]

        return torch.stack(tensor_x)

    return _base_rn(model, repacked_trans, embed_dims, device)


def _build_ms_rn(name: str, embed_dims: int, device: str):
    """ ref: https://huggingface.co/microsoft/resnet-50
    """
    processor = AutoImageProcessor.from_pretrained(name)
    model = ResNetForImageClassification.from_pretrained(name)
    model.to(device)
    model.eval()

    def repacked_trans(x: Union[np.ndarray, torch.Tensor, Iterable]):
        if isinstance(x, torch.Tensor):
            x = (x * 255).type(torch.uint8).cpu().numpy()
        return processor(x, return_tensors="pt")["pixel_values"]

    def repacked_encoder(x):
        out = model(x, output_hidden_states=True).hidden_states[-1]
        out = torch.nn.functional.adaptive_avg_pool2d(out, output_size=(1, 1))
        return torch.flatten(out, 1)

    return _base_rn(repacked_encoder, repacked_trans, embed_dims, device)


def _build_clip(name: str, embed_dims: int, device: str, pooling: bool):
    processor = CLIPImageProcessor.from_pretrained(name)
    model = CLIPVisionModel.from_pretrained(name)
    model.to(device)
    model.eval()

    def repacked_trans(x: Union[np.ndarray, torch.Tensor, Iterable]):
        if isinstance(x, torch.Tensor):
            x = (x * 255).type(torch.uint8).cpu().numpy()
        return processor(x, return_tensors="pt")["pixel_values"]

    def repacked_encoder(x):
        if pooling:
            out = model(x).pooler_output
        else:
            out = model(x).last_hidden_state
        return out

    return _base_rn(repacked_encoder, repacked_trans, embed_dims, device)


def rn34_hf(device: str, pooling: bool):
    return _build_timm("resnet34.a1_in1k", 512, device)


def rn50_hf(device: str, pooling: bool):
    return _build_timm("resnet50.a1_in1k", 2048, device)


def mobilenetv3_hf(device: str, pooling: bool):
    return _build_timm("mobilenetv3_large_100.ra_in1k", 1280, device)


def rn34_ms(device: str, pooling: bool):
    return _build_ms_rn("microsoft/resnet-34", 512, device)


def rn50_ms(device: str, pooling: bool):
    return _build_ms_rn("microsoft/resnet-50", 2048, device)


def clip_vit_l14(device: str, pooling: bool):
    return _build_clip("openai/clip-vit-large-patch14", 768, device, pooling)


def clip_vit_l14_336(device: str, pooling: bool):
    return _build_clip("openai/clip-vit-large-patch14-336", 768, device, pooling)


def clip_vit_b16(device: str, pooling: bool):
    return _build_clip("openai/clip-vit-base-patch16", 512, device, pooling)


def clip_vit_b32(device: str, pooling: bool):
    return _build_clip("openai/clip-vit-base-patch32", 512, device, pooling)

def openclip_vit_b32(device: str, pooling: bool):
    return _build_clip("laion/CLIP-ViT-B-32-laion2B-s34B-b79K", 512, device, pooling)


def openclip_vit_l14(device: str, pooling: bool):
    return _build_clip("laion/CLIP-ViT-L-14-laion2B-s32B-b82K", 768, device, pooling)


def openclip_vit_h14(device: str, pooling: bool):
    return _build_clip("laion/CLIP-ViT-H-14-laion2B-s32B-b79K", 1024, device, pooling)


__model_zoo__ = {
    "rn34-hf": rn34_hf,
    "rn50-hf": rn50_hf,
    "mobilenetv3-hf": mobilenetv3_hf,
    "rn34-ms": rn34_ms,
    "rn50-ms": rn50_ms,
    "clip-vit-l14": clip_vit_l14,
    "clip-vit-l14-336": clip_vit_l14_336,
    "clip-vit-b16": clip_vit_b16,
    "clip-vit-b32": clip_vit_b32,
    "openclip-vit-b32": openclip_vit_b32,
    "openclip-vit-l14": openclip_vit_l14,
    "openclip-vit-h14": openclip_vit_h14,
}

def get_img_encoder(name: str, device: str = "cuda:0", pooling: bool = True):
    return __model_zoo__[name](device, pooling)
