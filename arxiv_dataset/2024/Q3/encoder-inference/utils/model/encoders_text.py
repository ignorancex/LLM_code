from typing import List, Iterable, Callable
# from diffusers import StableDiffusionPipeline
import torch

from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel
from transformers import RobertaForMaskedLM, CLIPTextModel


class _base_encoder:
    def __init__(self, model: Callable, trans: Callable, embed_dims: int, device: str, pooling: bool=True):
        self.model = model
        self.trans = trans
        self.embed_dims = embed_dims
        self.device = device
        self.pooling = pooling

    @torch.inference_mode()
    def __call__(self, x):
        return self.model(self.trans(x), self.pooling)


def _build_bert_hf(name: str, embed_dims: int, max_length: int, device: str, pooling: bool):
    """ ref: https://huggingface.co/google-bert/bert-base-uncased
    """
    model = BertModel.from_pretrained(name)
    model = model.to(device)
    model = model.eval()

    tokenizer = AutoTokenizer.from_pretrained(name)

    def repacked_trans(x: Iterable):
        out = tokenizer(x, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        out = {key: out[key].to(device) for key in out.keys()}
        return out

    def repacked_encoder(x_dict: dict, pooling=True):
        if pooling:
            out = model(**x_dict).pooler_output
        else:
            out = model(**x_dict).last_hidden_state
        return out

    return _base_encoder(repacked_encoder, repacked_trans, embed_dims, device, pooling)


def _build_t5_hf(name: str, embed_dims: int, max_length: int, device: str, pooling: bool):
    """ ref: https://huggingface.co/google-t5/t5-base
    """
    model = AutoModel.from_pretrained(name)
    model = model.to(device)
    model = model.eval()

    tokenizer = AutoTokenizer.from_pretrained(name)

    def repacked_trans(x: Iterable):
        out = tokenizer(x, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        out = {key: out[key].to(device) for key in out.keys()}
        return out

    def repacked_encoder(x_dict: dict, pooling=True):
        out = model.encoder(**x_dict)["last_hidden_state"]

        if pooling:
            """ Follow `https://arxiv.org/abs/2108.08877`
            to perform averaging sentence embedding.
            """
            msk = x_dict["attention_mask"].unsqueeze(-1)
            out = (out * msk / msk.sum(dim=1, keepdim=True)).sum(dim=1)

        return out

    return _base_encoder(repacked_encoder, repacked_trans, embed_dims, device, pooling)


def _build_roberta_hf(name: str, embed_dims: int, max_length: int, device: str, pooling: bool):
    """ ref: https://huggingface.co/FacebookAI/roberta-base
             https://github.com/huggingface/transformers/issues/9882#issuecomment-770675232
    """
    model = RobertaForMaskedLM.from_pretrained(name)
    model = model.to(device)
    model = model.eval()

    tokenizer = AutoTokenizer.from_pretrained(name)

    def repacked_trans(x: Iterable):
        out = tokenizer(x, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        out = {key: out[key].to(device) for key in out.keys()}
        return out

    def repacked_encoder(x_dict: dict, pooling=True):
        out = model.roberta(**x_dict)["last_hidden_state"]
        if pooling:
            msk = x_dict["attention_mask"].unsqueeze(-1)
            out = (out * msk / msk.sum(dim=1, keepdim=True)).sum(dim=1)
        return out

    return _base_encoder(repacked_encoder, repacked_trans, embed_dims, device, pooling)


def _build_clip(name: str, embed_dims: int, device: str, pooling: bool):
    """ ref: https://huggingface.co/openai/clip-vit-base-patch16
    """
    model = CLIPTextModel.from_pretrained(name)
    model = model.to(device)
    model = model.eval()

    tokenizer = AutoTokenizer.from_pretrained(name)

    def repacked_trans(x: Iterable):
        out = tokenizer(x, return_tensors="pt", padding="max_length", truncation=True)
        out = {key: out[key].to(device) for key in out.keys()}
        return out

    def repacked_encoder(x_dict: dict, pooling=True):
        out = model(**x_dict)["last_hidden_state"]
        if pooling:
            msk = x_dict["attention_mask"].unsqueeze(-1)
            out = (out * msk / msk.sum(dim=1, keepdim=True)).sum(dim=1)
        return out

    return _base_encoder(repacked_encoder, repacked_trans, embed_dims, device, pooling)


def bertbase(device: str, pooling: bool):
    return _build_bert_hf("bert-base-uncased", 768, 128, device, pooling)


def bertlarge(device: str, pooling: bool):
    return _build_bert_hf("bert-large-uncased", 1024, 128, device, pooling)


def t5small(device: str, pooling: bool):
    """ `max_length=128` follows `https://arxiv.org/abs/2108.08877` """
    return _build_t5_hf("google-t5/t5-small", 512, 128, device, pooling)


def t5base(device: str, pooling: bool):
    """ `max_length=128` follows `https://arxiv.org/abs/2108.08877` """
    return _build_t5_hf("google-t5/t5-base", 768, 128, device, pooling)


def robertabase(device: str, pooling: bool):
    return _build_roberta_hf("FacebookAI/roberta-base", 768, 128, device, pooling)


def clip_vit_l14(device: str, pooling: bool):
    return _build_clip("openai/clip-vit-large-patch14", 768, device, pooling)


def clip_vit_b16(device: str, pooling: bool):
    return _build_clip("openai/clip-vit-base-patch16", 512, device, pooling)


def clip_vit_b32(device: str, pooling: bool):
    return _build_clip("openai/clip-vit-base-patch32", 512, device, pooling)


def openclip_vit_b16(device: str, pooling: bool):
    return _build_clip("laion/CLIP-ViT-B-16-laion2B-s34B-b88K", 512, device, pooling)


def openclip_vit_b32(device: str, pooling: bool):
    return _build_clip("laion/CLIP-ViT-B-32-laion2B-s34B-b79K", 512, device, pooling)


def openclip_vit_l14(device: str, pooling: bool):
    return _build_clip("laion/CLIP-ViT-L-14-laion2B-s32B-b82K", 768, device, pooling)


def openclip_vit_h14(device: str, pooling: bool):
    return _build_clip("laion/CLIP-ViT-H-14-laion2B-s32B-b79K", 1024, device, pooling)


__model_zoo__ = {
    "bertbase": bertbase,
    "bertlarge": bertlarge,
    "t5small": t5small,
    "t5base": t5base,
    "robertabase": robertabase,
    "clip-vit-l14": clip_vit_l14,
    "clip-vit-b16": clip_vit_b16,
    "clip-vit-b32": clip_vit_b32,
    "openclip-vit-b16": openclip_vit_b16,
    "openclip-vit-b32": openclip_vit_b32,
    "openclip-vit-l14": openclip_vit_l14,
    "openclip-vit-h14": openclip_vit_h14,
}


def get_text_encoder(name: str, device: str = "cuda:0", pooling: bool = True):
    return __model_zoo__[name](device, pooling)
