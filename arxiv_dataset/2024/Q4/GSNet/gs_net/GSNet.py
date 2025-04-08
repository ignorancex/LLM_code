# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import ImageList
from detectron2.utils.memory import _ignore_torch_cuda_oom

from einops import rearrange
from .vision_transformer import vit_base
import os

def BuildRSIB(Weights):
    model = vit_base(patch_size=8, num_classes=0)
    if os.path.isfile(Weights):
        state_dict = torch.load(Weights, map_location='cpu')
        checkpoint_key = "teacher"
        if checkpoint_key is not None and checkpoint_key in state_dict:
            print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(Weights, msg))
        model = model.float()
        return model
    else:
        raise FileNotFoundError(f"Pretrained weights not found at {Weights}. Please check the file path.")
    
@META_ARCH_REGISTRY.register()
class GSNet(nn.Module):
    @configurable
    
    
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        size_divisibility: int,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        clip_pixel_mean: Tuple[float],
        clip_pixel_std: Tuple[float],
        train_class_json: str,
        test_class_json: str,
        sliding_window: bool,
        clip_finetune: str,
        backbone_multiplier: float,
        clip_pretrained: str,
        dino: nn.Module,
        use_clip: bool,
        clip_decod_guid_dim: list,
        dino_decod_guid_dim: list,
    ):
        """
        Args:
            sem_seg_head: a module that predicts semantic segmentation from backbone features
        """
        super().__init__()
        self.dino_model = dino
        self.use_clip = use_clip

        self.backbone = backbone
        self.clip_decod_dim = clip_decod_guid_dim
        self.dino_decod_dim = dino_decod_guid_dim
        self.sem_seg_head = sem_seg_head
        if size_divisibility < 0:
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility

        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        self.register_buffer("clip_pixel_mean", torch.Tensor(clip_pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("clip_pixel_std", torch.Tensor(clip_pixel_std).view(-1, 1, 1), False)
        
        self.train_class_json = train_class_json
        self.test_class_json = test_class_json

        self.clip_finetune = clip_finetune
        for name, params in self.sem_seg_head.predictor.clip_model.named_parameters():
            if clip_finetune == "freezeIMG":
                if "attn" in name:
                    # QV fine-tuning for attention blocks
                    params.requires_grad = True if "q_proj" in name or "v_proj" in name else False
                elif "position" in name:
                    params.requires_grad = True
                else: params.requires_grad = False
                if "visual" in name:
                    params.requires_grad = False
                
            elif "transformer" in name:
                if clip_finetune == "prompt":
                    params.requires_grad = True if "prompt" in name else False
                elif clip_finetune == "attention":
                    if "attn" in name:
                        # QV fine-tuning for attention blocks
                        params.requires_grad = True if "q_proj" in name or "v_proj" in name else False
                    elif "position" in name:
                        params.requires_grad = True
                    else:
                        params.requires_grad = False
                elif clip_finetune == "full":
                    params.requires_grad = True
                else:
                    params.requires_grad = False
            
            else:
                params.requires_grad = False


        self.sliding_window = sliding_window
        if clip_pretrained == "ViT-B/16": 
            self.clip_resolution = (384, 384)
        elif clip_pretrained == "RemoteCLIP-ViT-B-32":
            self.clip_resolution = (768,768)
        else: 
            self.clip_resolution = (336, 336)
        self.dino_resolution = (384,384)
        self.proj_dim = 768 if clip_pretrained == "ViT-B/16" or clip_pretrained == "RemoteCLIP-ViT-B-32" else 1024


        self.upsample1 = nn.ConvTranspose2d(self.proj_dim, 256, kernel_size=2, stride=2) if self.use_clip and self.clip_decod_dim[0]!=0 else None
        self.upsample2 = nn.ConvTranspose2d(self.proj_dim, 128, kernel_size=4, stride=4) if self.use_clip and self.clip_decod_dim[1]!=0 else None
        
        self.dino_decod_proj1 = nn.Conv2d(in_channels = 768, out_channels=256, kernel_size=1, stride=1, padding=0) if self.dino_model and self.dino_decod_dim[0]!=0 else None
        self.dino_decod_proj2 = nn.ConvTranspose2d(in_channels= 768, out_channels=128, kernel_size=2, stride=2) if self.dino_model and self.dino_decod_dim[0]!=0 else None
        
        self.dino_down_sample = nn.Conv2d(in_channels=768, out_channels=512, kernel_size=2, stride=2, padding=0) if self.dino_model else None
        self.layer_indexes = [3, 7] if clip_pretrained == "ViT-B/16" or clip_pretrained == "RemoteCLIP-ViT-B-32" else [7, 15] 
        self.layers = []
        if self.use_clip:
            for l in self.layer_indexes:
                self.sem_seg_head.predictor.clip_model.visual.transformer.resblocks[l].register_forward_hook(lambda m, _, o: self.layers.append(o))


    @classmethod
    def from_config(cls, cfg):
        backbone = None
        sem_seg_head = build_sem_seg_head(cfg, None)
        if cfg.MODEL.SEM_SEG_HEAD.USE_DINO_CORR:
            
            dino = BuildRSIB(os.getenv('RSIB_CKPT'))
            dino_ft = cfg.MODEL.SEM_SEG_HEAD.DINO_FINETUNE
            for name, params in dino.named_parameters():
                if dino_ft == "attention":
                    
                    if "attn.qkv.weight" in name:
                        params.requires_grad = True
                    elif "pos_embed" in name:
                        params.requires_grad = True
                    else:
                        params.requires_grad = False
                elif dino_ft == "full":
                    params.requires_grad = True
                else:
                    params.requires_grad = False

        else:
            dino = None
            

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "clip_pixel_mean": cfg.MODEL.CLIP_PIXEL_MEAN,
            "clip_pixel_std": cfg.MODEL.CLIP_PIXEL_STD,
            "train_class_json": cfg.MODEL.SEM_SEG_HEAD.TRAIN_CLASS_JSON,
            "test_class_json": cfg.MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON,
            "sliding_window": cfg.TEST.SLIDING_WINDOW,
            "clip_finetune": cfg.MODEL.SEM_SEG_HEAD.CLIP_FINETUNE,
            "backbone_multiplier": cfg.SOLVER.BACKBONE_MULTIPLIER,
            "clip_pretrained": cfg.MODEL.SEM_SEG_HEAD.CLIP_PRETRAINED,
            "dino": dino, 
            "use_clip":cfg.MODEL.SEM_SEG_HEAD.USE_CLIP_CORR, 
            "clip_decod_guid_dim":cfg.MODEL.SEM_SEG_HEAD.DECODER_CLIP_GUIDANCE_DIMS,
            "dino_decod_guid_dim":cfg.MODEL.SEM_SEG_HEAD.DECODER_DINO_GUIDANCE_DIMS
            
        }

    @property
    def device(self):
        return self.pixel_mean.device
    # @profile(precision=4,stream=open('./log.txt','w+',encoding="utf-8"))
    def forward(self, batched_inputs):

        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
        """
        if self.training:
            images = [x["image"].to(self.device) for x in batched_inputs]
            # images_shape: 384*384
            clip_images = [(x - self.clip_pixel_mean) / self.clip_pixel_std for x in images]
            clip_images = ImageList.from_tensors(clip_images, self.size_divisibility)
        
            self.layers = []

            clip_images_resized = F.interpolate(clip_images.tensor, size=self.clip_resolution, mode='bilinear', align_corners=False, )
            dino_images_resized = F.interpolate(clip_images.tensor, size=self.dino_resolution, mode='bilinear', align_corners=False, )
            # clip_features = self.sem_seg_head.predictor.clip_model.encode_image(clip_images_resized, dense=True)
        elif not self.sliding_window:
            with torch.no_grad():
                images = [x["image"].to(self.device) for x in batched_inputs]
                clip_images = [(x - self.clip_pixel_mean) / self.clip_pixel_std for x in images]
                clip_images = ImageList.from_tensors(clip_images, self.size_divisibility)
            
                self.layers = []

                clip_images_resized = F.interpolate(clip_images.tensor, size=self.clip_resolution, mode='bilinear', align_corners=False, )
                dino_images_resized = F.interpolate(clip_images.tensor, size=self.dino_resolution, mode='bilinear', align_corners=False, )
        elif self.sliding_window:
            with torch.no_grad():
                kernel=384
                overlap=0.333
                out_res=[640, 640]
                images = [x["image"].to(self.device, dtype=torch.float32) for x in batched_inputs]
                stride = int(kernel * (1 - overlap))
                unfold = nn.Unfold(kernel_size=kernel, stride=stride)
                fold = nn.Fold(out_res, kernel_size=kernel, stride=stride)

                image = F.interpolate(images[0].unsqueeze(0), size=out_res, mode='bilinear', align_corners=False).squeeze()
                image = rearrange(unfold(image), "(C H W) L-> L C H W", C=3, H=kernel)
                global_image = F.interpolate(images[0].unsqueeze(0), size=(kernel, kernel), mode='bilinear', align_corners=False)
                image = torch.cat((image, global_image), dim=0)

                images = (image - self.pixel_mean) / self.pixel_std
                clip_images = (image - self.clip_pixel_mean) / self.clip_pixel_std
                clip_images_resized = F.interpolate(clip_images, size=self.clip_resolution, mode='bilinear', align_corners=False, )
                dino_images_resized = F.interpolate(clip_images, size=self.dino_resolution, mode='bilinear', align_corners=False, )
                self.layers = []
                
        
        if self.dino_model is not None:
            dino_feat = self.dino_model.get_intermediate_layers(dino_images_resized, n=12) # actually only 12 layers, but use a large num to avoid ambiguity
            dino_patch_feat_last_unfold = rearrange(dino_feat[-1][:,1:,:],"B (H W) C -> B C H W", H=48)
            dino_feat_down = self.dino_down_sample(dino_patch_feat_last_unfold) # B,512,24,24
            dino_feat_L4 = rearrange(dino_feat[3][:,1:,:],"B (H W) C -> B C H W", H=48)
            dino_feat_L8 = rearrange(dino_feat[7][:,1:,:],"B (H W) C -> B C H W", H=48)
            
            dino_feat_L4_proj = self.dino_decod_proj1(dino_feat_L4) if self.dino_decod_proj1 is not None else None
            dino_feat_L8_proj = self.dino_decod_proj2(dino_feat_L8) if self.dino_decod_proj2 is not None else None
            dino_feat_guidance = [dino_feat_L4_proj,dino_feat_L8_proj]
        else:
            dino_feat_down, dino_feat_guidance = None, None
        
        if self.use_clip:
            clip_features = self.sem_seg_head.predictor.clip_model.encode_image(clip_images_resized, dense=True)
            clip_image_features = clip_features[:, 1:, :]
            res3 = rearrange(clip_image_features, "B (H W) C -> B C H W", H=24)
            res4 = rearrange(self.layers[0][1:, :, :], "(H W) B C -> B C H W", H=24)
            res5 = rearrange(self.layers[1][1:, :, :], "(H W) B C -> B C H W", H=24)
            res4 = self.upsample1(res4) if self.upsample1 is not None else None
            res5 = self.upsample2(res5) if self.upsample2 is not None else None
            

            clip_features_guidance = {'res5': res5, 'res4': res4, 'res3': res3,}
        else:
            clip_features, clip_features_guidance=None, None

        outputs = self.sem_seg_head(clip_features,dino_feat_down, clip_features_guidance, dino_feat_guidance)
        if self.training:
            targets = torch.stack([x["sem_seg"].to(self.device) for x in batched_inputs], dim=0)
            outputs = F.interpolate(outputs, size=(targets.shape[-2], targets.shape[-1]), mode="bilinear", align_corners=False)
            
            num_classes = outputs.shape[1]
            mask = targets != self.sem_seg_head.ignore_value

            outputs = outputs.permute(0,2,3,1)
            _targets = torch.zeros(outputs.shape, device=self.device)
            _onehot = F.one_hot(targets[mask], num_classes=num_classes).float()
            _targets[mask] = _onehot
            
            loss = F.binary_cross_entropy_with_logits(outputs, _targets)
            losses = {"loss_sem_seg" : loss}
            return losses
        elif self.sliding_window:
            with torch.no_grad():
                outputs = F.interpolate(outputs, size=kernel, mode="bilinear", align_corners=False)
                outputs = outputs.sigmoid()
                
                global_output = outputs[-1:]
                global_output = F.interpolate(global_output, size=out_res, mode='bilinear', align_corners=False,)
                outputs = outputs[:-1]
                outputs = fold(outputs.flatten(1).T) / fold(unfold(torch.ones([1] + out_res, device=self.device)))
                outputs = (outputs + global_output) / 2.

                height = batched_inputs[0].get("height", out_res[0])
                width = batched_inputs[0].get("width", out_res[1])
                output = sem_seg_postprocess(outputs[0], out_res, height, width)
                return [{'sem_seg': output}]
        
        else:
            with torch.no_grad():
                outputs = outputs.sigmoid()
                image_size = clip_images.image_sizes[0]
                height = batched_inputs[0].get("height", image_size[0])
                width = batched_inputs[0].get("width", image_size[1])

                output = sem_seg_postprocess(outputs[0], image_size, height, width)
                processed_results = [{'sem_seg': output}]
                return processed_results        
        