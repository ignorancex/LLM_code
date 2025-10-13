from PIL import Image
import torch

from transformers import AutoProcessor, SiglipVisionModel
from models.projection_models import MLPProjModel, QFormerProjModel
from models.attention_processor import FluxAttnProcessor


class Calligrapher:
    def __init__(self, sd_pipe, image_encoder_path, calligrapher_path, device, num_tokens):
        self.device = device
        self.image_encoder_path = image_encoder_path
        self.calligrapher_path = calligrapher_path
        self.num_tokens = num_tokens

        self.pipe = sd_pipe.to(self.device)
        self.set_attn_adapter()

        self.image_encoder = SiglipVisionModel.from_pretrained(image_encoder_path).to(self.device, dtype=torch.bfloat16)
        self.clip_image_processor = AutoProcessor.from_pretrained(self.image_encoder_path)
        self.image_proj_mlp, self.image_proj_qformer = self.init_proj()

        self.load_models()

    def init_proj(self):
        image_proj_mlp = MLPProjModel(
            cross_attention_dim=self.pipe.transformer.config.joint_attention_dim,
            id_embeddings_dim=1152,
            num_tokens=self.num_tokens,
        ).to(self.device, dtype=torch.bfloat16)

        image_proj_qformer = QFormerProjModel(
            cross_attention_dim=4096,
            id_embeddings_dim=1152,
            num_tokens=self.num_tokens,
            num_heads=8,
            num_query_tokens=32
        ).to(self.device, dtype=torch.bfloat16)
        return image_proj_mlp, image_proj_qformer

    def set_attn_adapter(self):
        transformer = self.pipe.transformer
        attn_procs = {}
        for name in transformer.attn_processors.keys():
            if name.startswith("transformer_blocks.") or name.startswith("single_transformer_blocks"):
                attn_procs[name] = FluxAttnProcessor(
                    hidden_size=transformer.config.num_attention_heads * transformer.config.attention_head_dim,
                    cross_attention_dim=transformer.config.joint_attention_dim,
                    num_tokens=self.num_tokens,
                ).to(self.device, dtype=torch.bfloat16)
            else:
                attn_procs[name] = transformer.attn_processors[name]
        transformer.set_attn_processor(attn_procs)

    def load_models(self):
        state_dict = torch.load(self.calligrapher_path, map_location="cpu")
        self.image_proj_mlp.load_state_dict(state_dict["image_proj_mlp"], strict=True)
        self.image_proj_qformer.load_state_dict(state_dict["image_proj_qformer"], strict=True)
        target_layers = torch.nn.ModuleList(self.pipe.transformer.attn_processors.values())
        target_layers.load_state_dict(state_dict["attn_adapter"], strict=False)

    @torch.inference_mode()
    def get_image_embeds(self, pil_image=None, clip_image_embeds=None):
        if pil_image is not None:
            if isinstance(pil_image, Image.Image):
                pil_image = [pil_image]
            clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
            clip_image_embeds = self.image_encoder(
                clip_image.to(self.device, dtype=self.image_encoder.dtype)).pooler_output
            clip_image_embeds = clip_image_embeds.to(dtype=torch.bfloat16)
        else:
            clip_image_embeds = clip_image_embeds.to(self.device, dtype=torch.bfloat16)
        image_prompt_embeds = self.image_proj_mlp(clip_image_embeds) \
                              + self.image_proj_qformer(clip_image_embeds)
        return image_prompt_embeds

    def set_scale(self, scale):
        for attn_processor in self.pipe.transformer.attn_processors.values():
            if isinstance(attn_processor, FluxAttnProcessor):
                attn_processor.scale = scale

    def generate(
            self,
            image=None,
            mask_image=None,
            ref_image=None,
            clip_image_embeds=None,
            prompt=None,
            scale=1.0,
            seed=None,
            num_inference_steps=30,
            **kwargs,
    ):
        self.set_scale(scale)

        image_prompt_embeds = self.get_image_embeds(
            pil_image=ref_image, clip_image_embeds=clip_image_embeds
        )

        if seed is None:
            generator = None
        else:
            generator = torch.Generator(self.device).manual_seed(seed)

        images = self.pipe(
            image=image,
            mask_image=mask_image,
            prompt=prompt,
            image_emb=image_prompt_embeds,
            num_inference_steps=num_inference_steps,
            generator=generator,
            **kwargs,
        ).images

        return images
