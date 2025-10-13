import typing as tp

import torch
import torch.utils.checkpoint

import ImageReward as image_reward_module
from src.constants.dataset import DatasetColumns
from src.reward_models.base_model import BaseModel

PROCESSOR_NAME = "ViT-H-14"
PRETRAINED_MODEL_NAME = "laion2B-s32B-b79K"
HUGGINGFACE_REPO = "xswu/HPSv2"
HUGGINGFACE_FILENAME = "HPS_v2.1_compressed.pt"

MODEL_SUFFIX = "ImageReward"
MODEL_NAME = "ImageReward-v1.0"


class ImageReward(BaseModel):
    def __init__(self, device: torch.device):
        super().__init__(
            model_suffix=MODEL_SUFFIX,
            reward_scale_factor=1.0333394966054072,
            reward_offset=-0.16717362830052426,
        )
        self.model = image_reward_module.load(MODEL_NAME, device=device)

        self.device = device

    def tokenize(self, caption: str) -> tp.Dict[str, torch.Tensor]:
        processed_caption = self.model.blip.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=35,
            return_tensors="pt",
        )

        return {
            f"{DatasetColumns.tokenized_text.name}_{self.model_suffix}": processed_caption.input_ids,
            f"{DatasetColumns.text_attention_mask.name}_{self.model_suffix}": processed_caption.attention_mask,
        }

    def _get_reward(
        self,
        batch: tp.Dict[str, torch.Tensor],
        image: torch.Tensor,
    ) -> torch.Tensor:
        image_embeds = self.model.blip.visual_encoder(image)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            self.device
        )

        text_output = self.model.blip.text_encoder(
            batch[f"{DatasetColumns.tokenized_text.name}_{self.model_suffix}"],
            batch[f"{DatasetColumns.text_attention_mask.name}_{self.model_suffix}"],
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        txt_features = text_output.last_hidden_state[:, 0, :].float()
        rewards = self.model.mlp(txt_features)
        return rewards
