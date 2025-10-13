import typing as tp

import torch
import torch.utils.checkpoint
from transformers import AutoModel, AutoProcessor

from src.constants.dataset import DatasetColumns
from src.reward_models.base_model import BaseModel

PROCESSOR_NAME = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
PRETRAINED_MODEL_NAME = "yuvalkirstain/PickScore_v1"

MODEL_SUFFIX = "PickScore"


class PickScore(BaseModel):
    def __init__(self, device: torch.device):
        super().__init__(
            model_suffix=MODEL_SUFFIX, reward_scale_factor=0.1, reward_offset=-20
        )

        self.processor = AutoProcessor.from_pretrained(PROCESSOR_NAME)
        self.model = AutoModel.from_pretrained(PRETRAINED_MODEL_NAME).eval()

        self.device = device

    def tokenize(self, caption: str) -> tp.Dict[str, torch.Tensor]:
        processed_caption = self.processor(
            text=caption,
            padding="max_length",
            truncation=True,
            max_length=77,
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
        image_embs = self.model.get_image_features(pixel_values=image)
        image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)

        text_embs = self.model.get_text_features(
            input_ids=batch[
                f"{DatasetColumns.tokenized_text.name}_{self.model_suffix}"
            ],
            attention_mask=batch[
                f"{DatasetColumns.text_attention_mask.name}_{self.model_suffix}"
            ],
        )
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)

        return self.model.logit_scale.exp() * torch.diag(text_embs @ image_embs.T)
