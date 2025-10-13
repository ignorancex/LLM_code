import typing as tp

import clip
import torch

from src.constants.dataset import DatasetColumns
from src.reward_models.base_model import BaseModel

MODEL_NAME = "ViT-B/32"

MODEL_SUFFIX = "ClipScore"


class ClipScore(BaseModel):
    def __init__(self, device: torch.device):
        super().__init__(
            model_suffix=MODEL_SUFFIX, reward_scale_factor=1, reward_offset=0
        )

        model, transform = clip.load(MODEL_NAME, device=device, jit=False)
        self.model = model
        self.transform = transform
        self.device = device

    def tokenize(self, caption: str) -> tp.Dict[str, torch.Tensor]:
        processed_caption = clip.tokenize(
            caption,
            truncate=True,
        )

        return {
            f"{DatasetColumns.tokenized_text.name}_{self.model_suffix}": processed_caption
        }

    def _get_reward(
        self,
        batch: tp.Dict[str, torch.Tensor],
        image: torch.Tensor,
    ) -> torch.Tensor:
        tokenized_caption = batch[
            f"{DatasetColumns.tokenized_text.name}_{self.model_suffix}"
        ]
        candidates = self.model.encode_text(tokenized_caption)
        images = self.model.encode_image(image)

        images = torch.nn.functional.normalize(images, dim=1)
        candidates = torch.nn.functional.normalize(candidates, dim=1)

        reward = torch.clip(torch.sum(images * candidates, axis=1), 0, None)
        return reward
