import typing as tp

import clip
import numpy as np
import torch
import torch.utils.checkpoint
from transformers import AutoProcessor

from src.reward_models.base_model import BaseModel

PROCESSOR_NAME = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
MODEL_NAME = "ViT-B/32"

MODEL_SUFFIX = "ClipDiversity"


def get_tril_elements_mask(linear_size):
    mask = np.zeros((linear_size, linear_size), dtype=np.bool_)
    mask[np.tril_indices_from(mask)] = True
    np.fill_diagonal(mask, False)
    return mask


def _diversity_from_embeddings_pairwise_cosines(imgs_encoded: torch.Tensor):
    data = (imgs_encoded @ imgs_encoded.T).to(torch.float32).detach().cpu().numpy()
    mask = get_tril_elements_mask(data.shape[0])
    masked = data[mask].astype(np.float64)
    return masked


class ClipDiversity(BaseModel):
    def __init__(self, images_per_batch: int, device: torch.device):
        super().__init__(
            model_suffix=MODEL_SUFFIX, reward_scale_factor=0.1, reward_offset=0
        )
        self.images_per_batch = images_per_batch
        self.processor = AutoProcessor.from_pretrained(PROCESSOR_NAME)
        self.clip_model, self.transform = clip.load(
            MODEL_NAME, device=device, jit=False
        )

        self.device = device

    def tokenize(self, caption: str) -> tp.Dict[str, torch.Tensor]:
        return {}

    def _get_reward(
        self,
        batch: tp.Dict[str, torch.Tensor],
        image: torch.Tensor,
    ) -> torch.Tensor:
        scores = []
        for i in range(0, image.shape[0], self.images_per_batch):
            image_features = self.clip_model.encode_image(
                image[i : i + self.images_per_batch]
            )
            image_features /= image_features.clone().norm(dim=-1, keepdim=True)
            masked = _diversity_from_embeddings_pairwise_cosines(image_features)
            scores.append(1 - masked.mean())
        return torch.tensor(scores)
