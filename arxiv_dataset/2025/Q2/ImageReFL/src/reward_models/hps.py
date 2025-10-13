import typing as tp

import huggingface_hub
import torch
import torch.utils.checkpoint
from hpsv2 import img_score

from src.constants.dataset import DatasetColumns
from src.reward_models.base_model import BaseModel

PROCESSOR_NAME = "ViT-H-14"
PRETRAINED_MODEL_NAME = "laion2B-s32B-b79K"
HUGGINGFACE_REPO = "xswu/HPSv2"
HUGGINGFACE_FILENAME = "HPS_v2.1_compressed.pt"

MODEL_SUFFIX = "HPS"


class HPS(BaseModel):
    def __init__(self, device: torch.device):
        super().__init__(
            model_suffix=MODEL_SUFFIX, reward_scale_factor=1, reward_offset=0
        )

        model, _, _ = img_score.create_model_and_transforms(
            PROCESSOR_NAME,
            PRETRAINED_MODEL_NAME,
            precision="amp",
            device=device,
            jit=False,
            force_quick_gelu=False,
            force_custom_text=False,
            force_patch_dropout=False,
            force_image_size=None,
            pretrained_image=False,
            image_mean=None,
            image_std=None,
            light_augmentation=True,
            aug_cfg={},
            output_dict=True,
            with_score_predictor=False,
            with_region_predictor=False,
        )

        cp = huggingface_hub.hf_hub_download(HUGGINGFACE_REPO, HUGGINGFACE_FILENAME)

        checkpoint = torch.load(cp, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])

        tokenizer = img_score.get_tokenizer(PROCESSOR_NAME)

        self.model = model
        self.tokenizer = tokenizer

        self.device = device

    def tokenize(self, caption: str) -> tp.Dict[str, torch.Tensor]:
        processed_caption = self.tokenizer(
            caption,
        )

        return {
            f"{DatasetColumns.tokenized_text.name}_{self.model_suffix}": processed_caption
        }

    def _get_reward(
        self,
        batch: tp.Dict[str, torch.Tensor],
        image: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.model(
            image, batch[f"{DatasetColumns.tokenized_text.name}_{self.model_suffix}"]
        )

        image_features, text_features = (
            outputs["image_features"],
            outputs["text_features"],
        )
        logits_per_image = image_features @ text_features.T

        return torch.diagonal(logits_per_image)
