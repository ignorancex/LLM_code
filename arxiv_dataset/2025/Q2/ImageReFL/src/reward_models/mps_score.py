import typing as tp

import torch
from transformers import AutoTokenizer

from src.constants.dataset import DatasetColumns
from src.reward_models.base_model import BaseModel

MODEL_NAME = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"

CONDITION = "light, color, clarity, tone, style, ambiance, artistry, shape, face, hair, hands, limbs, structure, instance, texture, quantity, attributes, position, number, location, word, things."

MODEL_SUFFIX = "MPS"


class MPS(BaseModel):
    def __init__(self, checkpoint_path: str, device: torch.device):
        super().__init__(
            model_suffix=MODEL_SUFFIX, reward_scale_factor=0.1, reward_offset=0
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME, trust_remote_code=True
        )
        self.tokenized_condition = self.tokenizer(
            CONDITION,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(device)

        self.model = torch.load(checkpoint_path, map_location=device)
        self.model.eval().to(device)

        # straightforward fix of original code
        self.model.model.text_model.eos_token_id = self.tokenizer.eos_token_id

    def tokenize(self, caption: str) -> tp.Dict[str, torch.Tensor]:
        processed_caption = self.tokenizer(
            caption,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids
        return {
            f"{DatasetColumns.tokenized_text.name}_{self.model_suffix}": processed_caption
        }

    def _fixed_forward(
        self, text_inputs=None, image_inputs=None, condition_inputs=None
    ):
        text_f, text_features_eos = self.model.model.get_text_features(
            text_inputs
        )  # B*77*1024

        image_f = self.model.model.get_image_features(image_inputs.half())
        condition_f, _ = self.model.model.get_text_features(condition_inputs)

        sim_text_condition = torch.einsum("b i d, b j d -> b j i", text_f, condition_f)
        sim_text_condition = torch.max(sim_text_condition, dim=1, keepdim=True)[0]
        sim_text_condition = sim_text_condition / sim_text_condition.max()
        mask = torch.where(sim_text_condition > 0.01, 0, float("-inf"))  # B*1*77

        mask = mask.repeat(1, image_f.shape[1], 1)  # B*257*77

        image_features = self.model.cross_model(image_f, text_f, mask.half())[:, 0, :]

        return text_features_eos, image_features

    def _get_reward(
        self,
        batch: tp.Dict[str, torch.Tensor],
        image: torch.Tensor,
    ) -> torch.Tensor:
        image_inputs = image

        text_inputs = batch[f"{DatasetColumns.tokenized_text.name}_{self.model_suffix}"]

        text_features, image_features = self._fixed_forward(
            text_inputs, image_inputs, self.tokenized_condition
        )
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        reward = self.model.logit_scale.exp() * torch.diag(
            torch.einsum("bd,cd->bc", text_features, image_features)
        )

        return reward
