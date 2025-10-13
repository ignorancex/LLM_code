import typing as tp
from abc import abstractmethod

import torch.utils.checkpoint


class BaseModel(torch.nn.Module):
    def __init__(
        self, model_suffix: str, reward_scale_factor: float, reward_offset: float
    ):
        super().__init__()
        self.model_suffix = model_suffix
        self.reward_scale_factor = reward_scale_factor
        self.reward_offset = reward_offset

    @abstractmethod
    def tokenize(self, caption: str) -> tp.Dict[str, torch.Tensor]:
        pass

    @abstractmethod
    def _get_reward(
        self,
        batch: tp.Dict[str, torch.Tensor],
        image: torch.Tensor,
    ) -> torch.Tensor:
        pass

    def score_grad(
        self,
        batch: tp.Dict[str, torch.Tensor],
    ) -> None:
        image = batch["image"]
        reward = self._get_reward(batch, image)
        loss = -(reward + self.reward_offset) * self.reward_scale_factor
        batch["loss"] += loss.mean()
        batch[self.model_suffix] = reward.mean().detach()

    def score(
        self,
        batch: tp.Dict[str, torch.Tensor],
    ) -> None:
        image = batch["image"]
        with torch.no_grad():
            reward = self._get_reward(batch, image)
        batch[self.model_suffix] = reward.mean().detach()
