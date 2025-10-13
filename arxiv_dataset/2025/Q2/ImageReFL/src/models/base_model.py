import typing as tp
from abc import abstractmethod

import torch


class BaseModel(torch.nn.Module):
    def __init__(
        self,
        pretrained_model_name: str,
        revision: str | None = None,
    ) -> None:
        super().__init__()
        self.pretrained_model_name_or_path = pretrained_model_name
        self.revision = revision

    @abstractmethod
    def train(self, mode: bool = True):
        pass

    @abstractmethod
    def eval(self, mode: bool = True):
        pass

    @abstractmethod
    def tokenize(self, caption: str) -> dict[str, tp.Any]:
        pass

    @abstractmethod
    def get_noisy_latents_from_images(
        self,
        images: torch.Tensor,
        timestep_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def set_timesteps(self, max_timestep: int, device) -> None:
        pass

    @abstractmethod
    def predict_next_latents(
        self,
        latents: torch.Tensor,
        timestep_index: int,
        encoder_hidden_states: torch.Tensor,
        batch: dict[str, torch.Tensor],
        return_pred_original: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def do_k_diffusion_steps(
        self,
        start_timestep_index: int,
        end_timestep_index: int,
        batch: dict[str, torch.Tensor],
        latents: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        return_pred_original: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def sample_image(
        self,
        latents: torch.Tensor | None,
        start_timestep_index: int,
        end_timestep_index: int,
        batch: dict[str, torch.Tensor],
        encoder_hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pass
