import torch
from torch.distributions import LogisticNormal
import torch.nn.functional as F

# some code are inspired by https://github.com/magic-research/piecewise-rectified-flow/blob/main/scripts/train_perflow.py
# and https://github.com/magic-research/piecewise-rectified-flow/blob/main/src/scheduler_perflow.py

class RFlowAudioScheduler_x0:
    def __init__(
        self,
        num_sampling_steps=10,
        num_timesteps=1000,
        sample_method="uniform",
        loc=0.0,
        scale=1.0,
        loss_type="mse",
    ):
        self.num_sampling_steps = num_sampling_steps
        self.num_timesteps = num_timesteps
        # sample method
        assert sample_method in ["uniform", "logit-normal"]
        self.sample_method = sample_method
        if sample_method == "logit-normal":
            self.distribution = LogisticNormal(torch.tensor([loc]), torch.tensor([scale]))
            self.sample_t = lambda x: self.distribution.sample((x.shape[0],))[:, 0].to(x.device)

        # loss type
        assert loss_type in ["l1", "mse"]
        self.loss_type = loss_type

    def training_losses(self, model, x_start, model_kwargs=None, noise=None, mask=None, t=None):
        """
        Compute training losses for a single timestep.
        Arguments format copied from opensora/schedulers/iddpm/gaussian_diffusion.py/training_losses
        Note: t is int tensor and should be rescaled from [0, num_timesteps-1] to [1,0]
        """
        if t is None:
            if self.sample_method == "uniform":
                t = torch.rand((x_start.shape[0],), device=x_start.device) * self.num_timesteps
            elif self.sample_method == "logit-normal":
                t = self.sample_t(x_start) * self.num_timesteps

        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape

        x_t = self.add_noise(x_start, noise, t)
        if mask is not None:
            t0 = torch.zeros_like(t)
            x_t0 = self.add_noise(x_start, noise, t0)
            x_t = torch.where(mask[:, None, :], x_t, x_t0)

        terms = {}
        x_pred = model(x_t, t, **model_kwargs)
        velocity_pred = x_pred - noise

        velocity = x_start - noise

        loss = self.compute_loss(velocity_pred, velocity, model_kwargs["x_mask_for_padding"] & mask if mask is not None else model_kwargs["x_mask_for_padding"])
        terms["loss"] = loss

        return terms

    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        compatible with diffusers add_noise()
        """
        timepoints = timesteps.float() / self.num_timesteps
        timepoints = 1 - timepoints  # [1,1/1000]
        timepoints = timepoints.unsqueeze(1).unsqueeze(1)
        return (1 - timepoints) * noise + timepoints * original_samples

    def compute_loss(self, pred, gt, mask):
        mask = mask.unsqueeze(-1)
        if self.loss_type == "l1":
            return F.l1_loss(pred * mask, gt * mask, reduction="sum") / (torch.sum(mask) * gt.shape[-1])
        elif self.loss_type == "mse":
            return F.mse_loss(pred * mask, gt * mask, reduction="sum") / (torch.sum(mask) * gt.shape[-1])
