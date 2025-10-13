import torch
from tqdm import tqdm

from .dps import PDESolverDPS


class PDESolverDPS_MultiRes(PDESolverDPS):
    def __init__(self, config):
        super().__init__(config)

        self.sigma_max_1 = config["sigma_max"]
        self.sigma_min_1 = config["guidance"]["sigma_min_1"]
        self.sigma_max_2 = config["guidance"]["sigma_max_2"]
        self.sigma_min_2 = config["sigma_min"]
        self.rho = config["rho"]

        self.weights = config["guidance"]["weights"]
        self.init_resolution = config["guidance"]["init_resolution"]
        self.upsampling_mode = config["guidance"]["upsampling_mode"]
        self.upsampling_step = config["guidance"]["upsampling_step"]

    def generate_latents(self, resolution=None):
        if resolution is None:
            resolution = self.resolution

        if self.config["init_latents"] == "white_noise":
            return torch.randn([self.batch_size, self.n_channels, resolution, resolution], device=self.device)
        elif self.config["init_latents"] == "rbf":
            if self.noise_sampler.Ln1 != resolution:
                from training.noise_samplers import RBFKernel

                self.noise_sampler = RBFKernel(self.n_channels, resolution, resolution, scale=self.config["rbf_scale"], device=self.device)
            return self.noise_sampler.sample(self.batch_size)
        else:
            raise ValueError("Invalid init_latents value")

    def interpolate_sample(self, x, target_res):
        # batch_size, channels, height, width = x.shape

        # Perform upsampling with specific size
        x_upsampled = torch.nn.functional.interpolate(
            x,
            size=(target_res, target_res),
            mode=self.upsampling_mode,
            align_corners=True,
        )
        return x_upsampled

    def generate_single_batch(self, observations, class_labels=None):
        """Generate a single batch of samples"""
        # Initialize latents
        latents = self.generate_latents(resolution=self.init_resolution)

        # Set up observations
        for obs in observations:
            obs.interpolation_mode = self.upsampling_mode

        # Set up sigma schedule
        step_indices = torch.arange(self.upsampling_step, dtype=torch.float64, device=self.device)
        sigma_t_steps = (self.sigma_max_1 ** (1 / self.rho) + step_indices / (self.upsampling_step - 1) * (self.sigma_min_1 ** (1 / self.rho) - self.sigma_max_1 ** (1 / self.rho))) ** self.rho
        sigma_t_steps = torch.cat([self.net.round_sigma(sigma_t_steps), torch.zeros_like(sigma_t_steps[:1])])

        num_steps_2 = self.num_steps - self.upsampling_step
        step_indices_2 = torch.arange(num_steps_2, dtype=torch.float64, device=self.device)
        sigma_t_steps_2 = (self.sigma_max_2 ** (1 / self.rho) + step_indices_2 / (num_steps_2 - 1) * (self.sigma_min_2 ** (1 / self.rho) - self.sigma_max_2 ** (1 / self.rho))) ** self.rho
        sigma_t_steps_2 = torch.cat([self.net.round_sigma(sigma_t_steps_2), torch.zeros_like(sigma_t_steps_2[:1])])

        sigma_t_steps = torch.cat([sigma_t_steps, sigma_t_steps_2])
        self.sigma_t_steps = sigma_t_steps

        # Initial state
        x_next = latents.to(torch.float64) * sigma_t_steps[0]
        intermediates = []
        loss_history = []

        for i, (sigma_t_cur, sigma_t_next) in enumerate(tqdm(zip(sigma_t_steps[:-1], sigma_t_steps[1:]), total=self.num_steps)):
            x_cur = x_next.detach().clone()

            if sigma_t_cur == 0 and i < self.num_steps - 1:
                print(f"\nUpsampling at step {i}: {x_cur.shape[-1]} x 2")
                target_res = 2 * x_cur.shape[-1]
                x_cur = self.interpolate_sample(x_cur, target_res=target_res)
                x_next = x_cur + self.generate_latents(resolution=target_res) * sigma_t_next
                continue

            x_cur.requires_grad_(True)
            sigma_t = self.net.round_sigma(sigma_t_cur)

            # Euler step
            x_N = self.net(x_cur, sigma_t, class_labels=class_labels).to(torch.float64)
            d_cur = (x_cur - x_N) / sigma_t
            x_next = x_cur + (sigma_t_next - sigma_t) * d_cur

            # 2nd order correction
            if sigma_t_next > 0:
                x_N = self.net(x_next, sigma_t_next, class_labels=class_labels).to(torch.float64)
                d_prime = (x_next - x_N) / sigma_t_next
                x_next = x_cur + (sigma_t_next - sigma_t) * (0.5 * d_cur + 0.5 * d_prime)

            denorm_x_N = self.normalizer.denormalize(x_N)
            denorm_x_next = self.normalizer.denormalize(x_next.detach())

            update = torch.zeros_like(x_cur)
            if sigma_t_next > 0:
                weight_ptr = 0
                step_losses = []
                active_losses = []

                for obs in observations:
                    loss = obs.get_observation_loss(denorm_x_N)
                    step_losses.append(loss.detach())

                    coef = self.get_coef(cur_step=i, obs_type=obs.type)
                    for c in range(obs.n_channels):
                        if coef != 0 and self.weights[weight_ptr] != 0:
                            active_losses.append((loss[:, c].sum(), coef * self.weights[weight_ptr]))  # NOTE: for numerical stability, we don't weight the loss here
                        weight_ptr += 1

                for idx, (loss, weight) in enumerate(active_losses):
                    flag_retain_graph = idx < len(active_losses) - 1
                    grad = torch.autograd.grad(loss, x_cur, retain_graph=flag_retain_graph)[0]
                    update = update + weight * grad

                loss_history.append(torch.cat(step_losses, dim=1))

            # Project update
            if getattr(self, "project_gradient", None) is not None:
                update = self.project_gradient(x_next, update)

            # Apply updates
            x_next = x_next - update

            if x_next.isnan().any():
                print(f"\nStep {i}: NaN detected!")
                break

            # Save intermediate results if requested
            if self.save_indices is not None and i in self.save_indices:
                denorm_x_updated = self.normalizer.denormalize(x_next.detach())
                intermediates.append(torch.cat([denorm_x_N.detach(), denorm_x_next, denorm_x_updated], dim=1))

        x_final = x_next.detach()
        if x_final.shape[-1] != self.resolution:
            assert x_final.shape[-1] > self.resolution
            print(f"Final downsampling from {x_final.shape[-1]} to {self.resolution}")
            x_final = self.interpolate_sample(x_final, target_res=self.resolution)
        pred = self.normalizer.transform(x_final, denormalize=True)
        aux = {"intermediates": intermediates, "loss_history": loss_history}
        return pred, aux

    def get_coef(self, cur_step, obs_type):
        if self.sigma_t_steps[cur_step] > 1.0:
            if obs_type == "pde":
                return 0
            else:
                return 1
        else:
            return self.sigma_t_steps[cur_step].item()
