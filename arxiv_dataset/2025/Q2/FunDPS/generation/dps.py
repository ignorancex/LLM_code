import torch
from tqdm import tqdm

from .base import PDESolver


class PDESolverDPS(PDESolver):
    def __init__(self, config):
        super().__init__(config)

        self.sigma_min = config["sigma_min"]
        self.sigma_max = config["sigma_max"]
        self.rho = config["rho"]

        self.weights = config["guidance"]["weights"]

    def load_data(self):
        super().load_data()
        self.normalizer = self.dataset.create_normalizer()

    def generate_single_batch(self, observations, class_labels=None):
        """Generate a single batch of samples"""
        # Initialize latents
        latents = self.generate_latents()

        # Set up sigma schedule
        step_indices = torch.arange(self.num_steps, dtype=torch.float64, device=self.device)
        sigma_t_steps = (self.sigma_max ** (1 / self.rho) + step_indices / (self.num_steps - 1) * (self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho))) ** self.rho
        sigma_t_steps = torch.cat([self.net.round_sigma(sigma_t_steps), torch.zeros_like(sigma_t_steps[:1])])
        self.sigma_t_steps = sigma_t_steps

        # Initial state
        x_next = latents.to(torch.float64) * sigma_t_steps[0]
        intermediates = []
        loss_history = []

        for i, (sigma_t_cur, sigma_t_next) in enumerate(tqdm(zip(sigma_t_steps[:-1], sigma_t_steps[1:]), total=self.num_steps)):
            x_cur = x_next.detach().clone()
            x_cur.requires_grad_(True)
            sigma_t = self.net.round_sigma(sigma_t_cur)

            # Euler step
            x_N = self.net(x_cur, sigma_t, class_labels=class_labels).to(torch.float64)
            d_cur = (x_cur - x_N) / sigma_t
            x_next = x_cur + (sigma_t_next - sigma_t) * d_cur

            # 2nd order correction
            if i < self.num_steps - 1:
                x_N = self.net(x_next, sigma_t_next, class_labels=class_labels).to(torch.float64)
                d_prime = (x_next - x_N) / sigma_t_next
                x_next = x_cur + (sigma_t_next - sigma_t) * (0.5 * d_cur + 0.5 * d_prime)

            denorm_x_N = self.normalizer.denormalize(x_N)
            denorm_x_next = self.normalizer.denormalize(x_next.detach())

            update = torch.zeros_like(x_cur)
            if i < self.num_steps - 1:
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
