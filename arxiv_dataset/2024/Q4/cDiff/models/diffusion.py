import torch
from .utils import *
from functools import partial
class VariancePreservingSDE(torch.nn.Module):
    """
    Implementation of the variance preserving SDE proposed by Song et al. 2021
    See eq (32-33) of https://openreview.net/pdf?id=PxTIG12RRHS
    """
    def __init__(self, action_dim, state_dim, device, beta_min=0.1, beta_max=20.0, T=1.0, t_epsilon=0.001):
        super().__init__()
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.T = T
        self.t_epsilon = t_epsilon
        self.device = device
        self.action_dim = action_dim
        self.state_dim = state_dim

    @property
    def logvar_mean_T(self):
        logvar = torch.zeros(1)
        mean = torch.zeros(1)
        return logvar, mean

    def beta(self, t):
        return self.beta_min + (self.beta_max-self.beta_min)*t

    def mean_weight(self, t):
        return torch.exp(-0.25 * t**2 * (self.beta_max-self.beta_min) - 0.5 * t * self.beta_min)

    def var(self, t):
        return 1. - torch.exp(-0.5 * t**2 * (self.beta_max-self.beta_min) - t * self.beta_min)

    def f(self, t, x_t):
        return - 0.5 * self.beta(t) * x_t

    def g(self, t, x_t):
        beta_t = self.beta(t)
        return beta_t**0.5

    def forward_process(self, t, x_0, return_noise=False):
        """
        sample yt | y0
        if return_noise=True, also return std and g for reweighting the denoising score matching loss
        """
        mu = self.mean_weight(t) * x_0
        std = self.var(t) ** 0.5
        epsilon = torch.randn_like(x_0)
        x_t = epsilon * std + mu
        if not return_noise:
            return x_t
        else:
            return x_t, epsilon, std, self.g(t, x_t)

    def diffusion_loss(self,snet, x_0, cond, t=None):
        if t is None:
            t = torch.rand([x_0.shape[0], 1]).to(x_0.device) * self.T
        x_t, target, std, g = self.forward_process(t, x_0, return_noise=True)
        s = snet(x_t, cond, t)

        # \eps(y,t)=\sigma*score, see cfg.
        return ((s * std + target) ** 2).view(x_0.size(0), -1).sum(1, keepdim=False) / 2

    # Drift of reverse sde
    # see Eq(6) and Eq(13), the coef of dt with negative sign.
    def mu(self, snet, t, x_t, cond, lmbd=1.):
        return - self.f(self.T - t, x_t) + \
            (1. - 0.5 * lmbd) * self.g(self.T - t, x_t).square() * snet(x_t, cond, self.T - t)

    # Diffusion
    def sigma(self, t, x_t, lmbd=0.):
        return (1. - lmbd) * self.g(self.T - t, x_t)

    @torch.enable_grad()
    def log_prob(self, snet, x_0, cond, vtype="rademacher"):
        """
        estimating the ELBO of the plug-in reverse SDE by sampling t uniformly between [0, T], and by estimating
        div(mu) using the Hutchinson trace estimator
        """
        t_ = torch.rand([x_0.size(0), ] + [1 for _ in range(x_0.ndim - 1)]).to(x_0) * self.T
        qt = 1 / self.T
        x_t = self.forward_process(t_, x_0).requires_grad_(True)

        s = snet(x_t, cond, t_)
        mu = self.g(t_, x_t).square() * s - self.f(t_, x_t)

        v = sample_v(x_0.shape, vtype=vtype).to(x_t)

        # \nable_{y}\mu * v
        Mu = - (
                torch.autograd.grad(mu, x_t, v, create_graph=self.training)[0] * v
        ).view(x_0.size(0), -1).sum(1, keepdim=False) / qt

        # ||s||_{gg^T}^2 = ||a||_2^2
        Nu = - ((self.g(t_, x_t) * s) ** 2).view(x_t.size(0), -1).sum(1, keepdim=False) / 2 / qt
        yT = self.forward_process(torch.ones_like(t_) * self.T, x_0)
        lp = log_normal(yT, torch.zeros_like(yT), torch.zeros_like(yT)).view(x_0.size(0), -1).sum(1)

        return lp + Mu + Nu

    #num_steps too small will make training bad
    @torch.enable_grad()
    def sample(self, snet, cond, num_steps=100, lmbd=1.,keep_all_samples=False):
        """
        Euler Maruyama method with a step size delta
        """
        # init
        batch_size = cond.size(0)
        delta = self.T / num_steps
        ts = torch.linspace(0, 1, num_steps + 1).to(self.device) * self.T

        # sample
        x_t = torch.randn((batch_size, self.action_dim)).to(self.device)
        t = torch.zeros((batch_size,1)).to(self.device)
        xs = []
        for i in range(num_steps):
            t.fill_(ts[i].item())
            mu = self.mu(snet, t, x_t, cond, lmbd=lmbd)
            sigma = self.sigma(t, x_t, lmbd=lmbd)
            x_t = x_t + delta * mu + delta ** 0.5 * sigma * torch.randn_like(
                x_t)  # one step update of Euler Maruyama method with a step size delta
            xs.append(x_t)
        log_prob = self.log_prob(snet, x_t,cond).reshape(-1,1)

        if keep_all_samples:
            return xs, log_prob, xs
        else:
            return x_t, log_prob , x_t


class KarrasSDE(nn.Module):
    def __init__(
            self,
            theta_dim: int,
            data_dim: int,
            device: str,
            sigma_data=0.5,
            sigma_min=0.002,
            sigma_max=80,
            sigma_sample_density_type: str = 'lognormal',
    ) -> None:
        super().__init__()

        self.device = device
        # use the score wrapper
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_sample_density_type = sigma_sample_density_type
        self.theta_dim = theta_dim
        self.data_dim = data_dim

    def get_diffusion_scalings(self, sigma):
        """
        Computes the scaling factors for diffusion training at a given time step sigma.

        Args:
        - self: the object instance of the model
        - sigma (float or torch.Tensor): the time step at which to compute the scaling factors

        , where self.sigma_data: the data noise level of the diffusion process, set during initialization of the model

        Returns:
        - c_skip (torch.Tensor): the scaling factor for skipping the diffusion model for the given time step sigma
        - c_out (torch.Tensor): the scaling factor for the output of the diffusion model for the given time step sigma
        - c_in (torch.Tensor): the scaling factor for the input of the diffusion model for the given time step sigma

        """
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        c_in = 1 / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        return c_skip, c_out, c_in

    def diffusion_train_step(self, model, x, cond, noise=None, t_chosen=None):
        """
        Computes the training loss and performs a single update step for the score-based model.

        Args:
        - self: the object instance of the model
        - x (torch.Tensor): the input tensor of shape (batch_size, dim)
        - cond (torch.Tensor): the conditional input tensor of shape (batch_size, cond_dim)

        Returns:
        - loss.item() (float): the scalar value of the training loss for this batch

        """
        model.train()
        x = x.to(self.device)
        cond = cond.to(self.device)
        if t_chosen is None:
            t_chosen = self.make_sample_density()(shape=(len(x),), device=self.device)

        loss = self.diffusion_loss(model, x, cond, t_chosen, noise)
        return loss

    def diffusion_loss(self, model, x, cond, t, noise):
        """
        Computes the diffusion training loss for the given model, input, condition, and time.

        Args:
        - self: the object instance of the model
        - x (torch.Tensor): the input tensor of shape (batch_size, channels, height, width)
        - cond (torch.Tensor): the conditional input tensor of shape (batch_size, cond_dim)
        - t (torch.Tensor): the time step tensor of shape (batch_size,)

        Returns:
        - loss (torch.Tensor): the diffusion training loss tensor of shape ()

        The diffusion training loss is computed based on the following equation from Karras et al. 2022:
        loss = (model_output - target)^2.mean()
        where,
        - noise: a tensor of the same shape as x, containing randomly sampled noise
        - x_1: a tensor of the same shape as x, obtained by adding the noise tensor to x
        - c_skip, c_out, c_in: scaling tensors obtained from the diffusion scalings for the given time step
        - t: a tensor of the same shape as t, obtained by taking the natural logarithm of t and dividing it by 4
        - model_output: the output tensor of the model for the input x_1, condition cond, and time t
        - target: the target tensor for the given input x, scaling tensors c_skip, c_out, c_in, and time t
        """
        if noise is None:
            noise = torch.randn_like(x)
        xt = x + noise * append_dims(t, x.ndim)
        c_skip, c_out, c_in = [append_dims(x, 2) for x in self.get_diffusion_scalings(t)]
        t = torch.log(t) / 4
        model_output = model(xt * c_in, cond, t)

        # denoised_x = c_out * model_output + c_skip * x_1
        target = (x - c_skip * xt) / c_out
        loss = (model_output - target).pow(2).mean()

        return loss

    def round_sigma(self, sigma):
        return torch.tensor(sigma)

    def denoise(self, model, xt, cond, t):
        c_skip, c_out, c_in = [append_dims(x, 2) for x in self.get_diffusion_scalings(t)]
        t = torch.log(t) / 4
        model_output = model(xt * c_in, cond, t)

        denoised_x = c_out * model_output + c_skip * xt
        return denoised_x

    def make_sample_density(self):
        """
        Returns a function that generates random timesteps based on the chosen sample density.

        Args:
        - self: the object instance of the model

        Returns:
        - sample_density_fn (callable): a function that generates random timesteps

        The method returns a callable function that generates random timesteps based on the chosen sample density.
        The available sample densities are:
        - 'lognormal': generates random timesteps from a log-normal distribution with mean and standard deviation set
                    during initialization of the model also used in Karras et al. (2022)
        - 'loglogistic': generates random timesteps from a log-logistic distribution with location parameter set to the
                        natural logarithm of the sigma_data parameter and scale and range parameters set during initialization
                        of the model
        - 'loguniform': generates random timesteps from a log-uniform distribution with range parameters set during
                        initialization of the model
        - 'uniform': generates random timesteps from a uniform distribution with range parameters set during initialization
                    of the model
        - 'v-diffusion': generates random timesteps using the Variational Diffusion sampler with range parameters set during
                        initialization of the model
        - 'discrete': generates random timesteps from the noise schedule using the exponential density
        - 'split-lognormal': generates random timesteps from a split log-normal distribution with mean and standard deviation
                            set during initialization of the model
        """
        sd_config = []

        if self.sigma_sample_density_type == 'lognormal':
            # loc = self.sigma_sample_density_mean  # if 'mean' in sd_config else sd_config['loc']
            # scale = self.sigma_sample_density_std  # if 'std' in sd_config else sd_config['scale']
            loc = - 1.2
            scale = 1.2
            return partial(rand_log_normal, loc=loc, scale=scale)

        if self.sigma_sample_density_type == 'loglogistic':
            loc = sd_config['loc'] if 'loc' in sd_config else math.log(self.sigma_data)
            scale = sd_config['scale'] if 'scale' in sd_config else 0.5
            min_value = sd_config['min_value'] if 'min_value' in sd_config else self.sigma_min
            max_value = sd_config['max_value'] if 'max_value' in sd_config else self.sigma_max
            return partial(rand_log_logistic, loc=loc, scale=scale, min_value=min_value, max_value=max_value)

        if self.sigma_sample_density_type == 'loguniform':
            min_value = sd_config['min_value'] if 'min_value' in sd_config else self.sigma_min
            max_value = sd_config['max_value'] if 'max_value' in sd_config else self.sigma_max
            return partial(rand_log_uniform, min_value=min_value, max_value=max_value)
        if self.sigma_sample_density_type == 'uniform':
            return partial(rand_uniform, min_value=self.sigma_min, max_value=self.sigma_max)

        if self.sigma_sample_density_type == 'v-diffusion':
            min_value = self.min_value if 'min_value' in sd_config else self.sigma_min
            max_value = sd_config['max_value'] if 'max_value' in sd_config else self.sigma_max
            return partial(rand_v_diffusion, sigma_data=self.sigma_data, min_value=min_value, max_value=max_value)
        if self.sigma_sample_density_type == 'discrete':
            sigmas = self.get_noise_schedule(self.n_sampling_steps, 'exponential')
            return partial(rand_discrete, values=sigmas)
        else:
            raise ValueError('Unknown sample density type')

    def edm_sampler(self, model, cond, randn_like=torch.randn_like,
            num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
            S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
    ):
        # Adjust noise levels based on what's supported by the network.
        sigma_min = max(sigma_min, self.sigma_min)
        sigma_max = min(sigma_max, self.sigma_max)

        latents = torch.randn((cond.size(0), self.theta_dim)).to(cond.device)

        # Time step discretization.
        step_indices = torch.arange(num_steps, device=latents.device)
        t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (
                    sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        t_steps = torch.cat([self.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])  # t_N = 0

        # Main sampling loop.
        x_next = latents * t_steps[0]
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N-1
            x_cur = x_next

            # Increase noise temporarily.
            gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
            t_hat = self.round_sigma(t_cur + gamma * t_cur)
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

            # Euler step.
            denoised = self.denoise(model, x_hat, cond, t_hat)
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            # Apply 2nd order correction.
            if i < num_steps - 1:
                denoised = self.denoise(model, x_next, cond, t_next)
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        return x_next

def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f'input has {x.ndim} dims but target_dims is {target_dims}, which is less')
    return x[(...,) + (None,) * dims_to_append]


def append_zero(action):
    return torch.cat([action, action.new_zeros([1])])


def rand_log_normal(shape, loc=0., scale=1., device='cpu', dtype=torch.float32):
    """Draws samples from a lognormal distribution."""
    return (torch.randn(shape, device=device, dtype=dtype) * scale + loc).exp()


def rand_log_logistic(shape, loc=0., scale=1., min_value=0., max_value=float('inf'), device='cpu', dtype=torch.float32):
    """Draws samples from an optionally truncated log-logistic distribution."""
    min_value = torch.as_tensor(min_value, device=device, dtype=torch.float64)
    max_value = torch.as_tensor(max_value, device=device, dtype=torch.float64)
    min_cdf = min_value.log().sub(loc).div(scale).sigmoid()
    max_cdf = max_value.log().sub(loc).div(scale).sigmoid()
    u = torch.rand(shape, device=device, dtype=torch.float64) * (max_cdf - min_cdf) + min_cdf
    return u.logit().mul(scale).add(loc).exp().to(dtype)


def rand_log_uniform(shape, min_value, max_value, device='cpu', dtype=torch.float32):
    """Draws samples from an log-uniform distribution."""
    min_value = math.log(min_value)
    max_value = math.log(max_value)
    return (torch.rand(shape, device=device, dtype=dtype) * (max_value - min_value) + min_value).exp()


def rand_uniform(shape, min_value, max_value, device='cpu', dtype=torch.float32):
    """Draws samples from an uniform distribution."""
    return torch.rand(shape, device=device, dtype=dtype) * (max_value - min_value) + min_value


def rand_discrete(shape, values, device='cpu', dtype=torch.float32):
    probs = [1 / len(values)] * len(values)  # set equal probability for all values
    return torch.tensor(np.random.choice(values, size=shape, p=probs), device=device, dtype=dtype)


def rand_v_diffusion(shape, sigma_data=1., min_value=0., max_value=float('inf'), device='cpu', dtype=torch.float32):
    """Draws samples from a truncated v-diffusion training timestep distribution."""
    min_cdf = math.atan(min_value / sigma_data) * 2 / math.pi
    max_cdf = math.atan(max_value / sigma_data) * 2 / math.pi
    u = torch.rand(shape, device=device, dtype=dtype) * (max_cdf - min_cdf) + min_cdf
    return torch.tan(u * math.pi / 2) * sigma_data


if __name__ == "__main__":
    from models.utils import ScoreNetwork
    import torch.optim as optim
    import matplotlib.pyplot as plt


    def generate_swiss_roll(num_samples=1000, noise=0.0):
        """
        Generate 2D Swiss roll dataset.

        Parameters:
        - num_samples: Number of data points to generate.
        - noise: Standard deviation of Gaussian noise added to the data.

        Returns:
        - data: A tensor of shape (num_samples, 2) containing the Swiss roll data.
        """
        t = np.linspace(0, 4 * np.pi, num_samples)
        x = t * np.cos(t)
        # y = t * np.sin(t)
        y = t * np.cos(t) * np.sin(t)

        # Add noise to the data
        x += np.random.normal(0, noise, size=x.shape)
        y += np.random.normal(0, noise, size=y.shape)

        data = np.stack([x, y], axis=1)  # Shape (num_samples, 2)

        # Convert to a PyTorch tensor
        return torch.tensor(data, dtype=torch.float32)

    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    model = ScoreNetwork(
        x_dim=2,
        hidden_dim=256,
        time_embed_dim=16,
        cond_dim=1,
        cond_mask_prob=0.0,
        num_hidden_layers=4,
        output_dim=2,
        device=device,
        cond_conditional=True).to(device)

    def plot_generated_data(data, filename):
        plt.figure(figsize=(8, 8))
        plt.scatter(data[:, 0].cpu().numpy(), data[:, 1].cpu().numpy(), s=10, color='blue')
        plt.title("Generated 2D Swiss Roll Data")
        plt.savefig(filename)
        plt.show()


    # Generate 2D Swiss roll data and condition
    # You might want to customize this data generation method
    data = generate_swiss_roll(num_samples=10000, noise=0.1)
    plot_generated_data(data, "swiss_roll.png")
    data = data.to(device)
    cond = torch.ones((data.shape[0], 1), device=device)  # Condition as all 1s

    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Define the diffusion process
    diffusion = KarrasSDE(theta_dim=1, data_dim=2, device=device)

    # Training loop
    num_epochs = 20000
    for epoch in range(num_epochs):
        shuffled_indices = torch.randperm(data.size(0))  # Generate a random permutation of indices
        data = data[shuffled_indices]
        loss = diffusion.diffusion_train_step(model, data, cond)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

    # Generate data from the trained model using the sampler
    generated_data = diffusion.edm_sampler(model, cond, num_steps=18)
    plot_generated_data(generated_data.detach(), "generated_swiss_roll.png")