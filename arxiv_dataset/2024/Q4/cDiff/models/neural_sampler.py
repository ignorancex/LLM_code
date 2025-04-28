import torch
from .summary import DeepSetSummary, BayesFlowEncoder, SetEmbedderClean
from .normalizing_flow import ConditionalNormalizingFlow
from .diffusion import KarrasSDE
from .utils import ScoreNetwork


class NormalizingFlowPosteriorSampler(torch.nn.Module):
    def __init__(self, y_dim, x_dim, n_summaries,
                 hidden_dim_decoder, n_flows_decoder,alpha,device,use_encoder,data_type="iid"):
        super().__init__()
        self.y_dim = y_dim
        self.x_dim = x_dim
        self.use_encoder = use_encoder
        self.n_summaries = n_summaries if self.use_encoder else y_dim

        if self.use_encoder:
            if data_type == "iid":
                self.summary = DeepSetSummary(y_dim, n_summaries).to(device)
                print("Encoder is for iid data. If not, please check it.")
            elif data_type == "time":
                self.summary = BayesFlowEncoder(y_dim, n_summaries).to(device)
                print("Encoder is for time dependent data. If not, please check it.")
            elif data_type == "set":
                num_head = 4
                num_seed = 4
                self.summary = SetEmbedderClean(y_dim, n_summaries, num_head, num_seed).to(device)
                print("Encoder is for time dependent data. If not, please check it.")
            else:
                raise ImportError("Other summary is not supported")
        else:
            pass

        self.decoder = ConditionalNormalizingFlow(x_dim=x_dim, y_dim=self.n_summaries, n_cond_layers=3,
                                                         hidden_dim=hidden_dim_decoder, n_params=2,
                                                         n_flows=n_flows_decoder,
                                                         alpha=alpha)

    def forward(self, x, y):
        # first dimension of y is number of samples in the batch
        # second dimension is number of data points per sample

        # get summary statistics
        s = self.summary(y) if self.use_encoder else y
        z, sum_log_abs_det = self.decoder(x=x, y=s)
        return s, z, sum_log_abs_det

    def backward(self, z, y):
        with torch.no_grad():
            s = self.summary(y) if self.use_encoder else y
            x = self.decoder.backward(z=z, y=s)
        return x

    @torch.no_grad()
    def sample(self, y):
        with torch.no_grad():
            s = self.summary(y) if self.use_encoder else y
            z = self.decoder.sample(s)
        return z

    def loss(self, x, y):
        s, z, sum_log_abs_det = self(x=x, y=y)
        return 0.5 * (z * z).sum(dim=1) - sum_log_abs_det

    @torch.no_grad()
    def sample_given_s(self, s):
        with torch.no_grad():
            z = self.decoder.sample(s)
        return z




class DiffusionPosteriorSampler(torch.nn.Module):
    def __init__(self, y_dim, x_dim, n_summaries,
                 num_hidden_layer,device,use_encoder, data_type="iid", sigma_data=0.5):
        super().__init__()
        self.y_dim = y_dim
        self.x_dim = x_dim
        self.use_encoder = use_encoder
        self.n_summaries = n_summaries if use_encoder else y_dim

        if self.use_encoder:
            if data_type == "iid":
                self.summary = DeepSetSummary(y_dim, n_summaries).to(device)
                print("Encoder is for iid data. If not, please check it.")
            elif data_type == "time":
                self.summary = BayesFlowEncoder(y_dim, n_summaries).to(device)
                print("Encoder is for time dependent data. If not, please check it.")
            elif data_type == "set":
                num_head = 4
                num_seed = 4
                self.summary = SetEmbedderClean(y_dim, n_summaries, num_head, num_seed).to(device)
                print("Encoder is for time dependent data. If not, please check it.")
            else:
                raise ImportError("Other summary is not supported")
        else:
            pass

        self.decoder = ScoreNetwork(
            x_dim=x_dim,
            hidden_dim=256,
            time_embed_dim=16,
            cond_dim=self.n_summaries,
            cond_mask_prob=0.0,
            num_hidden_layers=num_hidden_layer,
            output_dim=x_dim,
            device=device,
            cond_conditional=True).to(device)
        # self.diffusion = VariancePreservingSDE(action_dim=x_dim, state_dim=n_summaries, device=device)
        self.diffusion = KarrasSDE(theta_dim=x_dim, data_dim=self.n_summaries, device=device, sigma_data=sigma_data)

    @torch.no_grad()
    def sample(self, y, num_steps=18):
        with torch.no_grad():
            s = self.summary(y) if self.use_encoder else y
            # z, log_p,_ = self.diffusion.sample(self.decoder,s,num_steps)
            z = self.diffusion.edm_sampler(self.decoder, s, num_steps=num_steps)
        return z

    @torch.no_grad()
    def sample_given_s(self,s, num_steps=18):
        z = self.diffusion.edm_sampler(self.decoder, s, num_steps=num_steps)
        return z

    def loss(self, x, y):
        s = self.summary(y) if self.use_encoder else y
        # diffusion_loss = self.diffusion.diffusion_loss(self.decoder, x, s).mean()
        diffusion_loss = self.diffusion.diffusion_train_step(self.decoder, x, s)
        return diffusion_loss




if __name__ == "__main__":
    from models.utils import ScoreNetwork
    import torch.optim as optim
    import matplotlib.pyplot as plt
    import numpy as np


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
    model = DiffusionPosteriorSampler(y_dim=1, x_dim=2, n_summaries=4,
                 num_hidden_layer=4, device=device, use_encoder=False, data_type="iid")

    def plot_generated_data(data, filename):
        plt.figure(figsize=(8, 8))
        plt.scatter(data[:, 0].cpu().numpy(), data[:, 1].cpu().numpy(), s=10, color='blue')
        plt.title("Generated 2D Swiss Roll Data")
        plt.savefig(filename)
        plt.show()


    # Generate 2D Swiss roll data and condition
    # You might want to customize this data generation method
    x = generate_swiss_roll(num_samples=10000, noise=0.1)
    plot_generated_data(x, "swiss_roll.png")
    x = x.to(device)
    y = torch.ones((x.shape[0], 1), device=device)  # Condition as all 1s

    # Define the optimizer
    optimizer = optim.Adam(model.decoder.parameters(), lr=1e-4)


    # Training loop
    num_epochs = 10000
    for epoch in range(num_epochs):
        shuffled_indices = torch.randperm(y.size(0))  # Generate a random permutation of indices
        y = y[shuffled_indices]
        loss = model.loss(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

    # Generate data from the trained model using the sampler
    generated_data = model.sample(y, num_steps=18)
    plot_generated_data(generated_data.detach(), "generated_swiss_roll.png")