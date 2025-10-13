import torch


def construct_mog_dataset(n_samples_per_class: int = 100):
    """Generate a MoG dataset on 2D, with two classes."""
    generator = torch.Generator()
    generator.manual_seed(0)

    means = [(0, 1), (-1, 0), (0, -1), (1, 0)]
    means = [torch.tensor(m) for m in means]
    var = 0.05

    inputs, labels = [], []

    for idx in range(len(means)):
        # Generate input data by mu + x @ sqrt(cov)
        cov = (var**0.5) * torch.eye(2)  # Diagonal covariance matrix
        mu = means[idx]
        inputs.append(mu + torch.randn(n_samples_per_class, 2, generator=generator) @ cov)

        # Labels
        labels.append(torch.tensor(n_samples_per_class * [1.0 if idx < 2 else 0.0]))

    return torch.cat(inputs, dim=0), torch.cat(labels, dim=0)
