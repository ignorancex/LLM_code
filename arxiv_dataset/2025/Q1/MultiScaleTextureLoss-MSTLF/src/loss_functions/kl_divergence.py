import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt


def to_hounsfield(ct, min, max):
    return torch.round(0.5*(ct+1) * (max - min) + min)


def kl_divergence(p, q):
    """
    Compute the Kullback-Leibler Divergence between two probability distributions.

    Parameters:
    - p, q: torch Tensors representing the probability distributions.

    Returns:
    - KL Divergence between p and q.
    """
    # Ensure that the tensors have the same shape

    # Avoid division by zero by adding a small epsilon
    # epsilon = 1e-10
    # p = p + epsilon
    # q = q + epsilon

    # Compute KL Divergence
    kl_div = torch.sum(p * torch.log(p / q))

    return kl_div


class Histogramming(nn.Module):
    def __init__(self, num_bins, min_val, max_val, sigma=0.5):
        super(Histogramming, self).__init__()
        self.num_bins = num_bins
        self.sigma = sigma
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x):
        # print(self.min_val, self.max_val)
        # print(torch.round(x.min()).item(), torch.round(x.max()).item())
        edges = torch.linspace(torch.round(x.min()).item(), torch.round(x.max()).item(), self.num_bins + 1)
        # edges = torch.linspace(self.min_val, self.max_val, self.num_bins + 1)  # Define histogram bin edges
        x = x.unsqueeze(-1)
        centers = edges.to(x.device)
        #######
        # Calculate distances to each bin edge
        # distances = x.unsqueeze(-1) - centers.unsqueeze(0)

        # Apply Gaussian kernel to distances
        # kernel_values = torch.exp(-0.5 * (distances / self.sigma)**2)
        #####
        exponent = -((x - centers) ** 2) / (2 * self.sigma ** 2)

        kernel_values = torch.exp(exponent)

        sum_kernel_values = kernel_values.sum(dim=-1, keepdim=True)

        normalized_kernel_values = kernel_values / sum_kernel_values

        # Calculate the histogram by summing the weighted kernel values
        histogram = torch.sum(normalized_kernel_values, dim=-2)
        sum_histogram = torch.sum(histogram) * (edges[1] - edges[0])
        histogram /= sum_histogram

        return histogram


def compute_kl_divergence(image1, image2, min_val, max_val, hist):
    # print(min_val, max_val)
    # Go back to HU
    image1_hu = to_hounsfield(image1, min_val, max_val)
    image2_hu = to_hounsfield(image2, min_val, max_val)

    # Flatten the CT tensor images
    flat_image1 = image1_hu.flatten()
    flat_image2 = image2_hu.flatten()

    nhist1, _ = np.histogram(flat_image1.detach(), range=(torch.round(flat_image1.min()).item(), torch.round(flat_image1.max()).item()), bins=100, density=True)
    nhist2, _ = np.histogram(flat_image2.detach(), range=(-1200, -400), bins=100, density=True)
    # print(kl_divergence(torch.tensor(nhist1), torch.tensor(nhist2)))

    plt.figure()
    plt.plot(nhist1)
    plt.plot(nhist2)
    plt.show()
    plt.close()

    # Compute histograms
    pdf_1 = hist(flat_image1)
    pdf_2 = hist(flat_image2)

    plt.figure()
    plt.plot(pdf_1.detach())
    plt.plot(pdf_2.detach())
    plt.show()
    plt.close()
    # Compute KL divergence
    kl = kl_divergence(pdf_1, pdf_2)

    return kl


if __name__ == "__main__":
    import torch

    a = torch.rand(2, 1, 256, 256)
    a = torch.ones(2, 1, 256, 256)
    kl = compute_kl_divergence(a, a, 0, 1)
    print(kl)
