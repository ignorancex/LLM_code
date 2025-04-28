import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../../")

from .BayesDataStream import BayesDataStream
from torch.utils.data import DataLoader
import torch

scale = 4.
def sample_theta():
    theta1 = np.random.uniform(-1,1)
    theta2 = np.random.uniform(-1,1)
    return {"theta1":theta1, "theta2":theta2}

def sample_cos_data(parms_dict, sample_size):
    theta1 = parms_dict["theta1"] * np.pi
    theta2 = parms_dict["theta2"] * np.pi
    mu = (np.cos(theta1 -theta2) + np.cos(2 * theta1 + theta2) + np.cos(3 * theta1 - 4 * theta2))
    y = np.random.normal(mu,1,sample_size)
    return y.reshape(-1,1) / scale
    # return mu.reshape(-1,1) / scale

def return_cos_dl(n_batches=256,batch_size=128, n_sample=None, return_ds=False):
    if n_sample is not None:
        def my_gen_sample_size(n, low=n_sample, high=n_sample+1):
            return np.random.randint(low=low, high=high, size=n)
    else:
        def my_gen_sample_size(n, low=100, high=1000):
            return np.random.randint(low=low, high=high, size=n)
    ds = BayesDataStream(n_batches, batch_size, sample_theta, sample_cos_data, my_gen_sample_size)
    dl = DataLoader(ds, batch_size=batch_size, num_workers=0, pin_memory=True)
    ds.reset_batch_sample_sizes()
    if return_ds:
        return dl,ds
    else:
        return dl

def unnormalized_posterior(theta1,theta2,y_observed):
    model = (np.cos(theta1 -theta2) + np.cos(2 * theta1 + theta2) + np.cos(3 * theta1 - 4 * theta2))

    likelihood = np.exp(-0.5 * (y_observed - model)**2)

    return likelihood

def plot_posterior(y_observed, save_path=None):
    theta1 = np.linspace(-np.pi, np.pi, 200)
    theta2 = np.linspace(-np.pi, np.pi, 200)

    theta1_grid, theta2_grid = np.meshgrid(theta1, theta2)
    # y_observed = y_observed.cpu().numpy().squeeze()

    posterior_values = unnormalized_posterior(theta1_grid, theta2_grid, y_observed)
    plt.figure(figsize=(8, 6))

    plt.contourf(theta1_grid, theta2_grid, posterior_values, levels=50, cmap='viridis')

    plt.colorbar(label='Unnormalized Posterior')

    plt.xlabel(r'$\theta_1$')

    plt.ylabel(r'$\theta_2$')

    plt.title('Unnormalized Posterior Distribution in $[-\pi, \pi]^2$')

    if save_path:
        plt.savefig(f"{save_path}/cos_gt_posterior.png")

def sample_and_plot(y_observe, model, save_path, device,model_type,sample_steps, seed):
    y_observe = torch.tensor(y_observe).float()
    y_observe = y_observe.unsqueeze(0).repeat(5000, 1).to(device)
    with torch.no_grad():
        if model_type == "NormalizingFlow":
            theta_sampled = model.sample(y_observe / scale) * np.pi
        else:
            theta_sampled = model.sample(y_observe / scale, sample_steps) * np.pi
        theta_sampled = theta_sampled.detach()

    likelihood = unnormalized_posterior(theta_sampled[:, 0].cpu().numpy(), theta_sampled[:, 1].cpu().numpy(), y_observe[0].item())
    log_likelihood = np.log(likelihood).sum()
    # Plot the first dimension against the second dimension
    plt.scatter(theta_sampled[:, 0].cpu().numpy(), theta_sampled[:, 1].cpu().numpy(), alpha=0.2,s=5, c="red")

    # Add labels and title
    plt.xlabel('Theta 1')
    plt.ylabel('Theta 2')
    plt.title(f'Posterior log likelihood is {log_likelihood}')

    # Save the plot to the specified path
    plt.savefig(f"{save_path}/estimate_posterior_{model_type}_overlap_{seed}.png")
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.scatter(theta_sampled[:, 0].cpu().numpy(), theta_sampled[:, 1].cpu().numpy(), alpha=0.5, s=10)
    plt.xlabel('Theta 1')
    plt.ylabel('Theta 2')
    plt.title('Scatter Plot of Sampled Theta')
    plt.savefig(f"{save_path}/estimate_posterior_{model_type}_{seed}.png")
    plt.close()

if __name__ == "__main__":
    # Set grid for theta1 and theta2

    # theta1 = np.linspace(-np.pi,np.pi,200)
    # theta2 = np.linspace(-np.pi,np.pi,200)
    #
    # theta1_grid, theta2_grid = np.meshgrid(theta1,theta2)
    #
    #
    # # Observed y value
    # y_observed = 0.5 # You can change this to any specific observed value.
    #
    #
    # # model is (theta \sim U(-pi, pi)^2)
    #
    # # y \mid \theta \sim N(f(\theta), 1) where f(theta) is a sum of complicated cosines
    #
    # # Calculate the unnormalized posterior (proportional to the likelihood)
    #
    # def unnormalized_posterior(theta1,theta2,y_observed):
    #     model = (np.cos_layer4(theta1 -theta2) + np.cos_layer4(2 * theta1 + theta2) + np.cos_layer4(3 * theta1 - 4 * theta2))
    #
    #     likelihood = np.exp(-0.5 * (y_observed - model)**2)
    #
    #     return likelihood
    #
    #
    # posterior_values = unnormalized_posterior(theta1_grid,theta2_grid,y_observed)
    #
    #
    # # Plot the unnormalized posterior
    # plt.figure(figsize=(8, 6))
    #
    # plt.contourf(theta1_grid, theta2_grid, posterior_values, levels=50, cmap='viridis')
    #
    # plt.colorbar(label='Unnormalized Posterior')
    #
    # plt.xlabel(r'$\theta_1$')
    #
    # plt.ylabel(r'$\theta_2$')
    #
    # plt.title('Unnormalized Posterior Distribution in $[-\pi, \pi]^2$')
    #
    # plt.savefig("../fig/cos_layer4.png")

    dl = return_cos_dl(n_batches=1,batch_size=10)
    for batch in dl:
        theta, y = batch
        print(theta)
        print(y)