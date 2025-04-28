import torch
from tqdm import tqdm

import gpytorch
from gpytorch.kernels import SpectralMixtureKernel

from online_gp import models
from online_gp.models.stems import Identity
from online_gp.utils.cuda import try_cuda


# 模块化函数
def generate_samples(theta, num_samples):
    cov_matrix = torch.eye(1, dtype=torch.float32) * 0.6
    weights = torch.ones(2, dtype=torch.float32) / 2.0
    location = torch.tensor([[2.0], [-1.0]])
    if theta.dim() == 1:
        num_theta = 1
    else:
        num_theta, dim_theta = theta.shape
    likelihood = torch.distributions.MultivariateNormal(torch.tensor([0.0]), cov_matrix)
    if num_theta == 1:
        mode = torch.multinomial(weights, num_samples, replacement=True)
        samples = theta + location[mode, ] + likelihood.sample((num_samples,))
    elif num_samples == 1:
        mode = torch.multinomial(weights, num_theta, replacement=True)
        samples = theta + location[mode, ] + likelihood.sample((num_theta,))
    else:
        samples = torch.empty((num_theta, num_samples, 1))
        for k in range(num_theta):
            mode = torch.multinomial(weights, num_samples, replacement=True)
            samples[k, :, :] = theta[k, :] + location[mode, ] + likelihood.sample((num_samples,))
    return samples


def simulator_discrepancy(theta, num_samples):
    samples = generate_samples(theta, num_samples)
    return torch.abs(samples - y_obs)


def calculate_log_kernel(xx):
    if xx < epsilon:
        return torch.log(torch.tensor(1 / (2 * epsilon)))
    else:
        return torch.log(torch.tensor([1.0e-10]))


def normalize_fun(theta):
    re = (theta - (6 - 6) / 2) / ((6 + 6) / 2)
    return re


m = 2000
theta0 = torch.linspace(start=-6, end=6, steps=m).view(-1, 1)
Delta0 = torch.abs(generate_samples(theta0, 1) - torch.tensor([1.0]))
train_data = torch.cat((theta0, Delta0), dim=1)

init_theta = normalize_fun(theta0)
init_Delta = Delta0
# Train Model
stem = Identity(input_dim=1)
covar_module: SpectralMixtureKernel = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=2, ard_num_dims=1)
wiski_model = models.OnlineSKIRegression(stem, init_theta, init_Delta, lr=1e-1, grid_size=128, grid_bound=1,
                                         covar_module=covar_module)
wiski_model = try_cuda(wiski_model)
wiski_model.fit(init_theta, init_Delta, 200)  # pretrain model
wiski_model.eval()
# Updatad Model
# New_theta = torch.linspace(start=-3, end=3, steps=1000).view(-1, 1)
# New_Delta = torch.abs(generate_samples(New_theta, 1)-torch.tensor([1.0]))
# wiski_model.update(normalize_fun(New_theta), New_Delta)


Prior = torch.distributions.Uniform(-6, 6)
Local_proposal = torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([0.3]))
y_obs = torch.tensor([1.0])
epsilon = 0.6
num_ite = 50000
batch_size = 20
Initial = torch.tensor([0.0])
d_old = simulator_discrepancy(Initial, 1)
while d_old > epsilon:
    Initial = Prior.sample([1])
    d_old = simulator_discrepancy(Initial, 1)
Theta_Re = torch.empty(num_ite, 1)
d_Re = torch.empty(num_ite, 1)
Theta_Re[0, :] = Initial
Lambda = 2
Theta_New = torch.empty(1, 1)
d_New = torch.empty(1, 1)
if __name__ == '__main__':
    for i in tqdm(range(0, num_ite)):
        if i % 10000 == 0:  # Condition for updating GP model
            wiski_model.update(normalize_fun(Theta_New), d_New)
            Theta_New = torch.empty(1, 1)
            d_New = torch.empty(1, 1)
        Theta_Re[i, :] = Theta_Re[i - 1, :]
        Theta_prop = Local_proposal.sample([1]) + Theta_Re[i - 1, :].view(1, -1)
        log_w = torch.log(torch.rand(1))
        log_acc1 = Prior.log_prob(Theta_prop) - \
                   Prior.log_prob(Theta_Re[i - 1, :].view(1, -1)) - calculate_log_kernel(d_old)
        if log_w < log_acc1:
            pre_mean, pre_var = wiski_model.predict(normalize_fun(Theta_prop))
            hat_d = max(pre_mean - Lambda * pre_var ** (1 / 2), torch.zeros(1, 1))
            hat_log_acc = Prior.log_prob(Theta_prop) + calculate_log_kernel(hat_d) - \
                          Prior.log_prob(Theta_Re[i - 1, :].view(1, -1)) - calculate_log_kernel(d_old)
            if log_w < hat_log_acc:  # early rejection step based on GP
                d = simulator_discrepancy(Theta_prop, 1)
                log_acc = Prior.log_prob(Theta_prop) + calculate_log_kernel(d) - \
                          Prior.log_prob(Theta_Re[i - 1, :].view(1, -1)) - calculate_log_kernel(d_old)
                if log_w < log_acc:
                    Theta_Re[i, :] = Theta_prop
                    d_old = d
                # collect new training data
                Theta_New = torch.cat((Theta_New, Theta_prop), dim=0)
                d_New = torch.cat((d_New, d), dim=0)
# posterior sample Theta_Re
