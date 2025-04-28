import numpy as np
from .BayesDataStream import BayesDataStream
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import warnings
import sdeint
from scipy.integrate import odeint

# Convert RuntimeWarning to an exception
warnings.simplefilter("error", category=RuntimeWarning)

#
# def simulate_lotka_volterra(alpha, beta, gamma, delta, Q, R, initial_state, delta_t, num_steps):
#     """
#     Simulate the Lotka-Volterra predator-prey model in state-space form.
#
#     Parameters:
#     alpha (float): Natural growth rate of prey
#     beta (float): Rate of predation upon the prey
#     gamma (float): Natural death rate of predators
#     delta (float): Rate at which predators increase by consuming prey
#     Q (ndarray): Covariance matrix of process noise (2x2)
#     R (ndarray): Covariance matrix of observation noise (2x2)
#     initial_state (ndarray): Initial state vector [prey_population, predator_population]
#     delta_t (float): Time step
#     num_steps (int): Number of time steps to simulate
#
#     Returns:
#     x (ndarray): State vector at each time step (num_steps+1 x 2)
#     y (ndarray): Observation vector at each time step (num_steps x 2)
#     """
#     # Initialize state and observation arrays
#     x = np.zeros((num_steps + 1, 2))
#     y = np.zeros((num_steps, 2))
#
#     # Set the initial state
#     x[0] = initial_state
#
#     # Simulate the state-space model
#     for t in range(num_steps):
#         # Process noise
#         v_t = np.random.multivariate_normal([0, 0], Q)
#
#         # State transition
#         prey_t = x[t, 0]
#         predator_t = x[t, 1]
#         try:
#             x[t + 1, 0] = prey_t + (alpha * prey_t - beta * prey_t * predator_t) * delta_t + v_t[0]
#             x[t + 1, 1] = predator_t + (delta * prey_t * predator_t - gamma * predator_t) * delta_t + v_t[1]
#
#             x[t + 1, 0] = max(x[t + 1, 0], 0.)
#             x[t + 1, 1] = max(x[t + 1, 1], 0.)
#         except RuntimeWarning as e:
#             print(f"Warning captured: {e}")
#             print(alpha, beta, gamma, delta)
#             # Capture the current state of relevant variables
#             print(prey_t, predator_t, v_t[0], v_t[1])
#
#         # Observation noise
#         e_t = np.random.multivariate_normal([0, 0], R)
#
#         # Observation
#         y[t, 0] = x[t + 1, 0] + e_t[0]
#         y[t, 1] = x[t + 1, 1] + e_t[1]
#
#         y[t, 0] = max(y[t, 0],0)
#         y[t, 1] = max(y[t, 1],0)
#
#     return x, y


# dX = f(X,t)dt + g(X,t)dW
def simulate_lotka_volterra(X0, t_final, size, a, b, c, d, Q, R, sde=False):
    t = np.linspace(0, t_final, size)

    def f(X, t):
        # return np.array([a * X[0] - b * X[0] * X[1], -c * X[1] + d * b * X[0] * X[1]])
        return np.array([a * X[0] - b * X[0] * X[1], -c * X[1] + d * X[0] * X[1]])

    def g(X, t):
        return Q

    obs_noise = np.random.multivariate_normal([0, 0], R, size=(size,))
    if sde:
        data = sdeint.itoint(f, g, X0, t) + obs_noise
    else:
        data = odeint(f, X0, t) + obs_noise

    return data


def sample_theta():
    # Lotka-Volterra model parameters
    alpha = np.random.uniform(0.5, 1.0)
    beta = np.random.uniform(0.01, 0.1)
    gamma = np.random.uniform(0.01, 0.5)  # Centered around 0.3 with small variance
    delta = np.random.uniform(0.005, 0.05)  # Centered around 0.01 with small variance

    # Process noise covariance matrix Q parameters
    sigma_q1 = np.random.normal(0.05, 0.01)  # Standard deviation for prey process noise
    sigma_q2 = np.random.normal(0.05, 0.01)  # Standard deviation for predator process noise

    # Observation noise covariance matrix R parameters
    sigma_r1 = np.random.gamma(5, 1.0 / 5)  # Standard deviation for prey observation noise
    sigma_r2 = np.random.gamma(5, 1.0 / 5)  # Standard deviation for predator observation noise
    rho_r = np.random.uniform(-0.1, 0.1)  # Small correlation between prey and predator observation noise
    covariance_r = rho_r * sigma_r1 * sigma_r2  # Off-diagonal covariance

    return {
        'alpha': alpha,
        'beta': beta,
        'gamma': gamma,
        'delta': delta,
        'sigma_r1': sigma_r1,  # Standard deviation for prey observation noise
        'sigma_r2': sigma_r2,  # Standard deviation for predator observation noise
        'covariance_r': covariance_r  # Off-diagonal covariance for observation noise
    }


def sample_y(parms, n, delta_t=0.1, initial_state=(20, 5), sde=True):
    # Extract parameters from the dictionary
    alpha = parms['alpha']
    beta = parms['beta']
    gamma = parms['gamma']
    delta = parms['delta']

    # process noise -- for now basically nothing
    Q = np.array([
        [0.0001, 0.],
        [0., 0.0001]
    ])

    # Observation noise
    # Reconstruct R matrix
    sigma_r1 = parms['sigma_r1']
    sigma_r2 = parms['sigma_r2']
    covariance_r = parms['covariance_r']

    R = np.array([
        [sigma_r1 ** 2, covariance_r],
        [covariance_r, sigma_r2 ** 2]
    ])

    # Simulate the Lotka-Volterra system
    y = simulate_lotka_volterra(initial_state, t_final=30, size=n, a=alpha, b=beta, c=gamma, d=delta, Q=Q, R=R, sde=False)

    # _, y = simulate_lotka_volterra(alpha, beta, gamma, delta, Q, R, initial_state, delta_t, n)
    return y  # Return the observation matrix of size [L, 2]





# Sample size generator
def my_gen_sample_size(n, low=200, high=500):
    return np.random.randint(low=low, high=high, size=n)



def return_lotka_volterra_dl(n_batches = 256,batch_size = 128,n_sample=None,return_ds=False, num_workers=4):
    if n_sample is not None:
        def my_gen_sample_size(n, low=n_sample, high=n_sample+1):
            return np.random.randint(low=low, high=high, size=n)
    else:
        def my_gen_sample_size(n, low=200, high=500):
            return np.random.randint(low=low, high=high, size=n)
    ds = BayesDataStream(n_batches, batch_size, sample_theta, sample_y, my_gen_sample_size)
    dl = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    ds.reset_batch_sample_sizes()
    if return_ds:
        return dl,ds
    else:
        return dl


if __name__ == "__main__":
    import sys
    import time
    sys.path.append("..")  # Add the parent directory to the system path
    from my_model import BayesFlowEncoder  # Import the class from the module

    start_time = time.time()
    y_dim = 2
    embedding_dim = 64
    encoder = BayesFlowEncoder(y_dim, embedding_dim)
    dl,ds = return_lotka_volterra_dl(n_batches=2,batch_size=128,return_ds=True,n_sample=500)
    for batch in dl:
        theta, y = batch
        s = encoder(y)
        print(theta.shape)
        print(y.shape)
        print(s.shape)

    # use sde: 2 batch 128*500*2 data use 10s
    # use sde with 4 workers: 2 batch 128*500*2 data use 5.5s
    # James: 2 batch 128*500*2 data use 26.5s
    # James with 4 worker: 2 batch 128*500*2 data use 15.5s
    # use ode with 4 workers: 2 batch 128*500*2 data use 3 s
    print(f"Elps time is {time.time() - start_time} s.")

    theta, y = ds.__getitem__(3)
    print(y.shape)

    prey_population = y[:, 0].numpy()  # Convert to numpy array if it's a tensor
    predator_population = y[:, 1].numpy()

    # Create a time vector
    time_steps = np.arange(len(prey_population))

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(time_steps, prey_population, label='Prey Population', color='blue')
    plt.plot(time_steps, predator_population, label='Predator Population', color='red')
    plt.title('Predator and Prey Populations Over Time')
    plt.xlabel('Time Steps')
    plt.ylabel('Population')
    plt.legend()
    plt.grid(True)
    plt.savefig("../test/lotka_volterra.png")