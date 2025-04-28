import numpy as np
# Simulate Data
import numpy as np
from math import sqrt

# Generate fixed theta* in R^d
def generate_theta_star(dim):
    rng1 = np.random.RandomState(0)
    theta_star = rng1.randn(dim)  # Random parameters
    theta_star = theta_star / np.linalg.norm(theta_star)  # Make norm = 1
    return (theta_star)

# ---------------------------------LINEAR REGRESSION FUNCTIONS---------------------------
# Simulate training data for linear simulation
def sim_data_lin(dim, eps, n, rng, theta_star):
    b = rng.binomial(size=n, n=1, p=eps)  # (n,)
    # Normalize by dimension to small norm
    x = rng.normal(0, 1, size=(n, dim)) / dim  # (n, d)
    mu = (1-b) * rng.normal(0, 1, size=n) + b * \
        rng.normal(0, sqrt(10), size=n)  # (n,)
    y = np.matmul(x, theta_star) + mu
    return x, y

# Simulate contaminated point for linear simulation
def sim_contaminated_lin(dim, n, rng, theta_star):
    x = rng.normal(0, 1, size=(n, dim)) / dim  # (n, d)
    mu = rng.normal(0, sqrt(10), size=n)  # (n,)
    y = np.matmul(x, theta_star) + mu  # (n,)
    return x, y

# Runs n_sim number of simulations computing population and empirical IF on one contaminated point
def if_diff_sim_lin(dim, x_sim, y_sim, oracle_theta_dict, lambda_, n, n_sim, rng2, theta_star, if_emp, if_pop):
    oracle_theta = oracle_theta_dict[lambda_]
    if_emp_ls = []
    if_pop_ls = []
    x_con_ls, y_con_ls = sim_contaminated_lin(dim, n_sim, rng2, theta_star)
    for i in range(n_sim):
        x_con = x_con_ls[i]
        y_con = y_con_ls[i]
        if_emp_ls.append(if_emp(x_sim, y_sim, x_con, y_con, lambda_, n))
        if_pop_ls.append(if_pop(dim, x_con, y_con, oracle_theta, lambda_))
    return (if_emp_ls, if_pop_ls)

# Run simulations for different data size n
def run_sim_lin(n_sim, eps, n_ls, lambda_ls, oracle_theta_dict, dim, seed, theta_star, if_emp, if_pop):
    mean_diff_abs_total = []
    sd_diff_abs_total = []
    n_samp = []
    lambda_ = []
    rng = np.random.RandomState(1)
    rng2 = np.random.RandomState(2)
    for n in n_ls:
        x_sim, y_sim = sim_data_lin(dim, eps, n, rng, theta_star)
        for l in lambda_ls:
            if_emp_ls, if_pop_ls = if_diff_sim_lin(
                dim, x_sim, y_sim, oracle_theta_dict, l, n, n_sim, rng2, theta_star, if_emp, if_pop)
            bound_val = []
            for i in range(len(if_emp_ls)):
                diff_abs_total = np.abs(if_emp_ls[i]-if_pop_ls[i])
                bound_val.append(np.dot(np.matmul(diff_abs_total, np.eye(
                    dim) * (1 / dim**2 + l)), np.transpose(diff_abs_total)))
            mean_diff_abs_total.append(np.mean(bound_val))
            sd_diff_abs_total.append(np.std(bound_val))
            n_samp.append(n)
            lambda_.append(l)
    return (mean_diff_abs_total, sd_diff_abs_total, n_samp, lambda_)

# ---------------------------------LOGISTIC REGRESSION FUNCTIONS---------------------------
# Simulate training data for logistic simulation
def sim_data_log(dim, eps, n, rng, theta_star):
    b = rng.binomial(size=n, n=1, p=eps)
    # Normalize by dimension to small norm
    x = rng.normal(0, 1, size=(n, dim)) / dim  # (n, d)
    mu = (1-b) * rng.normal(0, 1, size=n) + b * \
        rng.normal(0, sqrt(10), size=n)  # (n,)
    prob = 1/(1+np.exp(-1*(np.matmul(x, theta_star) + mu)))
    y = rng.binomial(size=n, n=1, p=prob)
    return (x, y)

# Simulate contaminated point for logistic simulation
def sim_contaminated_log(dim, n, rng, theta_star):
    x = rng.normal(0, 1, size=(n, dim)) / dim  # (n, d)
    mu = rng.normal(0, sqrt(10), size=n)  # (n,)
    prob = 1/(1+np.exp(-1*(np.matmul(x, theta_star) + mu)))
    y = np.array([rng.binomial(size=n, n=1, p=prob)])
    return (x, y)

# Runs n_sim number of simulations computing population and empirical IF on one contaminated point
def if_diff_sim_log(dim, x_sim, y_sim, x_pop, y_pop, n_sim, rng, theta_star, emp_if_fn):
    if_emp_ls = []
    if_pop_ls = []
    x_con_ls, y_con_ls = sim_contaminated_log(dim, n_sim, rng, theta_star)
    for i in range(n_sim):
        x_con = x_con_ls[i].reshape([1, len(x_con_ls[i])])
        y_con = y_con_ls[0][i]
        if_emp, H_emp = emp_if_fn(x_sim, y_sim, x_con, y_con)
        if_emp_ls.append(if_emp)
        # Using a large sample size to generate a "population" IF
        if_pop, H_pop = emp_if_fn(x_pop, y_pop, x_con, y_con)
        if_pop_ls.append(if_pop)
    return (if_emp_ls, if_pop_ls, H_pop)

# Run simulations for different data size n
def run_sim_log(dim, eps, n_pop, n_ls, n_sim, emp_if_fn, theta_star):
    mean_diff_abs_total = []
    sd_diff_abs_total = []
    n_samp = []

    rng = np.random.RandomState(1)
    rng2 = np.random.RandomState(2)
    rng3 = np.random.RandomState(3)

    # Get population sample
    y_pop = {}
    while set(y_pop) != {0, 1}:
        x_pop, y_pop = sim_data_log(dim, eps, n_pop, rng2, theta_star)

    for n in n_ls:
        y_sim = {}
        # Make sure this is at least 1 of each category (in order to logistic regression)
        while set(y_sim) != {0, 1}:
            x_sim, y_sim = sim_data_log(dim, eps, n, rng, theta_star)
        if_emp_ls, if_pop_ls, H_pop = if_diff_sim_log(
            dim, x_sim, y_sim, x_pop, y_pop, n_sim, rng3, theta_star, emp_if_fn)
        bound_val = []
        for i in range(len(if_emp_ls)):
            diff_abs_total = np.abs(if_emp_ls[i]-if_pop_ls[i])
            bound_val.append(
                np.dot(np.matmul(diff_abs_total, H_pop), np.transpose(diff_abs_total)))
        mean_diff_abs_total.append(np.mean(bound_val))
        sd_diff_abs_total.append(np.std(bound_val))
        n_samp.append(n)
    return (x_pop, y_pop, mean_diff_abs_total, n_samp, sd_diff_abs_total)
