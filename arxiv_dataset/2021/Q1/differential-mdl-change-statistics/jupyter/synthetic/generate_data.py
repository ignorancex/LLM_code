import numpy as np

def generate_multiple_jumping_mean(N=10, n_seq=1000, coef=0.3, sigma=1.0, seed=0):
    np.random.seed(seed)
    x_list = [np.random.normal(0.0, sigma, n_seq)]
    for i in range(1, N+1):
        x_list.append(np.random.normal(coef*np.sum(np.arange(9, 10-i-1, -1)), sigma, n_seq))

    x = np.hstack(x_list)
    
    return x



def generate_multiple_jumping_variance(N=10, n_seq=1000, coef=0.1, mu=0.0, seed=0):
    np.random.seed(seed)
    x_list = []
    for i in range(N):
        x_list.append(np.random.normal(mu, np.exp(coef*np.sum(np.arange(9, 10-i-1, -1))), n_seq))

    x = np.hstack(x_list)
    
    return x


def generate_multiple_changing_mean_gradual(N=10, n_seq=1000, coef=0.3, sigma=1.0, win_slope=300, seed=0):
    np.random.seed(seed)
    x_list = [np.random.normal(0.0, sigma, n_seq)]
    for i in range(1, N+1):
        x_list.append(np.random.normal(coef*np.sum(np.arange(9, 10-i, -1)) + np.linspace(0.0, coef*(10-i), win_slope), sigma, win_slope))
        x_list.append(np.random.normal(coef*np.sum(np.arange(9, 10-i-1, -1)), sigma, n_seq-win_slope))

    x = np.hstack(x_list)
    
    return x


def generate_multiple_changing_variance_gradual(N=10, n_seq=1000, coef=0.1, mu=0.0, win_slope=300, seed=0):
    np.random.seed(seed)
    x_list = []
    for i in range(N):
        x_list.append(np.random.normal(mu, np.exp(coef*np.sum(np.arange(9, 10-i, -1)) + np.linspace(0.0, coef*(10-i), win_slope)), win_slope))
        x_list.append(np.random.normal(mu, np.exp(coef*np.sum(np.arange(9, 10-i-1, -1))), n_seq - win_slope))

    x = np.hstack(x_list)
    
    return x