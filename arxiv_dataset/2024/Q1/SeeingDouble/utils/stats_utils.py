import numpy as np


def gaussian_pdf(x, mu, sigma):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def symmetrized_exp_pdf(x, mu, sigma):
    return 0.5 * np.exp(-np.abs(x - mu) / sigma) / sigma


def gaussian_log_likelihood_function(z1_prime, z2_prime, z1, z2, sigma1, sigma2, Lambda):

    z1_term = -(z1_prime - z1) ** 2 / (2 * sigma1 ** 2) + np.log(1 / (sigma1 * np.sqrt(2 * np.pi)))
    z2_term = -(z2_prime - z2) ** 2 / (2 * sigma2 ** 2) + np.log(1 / (sigma2 * np.sqrt(2 * np.pi)))

    correlation = -(z1_prime - z2_prime) ** 2 / (2 * Lambda ** 2) + np.log(1 / (Lambda * np.sqrt(2 * np.pi)))

    return z1_term + z2_term + correlation


def symmetrized_exp_log_likelihood_function(z1_prime, z2_prime, z1, z2, sigma1, sigma2, Lambda_e):

    z1_term = -(z1_prime - z1) ** 2 / (2 * sigma1 ** 2) + np.log(1 / (sigma1 * np.sqrt(2 * np.pi)))
    z2_term = -(z2_prime - z2) ** 2 / (2 * sigma2 ** 2) + np.log(1 / (sigma2 * np.sqrt(2 * np.pi)))

    correlation = -np.abs(z1_prime - z2_prime) / Lambda_e - np.log(2 * Lambda_e)

    return z1_term + z2_term + correlation





def correlation_improved_estimates(z1, z2, sigma1, sigma2, Lambda, mode):

    # mode should either be "GAUSSIAN" or "EXPONENTIAL"

    if mode == "GAUSSIAN":

        sigma = np.sqrt(sigma1**2 + sigma2**2 + Lambda**2)

        def calc_z1_prime(z1, z2, sigma1, sigma2, Lambda):
            return (z1*(Lambda**2 + sigma2**2) + z2*sigma1**2)/(sigma**2)

        def calc_z2_prime(z1, z2, sigma1, sigma2, Lambda):
            return (z2*(Lambda**2 + sigma1**2) + z1*sigma2**2)/(sigma**2)

        def calc_sigma1_prime(z1, z2, sigma1, sigma2, Lambda):    
            return sigma1 * np.sqrt(sigma2**2 + Lambda**2) / sigma

        def calc_sigma2_prime(z1, z2, sigma1, sigma2, Lambda):
            return sigma2 * np.sqrt(sigma1**2 + Lambda**2) / sigma
        

        return calc_z1_prime(z1, z2, sigma1, sigma2, Lambda), calc_z2_prime(z1, z2, sigma1, sigma2, Lambda), calc_sigma1_prime(z1, z2, sigma1, sigma2, Lambda), calc_sigma2_prime(z1, z2, sigma1, sigma2, Lambda)

 
    elif mode == "EXPONENTIAL":


        # 3 possible cases: critical point and two choices of sign
        z1_case0 = (z1/sigma1**2 + z2/sigma2**2) / (1/sigma1**2 + 1/sigma2**2)
        z2_case0 = (z1/sigma1**2 + z2/sigma2**2) / (1/sigma1**2 + 1/sigma2**2)
        sigma1_case0 = np.sqrt(1 / (1/sigma1**2 + 1/sigma2**2))
        sigma2_case0 = np.sqrt(1 / (1/sigma1**2 + 1/sigma2**2))

        z1_case1 = z1 - (sigma1**2 / Lambda)
        z2_case1 = z2 + (sigma2**2 / Lambda)
        sigma1_case1 = sigma1
        sigma2_case1 = sigma2

        z1_case2 = z1 + (sigma1**2 / Lambda)
        z2_case2 = z2 - (sigma2**2 / Lambda)
        sigma1_case2 = sigma1
        sigma2_case2 = sigma2

        # Compute the likelihood for each case
        likelihood_case0 = symmetrized_exp_log_likelihood_function(z1_case0, z2_case0, z1, z2, sigma1, sigma2, Lambda)
        likelihood_case1 = symmetrized_exp_log_likelihood_function(z1_case1, z2_case1, z1, z2, sigma1, sigma2, Lambda)
        likelihood_case2 = symmetrized_exp_log_likelihood_function(z1_case2, z2_case2, z1, z2, sigma1, sigma2, Lambda)

        # Choose the case with the highest likelihood
        likelihoods = np.array([likelihood_case0, likelihood_case1, likelihood_case2])
        max_likelihood_index = np.argmax(likelihoods, axis = 0)

        z1_prime = np.zeros(len(max_likelihood_index))
        z2_prime = np.zeros(len(max_likelihood_index))
        sigma1_prime = np.zeros(len(max_likelihood_index))
        sigma2_prime = np.zeros(len(max_likelihood_index))

        for i in range(len(max_likelihood_index)):
            z1_prime[i] = np.array([z1_case0[i], z1_case1[i], z1_case2[i]])[max_likelihood_index[i]]
            z2_prime[i] = np.array([z2_case0[i], z2_case1[i], z2_case2[i]])[max_likelihood_index[i]]
            sigma1_prime[i] = np.array([sigma1_case0[i], sigma1_case1[i], sigma1_case2[i]])[max_likelihood_index[i]]
            sigma2_prime[i] = np.array([sigma2_case0[i], sigma2_case1[i], sigma2_case2[i]])[max_likelihood_index[i]]



        return z1_prime, z2_prime, sigma1_prime, sigma2_prime


    else:
        raise ValueError("kwarg mode should either be 'GAUSSIAN' or 'EXPONENTIAL'")