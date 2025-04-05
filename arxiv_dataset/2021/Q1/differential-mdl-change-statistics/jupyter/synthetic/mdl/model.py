import numpy as np
from scipy.special import gamma, gammaln
import scipy.stats

def gammaln_m(x, m):
    """
    \gamma_{m}(x) = pi^(m(m-1)/4) * \prod_{j=1}^{m} \gamma(x + (1-j)/2)
    \log{\gamma_{m}(x)} = m(m-1)/4 * \log{\pi} + \sum_{j=1}^{m} \log{\gamma(x+(1-j)/2)}
    :param x:
    :param m:
    :return:
    """
    return m*(m-1)/4 * np.log(np.pi) + np.sum([gammaln(x + (1-j)/2) for j in range(1, m+1)])

class Model():
    def __init__(self):
        """
        initialize parameters
        :param h: window size
        :param T: sequence length
        :return:
        """
        #self.h = h
        #self.T = T
        pass


class Norm(Model):
    def __init__(self):
        """
        initialize parameters
        :param h: window size
        :param T: sequence length
        :return:
        """
        super().__init__()

    def _log_NML_normalizer(self, k, m, mu_max, sigma_min):
        """
        Normalized Maximum Likelihood normalizer
        :param k:
        :param m:
        :param mu_max: maximum value of mu
        :param sigma_min: minimum value of sigma
        :return: log normalized maximum likelihood normalizer
        """
        return -(m+1)*np.log(m/2) + m/2*np.log(mu_max) - m**2/2*np.log(sigma_min) + \
               m*k/2.0*(np.log(k/2) - 1.0) - gammaln(m/2) - gammaln_m((k-1)/2, m)
               #m*k/2.0*(np.log(k/2) - 1.0) - gammaln(m/2) - gammaln((k-1)/2)

    def calc_change_score(self, x, h, mu_max, sigma_min):
        """
        calculate change score for a given point
        :param x: point (time points * features)
        :param mu_max: maximum value of mu
        :param sigma_min: minimum value of sigma
        :return: change statistic
        """

        mu0 = np.mean(x, axis=0)
        mu1 = np.mean(x[:h, :], axis=0)
        mu2 = np.mean(x[h:, :], axis=0)

        cor0 = np.cov(x, rowvar=False)
        cor1 = np.cov(x[:h, :], rowvar=False)
        cor2 = np.cov(x[h:, :], rowvar=False)

        cor_inv0 = np.linalg.inv(cor0)
        cor_inv1 = np.linalg.inv(cor1)
        cor_inv2 = np.linalg.inv(cor2)
        
        dim = x.shape[1]

        # calculate change statistic
        term0 = 1/(2*h) * np.sum(-scipy.stats.multivariate_normal(mu0, cor0).logpdf(x))
        term1 = 1/(2*h) * np.sum(-scipy.stats.multivariate_normal(mu1, cor1).logpdf(x[:h, :]))
        term2 = 1/(2*h) * np.sum(-scipy.stats.multivariate_normal(mu2, cor2).logpdf(x[h:, :]))
        change_statistic = term0 - term1 - term2 + \
                           1.0/(2*h) * self._log_NML_normalizer(2*h, dim, mu_max, sigma_min) - \
                           1.0/h * self._log_NML_normalizer(h, dim, mu_max, sigma_min)
        return change_statistic

    def calc_change_score_1st(self, x, h, mu_max, sigma_min):
        """
        calculate change score for a given point
        :param x: point (time points * features)
        :param mu_max: maximum value of mu
        :param sigma_min: minimum value of sigma
        :return: change statistic
        """
       
        mu1_1 = np.mean(x[:h, :], axis=0)
        mu2_1 = np.mean(x[h:, :], axis=0)

        mu1_2 = np.mean(x[:h+1, :], axis=0)
        mu2_2 = np.mean(x[h+1:, :], axis=0)

        cor0 = np.cov(x, rowvar=false)
        cor1_1 = np.cov(x[:h, :], rowvar=false)
        cor2_1 = np.cov(x[h:, :], rowvar=false)

        cor_inv0 = np.linalg.inv(cor0)
        cor_inv1_1 = np.linalg.inv(cor1_1)
        cor_inv2_1 = np.linalg.inv(cor2_1)
        
        cor1_2 = np.cov(x[:h+1, :], rowvar=false)
        cor2_2 = np.cov(x[h+1:, :], rowvar=false)
        
        cor_inv1_2 = np.linalg.inv(cor1_2)
        cor_inv2_2 = np.linalg.inv(cor2_2)
        
        dim = x.shape[1]

        # calculate change statistic
        term1_1 = 1/h * np.sum(-scipy.stats.multivariate_normal(mu1_1, cor1_1).logpdf(x[:h, :])) + \
                  1/h * self._log_nml_normalizer(h, dim, mu_max, sigma_min)
        term2_1 = 1/h * np.sum(-scipy.stats.multivariate_normal(mu2_1, cor2_1).logpdf(x[h:, :])) + \
                  1/h * self._log_nml_normalizer(h, dim, mu_max, sigma_min)
        
        term1_2 = 1/(h+1) * np.sum(-scipy.stats.multivariate_normal(mu1_2, cor1_2).logpdf(x[:h+1, :])) + \
                  1/(h+1) * self._log_NML_normalizer(h+1, dim, mu_max, sigma_min)
        term2_2 = 1/(h-1) * np.sum(-scipy.stats.multivariate_normal(mu2_2, cor2_2).logpdf(x[h+1:, :])) + \
                  1/(h-1) * self._log_NML_normalizer(h-1, dim, mu_max, sigma_min)
        
        change_statistic = term1_1 + term2_1 - term1_2 - term2_2 
        
        return change_statistic

 
class Norm1D(Model):
    def __init__(self):
        """
        initialize parameters
        :param h: window size
        :param T: sequence length
        :return:
        """
        super().__init__()

    def _log_NML_normalizer(self, k, mu_max, sigma_min):
        """
        Normalized Maximum Likelihood normalizer
        :param k:
        :param mu_max: maximum value of mu
        :param sigma_min: minimum value of sigma
        :return: log normalized maximum likelihood normalizer
        """
        return 0.5 * np.log(16 * np.abs(mu_max) / (np.pi * sigma_min**2)) + \
               k/2 * np.log(k/2) - k/2 - gammaln((k-1)/2)

    def calc_change_score(self, x, h, mu_max, sigma_min):
        """
        calculate change score for a given point
        :param x: point
        :param mu_max: maximum value of mu
        :param sigma_min: minimum value of sigma
        :return: change statistic
        """
        window = len(x)
        
        sigma0 = np.std(x)
        sigma1 = np.std(x[:h])
        sigma2 = np.std(x[h:])
        
        # calculate change statistic
        change_statistic = 0.5 * np.log(sigma0**2/(sigma1 * sigma2)) + \
                           (self._log_NML_normalizer(window, mu_max, sigma_min) 
                             -self._log_NML_normalizer(t, mu_max, sigma_min) -self._log_NML_normalizer(window-t, mu_max, sigma_min)
                           ) / (2.0 * window)
       
        return change_statistic
    
    def calc_change_score_1st(self, x, t, mu_max, sigma_min):
        """
        calculate change score for a given point
        :param x: point
        :param mu_max: maximum value of mu
        :param sigma_min: minimum value of sigma
        :return: change statistic
        """
        
        # calculate change statistic
        change_statistic = self.calc_change_score(x, t+1, mu_max, sigma_min) -  \
                           self.calc_change_score(x, t, mu_max, sigma_min)
        
        return change_statistic
     
        
    def calc_change_score_2nd(self, x, t, mu_max, sigma_min):
        """
        calculate change score for a given point
        :param x: point
        :param mu_max: maximum value of mu
        :param sigma_min: minimum value of sigma
        :return: change statistic
        """
        
        # calculate change statistic
        change_statistic = self.calc_change_score(x, t+1, mu_max, sigma_min) - \
                            2.0 * self.calc_change_score(x, t, mu_max, sigma_min) + \
                            self.calc_change_score(x, t-1, mu_max, sigma_min)
        
        return change_statistic
    

class Poisson1D(Model): 
    def __init__(self):
        super().__init__()

    def _log_NML_normalizer(self, k, lambda_max):
        return 0.5 * np.log(k/(2*np.pi)) + (1 + lambda_max/2)*np.log(2) + np.log(lambda_max)

    def calc_change_score(self, x, h, **kwargs):
        """
        calculate change score for a given point
        :param x: point
        :return:
        """
        lambda0 = np.mean(x)
        lambda1 = np.mean(x[:h])
        lambda2 = np.mean(x[h:])

        # calculate change statistic
        change_statistic = -2*h*(lambda0*np.log(lambda0) - 0.5*(lambda1*np.log(lambda1) + lambda2*np.log(lambda2))) + \
                           self._log_NML_normalizer(2*h) - self._log_NML_normalizer(h)
        return change_statistic

class LinearRegression(Model):
    def __init__(self):
        super().__init__()

    def calc_change_score(self, x, h, sigma_min=1.0, R=10.0):
        h = self.h
        sigma0 = np.std(x)
        sigma1 = np.std(x[:h])
        sigma2 = np.std(x[h:])

        # calculate change statistic
        change_statistic = h * np.log(sigma0**2/(sigma1 * sigma2)) - \
                            np.log(R/sigma_min**2) - \
                            np.log(gammaln(h-1)) - 2.0 * np.log(gammaln(h/2-1)) + \
                            h * np.log(2.0)
        return change_statistic
    
class Bernoulli(Model):
    def __init__(self):
        super().__init__()
        
    def calc_change_score(self, x, h):
        mu0 = np.sum(x)
        mu1 = np.sum(x[:h])
        mu2 = np.sum(x[h:])
        
        # calculate change statistic
        change_statistic = 1/(2*h)*(-mu0 * np.log(mu0/(2.0*h)) - (2*h-mu0) * np.log(1.0 - mu0/(2.0*h))) - \
                            1/(2*h)*(-mu1 * np.log(mu1/h) - (h-mu1) * np.log(1.0 - mu1/h)) - \
                            1/(2*h)*(-mu2 * np.log(mu2/h) - (h-mu2) * np.log(1.0 - mu2/h))

        return change_statistic
