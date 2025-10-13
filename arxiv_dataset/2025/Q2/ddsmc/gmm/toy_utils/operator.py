import torch
from ddsmc.ddsmc_sampler import DDSMCDDPM

class GaussianOperator:
    def __init__(self, A, sigma, prime=False):
        self.A = A
        ydim, xdim = A.shape
        self.device = A.device
        self.sigma = sigma
        self.L, self.Q = torch.linalg.eigh(A.T @ A)
        self.prime = prime
        self.U, self.singulars, Vt = torch.linalg.svd(A)
        self.S = torch.zeros((ydim, xdim), device=A.device)
        self.S[:ydim, :ydim] = torch.diag(self.singulars)
        self.V = Vt.T

        self.STS_diag = torch.diag(self.S.T @ self.S)
    
    def log_likelihood_prime(self, x0hat_prime, y_prime, r_t=0.):
        mean = (self.S @ x0hat_prime.T).T
        sigma2 = self.sigma**2 + r_t**2 * self.singulars**2
        return DDSMCDDPM.diag_gauss_logpdf(y_prime, mean, sigma2)
    
    def log_likelihood_regular(self, x0hat, y, r_t=0.):
        mean = x0hat @ self.A.T
        cov = self.sigma**2 * torch.eye(y.shape[1], device=self.device) + r_t**2 * (self.A @ self.A.T)
        distr = torch.distributions.MultivariateNormal(mean, cov, validate_args=False)
        return distr.log_prob(y)
    
    def log_likelihood(self, x0hat, y, r_t=0.):
        if self.prime:
            return self.log_likelihood_prime(x0hat, y, r_t)
        else:
            return self.log_likelihood_regular(x0hat, y, r_t)
        
    def measure(self, x):
        y0 = (self.A @ x.T).T
        return y0 + self.sigma * torch.rand_like(y0)

    def error(self, x, y):
        Ax = (self.A @ x.T).T
        return ((Ax - y) ** 2).flatten(1).sum(-1)


def multiply_by_matrix(M, vec):
    vec = vec.reshape((vec.shape[0], -1))
    return (M @ vec.T).T

class Toy_Hfunc:
    """
    """

    def __init__(self, A):
        self._U, self._singulars, Vt = torch.linalg.svd(A)
        self._V = Vt.T
        self._A = A
        self.ydim, self.xdim = self._A.shape

    def V(self, vec):
        """
        Multiplies the input vector by V
        """
        return multiply_by_matrix(self._V, vec)

    def Vt(self, vec, for_H=True):
        """
        Multiplies the input vector by V transposed
        """
        return multiply_by_matrix(self._V.T, vec)

    def U(self, vec):
        """
        Multiplies the input vector by U
        """
        return multiply_by_matrix(self._U, vec)

    def Ut(self, vec):
        """
        Multiplies the input vector by U transposed
        """
        return multiply_by_matrix(self._U.T, vec)

    def singulars(self):
        """
        Returns a vector containing the singular values. The shape of the vector should be the same as the smaller dimension (like U)
        """
        return self._singulars

    def add_zeros(self, vec):
        """
        Adds trailing zeros to turn a vector from the small dimension (U) to the big dimension (V)
        """
        return torch.cat([vec, torch.zeros((vec.shape[0], self.xdim - self.ydim), device=vec.device)], dim=1)

    def H(self, vec):
        """
        Multiplies the input vector by H
        """
        #temp = self.Vt(vec)
        #singulars = self.singulars()
        #return self.U(singulars * temp[:, :singulars.shape[0]])
        return multiply_by_matrix(self._A, vec)

    def Ht(self, vec):
        """
        Multiplies the input vector by H transposed
        """
        #temp = self.Ut(vec)
        #singulars = self.singulars()
        #return self.V(self.add_zeros(singulars * temp[:, :singulars.shape[0]]))
        return multiply_by_matrix(self._A.T, vec)

    def H_pinv(self, vec):
        """
        Multiplies the input vector by the pseudo inverse of H
        """
        temp = self.Ut(vec)
        singulars = self.singulars()
        temp[:, :singulars.shape[0]] = temp[:, :singulars.shape[0]] / singulars
        return self.V(self.add_zeros(temp))