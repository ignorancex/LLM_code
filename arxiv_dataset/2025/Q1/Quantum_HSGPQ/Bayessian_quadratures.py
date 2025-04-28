import numpy as np
from scipy.special import erf

"""
Different classes are created for each quadrature method.
Here we consider the Gaussian process quadrature using the exponential kernel and
Hilbert Space approximations of the Kernel.
"""

class Gaussian_process_quadrature():
    """
    Class for a GP model
    """
    def __init__(self, data, sigma2, alpha=1, scale=1, domain=[-1.0, 1.0]):
        """
        Initialize your GP class.
        Parameters
        ----------
        data : Tuple of regression data input and observation e.g. (x, y).
        sigma2 : Float of likelihood variance.
        length_scale: Float for kernel lengthscale.
        variance: Float for kernel variance.
        """
        
        self.data = data
        self.sigma2 = sigma2
        self.alpha = alpha
        self.scale = scale
        ## here we consider a symmetric domain for the quadrature approximation
        self.domain = domain

    def gaussian_kernel(self, X1, X2):
        """ 
        returns the NxM kernel matrix between the two sets of input X1 and X2 
        
        arguments:
        X1    -- NxD matrix
        X2    -- MxD matrix
        alpha -- scalar 
        scale -- scalar
        
        returns NxM matrix    
        """
        alpha = self.alpha
        scale = self.scale

        d2 = np.sum((X1[:,None,:]-X2[None,:,:])**2, axis=-1)
        K = alpha*np.exp(-0.5*d2/scale**2)
        return K
    
    def k_mu_vector(self, x):
        """
        Implementation of one of the kernel variables integrated over the domain [-L,L] 
        of the approximation with weight function mu = 1.
        """

        alpha = self.alpha
        scale = self.scale
        a = self.domain[0]
        b = self.domain[1]
        k_mu = np.sqrt(np.pi/2)* alpha * scale * (erf((x-a)/(np.sqrt(2)*scale)) - erf((x-b)/(np.sqrt(2)*scale)))
        return k_mu
    
    def mu_k_mu(self):
        """
        Implementation of the integral of the kernel variable k_mu over the domain [-L,L]
        with weight function mu = 1.
        """
        alpha = self.alpha
        scale = self.scale
        a = self.domain[0]
        b = self.domain[1]
        return 2*alpha * scale * (scale * (np.exp(-0.5*(a-b)**2/scale**2) - 1) + np.sqrt(np.pi/2)*(a-b) * erf((a-b)/(scale*np.sqrt(2))))

    def quadrature_posterior(self):
        """ 
        returns the posterior distribution of f evaluated at each of the points in Xp conditioned on (X, y)
        using the squared exponential kernel.
        """
        kernel = self.gaussian_kernel
        x_train, y_train = self.data
        sigma2 = self.sigma2
        
        k_mu = self.k_mu_vector(x_train)
        K = kernel(x_train,x_train)
        mu_k_mu = self.mu_k_mu()

        jitter = 1e-6
        
        mu_q = k_mu.T @ np.linalg.inv(K + sigma2*np.identity(len(x_train))) @ y_train
        sigma_q = mu_k_mu - k_mu.T @ np.linalg.inv(K + sigma2*np.identity(len(x_train))) @ k_mu
        return mu_q, sigma_q
    
        
    def GK_spectral(self, t):

        ## Spectral density of the Gaussian Kernel
        alpha = self.alpha
        scale = self.scale

        s = alpha * np.sqrt(2*np.pi*scale**2) * np.exp(-0.5*(scale*t)**2)
        return s 

class Sine_HSQ():
    """
    Class for Hilbert Space approximation of the GPR based on the paper
    from Solin and Särkkä (2020) https://link.springer.com/article/10.1007/s11222-019-09886-w
    """
    def __init__(self, data, sigma2, M, L, alpha=1, scale=1, domain=[1.0, 1.0]):
        """
        Initialize the Hilbert Space approximation GP class.
        Parameters
        ----------
        data : Tuple of regression data input and observation e.g. (x, y).
        sigma2 : Float of likelihood variance.
        alpha: Amplitude of the kernel.
        scale: Float for kernel lengthscale.
        M: Approximation degree.
        """
        self.data = data
        self.sigma2 = sigma2
        self.alpha = alpha
        self.scale = scale
        self.M = M
        self.L = L
        self.domain = domain

    def phi_j(self, lam_j, x):
        """
        returns the eigenfunction j evaluated ax the point x.
        """
        L = self.L
        return np.sin((lam_j)*(x+L))/(np.sqrt(L))
    
    def spectral_density(self, t):
        """
        Evaluates the spectral density of the exponential kernel ar the point t.
        """
        ## Spectral density of the Gaussian Kernel
        alpha = self.alpha
        scale = self.scale

        s = alpha * np.sqrt(2*np.pi*scale**2) * np.exp(-0.5*(scale*t)**2)
        return s
    
    def phi_vector(self, x):
        """
        Returns the vector of x evaluated at each of the eigenfunctions from 1 to M.
        """
        M = self.M
        L = self.L
        phi_j = self.phi_j

        phi = None
        for j in range(M):
            lam_j = np.pi*(j+1)/(2*L)
            phi = np.append(phi, phi_j(lam_j, x))
        phi = phi[1:]
        
        return phi[:,None].astype('float64')

    def Phi_matrix(self, X):
        """
        returns the data matrix of the input points X evaluated at each of the eigenfunctions from 1 to M.
        """
        M = self.M
        phi_vector = self.phi_vector
        
        return np.reshape(np.apply_along_axis(phi_vector, 1, X), (len(X) , M))
    
    def phi_mu_vector(self):
        """
        returns the vector of the eigenfunctions integrated over the domain [-L,L] 
        of the approximation with weight function mu = 1.
        """
        M = self.M
        L = self.L
        a = self.domain[0]
        b = self.domain[1]
        phi_mu = np.zeros(M)
        for j in range(M):
            phi_mu[j] = 2*np.sqrt(L)*(np.cos(np.pi*(j+1)*(a+L)/(2*L)) - np.cos(np.pi*(j+1)*(b+L)/(2*L)))/(np.pi*(j+1))

        return phi_mu
    
    def Lambda(self):
        """
        returns the diagonal matrix of the eigenvalues.
        """
        M = self.M
        L = self.L
        spectral_density = self.spectral_density

        lams = []
        for j in range(M):
            lam_j = np.pi*(j+1)/(2*L)
            lams.append(lam_j)
        lams = np.array(lams)

        Lambda = np.diag(spectral_density(lams))
        return Lambda
    
    def kernel_approx(self, X1, X2):
        """ 
        returns the NxM kernel matrix between the two sets of input X1 and X2 
        
        arguments:
        X1    -- NxD matrix
        X2    -- MxD matrix
    
        returns NxM matrix    
        """
    
        M = self.M
        alpha = self.alpha
        scale = self.scale
        L = self.L
        spectral_density = self.spectral_density
        phi_j = self.phi_j


        k = 0
        for j in range(M):
            lam_j = np.pi*(j+1)/(2*L)
            Phi_x1 = phi_j(X1,lam_j,L)
            Phi_x2 = phi_j(X2,lam_j,L)

            s_j = spectral_density(lam_j, alpha, scale)
            Knew = s_j * np.kron(Phi_x1, Phi_x2.T)
            k = Knew + k
        
        return k
    
    def quadrature_posterior(self):
        """ 
        returns the posterior distribution of f evaluated at each of the points in Xp conditioned on (X, y)
        using the approximated squared exponential kernel using the Hilbert Space approximation.
        """
                

        x_train, y_train = self.data
        sigma2 = self.sigma2
        M = self.M
        Phi_matrix = self.Phi_matrix

        jitter = 1.0E-10


        phi_mu = self.phi_mu_vector()
        Phif = Phi_matrix(x_train)


        Lambda = self.Lambda()
        X = Phif@np.sqrt(Lambda)
        X_mu = np.sqrt(Lambda)@phi_mu
        
        y_train = y_train[:,None]       
        
        B = X.T @ X + (jitter +  sigma2)*np.identity(M)
        B_1 = np.linalg.inv(B)
        print("X_mu", X_mu)
        mu = X_mu.T @ B_1 @ X.T @ y_train      
        Var = sigma2 * X_mu.T @ B_1 @ X_mu
        
        return mu, Var
