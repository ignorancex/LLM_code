import numpy as np
import torch
    
    
class Simulator_Additive:
    def __init__(self, Nbins, sigma, bounds=5, fraction=None, sample_fraction=False, bkg=False, device='cpu', dtype=torch.float64):
        """
        Args:
        - Nbins (int): Number of bins in the histogram.
        - sigma (float): Standard deviation of the Gaussian noise.
        - bounds (float): Bounds for the uniform distribution of the additive noise.
        - fraction (float): Fraction of bins to be perturbed by the additive noise. If None, just one bin is perturbed.
        - bkg (bool): If True, the simulator generates background events.
        - device (str): Device to run the tensors on.
        - dtype (torch.dtype): Data type of the tensors.
        """
        self.device = device
        self.dtype = dtype
        self.Nbins = Nbins
        self.sigma = sigma
        self.bounds = bounds
        self.bkg = bkg
        self.fraction = fraction
        self.sample_fraction = sample_fraction
        self.grid = torch.linspace(-10, 10, Nbins, device=device, dtype=dtype)
        
    def get_theta(self, Nsims: int) -> torch.Tensor:
        return torch.rand(Nsims, 3, device=self.device, dtype=self.dtype) * 2 - 1

    def get_mu(self, theta: torch.Tensor) -> torch.Tensor:
        grid = self.grid.unsqueeze(0)  # Shape: (1, Nbins)
        # Compute mu for all simulations at once
        mu = (
            torch.sin(grid + 0.5*theta[:, 0:1]) +
            theta[:, 1:2] * grid/10 +
            0.5*theta[:, 2:3]
        )
        return mu  # Shape: (Nsims, Nbins)
    
    def get_x_H0(self, Nsims: int, mu: torch.Tensor = 0) -> torch.Tensor:
        x_shape = (Nsims, self.Nbins)
        return (mu + torch.randn(x_shape, device=self.device, dtype=self.dtype) * self.sigma).to(self.dtype)
    
    def get_ni(self, x: torch.Tensor) -> torch.Tensor:
        if self.fraction is None:
            """Standard basis vectors"""
            batch_size, N_bins = x.shape
            ni = torch.zeros(batch_size, N_bins, device=self.device, dtype=self.dtype)
            indices = torch.randint(0, N_bins, (batch_size,), device=self.device)
            ni[torch.arange(batch_size), indices] = 1
        else:
            """Fraction of bins are distorted"""
            if self.sample_fraction:
                fr = np.random.uniform(0.01, self.fraction)
            else:   
                fr = self.fraction
            prob = fr*self.Nbins/100
            random_vals = torch.rand_like(x)
            ni = (random_vals < prob).type(self.dtype)  # fr% chance
        return ni
    
    def get_epsilon(self, ni: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return (2 * self.bounds * torch.rand(x.shape, device=self.device, dtype=self.dtype) - self.bounds) * ni
    
    def get_x_Hi(self, epsilon: torch.Tensor, ni: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return x + epsilon * ni
    
    def _sample(self, Nsims: int) -> dict:
        sample = {}
        if self.bkg:
            theta = self.get_theta(Nsims)
            mu = self.get_mu(theta)
            sample['theta'] = theta
            sample['mu'] = mu
            x0 = self.get_x_H0(Nsims, mu)
        else:
            x0 = self.get_x_H0(Nsims, 0)
        ni = self.get_ni(x0)
        epsilon = self.get_epsilon(ni, x0)
        xi = self.get_x_Hi(epsilon, ni, x0)
        sample.update({'x0': x0, 'epsilon': epsilon, 'ni': ni, 'xi': xi})
        return sample
    
    def _resample(self, sample: dict) -> dict:
        Nsims = sample['x0'].shape[0] if sample['x0'].ndim == 2 else 1
        if self.bkg:
            sample['x0'] = self.get_x_H0(Nsims, sample['mu'])
        else:  
            sample['x0'] = self.get_x_H0(Nsims, 0)
        sample['ni'] = self.get_ni(sample['x0'])
        sample['epsilon'] = self.get_epsilon(sample['ni'], sample['x0'])
        sample['xi'] = self.get_x_Hi(sample['epsilon'], sample['ni'], sample['x0'])
        return sample
    
    def sample(self, Nsims: int = 1) -> dict:
        sample = self._sample(Nsims)
        return sample
        
        
class Simulator_Additive_Correlated(Simulator_Additive):
    def __init__(self, Nbins, sigma, correlation_scales = 2**torch.linspace(0, 5, 6).int(), bounds=5, fraction=None, sample_fraction=False, bkg=False, device='cpu', dtype=torch.float64):
        super().__init__(Nbins, sigma, bounds, fraction, sample_fraction, bkg, device, dtype)
   
        self.correlation_scales = correlation_scales
    
    def _conv1d(self, ni: torch.Tensor, c: int) -> torch.Tensor:
        w = torch.linspace(-3, 3, 1+int(c)*2, device=ni.device, dtype=ni.dtype).unsqueeze(0).unsqueeze(0)
        w = torch.exp(-0.5*w**2)
        w = w/w.max() # Normalize maximum to 1
        y = torch.nn.functional.conv1d(ni.unsqueeze(1), w.to(ni.dtype), padding = int(c)).squeeze(1)
        return y
    
    def get_correlation(self, ni, epsilon) -> torch.Tensor:
        if isinstance(self.bounds, float) or isinstance(self.bounds, int):
            cc = torch.stack([self._conv1d(ni*epsilon, int(c)) for c in self.correlation_scales], axis=1)
        elif len(self.bounds)==len(self.correlation_scales):
            cc = torch.stack([self._conv1d(ni*epsilon[:, i_c], int(c)) for i_c, c in enumerate(self.correlation_scales)], axis=1)
        return cc
    
    def get_epsilon(self, ni: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if isinstance(self.bounds, float) or isinstance(self.bounds, int):
            eps = (2 * self.bounds * torch.rand(x.shape, device=self.device, dtype=self.dtype) - self.bounds) * ni
        elif len(self.bounds)==len(self.correlation_scales):
            eps = (2 * self.bounds.unsqueeze(0).unsqueeze(2) * torch.rand(x.shape, device=self.device, dtype=self.dtype).unsqueeze(1).to(self.dtype) - self.bounds.unsqueeze(0).unsqueeze(2)) * ni.unsqueeze(1)
        return eps.to(self.dtype)
        
    def get_x_Hi(self, cni: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return x +  cni
    
    def _sample(self, Nsims: int) -> dict:
        sample = {}
        if self.bkg:
            theta = self.get_theta(Nsims)
            mu = self.get_mu(theta)
            sample['theta'] = theta
            sample['mu'] = mu
            x0 = self.get_x_H0(Nsims, mu)
        else:
            x0 = self.get_x_H0(Nsims, 0)
        ni = self.get_ni(x0)
        epsilon = self.get_epsilon(ni, x0)
        cni = self.get_correlation(ni, epsilon)
        xi = self.get_x_Hi(cni, x0.unsqueeze(1))
        sample.update({'x0': x0, 'epsilon': epsilon, 'ni': ni, 'cni': cni, 'xi': xi})
        return sample
    
    def _resample(self, sample: dict) -> dict:
        Nsims = sample['x0'].shape[0] if sample['x0'].ndim == 2 else 1
        if self.bkg:
            sample['x0'] = self.get_x_H0(Nsims, sample['mu'])
        else:  
            sample['x0'] = self.get_x_H0(Nsims, 0)
        sample['ni'] = self.get_ni(sample['x0'])
        sample['epsilon'] = self.get_epsilon(sample['ni'], sample['x0'])
        sample['cni'] = self.get_correlation(sample['ni'],  sample['epsilon'])
        sample['xi'] = self.get_x_Hi(sample['cni'], sample['x0'].unsqueeze(1))
        return sample