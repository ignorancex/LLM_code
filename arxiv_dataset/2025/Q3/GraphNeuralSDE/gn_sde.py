import math
import torch
import torchsde
import torch
from torch import distributions, nn

def _stable_division(a, b, epsilon=1e-7):
    b = torch.where(b.abs().detach() > epsilon, b, torch.full_like(b, fill_value=epsilon) * b.sign())
    return a / b

class LatentGraphSDE(torchsde.SDEIto):

    def __init__(self, in_net, drift_net, out_net, theta=1.0, mu=0.0, sigma=1.0, adaptive=False, method="srk", rtol =0.01, atol = 0.01, sde_output_dim=64, t0=0, t1=0.001, device=None, opt=None, epoch=0):
        super(LatentGraphSDE, self).__init__(noise_type="diagonal")
        
        
        sigma = opt['sigma'] if opt is not None else sigma
        
        logvar = math.log(sigma ** 2 / (2. * theta))
        self.sde_output_dim = sde_output_dim
        # Prior drift
        self.register_buffer("theta", torch.full((1, sde_output_dim), theta))
        self.register_buffer("mu", torch.full((1, sde_output_dim), mu))
        self.register_buffer("sigma", torch.full((1, sde_output_dim), sigma))

        # p(y0)
        self.register_buffer("py0_mean", torch.full((1, sde_output_dim), mu))
        self.register_buffer("py0_logvar", torch.full((1, sde_output_dim), logvar))

        # Approximate posterior drift
        self.net = drift_net

        # q(y0)
        self.qy0_mean = nn.Parameter(torch.full((1, sde_output_dim), mu), requires_grad=True)
        self.qy0_logvar = nn.Parameter(torch.full((1, sde_output_dim), logvar), requires_grad=True)

        self.in_net = in_net
        self.projection_net = out_net
        
        
        # Arguments
        if device == None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else: 
            self.device = device
        self.adaptive = adaptive
        self.method = method
 
        if opt is not None:
                self.rtol = opt['rtol']
                self.atol = opt['rtol']
                t1 = opt['sde_t']
                self.ts_vis = torch.tensor([t0, t1]).float().to(self.device)    
                self.t0 = t0
                self.t1 = t1
        else:
            self.rtol = rtol
            self.atol = atol
            self.ts_vis = torch.tensor([t0, t1]).float().to(self.device)    
            self.t0 = t0
            self.t1 = t1
        
        self.opt = opt
        self.ts_fixed = torch.linspace(self.t0, self.t1, 11).to(self.device) 
        self.collect_embeddings = False
        self.embeddings = {f'{ts.item():.3g}': 0 for ts in self.ts_fixed.cpu().numpy()}
        self.epoch = epoch
        print('self.embeddings', self.embeddings)
        print('using file gn_sde.py')
        print('sigma', sigma)
        print("rtol",self.rtol)
        print("t1", self.t1)


    def f(self, t, y):
        if t.dim() == 0:
            t = t.unsqueeze(0).expand_as(y[:, :1])
        
        # Positional encoding in transformers for time-inhomogeneous posterior.
        z = self.net(torch.cat((t, y), dim=-1))
        return z


    def g(self, t, y):  # Shared diffusion.
        return self.sigma.repeat(y.size(0), 1)

    def h(self, t, y):  # Prior drift.
        prioir_drift = self.theta * (self.mu - y)
        return prioir_drift

    def f_aug(self, t, y):  # Drift for augmented dynamics with logqp term.
        y = y[:, 0:self.sde_output_dim]
        f, g, h = self.f(t, y), self.g(t, y), self.h(t, y)
        u = _stable_division(f - h, g)
        f_logqp = .5 * (u ** 2).sum(dim=1, keepdim=True)
        return torch.cat([f, f_logqp], dim=1)

    def g_aug(self, t, y):  # Diffusion for augmented dynamics with logqp term.
        y = y[:, 0:self.sde_output_dim]
        g = self.g(t, y)
        g_logqp = torch.zeros(y.shape[0], 1).to(y)
        return torch.cat([g, g_logqp], dim=1)
    
    @property
    def py0_std(self):
        return torch.exp(.5 * self.py0_logvar)

    @property
    def qy0_std(self):
        return torch.exp(.5 * self.qy0_logvar)
    
    def _init_brownian_motion(self, batch_size, aug_y0):
        # We need space-time Levy area to use the SRK solver
        bm =  torchsde.BrownianInterval(
            t0=self.ts_vis[0],
            t1=self.ts_vis[-1],
            size=(batch_size, aug_y0.shape[1]),
            device=self.device,
            levy_area_approximation='space-time'
        )
        return bm

    
    def forward(self, ts):
        batch_size = ts.shape[0]
        qy0 = distributions.Normal(loc=self.qy0_mean, scale=self.qy0_std)
        py0 = distributions.Normal(loc=self.py0_mean, scale=self.py0_std)
        logqp0 = distributions.kl_divergence(qy0, py0).sum(dim=1)

        y0 = self.in_net(ts)
        aug_y0 = torch.cat([y0, torch.zeros(y0.shape[0], 1).to(ts)], dim=1)
        
        bm = self._init_brownian_motion(batch_size, aug_y0)
        
        
        aug_ys = torchsde.sdeint(
            sde=self,
            y0=aug_y0,
            ts=torch.tensor([self.t0, self.t1]).float().to(self.device),
            # ts=self.ts_fixed,
            method=self.method,
            bm=bm,
            adaptive=self.adaptive,
            rtol=self.rtol,
            atol=self.atol,
            names={'drift': 'f_aug', 'diffusion': 'g_aug'}
        )
        ys, logqp_path = aug_ys[:, :, 0:self.sde_output_dim], aug_ys[-1, :, self.sde_output_dim]
        

            
        ys_out = self.projection_net(ys[-1])
        logqp = (logqp0 + logqp_path).mean(dim=0)  # KL(t=0) + KL(path).
        
        # if self.collect_embeddings:
        #     self.layer_embeddings = [y0, ys, ys_out]
            
        return ys_out, logqp

