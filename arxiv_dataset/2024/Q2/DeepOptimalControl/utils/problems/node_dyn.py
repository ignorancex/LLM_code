import torch
import numpy as np


class Dynamics_Quadrotor(torch.nn.Module):
    def __init__(self, device=None):
        super(Dynamics_Quadrotor, self).__init__()
        self.device = device

        self.m = 2.
        g0 = 9.81
        self.ud = torch.tensor([[self.m*g0], [0.], [0.], [0.]], device=self.device)
        self.g = torch.tensor([[0.],[0.],[g0]],device=self.device)

        Jx = 1.2416
        Jy = 1.2416
        Jz = 2 * 1.2416
        Jinv = torch.tensor([1/Jx, 1/Jy, 1/Jz], device=self.device)
        J = torch.tensor([Jx, Jy, Jz], device=self.device)
        self.Jinv = torch.diag(Jinv)
        self.J = torch.diag(J)
        self.A = torch.tensor([[0.,0.,0.,0.], [0.,0.,0.,0.], [1.,0.,0.,0.]], device=self.device)
        self.B = torch.tensor([[0.,1.,0.,0.], [0.,0.,1.,0.], [0.,0.,0.,1.]], device=self.device)
        self.Q = torch.diag(torch.tensor([1., 1., 1., 1.], device=self.device))
        self.Q_diag = torch.tensor([1., 1., 1., 1.], device=self.device)

    def forward(self, t, x, u):
        u0 = u.unsqueeze(2)
        batch_size = x.shape[0]
        ones = torch.ones((batch_size, 1), device=self.device)
        zeros = torch.zeros((batch_size, 1), device=self.device)

        theta = x
        v = theta[:, 3:6].unsqueeze(2)
        eta = theta[:, 6:9]
        w = theta[:, 9:12]

        sin_eta = torch.sin(eta)
        cos_eta = torch.cos(eta)
        sin_phi = sin_eta[:, 0:1]
        cos_phi = cos_eta[:, 0:1]
        sin_theta = sin_eta[:, 1:2]
        cos_theta = cos_eta[:, 1:2]
        tan_theta = torch.tan(eta[:, 1:2])
        sin_psi = sin_eta[:, 2:3]
        cos_psi = cos_eta[:, 2:3]

        R = torch.cat([cos_theta * cos_psi, cos_theta * sin_psi, -sin_theta,
                        sin_phi * sin_theta * cos_psi - cos_phi * sin_psi, sin_phi * sin_theta * sin_psi + cos_phi * cos_psi, cos_theta * sin_phi,
                        cos_phi * sin_theta * cos_psi + sin_phi * sin_psi, cos_phi * sin_theta * sin_psi - sin_phi * cos_psi, cos_theta * cos_phi], dim=1)
        K = torch.cat([ones, sin_phi * tan_theta, cos_phi * tan_theta,
                        zeros, cos_phi, -sin_phi,
                        zeros, sin_phi / cos_theta, cos_phi / cos_theta], dim=1)
        S = torch.cat([zeros, w[:, 2:3], -w[:, 1:2],
                       -w[:, 2:3], zeros, w[:, 0:1],
                       w[:, 1:2], -w[:, 0:1], zeros], dim=1)

        R = R.reshape(batch_size, 3, 3)
        K = K.reshape(batch_size, 3, 3)
        S = S.reshape(batch_size, 3, 3)
        R_T = torch.transpose(R, 1, 2)
        w = w.unsqueeze(2)

        dpdt = torch.matmul(R_T, v)
        dvdt = torch.matmul(S, v) - torch.matmul(R, self.g) + 1 / self.m * torch.matmul(self.A, (u0 + self.ud))
        detadt = torch.matmul(K, w)
        dwdt = torch.matmul(self.Jinv, torch.matmul(S, torch.matmul(self.J, w)))\
               + torch.matmul(self.Jinv, torch.matmul(self.B, (u0 + self.ud)))
        dldt = torch.matmul(self.Q_diag, u0**2)

        dxdt = torch.cat((dpdt.squeeze(2), dvdt.squeeze(2), detadt.squeeze(2), dwdt.squeeze(2), dldt), 1)
        return dxdt


class Dynamics_Satellite(torch.nn.Module):
    def __init__(self, w1, w2 ,w3, device=None):
        super(Dynamics_Satellite, self).__init__()
        self.device = device

        self.w1 = w1
        self.w2 = w2
        self.w3 = w3

        Jinv = torch.tensor([1/2, 1/3, 1/4], device=self.device)
        self.Jinv = torch.diag(Jinv)
        self.B = torch.tensor([[1., 1/20, 1/10], [1/15, 1., 1/10], [1/10, 1/15, 1.]]).to(self.device)
        self.H = torch.tensor([[1.], [1.], [1.]]).to(self.device)

    def forward(self, t, x, u=None):
        u = u.unsqueeze(2)
        batch_size = x.shape[0]
        ones = torch.ones((batch_size, 1), device=self.device)
        zeros = torch.zeros((batch_size, 1), device=self.device)
        
        theta = x
        v = theta[:, :3]
        w = theta[:, 3:6]
    
        sin_v = torch.sin(v)
        cos_v = torch.cos(v)

        sin_phi = sin_v[:, 0:1]
        cos_phi = cos_v[:, 0:1]
        sin_theta = sin_v[:, 1:2]
        cos_theta = cos_v[:, 1:2]
        tan_theta = torch.tan(v[:, 1:2])
        sin_psi = sin_v[:, 2:3]
        cos_psi = cos_v[:, 2:3]

        R = torch.cat([cos_theta * cos_psi, cos_theta * sin_psi, -sin_theta,
                        sin_phi * sin_theta * cos_psi - cos_phi * sin_psi, sin_phi * sin_theta * sin_psi + cos_phi * cos_psi, cos_theta * sin_phi,
                        cos_phi * sin_theta * cos_psi + sin_phi * sin_psi, cos_phi * sin_theta * sin_psi - sin_phi * cos_psi, cos_theta * cos_phi], dim=1)
        K = torch.cat([ones, sin_phi * tan_theta, cos_phi * tan_theta,
                        zeros, cos_phi, -sin_phi,
                        zeros, sin_phi / cos_theta, cos_phi / cos_theta], dim=1)
        S = torch.cat([zeros, w[:, 2:3], -w[:, 1:2],
                       -w[:, 2:3], zeros, w[:, 0:1],
                       w[:, 1:2], -w[:, 0:1], zeros], dim=1)

        R = R.reshape(batch_size, 3, 3)
        K = K.reshape(batch_size, 3, 3)
        S = S.reshape(batch_size, 3, 3)
        w = w.unsqueeze(2)

        dvdt = torch.matmul(K, w)
        dwdt = torch.matmul(S, torch.matmul(R, self.H)) + torch.matmul(self.B, u)
        dwdt = torch.matmul(self.Jinv, dwdt)

        w = w.squeeze(2)
        u = u.squeeze(2)
        dldt = self.w1 * (v**2).sum(1, True) + self.w2 * (w**2).sum(1, True) + self.w3 * (u**2).sum(1, True)
        dxdt = torch.cat([dvdt.squeeze(2), dwdt.squeeze(2), dldt], dim=1)
        return dxdt


class NNCDynamics(torch.nn.Module):
    def __init__(self, dynamics, neural_network, scaling=None, device="cpu"):
        super().__init__()
        self.nnc = neural_network
        self.dynamics = dynamics
        if scaling != None:
            self.scaling = True
            self.lb = torch.tensor(scaling['lb'].T.astype(np.float32), device=device)
            self.ub = torch.tensor(scaling['ub'].T.astype(np.float32), device=device)
            self.T = torch.tensor(scaling['T'], device=device)
        else:
            self.scaling = False
        
    def get_control(self, t, x):
        if self.scaling:
            t_scaled = 2. * t / self.T - 1.
            x_scaled = 2. * (x[:, :-1] - self.lb) / (self.ub - self.lb) - 1.
            u = self.nnc(t_scaled, x_scaled)
        else:
            u = self.nnc(t, x[:, :-1])
        return u

    def forward(self, t, x):
        u = self.get_control(t, x)
        dx = self.dynamics(t=t, u=u, x=x)
        return dx
