import torch
import torch.nn as nn
import numpy as np
from scipy.linalg import svd, inv
import math

from .noise import DiffusionAwareNoiseScheduler
from .utils import _strip_unpicklable, safe_sqrt, fft_density, EPS

try:
    import mrcfile
    HAS_MRCFILE = True
except ImportError:
    HAS_MRCFILE = False

class AdaptiveMeasurement(nn.Module):
    def __init__(self, name: str, config: dict, device: str):
        super().__init__()
        self.name = name
        self.config = config
        self.device = device
        sigma_init = float(config.get("noise_level", 1.0))
        L_init = float(config.get("lipschitz_constant", 1.0))
        self.noise_scheduler = DiffusionAwareNoiseScheduler(sigma_init, L_init, device)
        self.current_sigma = sigma_init
        self.data: dict = {}
        self.auto_lipschitz = config.get("auto_lipschitz", True)
        self.lipschitz_estimated = False

    def __getstate__(self):
        return _strip_unpicklable(self.__dict__)

    def __setstate__(self, state):
        self.__dict__.update(state)

    def update_effective_noise(self, t: float, noise_schedule):
        self.current_sigma = self.noise_scheduler.get_effective_sigma(t, noise_schedule)

    def get_learned_sigma_meas(self) -> float:
        return float(self.noise_scheduler.sigma_meas.item())

    def estimate_lipschitz_constant(self, X_sample: torch.Tensor, C_sample: torch.Tensor) -> float:
        raise NotImplementedError

    def forward(self, X: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def log_likelihood(self, X: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class PartialStructureMeasurement(AdaptiveMeasurement):
    def __init__(self, name, config, X_true, C, backbone_network, device):
        super().__init__(name, config, device)
        self.backbone_network = backbone_network
        self.multiply_R = backbone_network.noise_perturb.base_gaussian._multiply_R
        self.multiply_R_inverse = backbone_network.noise_perturb.base_gaussian._multiply_R_inverse
        self._create_measurement(X_true, C)
        if self.auto_lipschitz and not self.lipschitz_estimated:
            L_est = self.estimate_lipschitz_constant(X_true, C)
            self.noise_scheduler.update_lipschitz_constant(L_est)
            self.lipschitz_estimated = True
            print(f"Partial structure: Estimated Lipschitz constant = {L_est:.3f}")

    def _create_measurement(self, X_true: torch.Tensor, C: torch.Tensor):
        X_flat = X_true[0] if (X_true.dim() == 4 and X_true.shape[0] == 1) else X_true
        c0 = C[0] if (C.dim() == 2 and C.shape[0] == 1) else C.reshape(-1)
        n = X_flat.shape[0]

        # Handle both fixed interval and target number of observations
        if "n_observations" in self.config:
            # Calculate fix_every to achieve target observations
            valid_residues = (c0 == 1).sum().item()
            target_obs = min(self.config["n_observations"], valid_residues)
            fix_every = max(1, int(valid_residues / target_obs))
        else:
            fix_every = self.config.get("fix_every", 1)

        mask_flat = torch.zeros(n, device=self.device)
        mask_flat[::fix_every] = 1.0
        mask_flat[c0 != 1] = 0.0
        mask = mask_flat > 0.5
        Y_clean = X_flat[mask].clone().unsqueeze(0)
        sigma_true = float(self.config.get("true_noise_level", self.current_sigma))
        Y = Y_clean + torch.randn_like(Y_clean) * sigma_true
        self.data = {
            "mask": mask,
            "Y": Y[0],
            "Y_clean": Y_clean[0],
            "true_noise_level": sigma_true,
            "n_observed": int(mask.sum().item()),
            "n_total": n
        }
        
        if self.config.get("use_preconditioning", True):
            def mR_fun(z):
                z_r = z.reshape(-1, n, 4, 3)
                out = self.multiply_R(z_r, c0.unsqueeze(0).expand(z_r.shape[0], -1))
                sel = out[:, mask]
                return sel.reshape(z_r.shape[0], -1).permute(1, 0)
            
            I = torch.eye(n * 12, device=self.device)
            M_np = mR_fun(I).cpu().numpy()
            U_np, S_np, Vh_np = svd(M_np, full_matrices=False)
            S_reg = S_np + 1e-6 * S_np[0]
            self.data["precon"] = {
                "U": torch.tensor(U_np, device=self.device, dtype=torch.float32),
                "S": torch.tensor(S_reg, device=self.device, dtype=torch.float32),
                "Vh": torch.tensor(Vh_np, device=self.device, dtype=torch.float32),
                "U_inv": torch.tensor(inv(U_np), device=self.device, dtype=torch.float32)
            }

    def estimate_lipschitz_constant(self, X_sample: torch.Tensor, C_sample: torch.Tensor) -> float:
        X0 = X_sample.unsqueeze(0) if X_sample.ndim == 3 else X_sample
        B = X0.shape[0]
        with torch.no_grad():
            F0 = self.forward(X0, C_sample)
            eps = 1e-3
            best = 0.0
            for _ in range(5):
                dX = torch.randn_like(X0)
                dX_flat = dX.reshape(B, -1)
                norms = dX_flat.norm(dim=1, keepdim=True)
                dX = dX / (norms.view(B,1,1,1) + EPS)
                F1 = self.forward(X0 + eps * dX, C_sample)
                Jv = (F1 - F0) / eps
                L = Jv.reshape(B, -1).norm(dim=1).max().item()
                best = max(best, L)
        return best

    def forward(self, X, C):
        return X[:, self.data["mask"]]

    def log_likelihood(self, X, C):
        X_obs = self.forward(X, C)
        Y_true = self.data["Y"].unsqueeze(0).expand(X.shape[0], -1, -1, -1)
        diff = X_obs - Y_true
        sigma = float(self.current_sigma)
        ll = -0.5 * torch.sum(diff * diff, dim=[1,2,3]) / (sigma*sigma)
        n_elem = diff[0].numel()
        ll = ll - 0.5 * n_elem * math.log(2*math.pi*sigma*sigma)
        return ll

class DistanceMeasurement(AdaptiveMeasurement):
    def __init__(self, name, config, X_true, C, device):
        super().__init__(name, config, device)
        self._create_measurement(X_true, C)
        if self.auto_lipschitz and not self.lipschitz_estimated:
            L_est = self.estimate_lipschitz_constant(X_true, C)
            self.noise_scheduler.update_lipschitz_constant(L_est)
            self.lipschitz_estimated = True
            print(f"Distances: Estimated Lipschitz constant = {L_est:.3f}")

    def _create_measurement(self, X_true, C):
        X_flat = X_true[0] if (X_true.dim() == 4 and X_true.shape[0] == 1) else X_true
        c0 = C[0] if (C.dim() == 2 and C.shape[0] == 1) else C.reshape(-1)
        n = X_flat.shape[0]
        D = self._compute_distance_matrix(X_flat.unsqueeze(0))[0]
        mask = torch.zeros_like(D)
        if self.config["n_distances"] == -1:
            mask = torch.ones_like(D) - torch.eye(n, device=self.device)
        else:
            valid = torch.where(c0 == 1)[0].tolist()
            cats = {0: [], 1: [], 2: []}
            for i in range(len(valid)):
                for j in range(i+1, len(valid)):
                    d = abs(valid[i]-valid[j])
                    cats[0 if d<5 else 1 if d<20 else 2].append((valid[i],valid[j]))
            per_cat = self.config["n_distances"]//3
            pairs = []
            for k in cats:
                if cats[k]:
                    idxs = np.random.choice(len(cats[k]), min(per_cat,len(cats[k])),replace=False)
                    pairs += [cats[k][i] for i in idxs]
            for (i,j) in pairs:
                mask[i,j] = mask[j,i] = 1.0
        Y_clean = mask * D
        sigma_true = float(self.config.get("true_noise_level", self.current_sigma))
        Y = Y_clean + torch.randn_like(Y_clean)*sigma_true*mask
        Y = torch.clamp(Y, min=0.0)
        self.data = {
            "mask": mask,
            "Y": Y,
            "Y_clean": Y_clean,
            "true_noise_level": sigma_true,
            "n_observed": int((mask.sum()/2).item()),
            "n_total": int(n*(n-1)/2)
        }

    def estimate_lipschitz_constant(self, X_sample: torch.Tensor, C_sample: torch.Tensor) -> float:
        X0 = X_sample.unsqueeze(0) if X_sample.ndim == 3 else X_sample
        B = X0.shape[0]
        with torch.no_grad():
            F0 = self.forward(X0, C_sample)
            eps = 1e-3
            best = 0.0
            for _ in range(5):
                dX = torch.randn_like(X0)
                dX_flat = dX.reshape(B, -1)
                norms = dX_flat.norm(dim=1, keepdim=True)
                dX = dX / (norms.view(B,1,1,1) + EPS)
                F1 = self.forward(X0 + eps * dX, C_sample)
                Jv = (F1 - F0) / eps
                L = Jv.reshape(B, -1).norm(dim=1).max().item()
                best = max(best, L)
        return best

    def _compute_distance_matrix(self, X):
        CA = X[:,:,1,:]
        return torch.cdist(CA,CA)

    def forward(self, X, C):
        return self._compute_distance_matrix(X)*self.data["mask"]

    def log_likelihood(self, X, C):
        pred = self.forward(X,C)
        Y = self.data["Y"].unsqueeze(0).expand_as(pred)
        diff = (pred-Y)*self.data["mask"]
        sigma = float(self.current_sigma)
        ll = -0.5 * torch.sum(diff*diff, dim=[1,2])/(sigma*sigma)
        n_obs = float(self.data["mask"].sum().item())
        ll = ll - 0.5 * n_obs*math.log(2*math.pi*sigma*sigma)
        return ll

class DensityMeasurement(AdaptiveMeasurement):
    def __init__(self, name: str, config: dict, X_true: torch.Tensor, device: str):
        super().__init__(name, config, device)
        if not HAS_MRCFILE:
            raise ImportError("mrcfile is required for density measurements")

        self.resolution = float(config.get("resolution", 2.0))
        self.n_fourier = int(config.get("n_fourier_samples", 8000))
        self.mrc_path = config["mrc_path"]

        with mrcfile.open(self.mrc_path) as mrc:
            data = torch.tensor(mrc.data, dtype=torch.float32, device=device)
            self.nx, self.ny, self.nz = int(mrc.header.nx), int(mrc.header.ny), int(mrc.header.nz)
            
            try:
                voxel_x = float(mrc.voxel_size.x)
                voxel_y = float(mrc.voxel_size.y)
                voxel_z = float(mrc.voxel_size.z)
            except:
                voxel_x = mrc.header.cella.x / self.nx
                voxel_y = mrc.header.cella.y / self.ny
                voxel_z = mrc.header.cella.z / self.nz
            
            self.voxel_size = torch.tensor([voxel_x, voxel_y, voxel_z], device=device, dtype=torch.float32)
            
            try:
                origin_x = float(mrc.header.origin.x)
                origin_y = float(mrc.header.origin.y)
                origin_z = float(mrc.header.origin.z)
            except:
                origin_x = float(mrc.header.nxstart) * voxel_x
                origin_y = float(mrc.header.nystart) * voxel_y
                origin_z = float(mrc.header.nzstart) * voxel_z
            
            self.origin = torch.tensor([origin_x, origin_y, origin_z], device=device, dtype=torch.float32)
            self.box_size = max(self.nx, self.ny, self.nz)

        X_flat = X_true[0] if X_true.dim() == 4 else X_true
        protein_com = X_flat.reshape(-1, 3).mean(dim=0)
        density_center = torch.tensor([
            self.nx * self.voxel_size[0] / 2.0,
            self.ny * self.voxel_size[1] / 2.0,
            self.nz * self.voxel_size[2] / 2.0
        ], device=device) + self.origin
        self.coord_offset = density_center - protein_com

        f_exp = fft_density(data, norm='backward').flatten()
        freq_x = torch.fft.fftfreq(self.nx, d=float(self.voxel_size[0]), device=device)
        freq_y = torch.fft.fftfreq(self.ny, d=float(self.voxel_size[1]), device=device)
        freq_z = torch.fft.fftfreq(self.nz, d=float(self.voxel_size[2]), device=device)
        fx, fy, fz = torch.meshgrid(freq_x, freq_y, freq_z, indexing='ij')
        freq_mag = torch.sqrt(fx*fx + fy*fy + fz*fz).flatten()
        freq_max = 1.0 / self.resolution
        shell_width = 0.05 / self.resolution
        valid = torch.where(freq_mag <= freq_max + shell_width)[0]

        if valid.numel() > self.n_fourier:
            sel = torch.randperm(valid.numel(), device=device)[:self.n_fourier]
            freq_idx = valid[sel]
        else:
            freq_idx = valid

        self.data = {
            "Y": f_exp[freq_idx].detach(),
            "fourier_idx": freq_idx,
            "n_observed": freq_idx.numel()
        }

        self.gauss_sigma_phys = self.resolution / (2.0 * math.sqrt(2.0 * math.log(2.0)))
        self.gauss_sigma_voxels = self.gauss_sigma_phys / self.voxel_size.mean().item()
        
        x_coords = torch.arange(self.nx, device=device, dtype=torch.float32) * self.voxel_size[0] + self.origin[0]
        y_coords = torch.arange(self.ny, device=device, dtype=torch.float32) * self.voxel_size[1] + self.origin[1]
        z_coords = torch.arange(self.nz, device=device, dtype=torch.float32) * self.voxel_size[2] + self.origin[2]
        self.grid_x, self.grid_y, self.grid_z = torch.meshgrid(x_coords, y_coords, z_coords, indexing='ij')

        if self.auto_lipschitz and not self.lipschitz_estimated:
            L_est = self.estimate_lipschitz_constant(X_true, None)
            self.noise_scheduler.update_lipschitz_constant(L_est)
            self.lipschitz_estimated = True
            print(f"Density: Estimated Lipschitz constant = {L_est:.3f}")

    def estimate_lipschitz_constant(self, X_sample: torch.Tensor, C_sample: torch.Tensor) -> float:
        X0 = X_sample.unsqueeze(0) if X_sample.ndim == 3 else X_sample
        B = X0.shape[0]
        with torch.no_grad():
            F0 = self.forward(X0, C_sample)
            eps = 1e-3
            best = 0.0
            for _ in range(5):
                dX = torch.randn_like(X0)
                dX_flat = dX.reshape(B, -1)
                norms = dX_flat.norm(dim=1, keepdim=True)
                dX = dX / (norms.view(B,1,1,1) + EPS)
                with torch.no_grad():
                    F1 = self.forward(X0 + eps * dX, C_sample)
                Jv = (F1 - F0) / eps
                if torch.is_complex(Jv):
                    r = Jv.real.reshape(B, -1)
                    i = Jv.imag.reshape(B, -1)
                    stacked = torch.cat([r, i], dim=1)
                    L = stacked.norm(dim=1).max().item()
                else:
                    L = Jv.reshape(B, -1).norm(dim=1).max().item()
                best = max(best, L)
        return max(best, 1e-3)

    def _coords_to_density_differentiable(self, X: torch.Tensor) -> torch.Tensor:
        B, R, _, _ = X.shape
        atoms = X.reshape(B, R * 4, 3) + self.coord_offset.unsqueeze(0).unsqueeze(0)
        density = torch.zeros(B, self.nx, self.ny, self.nz, device=X.device, dtype=torch.float32)
        for b in range(B):
            grid_x = self.grid_x.unsqueeze(0)
            grid_y = self.grid_y.unsqueeze(0)
            grid_z = self.grid_z.unsqueeze(0)
            atom_x = atoms[b, :, 0:1].unsqueeze(1).unsqueeze(1)
            atom_y = atoms[b, :, 1:2].unsqueeze(1).unsqueeze(1)
            atom_z = atoms[b, :, 2:3].unsqueeze(1).unsqueeze(1)
            dx2 = (grid_x - atom_x) ** 2
            dy2 = (grid_y - atom_y) ** 2
            dz2 = (grid_z - atom_z) ** 2
            dist2 = dx2 + dy2 + dz2
            gaussians = torch.exp(-dist2 / (2.0 * self.gauss_sigma_phys ** 2))
            density[b] = gaussians.sum(dim=0)
        return density

    def forward(self, X: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        dens = self._coords_to_density_differentiable(X)
        f = fft_density(dens, norm='backward').reshape(X.shape[0], -1)
        return f[:, self.data["fourier_idx"]]

    def log_likelihood(self, X: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        pred = self.forward(X, C)
        Y = self.data["Y"].unsqueeze(0).expand_as(pred)
        diff = pred - Y
        sigma2 = self.current_sigma ** 2
        ll = -0.5 * torch.sum(diff.real**2 + diff.imag**2, dim=1) / sigma2
        ll -= self.data["n_observed"] * math.log(2 * math.pi * sigma2)
        return ll