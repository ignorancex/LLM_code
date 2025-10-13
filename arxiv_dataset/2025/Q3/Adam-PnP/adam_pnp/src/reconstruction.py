import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm
import math
from typing import Dict

from chroma import Chroma
from chroma.layers.structure.rmsd import CrossRMSD

from .measurements import AdaptiveMeasurement, PartialStructureMeasurement, DistanceMeasurement, DensityMeasurement, HAS_MRCFILE
from .utils import safe_sqrt, check_nan_inf

class AdaptiveMultiModalReconstruction(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.device = config["device"]
        torch.manual_seed(config["seed"])
        np.random.seed(config["seed"])
        print("Loading Chroma model...")
        self.chroma = Chroma()
        self.backbone_network = self.chroma.backbone_network
        self.design_network = self.chroma.design_network
        self.multiply_R = lambda Z,C: self.backbone_network.noise_perturb.base_gaussian._multiply_R(Z,C)
        self.multiply_R_inverse = lambda X,C: self.backbone_network.noise_perturb.base_gaussian._multiply_R_inverse(X,C)
        self.multiply_covariance = lambda g,C: self.backbone_network.noise_perturb.base_gaussian.multiply_covariance(g,C)
        self.rmsd_calc = CrossRMSD()
        self.t_threshold = config.get("t_sigma_update", 0.05)
        self.sigma_update_every = config.get("sigma_update_every", 10)
        self.sigma_ema_decay = config.get("sigma_ema_decay", 0.8)
        self.enable_noise_estimation = config.get("enable_noise_estimation", True)

    def create_measurements(self, X_gt, C_gt, S_gt) -> Dict[str, AdaptiveMeasurement]:
        measurements = {}
        mc = self.config["measurements"]
        if mc["partial_structure"]["enabled"]:
            measurements["partial_structure"] = PartialStructureMeasurement(
                "partial_structure", mc["partial_structure"], X_gt, C_gt,
                self.backbone_network, self.device
            )
            obs = measurements["partial_structure"].data["n_observed"]
            tot = measurements["partial_structure"].data["n_total"]
            print(f"Partial structure: {obs}/{tot} residues observed")
        if mc["distances"]["enabled"]:
            measurements["distances"] = DistanceMeasurement(
                "distances", mc["distances"], X_gt, C_gt, self.device
            )
            print(f"Distances: {measurements['distances'].data['n_observed']} pairs observed")
        if mc["density"]["enabled"]:
            if not HAS_MRCFILE:
                print("Warning: Density measurements disabled - mrcfile not available")
            else:
                measurements["density"] = DensityMeasurement(
                    "density", mc["density"], X_gt, self.device
                )
                print(f"Density: {measurements['density'].data['n_observed']} Fourier coeff observed")
        return measurements

    def temporal_schedule(self, epoch: int) -> float:
        t0, t1 = self.config["t_start"], self.config["t_end"]
        T = self.config["epochs"]
        sched = self.config["temporal_schedule"]
        if sched == "linear":
            return (t1-t0)*epoch/T + t0
        if sched == "sqrt":
            return (t0 - t1)*(1 - np.sqrt(epoch/T)) + t1
        if sched == "cosine":
            return t1 + 0.5*(t0-t1)*(1 + math.cos(math.pi*epoch/T))
        return t0

    def update_measurement_noise(self, X, C, measurements, t, t_tensor, epoch):
        with torch.no_grad():
            X0 = self.backbone_network.denoise(X.detach(), C, t_tensor) if self.config["use_diffusion"] else X.detach()
            for name, meas in measurements.items():
                if not meas.config.get("learn_noise", True):
                    continue
                
                update_freq = meas.config.get("sigma_update_every", self.sigma_update_every)
                t_thresh = meas.config.get("t_sigma_update", self.t_threshold)
                warmup_epochs = meas.config.get("sigma_update_warmup", 50)
                
                if (t >= t_thresh) or (epoch < warmup_epochs) or (epoch % update_freq != 0):
                    continue
                
                if name == "partial_structure":
                    X_obs = meas.forward(X0, C)
                    Y = meas.data["Y"].unsqueeze(0).expand_as(X_obs)
                    residuals = X_obs - Y
                    var_emp = torch.median((residuals**2).reshape(-1)).item()
                elif name == "distances":
                    pred = meas.forward(X0, C)
                    Y = meas.data["Y"].unsqueeze(0).expand_as(pred)
                    mask = meas.data["mask"].unsqueeze(0).expand_as(pred)
                    diffs = (pred - Y) * mask
                    nonzero = diffs.reshape(-1)[mask.reshape(-1) > 0.5]
                    var_emp = torch.median(nonzero**2).item() if nonzero.numel() > 0 else (diffs**2).sum().item() / mask.sum().item()
                else:
                    pred = meas.forward(X0, C)
                    Y = meas.data["Y"].unsqueeze(0).expand_as(pred)
                    diff = pred - Y
                    sq = (diff.real**2 + diff.imag**2).reshape(-1)
                    var_emp = torch.median(sq).item()
                
                L = meas.noise_scheduler.L
                sigma_t = float(self.backbone_network.noise_perturb.noise_schedule.sigma(t_tensor).item())
                bias_factor = meas.config.get("bias_correction_factor", 0.5) * (1 - t)
                bias = bias_factor * (L * sigma_t) ** 2
                sigma2_new = max(1e-12, var_emp - bias)
                sigma_meas_new = math.sqrt(sigma2_new)
                
                current_sigma = meas.get_learned_sigma_meas()
                max_change_ratio = 2.0
                sigma_meas_new = np.clip(
                    sigma_meas_new,
                    current_sigma / max_change_ratio,
                    current_sigma * max_change_ratio
                )
                
                min_sigma = meas.config.get("min_sigma", 0.01)
                max_sigma = meas.config.get("max_sigma", 10.0)
                sigma_meas_new = float(np.clip(sigma_meas_new, min_sigma, max_sigma))
                
                if abs(np.log(sigma_meas_new / current_sigma)) > np.log(1.5):
                    print(f" Large sigma change for {name}: {current_sigma:.3f} -> {sigma_meas_new:.3f}")
                
                base_decay = meas.config.get("sigma_ema_decay", self.sigma_ema_decay)
                ema_decay = base_decay + (1 - base_decay) * t
                meas.noise_scheduler.update_sigma(sigma_meas_new, ema_decay)

    def get_gradient_measurements(self, Z, C, measurements, epoch, t):
        grads, losses, noise_levels = {}, {}, {}
        Cb = C.unsqueeze(0).expand(Z.shape[0], -1) if C.dim()==1 else C
        
        for name, meas in measurements.items():
            Zl = Z.clone().detach().requires_grad_(True)
            Xl = self.multiply_R(Zl, Cb)
            
            if name == "partial_structure" and meas.data.get("precon") is not None:
                pre = meas.data["precon"]
                Yf = meas.data["Y"].reshape(-1,1)
                batch = Zl.shape[0]
                Um1 = pre["U_inv"].unsqueeze(0).expand(batch,-1,-1)
                Vh = pre["Vh"].unsqueeze(0).expand(batch,-1,-1)
                S = pre["S"]
                Yb = Yf.unsqueeze(0).expand(batch,-1,1)
                Um1Y = torch.bmm(Um1, Yb).squeeze(-1)
                Zf = Zl.reshape(batch, -1, 1)
                VhZ = torch.bmm(Vh, Zf).squeeze(-1)
                res = VhZ[:,:Um1Y.shape[1]] - Um1Y/S.unsqueeze(0)
                loss = 0.5*torch.sum(res*res)/(meas.current_sigma**2)
                loss.backward()
                grad = Zl.grad.clone()
            else:
                ll = meas.log_likelihood(Xl, Cb).sum()
                neg_ll = -ll
                neg_ll.backward()
                loss = neg_ll
                grad = Zl.grad.clone() if Zl.grad is not None else torch.zeros_like(Zl)
            
            grad = torch.nan_to_num(grad)
            norm = torch.norm(grad)
            thresh = self.config["gradient_clip"]
            if norm > thresh:
                grad *= (thresh / norm)
            grads[name] = grad
            losses[name] = loss.item()
            noise_levels[name] = meas.current_sigma
        
        if self.config.get("auto_scale_modalities", True):
            base_weights = {}
            for name in measurements:
                base_weights[name] = 1.0 / (noise_levels[name] + 1e-6)
            
            total_weight = sum(base_weights.values())
            n_modalities = len(measurements)
            for name in measurements:
                scale_factor = base_weights[name] / total_weight * n_modalities
                grads[name] = grads[name] * scale_factor
                self.config[f"weight_{name}"] = scale_factor
        
        return grads, losses

    def run_algorithm(self, measurements: Dict[str,AdaptiveMeasurement], X_gt, C_gt, S_gt):
        pop = self.config["population_size"]
        c0 = C_gt[0] if (C_gt.dim()==2 and C_gt.shape[0]==1) else C_gt.reshape(-1)
        Cb = c0.unsqueeze(0).expand(pop, -1)
        s0 = S_gt[0] if (S_gt.dim()==2 and S_gt.shape[0]==1) else S_gt.reshape(-1)
        Sb = s0.unsqueeze(0).expand(pop, -1)
        Z = torch.randn(pop, c0.numel(), 4, 3, device=self.device)
        X = self.multiply_R(Z, Cb)
        V_sum = torch.zeros_like(Z)
        
        metrics = {
            "epoch": [],
            "rmsd": [],
            "rmsd_ca": [],
            "t": [],
            "best_rmsd": [],
            "best_rmsd_epoch": 0,
            "best_rmsd_overall": float("inf")
        }
        for name in measurements:
            metrics[f"loss_{name}"] = []
            metrics[f"sigma_meas_{name}"] = []
            metrics[f"lipschitz_{name}"] = []
        
        best_rmsd = float("inf")
        best_X = None
        best_epoch = 0
        
        pbar = tqdm(range(self.config["epochs"]), 
                   desc=f"Adam-PnP {'(Adaptive sigma)' if self.enable_noise_estimation else '(Fixed sigma)'}")
        
        for epoch in pbar:
            try:
                t = self.temporal_schedule(epoch)
                t_tensor = torch.tensor(t, device=self.device, dtype=torch.float32)
                for meas in measurements.values():
                    meas.update_effective_noise(t, self.backbone_network.noise_perturb.noise_schedule)
                if self.enable_noise_estimation and (epoch>0) and (epoch % self.sigma_update_every==0):
                    self.update_measurement_noise(X, Cb, measurements, t, t_tensor, epoch)
                
                losses_dict = {}
                
                X = check_nan_inf(X, "X before denoise")
                X0 = (self.backbone_network.denoise(X.detach(), Cb, t_tensor)
                      if self.config["use_diffusion"] else X)
                X0 = check_nan_inf(X0, "X0 after denoise")
                Z0 = self.multiply_R_inverse(X0, Cb)
                Z0 = check_nan_inf(Z0, "Z0")
                grads, losses_dict = self.get_gradient_measurements(Z0, Cb, measurements, epoch, t)
                
                grad_sum = torch.zeros_like(Z0)
                for n, g in grads.items():
                    weight = self.config.get(f"weight_{n}", 1.0)
                    grad_sum += weight * g
                grad_sum = check_nan_inf(grad_sum, "grad_sum")
                
                rho = self.config["rho_adam_pnp"]
                eta_base = self.config["lr_adam_pnp"]
                
                if eta_base < 1e-8 or eta_base > 1.0:
                    print(f"Warning: learning rate {eta_base} out of bounds, using default 0.01")
                    eta_base = 0.01
                    self.config["lr_adam_pnp"] = eta_base
                
                eta = eta_base * (1 + t)
                V_sum = rho * V_sum + eta * grad_sum
                V_sum = check_nan_inf(V_sum, "V_sum")
                Z0 = Z0 - V_sum
                Z0 = check_nan_inf(Z0, "Z0 after update")
                
                if self.config["use_diffusion"] and epoch < self.config["epochs"]-1:
                    t2 = self.temporal_schedule(epoch+1)
                    alpha = self.backbone_network.noise_perturb.noise_schedule.alpha(
                        torch.tensor(t2, device=self.device, dtype=torch.float32))
                    sigma = self.backbone_network.noise_perturb.noise_schedule.sigma(
                        torch.tensor(t2, device=self.device, dtype=torch.float32))
                    Z = alpha*Z0 + sigma*torch.randn_like(Z0)
                else:
                    Z = Z0
                Z = check_nan_inf(Z, "Z final")
                X = self.multiply_R(Z, Cb)
                X = check_nan_inf(X, "X final")
                
                if (epoch+1)%self.config["log_every"]==0 or epoch==self.config["epochs"]-1:
                    with torch.no_grad():
                        Xpop = self.multiply_R(Z, Cb)
                        rmsds, rmsd_cas = self._compute_rmsd(Xpop, X_gt, Cb, C_gt)
                        idx = int(np.argmin(rmsds))
                        current_best_rmsd = rmsds[idx]
                        
                        if current_best_rmsd < best_rmsd:
                            best_rmsd = current_best_rmsd
                            best_X = Xpop[idx:idx+1].clone()
                            best_epoch = epoch + 1
                        
                        metrics["epoch"].append(epoch+1)
                        metrics["rmsd"].append(rmsds)
                        metrics["rmsd_ca"].append(rmsd_cas)
                        metrics["t"].append(t)
                        metrics["best_rmsd"].append(best_rmsd)
                        metrics["best_rmsd_epoch"] = best_epoch
                        metrics["best_rmsd_overall"] = best_rmsd
                        
                        for n in measurements:
                            metrics[f"loss_{n}"].append(losses_dict.get(n,0.0))
                            metrics[f"sigma_meas_{n}"].append(measurements[n].get_learned_sigma_meas())
                            metrics[f"lipschitz_{n}"].append(measurements[n].noise_scheduler.L)
                        
                        postfix = {"RMSD":f"{best_rmsd:.2f}Å","t":f"{t:.3f}"}
                        if self.enable_noise_estimation:
                            for n in measurements:
                                postfix[f"sigma_{n[:4]}"] = f"{measurements[n].get_learned_sigma_meas():.3f}"
                        else:
                            postfix["sigma"] = "fixed"
                        
                        if self.config.get("auto_scale_modalities", True) and epoch > 0:
                            weights_str = ""
                            for n in measurements:
                                w = self.config.get(f"weight_{n}", 1.0)
                                weights_str += f"{n[0]}:{w:.2f} "
                            postfix["weights"] = weights_str.strip()
                        pbar.set_postfix(postfix)
                        
            except RuntimeError as e:
                if "CUDA" in str(e) and "illegal memory access" in str(e):
                    print(f"\nCUDA error at epoch {epoch}, attempting to recover...")
                    torch.cuda.empty_cache()
                    Z = Z.detach().cpu()
                    Z = Z + 0.01 * torch.randn_like(Z)
                    Z = Z.to(self.device)
                    X = self.multiply_R(Z, Cb)
                    V_sum = torch.zeros_like(Z)
                else:
                    raise e
        
        if best_X is None:
            best_X = self.multiply_R(Z, Cb)[0:1]
        
        final_X = self.multiply_R(Z, Cb)[0:1]
        
        return best_X, final_X, metrics

    def _compute_rmsd(self, X, X_gt, C, C_gt):
        pop = X.shape[0]
        rmsd_all = []
        rmsd_ca = []
        c_pop = C[0] if C.dim()==2 else C
        c_gt = C_gt[0] if C_gt.dim()==2 else C_gt
        for i in range(pop):
            pred_all = X[i, c_pop==1].cpu().reshape(1,-1,3)
            true_all = X_gt[0, c_gt==1].cpu().reshape(1,-1,3)
            r_all,_ = self.rmsd_calc.pairedRMSD(pred_all, true_all, compute_alignment=True)
            rmsd_all.append(r_all.item())
            pred_ca = X[i, c_pop==1,1,:].cpu().reshape(1,-1,3)
            true_ca = X_gt[0,c_gt==1,1,:].cpu().reshape(1,-1,3)
            r_ca,_ = self.rmsd_calc.pairedRMSD(pred_ca, true_ca, compute_alignment=True)
            rmsd_ca.append(r_ca.item())
        return rmsd_all, rmsd_ca

    def run(self, X_gt, C_gt, S_gt):
        measurements = self.create_measurements(X_gt, C_gt, S_gt)
        X_best, X_final, metrics = self.run_algorithm(measurements, X_gt, C_gt, S_gt)
        return X_best, X_final, metrics, measurements