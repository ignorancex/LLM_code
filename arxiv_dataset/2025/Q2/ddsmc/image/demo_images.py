from omegaconf import OmegaConf
import os
from posterior_samplers.cond_sampling_algos import ddsmc

@dataclass
class Config:
    ddsmc_recon = "tweedie"
    ddsmc_eta = 0.5

elif args.algo == "ddsmc":
    ddsmc_config = OmegaConf.load(os.path.join("configs", f"ddsmc_{args.model}_{args.std}.yaml"))
    ddsmc_config["reconstruct_fn"] = args.ddsmc_recon
    ddsmc_config["eta"] = args.ddsmc_eta
    samples = ddsmc(initial_noise, epsilon_net.net, y_0, H_funcs, sigma_y,
        device, 
        ddsmc_config.smc_config, 
        ddsmc_config.annealing_scheduler_config,
        ddsmc_config.rho_scheduler_config, 
        ddsmc_config.diffusion_scheduler_config,
        ddsmc_config.eta,
        ddsmc_config.reconstruct_fn).clamp(-1, 1)