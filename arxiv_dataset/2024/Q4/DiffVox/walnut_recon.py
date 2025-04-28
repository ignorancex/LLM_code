import argparse
import torch
from datetime import datetime
from pathlib import Path
from torchvision.transforms import Resize

from diffdrr.data import read 

from diffvox.optimize import optimize
from diffvox.data import Dataset_DiffVox, FastTensorDataLoader

from utils.walnut_data import load, get_source_target_vec
from utils.construct_ground_truth import reconstruct as reconstruct_trad

normalize = lambda x: (x - x.min()) / (x.max() - x.min())


class Dataset_Walnut(Dataset_DiffVox):
    def __init__(self, walnut_id, tube=[2], downsample=1, n_views=15, half_orbit=False, initialize_alg='None'):
        main_dir = Path(f'data/Walnut{walnut_id}/')
        
        self.walnut_id = walnut_id
        self.n_views = n_views
        self.gt_projs, vecs = load(main_dir / 'Projections/', 972, 768, n_views, orbits_to_recon=tube, half_orbit=half_orbit)

        self.gt_projs = torch.tensor(self.gt_projs)
        self.sources, self.targets = get_source_target_vec(vecs)
        self.sources = torch.stack(self.sources)
        self.targets = torch.stack(self.targets)
        self.subject = read(main_dir / 'gt.nii.gz')

        if downsample != 1:
            resizer = Resize((972//downsample, 768//downsample))
            self.gt_projs = resizer(self.gt_projs)
            self.sources = resizer(self.sources.permute(0,3,1,2)).permute(0,2,3,1)
            self.targets = resizer(self.targets.permute(0,3,1,2)).permute(0,2,3,1)


        self.sources = self.sources.reshape(1, -1, 3)
        self.targets = self.targets.reshape(1, -1, 3)
        self.gt_projs = self.gt_projs.reshape(1, 1, -1)
        self.init_vol = self.initialize_vol(initialize_alg)

    
    def initialize_vol(self, initialize_alg='None'):
        if initialize_alg == 'None':
            initialize_vol = None
            init_time = 0
        else:
            with torch.no_grad():
                if initialize_alg == "fdk":
                    n_itrs = 1
                elif initialize_alg == "cgls":
                    n_itrs = 20
                elif initialize_alg == "sirt":
                    n_itrs = 500
                elif initialize_alg == "nesterov":
                    n_itrs = 500
                initialize_vol, init_time = reconstruct_trad(walnut_id=self.walnut_id, n_views=self.n_views, algorithm=initialize_alg, n_itrs=n_itrs)
                initialize_vol = torch.from_numpy(initialize_vol)
                torch.cuda.empty_cache() # release memroy used for initialization
        return initialize_vol, init_time



def initialize(walnut_id, n_views, downsample, batch_size, half_orbit)-> FastTensorDataLoader:
    """get the data from dataset and initialize the dataloader"""
    dataset = Dataset_Walnut(walnut_id=walnut_id, downsample=downsample, n_views=n_views, half_orbit=half_orbit)
    projections, sources, targets, subject = dataset.get_data()
    init_vol, init_time = dataset.initialize_vol()
    print(f"Data loaded, using {n_views} projections")
    return FastTensorDataLoader(sources, targets, projections, subject, init_vol, batch_size=batch_size), init_time


def run(
        walnut_id: int, # walnut id to reconstruct
        n_views: int, # number of views to use
        downsample: int | float = 1, # downsample factor
        batch_size:int = 1_800_000, # batch size (number of rays to process at once)
        half_orbit: bool =False, # use half orbit or full orbit
        n_itr: int = 50, # optimization iterations
        lr: float = 0.01, # optimizer learning rate
        tv_coeff: float = 15, # tv loss coefficient
        shift: float = 0, # shif value for sigmoid
        loss_fn: str = "l1", # loss function to use (l1, l2, pcc, ncc)
        drr_params={'renderer': 'trilinear', 'sdd': 199.006188, 'height': 768, 'width': 972, 'delx':0.074800, 'patch_size': None, 'n_points': 500},
        density_regulator: str = 'softplus', # density regulator to use (sigmoid, clamp, softplus, None)
        tv_type: str = 'vl1', # method to calculate tv loss (vl1, l1, l2)
        drr_scale: float = 1.0, # scale values of the drr
        proj_name: str = 'walnut_recon', # wandb project name
        initialize_alg: str = 'None', # algorithm to initialize the density
        beta = 20,
        **kwargs,
) -> None:
    drr_params['n_points'] = kwargs.get('n_points', 500)
    drr_params['renderer'] = kwargs.get('renderer', 'trilinear')
    now_time = datetime.now().strftime("%m-%d__%H:%M") # used fo logging
    dataloader, init_time = initialize(walnut_id, n_views, downsample, batch_size, half_orbit)

    if log_wandb:
        import wandb
        wandb.login() # replace your wandb key here!
        wandb.init(
            project=proj_name,
            config={
                "walnut_id": walnut_id,
                "n_views": n_views,
                "downsample": downsample,
                "batch_size": batch_size,
                "n_itr": n_itr,
                "lr": lr,
                "tv_coeff": tv_coeff,
                "shift": shift,
                "loss_fn": loss_fn,
                "drr_params": drr_params,
                "density_regulator": density_regulator,
                "tv_type": tv_type,
                "half_orbit": half_orbit,
                "drr_scale": drr_scale,
                "lr_scheduler_decay": 1,
                'beta': beta,
            },
            name = f"w{walnut_id}_{n_views}_{drr_params['renderer']}_{now_time}",
            )
    density, losses, tvs, set_ssim, set_psnr, set_pcc, set_times = optimize(
        dataloader,
        n_itr,
        lr, 
        tv_coeff, 
        shift, 
        loss_fn, 
        drr_params, 
        density_regulator, 
        tv_type,
        drr_scale,
        beta,
        log_wandb
    )

    total_time = sum(set_times) + init_time
    if log_wandb:
        wandb.run.summary['total_time'] = total_time / 60 # in minutes
        wandb.run.summary['ssim'] = set_ssim[-1]
    save_loc = Path(f'./results/{proj_name}')
    save_loc.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            'tensors':{
                'est': density.cpu(), 
            },
            'metrics':{
                'loss': losses,
                'tv': tvs,
                'ssim': set_ssim,
                'psnr': set_psnr,
                'pcc': set_pcc,
                'tota_time': total_time,
                'time_delta': set_times,
            },
            'hyperparameters':{
                "walnut_id": walnut_id,
                "n_views": n_views,
                "downsample": downsample,
                "batch_size": batch_size,
                "n_itr": n_itr,
                "lr": lr,
                "tv_coeff": tv_coeff,
                "shift": shift,
                "loss_fn": loss_fn,
                "drr_params": drr_params,
                "density_regulator": density_regulator,
                "tv_type": tv_type,
                "half_orbit": half_orbit,
                'beta': beta
            }
        },
        save_loc / f'walnut{walnut_id}_{n_views}_{drr_params['renderer']}.pt',

    )
    if log_wandb:
        wandb.finish()



def main(**kwargs):
    run(**kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description= "Run optimization on walnut data")
    parser.add_argument("--walnut_id", type=int, default=3)
    parser.add_argument("--n_views", type=int, default=15)
    parser.add_argument("--downsample", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1_800_000)
    parser.add_argument("--n_itr", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--tv_coeff", type=float, default=15)
    parser.add_argument("--shift", type=float, default=0)
    parser.add_argument("--beta", type=float, default=20)
    parser.add_argument("--loss_fn", type=str, default="l1")
    parser.add_argument("--renderer", type=str, default='trilinear')
    parser.add_argument("--n_points", type=int, default=500)
    parser.add_argument("--drr_params", type=dict, default={'sdd': 199.006188, 'height': 768, 'width': 972, 'delx':0.074800, 'patch_size': None}, required=False)
    parser.add_argument("--density_regulator", type=str, default='softplus')
    parser.add_argument("--tv_type", type=str, default='vl1')
    parser.add_argument("--half_orbit", type=bool, default=False)
    parser.add_argument("--drr_scale", type=float, default=1.0)
    parser.add_argument("--proj_name", type=str, default='walnut_recon')
    parser.add_argument("--initialize_alg", type=str, default='None')
    parser.add_argument("--log_wandb", type=bool, default=False)
    args = parser.parse_args()
    # set log_wandb as global and pop it from the args
    global log_wandb
    log_wandb = args.log_wandb
    del args.log_wandb

    main(**vars(args))
    
