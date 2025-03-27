import argparse
import numpy as np
import os, sys
import time
from functools import partial
import ray
from ray import tune
from ray.air import RunConfig

from ray.train.torch import TorchTrainer, TorchConfig
from ray.air.config import ScalingConfig

from ray.tune.tuner import Tuner, TuneConfig

from ray.tune.schedulers import ResourceChangingScheduler, ASHAScheduler

def parsIni():
    parser = argparse.ArgumentParser(description='Ray Tune Adaptive Example')
    parser.add_argument('--num-samples', type=int, default=24,
                    help='number of samples to train (default: 32)')
    parser.add_argument('--scheduler', type=str, default='RAND',
                    help='scheduler for tuning (default: RandomSearch)')
    parser.add_argument('--res-changer', action='store_true',
                    help='use the resource changing schedulers (default: false)')
    parser.add_argument('--seed', type=int, default='111',
                    help='seed to use (default: 111)')
    parser.add_argument('--amp', action='store_true',
                    help='use the automated mixed precision (default: false)')
    parser.add_argument('--data-dir', type=str, default='None',
                    help='dataset directory')
    parser.add_argument('--storage-path', type=str, default='None',
                    help='directory to store ray results')
    parser.add_argument('--report-interval', type=int, default=1,
                    help='after how many epochs to report a result (default: 1)')
    parser.add_argument('--num-epochs', type=int, default=10,
                    help='How many epochs to run per trial at max (default: 10)')
    parser.add_argument('--grace-period', type=int, default=1,
                    help='grace period for the ASHA scheduler (default: 1)')
    parser.add_argument('--metric', type=str, default='val_acc',
                    help='which metric to use (default: val_acc)')
    parser.add_argument('--mode', type=str, default='max',
                    help='min for minimizing the objective, max for maximizing')
    
    # arguments for the resource allocation per trial
    parser.add_argument('--base-cpus', type=int, default=1,
                    help='base cpus to use per trial for the resource adaptive scheduler (default: 1)')
    parser.add_argument('--base-gpus', type=int, default=1,
                    help='base gpus to use per trial for the resource adaptive scheduler (default: 1)')
    parser.add_argument('--num-workers', type=int, default=1, 
                    help='number of parallel workers (=base_cpus and base_gpus) to use to train a single trial (default: 1 = (1* base-cpu, 1*base-gpu)')
    parser.add_argument('--reduction-factor', type=int, default=2, 
                    help='reduction factor to use for ASHA (default: 2)')    
    parser.add_argument('--scaling-factor', type=int, default=2, 
                    help='reduction factor to use for RASDA, should be same as reduction factor (default: 2)')    
    return parser


def main(args):
        
    ray.init(address=os.environ['ip_head'], _node_ip_address=str(os.environ["SLURMD_NODENAME"] + "i"))     
    
    # set seed
    np.random.seed(args.seed)
    
    # set search space
    
    config = {
        "lr": tune.loguniform(1e-5, 1),
        "wd": tune.uniform(0, 1e-1),
        "optimizer": tune.choice(["adam", "sgd"]),
        "activation_function": tune.choice(["ReLU", "LeakyReLU", "SELU", "Tanh", "Sigmoid"]),
        "init_warmup_steps": tune.choice([1,2,3,4,5]),
        "rescale_warmup_steps": tune.choice([1,2]),
        "kernel_size": tune.choice([3]),
        "conv_init": tune.choice(["kaiming", "xavier"]),
        "model": tune.choice([50]),
        "bs": tune.choice([128]),    
        "data_dir": tune.choice([args.data_dir]),
        "workers": tune.choice([args.num_workers]),
        "seed": tune.choice([args.seed]),
        "amp": tune.choice([args.amp]),
        "report_interval": tune.choice([args.report_interval]),
        "num_epochs": tune.choice([args.num_epochs]),
    }


    if (args.scheduler == "ASHA"):
        scheduler = ASHAScheduler(
               max_t=args.num_epochs,
               grace_period=args.grace_period,
               reduction_factor=args.reduction_factor)

        search_alg = None
        
    if (args.res_changer):
        from res_allocate import create_resource_allocation 

        resource_allocation_function = create_resource_allocation(base_cpus = args.base_cpus, base_gpus = args.base_gpus, num_workers=args.num_workers, scaling_factor = args.scaling_factor, min_t=args.grace_period, max_t=args.num_epochs)
        res_scheduler = ResourceChangingScheduler(
                base_scheduler=scheduler,
                resources_allocation_function=resource_allocation_function
            )
        
        scheduler = res_scheduler
        
        
    run_config = RunConfig(
        # Directory to store results in (will be local_dir/name).
        storage_path=str(args.storage_path),
        # Name of the training run (directory name).
        name=f'scheduler_{args.scheduler}_res_changer_{args.res_changer}_seed_{args.seed}',
        # Low training verbosity.
        verbose=1,
        stop={"training_iteration": args.num_epochs}

    )
    
    from cases.ImagenetTrainLoopDALI import imagenet_dali_train_loop_per_worker as train_loop_per_worker
    
    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        scaling_config=ScalingConfig(trainer_resources={"CPU": 0}, num_workers=args.num_workers, use_gpu=True, resources_per_worker={"CPU": args.base_cpus, "GPU": args.base_gpus}),
        torch_config=TorchConfig(backend="nccl", timeout_s=3600),
    )
    
    tuner = Tuner(
        trainer,
        param_space={"train_loop_config": config,
                    },
        tune_config=TuneConfig(
           num_samples=args.num_samples,
           metric=str(args.metric),
           mode=str(args.mode),
           scheduler=scheduler,
           search_alg=search_alg,
           ),
        run_config=run_config
    )
    
    start_time = time.time()
    
    result = tuner.fit()
    
    total_time = time.time() - start_time
    
    print("Total runtime: ", total_time)
    
    result_df = result.get_dataframe()
    result_df.to_csv(f'scheduler_{args.scheduler}_res_changer_{args.res_changer}_seed_{args.seed}.csv')

    best_result = result.get_best_result(metric=str(args.metric), mode=str(args.mode))    
    
    print("Best results: ", best_result) 
    

if __name__ == "__main__":
    
    parser = parsIni()
    args = parser.parse_args()

    main(args)
