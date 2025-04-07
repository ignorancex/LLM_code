import tensorflow as tf
from design_baselines.logger import Logger
import os
os.environ["RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE"] = "1"  
import glob

for epsilon in [0.1]:
    for lambda_ in [0.001]:
        for eta_lambda_ in [1e-3]:
            for rho in [0.05]:
                for r in [0.05]:
                    for task in ['ant', 'dkitty', 'tf-bind-8', 'tf-bind-10']:
                        for algo in ['bo-qei']:
                                os.system(f'{algo}-IGNITE {task} --local-dir results/IGNITE/epsilon_{epsilon}/lambda_{lambda_}/eta_lambda_{eta_lambda_}/rho_{rho}/r_{r}/{algo}-IGNITE/{task} --cpus 5 \
                                        --gpus 1 \
                                        --num-parallel 1 \
                                        --num-samples 16 \
                                        --lambda_ {lambda_} \
                                        --eta_lambda_ {eta_lambda_} \
                                        --epsilon {epsilon} \
                                        --rho {rho} \
                                        --r {r} ')

