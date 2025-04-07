import tensorflow as tf
from design_baselines.logger import Logger
import os
os.environ["RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE"] = "1"  
import glob

for lambda_ in [0.01]:
    for rho in [0.2]:
        for r in [0.2]:
            for task in ['ant', 'dkitty', 'tf-bind-8', 'tf-bind-10']:
                for algo in ['gradient-ascent-mean-ensemble']:
                    os.system(f'{algo}-IGNITE2 {task} --local-dir results/ours/lambda_{lambda_}/rho_{rho}/r_{r}/{algo}-IGNITE2/{task} --cpus 5 \
                                --gpus 1 \
                                --num-parallel 1 \
                                --num-samples 16 \
                                --lambda_ {lambda_} \
                                --rho {rho} \
                                --r {r} ')

