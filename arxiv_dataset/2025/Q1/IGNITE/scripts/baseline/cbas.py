import tensorflow as tf
from design_baselines.logger import Logger
import os
os.environ["RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE"] = "1"  
import glob

for task in ['ant', ]:
    for algo in ['cbas']:
        os.system(f'{algo} {task} --local-dir results/baselines/{algo}/{task} --cpus 5 \
                    --gpus 1 \
                    --num-parallel 1 \
                    --num-samples 1')

