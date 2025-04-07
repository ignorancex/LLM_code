import tensorflow as tf
from design_baselines.logger import Logger
import os
os.environ["RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE"] = "1"  
import glob

# for task in ['ant', 'dkitty', 'tf-bind-8', 'tf-bind-10']:
for task in ['ant']:
    for algo in ['gradient-ascent']:
        os.system(f'{algo} {task} --local-dir results/baselines/{algo}/{task} --cpus 5 \
                    --gpus 1 \
                    --num-parallel 1 \
                    --num-samples 1')

