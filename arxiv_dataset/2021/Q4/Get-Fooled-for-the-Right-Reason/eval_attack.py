from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import subprocess

directory = "models/modelGTP_cifar10"
subprocess.run("python PGD_attack.py -d cifar10 --data_path datasets/cifar10 --attack_name fgsm --save_eval_log --num_steps 1 --no-random_start --step_size 8 --model_dir {} ; python run_attack.py -d cifar10 --data_path datasets/cifar10 --attack_name fgsm --save_eval_log --model_dir {} ; python PGD_attack.py -d cifar10 --data_path datasets/cifar10 --attack_name pgds5 --save_eval_log --num_steps 5 --model_dir {} ; python run_attack.py -d cifar10 --data_path datasets/cifar10 --attack_name pgds5 --save_eval_log --num_steps 5 --model_dir {} ; python PGD_attack.py -d cifar10 --data_path datasets/cifar10 --attack_name pgds10 --save_eval_log --num_steps 10 --model_dir {} ; python run_attack.py -d cifar10 --data_path datasets/cifar10 --attack_name pgds10 --save_eval_log --num_steps 10 --model_dir {} ; python PGD_attack.py -d cifar10 --data_path datasets/cifar10 --attack_name pgds20 --save_eval_log --num_steps 20 --model_dir {} ; python run_attack.py -d cifar10 --data_path datasets/cifar10 --attack_name pgds20 --save_eval_log --num_steps 20 --model_dir {}".format(directory, directory, directory, directory, directory, directory, directory, directory, directory, directory), shell=True)
print("{}: Ended evaluation on fgsm and pgd  attacks".format(datetime.now()))

