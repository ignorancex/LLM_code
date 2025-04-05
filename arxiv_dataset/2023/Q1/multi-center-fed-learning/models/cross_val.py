import os
import sys
import time

from utils_io import get_job_config
from utils.args import parse_job_args

def read_yamlconfig(args):
    yaml_file = os.path.join("..", "configs", args.experiment, "job.yaml")
    job = get_job_config(yaml_file)
    params = job['PARAMS']

    rounds = params['num-rounds']
    print("config rounds: ", rounds)

    lr = params['lr']
    print("config lr: ", lr )    
    
    epochs = params['epochs']
    print("config epochs: ", epochs)
    
    return params

def main():
    args = parse_job_args()
    config = read_yamlconfig(args)

    for i in config["exp-seeds"]:
        print(int(i))
    
    config["seed"] = 9999
    print(config["seed"])
    
main()
