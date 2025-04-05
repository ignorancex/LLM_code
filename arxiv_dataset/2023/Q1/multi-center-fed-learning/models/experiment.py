
import os
import sys
import numpy as np
import time
import pandas as pd

from utils_io import get_job_config
from utils.model_utils import read_data
from utils.args import parse_job_args
from mlhead_utilfuncs import save_clustereva_file


from fedsem import Fedsem_Trainer
from fedavg import Fedavg_Trainer
from fedprox import Fedprox_Trainer
from fedsgd import Fedsgd_Trainer
from fedbayes import Fedbayes_Sing_Trainer
from fedrobust import Fedrobust_Trainer
from modelsaver import Model_Saver
from modelpoison import Model_Poison

def read_yamlconfig(args):
    yaml_file = os.path.join("..", "configs", args.experiment, args.configuration)
    job = get_job_config(yaml_file)
    params = job['PARAMS']

    rounds = params['num-rounds']
    print("config rounds: ", rounds)

    lr = params['lr']
    print("config lr: ", lr )    
    
    epochs = params['epochs']
    print("config epochs: ", epochs)
    
    clients_per_round = params['clients-per-round']
    print("config clients per round: ", clients_per_round)
    
    if ('poisoning' not in params.keys()):
        poisoning = False
        poi_agents = 0
    else:
        poisoning = True
        poi_agents = int(params['num_agents'])
    print("config poisoning: ", poisoning)
    print("config num agents: ", poi_agents)
    params['poisoning'] = poisoning
    
    return params

def main():
    args = parse_job_args()
    config = read_yamlconfig(args)
    
    # changed 29/08/2021 the follow lines are for google cloud dir
    base_dir =  os.path.join(os.path.expanduser('~'), 'leaf')
    train_data_dir = os.path.join(base_dir, 'data', args.dataset, 'data', 'train')
    test_data_dir = os.path.join(base_dir, 'data', args.dataset, 'data', 'test')   
    users, groups, train_data, test_data = read_data(train_data_dir, test_data_dir)
    
    exp_seeds, book_keep = config["exp-seeds"], [0.] * len(config["exp-seeds"])
    metrics_list = {"bic": [], "db_score": []}
    config["benchmark"] = 0
    
    for j, rnd_sed in enumerate(exp_seeds):
        config["seed"] = rnd_sed
        if args.experiment == 'fedavg':
            trainer = Fedavg_Trainer(users, groups, train_data, test_data)
            metric = trainer.begins(config, args)
            trainer.ends()
        elif args.experiment == 'fedprox':
            trainer = Fedprox_Trainer(users, groups, train_data, test_data)
            metric = trainer.begins(config, args)
            trainer.ends()
        elif args.experiment == 'fedrobust':
            trainer = Fedrobust_Trainer(users, groups, train_data, test_data)
            metric = trainer.begins(config, args)
            trainer.ends()            
        elif args.experiment == 'fedcluster':
            pass
        elif args.experiment == 'fedsgd':
            trainer = Fedsgd_Trainer(users, groups, train_data, test_data)
            metric =trainer.begins(config, args)
            trainer.ends()            
        elif args.experiment == 'fedbayes':
            trainer = Fedbayes_Sing_Trainer(users, groups, train_data, test_data)
            metric =trainer.begins(config, args)
            trainer.ends()
        elif args.experiment == 'modelsaver':
            trainer = Model_Saver(users, groups, train_data, test_data)
            metric = trainer.begins(config, args)
            trainer.ends()        
        elif args.experiment == 'fedsem':
            trainer = Fedsem_Trainer(users, groups, train_data, test_data) 
            metric = trainer.begins(config, args)
            trainer.ends() 
        elif args.experiment == 'fedpoison':
            trainer = Model_Poison(users, groups, train_data, test_data) 
            metric = trainer.begins(config, args)
            trainer.ends()
        elif args.experiment == 'fedrobust_benchmark':
            config["benchmark"] = 1
            trainer = Fedrobust_Trainer(users, groups, train_data, test_data)
            metric = trainer.begins(config, args)
            trainer.ends()
            metrics_list["bic"].append(trainer._bic)
            metrics_list["db_score"].append(trainer._db_score)
        elif args.experiment == 'fedsem_benchmark':
            config["benchmark"] = 1
            trainer = Fedsem_Trainer(users, groups, train_data, test_data) 
            metric = trainer.begins(config, args)
            trainer.ends() 
            metrics_list["bic"].append(trainer._bic)
            metrics_list["db_score"].append(trainer._db_score)
        else:
            print("Applications not defined. Please check configs directory if the name is right.")
            break
        book_keep[j] = metric
        
    finals = np.array(book_keep) * 100
    print(finals)
    
    if args.experiment in ["fedrobust_benchmark", "fedsem_benchmark"]:
        df = pd.DataFrame(metrics_list)
        save_clustereva_file(args.experiment, df)
#     print("{} runs - std: {}, med: {}".format(len(exp_seeds), 
#                                               np.var(finals),
#                                              np.median(finals)))        
main()
