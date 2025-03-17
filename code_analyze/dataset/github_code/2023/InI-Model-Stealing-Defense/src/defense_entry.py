import os
import random
import numpy as np
import yaml
import argparse
from easydict import EasyDict
from pprint import pprint
import torch

from defenses.nd.nd import ND
from defenses.ini.ini import INI



def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="config.yaml")

args = parser.parse_args()


with open(args.config + "/config.yaml", "r") as fp:
    args = yaml.load(fp, Loader=yaml.FullLoader)

args["defense_args"]["device"] = torch.device(args["device"])
args["defense_args"]["batch_size"] = args["batch_size"]
args["defense_args"]["num_workers"] = args["num_workers"]
args["defense_args"]["dataset"] = args["dataset"]
args["defense_args"]["dataset_root"] = args["dataset_root"]

args["attack_args"]["device"] = torch.device(args["device"])
args["attack_args"]["batch_size"] = args["batch_size"]
args["attack_args"]["num_workers"] = args["num_workers"]
args["attack_args"]["dataset"] = args["dataset"]
args["attack_args"]["dataset_root"] = args["dataset_root"]

args = EasyDict(args)
pprint(args)

set_seed(args.seed)

os.makedirs(args.defense_args.exp_path, exist_ok=True)

if args.defense == "nd":
    defense = ND(**args.defense_args)
elif args.defense == "ini":
    defense = INI(attack_args=args.attack_args, **args.defense_args)

defense.train()
