import os
import random
import yaml
import argparse
import numpy as np
from easydict import EasyDict
from pprint import pprint

import torch

from defenses.nd.nd import ND

from attacks.knockoffnets.knockoff import Knockoff
from attacks.jbda.jbda import JBDA
from datasets import REGISTRY_INFO, REGISTRY_TEST_DATASET, REGISTRY_QUERY_DATASET
from models import get_family, REGISTRY_MODEL

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

# dataset
family = get_family(args.dataset)
print("Building dataset: {}, {} family, num_classes {}, image_size {}, channels {}.".format(args.dataset, family, *REGISTRY_INFO[args.dataset]))
test_set = REGISTRY_TEST_DATASET[args.dataset](root=args.dataset_root)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)


model = ND(**args.defense_args)
model.load(args.attack_args.checkpoint_path)

def test(test_loader, model, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    accuracy = correct / len(test_loader.dataset)
    return accuracy

victim_acc = test(test_loader, model, args.device)
print('Victim accuracy: {:.2f}%'.format(victim_acc*100))
if args.attack == "knockoff":
    attack = Knockoff(teacher=model, **args.attack_args)
    sur_set = REGISTRY_QUERY_DATASET[args.attack_args.sur_dataset](root=args.attack_args.sur_dataset_root)
    sur_loader = torch.utils.data.DataLoader(sur_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    acc_list = attack.run(test_loader, sur_loader, victim_acc)
elif args.attack == "jbda":
    attack = JBDA(teacher=model, **args.attack_args)
    train_set = REGISTRY_QUERY_DATASET[args.attack_args.train_dataset](root=args.attack_args.train_dataset_root)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    acc_list = attack.run(test_loader, train_loader, victim_acc)
print(acc_list)
