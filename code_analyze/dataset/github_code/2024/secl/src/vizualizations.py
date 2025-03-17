import time
import os
import sys
from ast import literal_eval

import configargparse

sys.path.append("src")

from dataset import DataIterativeLoader
from trainer import BaseDATrainer, UnlabeledDATrainer, get_trainer
from util import arguments_parsing, set_seed, wandb_logger
import torch

import wandb
from evaluation import evaluation
from model import ProtoClassifier, ResModel, SENNModel
from util import (
    TIMING_TABLE,
    BaseTrainerConfig,
    LR_Scheduler,
    MetricMeter,
    SLATrainerConfig,
)

import pdb
import matplotlib.pyplot as plt



def save_or_show(self,img, save_path):
    print(save_path)
    img = img.clone().squeeze()
    npimg = img.cpu().numpy()
    if len(npimg.shape) == 2:
        if save_path is None:
            plt.imshow(npimg, cmap='Greys')
            plt.show()
        else:
            plt.imsave(save_path, npimg, cmap='Greys')
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')
        if save_path is None:
            plt.show()
        else:
            plt.savefig(save_path)
    plt.clf()

args = arguments_parsing("config-viz.yml")
# replace the configuration
args.dataset = args.dataset_cfg["dataset_cfg"][args.dataset]


loaders = DataIterativeLoader(args, strong_transform=False)
loaders = loaders

# iter_loaders = iter(loaders.loaders["target_validation"])
iter_loaders = iter(loaders)
backbone="resnet34"

model = SENNModel(backbone, output_dim=args.dataset["num_classes"]).cuda()
model.load_state_dict(torch.load("/u/ss7mu/phd/robust-prototype/sla/SLA/wandb/run-20240107_150715-c2ibkgme/files/model-5100.h5"))

s_concept_activations = []
s_imgs = []

t_concept_activations = []
t_imgs = []


s_acc=0
t_acc=0
cnt=0
for n in range(1,20):
    # pdb.set_trace()
    print(n)
    (sx, sy), (tx, ty), ux = next(iter_loaders)
    # tx, ty = next(iter_loaders)
    # tx, ty = tx.float().cuda(), ty.long().cuda()
    with torch.no_grad():
        aggregates, s_concepts, relevances, reconstructions, sufficient_output = model(sx)
        pred = aggregates.argmax(dim=1)
        s_acc += (pred == sy).float().sum().item()
        aggregates, t_concepts, relevances, reconstructions, sufficient_output = model(tx)
        pred = aggregates.argmax(dim=1)
        t_acc += (pred == ty).float().sum().item()
        cnt += len(tx)
    
    s_concept_activations.append(s_concepts)
    s_imgs.append(sx)
    t_concept_activations.append(t_concepts)
    t_imgs.append(tx)

num_prototypes = 9
s_c = torch.cat(s_concept_activations,dim=0)
s_idx = torch.topk(s_c,dim=0,k=num_prototypes)
s_imgs = torch.cat(s_imgs,dim=0)
t_c = torch.cat(t_concept_activations,dim=0)
t_idx = torch.topk(t_c,dim=0,k=num_prototypes)
t_imgs = torch.cat(t_imgs,dim=0)

pdb.set_trace()
concepts = 10
for c in range(concepts):
    one_row
    s_imgs[s_idx[0][:,c]]

s_plot = s_imgs[s_idx[0]]
t_plot = t_imgs[t_idx[0]]

plt.rcdefaults()
fig, ax = plt.subplots()
concept_names = ['Concept {}'.format(i + 1) for i in range(num_concepts)]

start = 0.0
end = concepts * x.size(-1)
stepsize = abs(end - start) / concepts
ax.yaxis.set_ticks(np.arange(start + 0.5 * stepsize, end - 0.49 * stepsize, stepsize))
ax.set_yticklabels(concept_names)
plt.xticks([])
ax.set_xlabel('{} most prototypical data examples per concept'.format(num_prototypes))
ax.set_title('Concept Prototypes: ')
save_or_show(make_grid(top_examples, nrow=num_prototypes, pad_value=1), save_path)
plt.rcdefaults()

# print(s_acc/cnt)
print(t_acc/cnt)