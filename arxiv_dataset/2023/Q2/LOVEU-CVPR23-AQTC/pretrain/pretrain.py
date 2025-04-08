import os
import torch
import random
import numpy as np
import logging
from parameters import parse_args
from pathlib import Path
from utils import set_seed, setuplogging, compute_metrics
from model import get_model
from data import get_dataloader
from trainer import get_trainer

def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device('cuda')
    args.device = device
    print(torch.cuda.current_device())

    set_seed(args.seed)

    model = get_model(args.model)(args)
    dataloader = get_dataloader(args.dataset)(args,model)
    trainer = get_trainer(args.trainer)(args,model,dataloader)
    trainer.start()

if __name__=='__main__':
    args = parse_args()
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir+'/model').mkdir(parents=True, exist_ok=True)
    setuplogging(args,0)
    logging.info(args)
    main(args)