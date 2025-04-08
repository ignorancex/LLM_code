#!/usr/bin/env python3
 
import os
import re
from argparse import ArgumentParser
from utils import yaml
import torch.nn as nn
from pytorch_lightning import seed_everything
import Models
from lightning import getModules
from lightning import getModulesSVDD


def train(configAE, configSVDD, fast_dev_run=False, accelerator="gpu", checkpoint=None, pretrained=True):
    seed_everything(42, workers=True)

    trainer, modelTrainer, dataModule = getModules(configAE, fast_dev_run, accelerator)   
    if pretrained:
    	print('deep_SVDD.pretrain')
    	trainer.fit(modelTrainer, dataModule, ckpt_path=checkpoint) # train the AEmodel
    # self.init_network_weights_from_pretraining()
    trainerSVDD, modelTrainerSVDD, dataModuleSVDD = getModulesSVDD(configSVDD, fast_dev_run, accelerator,pretrained)
    print('deep_SVDD.train')
    trainerSVDD.fit(modelTrainerSVDD, dataModuleSVDD, ckpt_path=checkpoint) # train the SVDDmodel  


if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--configAE',type=str, help='Config file full path')
    parser.add_argument('--configSVDD',type=str, help='Config file full path')
    parser.add_argument('--fast-dev-run', default=False, action='store_true', help='if set, only run a single batch')
    parser.add_argument('--accelerator', type=str, default="gpu", help='accelerator to use')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint')
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--no-pretrained', dest='pretrained', action='store_false')
    parser.set_defaults(pretrained=True)

    params = parser.parse_args()

    config_name = os.path.basename(params.configAE)
    config_name = re.sub(r'.ya?ml$', '', config_name)
    configAE = {'name': config_name, **yaml.load(params.configAE)}

    config_name = os.path.basename(params.configSVDD)
    config_name = re.sub(r'.ya?ml$', '', config_name)
    configSVDD = {'name': config_name, **yaml.load(params.configSVDD)}

    train(configAE, configSVDD, params.fast_dev_run, params.accelerator, params.checkpoint, params.pretrained)
