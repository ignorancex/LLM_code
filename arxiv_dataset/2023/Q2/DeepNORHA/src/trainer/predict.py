#!/usr/bin/env python3
# python predict.py --config '/home/duilio/Downloads/seg/configs/test1.yml' --checkpoint '/home/duilio/Downloads/seg/src/results/test1/version_0/checkpoints/last.ckpt' 
from nilearn.plotting import plot_anat

import os
import re
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from pytorch_lightning import seed_everything

import Models
from lightning import getModules
from lightning import getModulesSVDD
from utils import yaml

import nibabel as nb
from pytorch_lightning import Trainer

import DataModule
import Models
from Training import TrainFactory
from Training import TrainFactorySVDD


def predict(config, accelerator="gpu", checkpoint=None, output_path=None):
    trainer, modelTrainer, dataModule = getModules(config, accelerator=accelerator)    
    
    # # Returns a list of dictionaries, one for each provided dataloader containing their respective predictions.
    results = trainer.predict(model=modelTrainer, datamodule=dataModule, ckpt_path=checkpoint)


    testdata = dataModule._test_data
    filenames = testdata.filenames
    batch_size = dataModule.test_batch
    for i,batch in enumerate(results):
        for j,pred in enumerate(batch):
            name = filenames[i*batch_size]
            element = (name, testdata.get(name), pred)
            # print(name)


    # nb.Nifti1Image(crop_image.astype(float), raw_image.affine) 
    # print(len(results), results[0])
    print(element[0],element[1][0].shape,element[2].shape)
    # exit()
    x = element[2]
    x = torch.squeeze(x[0], dim=0)
    plt.figure()
    plt.imshow(x[23,:,:])


    y = element[1][0]
    plt.figure()
    plt.imshow(y[23,:,:])
    plt.show()

    plt.figure()
    plt.imshow(x[:,30,:])


    plt.figure()
    plt.imshow(y[:,30,:])
    plt.show()
    


    trainer, modelTrainer, dataModule = getModulesSVDD(config, accelerator=accelerator)    

    # # Returns a list of dictionaries, one for each provided dataloader containing their respective predictions.
    results = trainer.predict(model=modelTrainer, datamodule=dataModule, ckpt_path=checkpoint)
    exit()

    model = torch.load(checkpoint)

    model_encoder_decoder = Models.get(config['Model'])

    model_encoder = Models.get_SVDD(config['Model'])
    net_dict = model_encoder.state_dict()

    # model_encoder_decoder.load_state_dict(net_dict)
    # model_encoder_decoder.eval()

    ae_net_dict = {k: v for k, v in model['state_dict'].items() if k in net_dict}
    net_dict.update(ae_net_dict)

    model_encoder.load_state_dict(net_dict)
    model_encoder.eval()
    # trainFactory = TrainFactorySVDD()
    # modelTrainer = trainFactory.getTrainer(model=net_dict, dataModule=dataModule, **config['Train'])
    # # logger = trainFactory.getLogger(config['name'], config['Logger'])


    net_dict = model_encoder_decoder.state_dict()
    ae_net_dict = {k: v for k, v in model['state_dict'].items() if k in net_dict}
    net_dict.update(ae_net_dict)
    model_encoder_decoder.load_state_dict(net_dict)
    model_encoder_decoder.eval()
    
    x = testdata.get('r_HEC_CON_001right')
    # print(x[0].shape,x[1])
    # # print(net_dict)
    x = torch.unsqueeze(x[0], dim=0)
    x = torch.unsqueeze(x, dim=0)
    # print(x.shape)
    # x =  x[0].reshape(1,1,64,64,64)
    # print(x.shape)
    # x = torch.unsqueeze(x[0], dim=0)
    # print(x.shape)
    result = model_encoder.predict(x)
    # print(result)
    print(result.shape)

    result = model_encoder_decoder.predict(x)
    # print(result)
    print(result.shape)
    # batch_size = dataModule.test_batch
    # for i,batch in enumerate(results):
    #     for j,pred in enumerate(batch):
    #         name = filenames[i*batch_size+j]
    #         element = (name, testdata.get(name), pred)
    #         for filter in pipe:
    #             element = filter(element)
    uotput_img = testdata.get('r_HEC_CON_001right')
    uotput_img = uotput_img[0]

    print('shape',uotput_img.shape)
    plt.figure()
    plt.imshow(uotput_img[23,:,:])
    # new_image_izq = ni.Nifti1Image(uotput_img.astype(float), img_aff)
    # plot_anat(new_image_izq, title='raw',
    #         display_mode='ortho', dim=-1, draw_cross=False, annotate=False);

    # outputs = ae_net(inputs)
    uotput_img = result[0]
    # uotput_img = uotput_img[0]
    uotput_img = uotput_img[0].cpu()
    uotput_img = uotput_img.detach().numpy()
    print('shape',uotput_img.shape)
    plt.figure()
    plt.imshow(uotput_img[23,:,:])
    plt.show()



if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--config',type=str, help='Config file full path')
    parser.add_argument('--accelerator', type=str, default="gpu", help='accelerator to use')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint')
    parser.add_argument('--output-path', type=str, default=None, help='Output path')
    
    params = parser.parse_args()

    config_name = os.path.basename(params.config)
    config_name = re.sub(r'.ya?ml$', '', config_name)
    config = {'name': config_name, **yaml.load(params.config)}

    predict(config, params.accelerator, params.checkpoint, params.output_path)
