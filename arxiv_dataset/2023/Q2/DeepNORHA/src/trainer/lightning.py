from logging import logMultiprocessing
from typing import Tuple
import torch
from pytorch_lightning import Trainer
import DataModule
import Models
from Training import TrainFactory
from Training import TrainFactorySVDD


def getModules(config, fast_dev_run=False, accelerator="gpu") -> Tuple[Trainer, torch.nn.Module, DataModule.DataModule]:
    dataModule = DataModule.get(config['DataModule'])
    model = Models.get(config['Model'])
    # print(model)
    trainFactory = TrainFactory()
    modelTrainer = trainFactory.getTrainer(model=model, dataModule=dataModule, **config['Train'])
    logger = trainFactory.getLogger(config['name'], config['Logger'])

    #generate the trainer
    trainer = Trainer(
            callbacks=trainFactory.getCallbacks(config.get('Callbacks')),
            auto_lr_find=False,
            auto_scale_batch_size= False,
            max_epochs=config['Train']['epochs'],
            accelerator=accelerator,
            # gpus=1,
            logger=logger,
            num_sanity_val_steps=0, #set to 0 to skip the sanity check
            log_every_n_steps=15,
            fast_dev_run=fast_dev_run,
            limit_val_batches=0, #if you dont wanna run validation
            default_root_dir="../results")

    return trainer, modelTrainer, dataModule


def getModulesSVDD(config, fast_dev_run=False, accelerator="gpu", pretrained=True) -> Tuple[Trainer, torch.nn.Module, DataModule.DataModule]:
    dataModule = DataModule.get(config['DataModule'])
    model_encoder = Models.get_SVDD(config['Model'])
    # print(model_encoder)

    if pretrained:
        print('Initialize the Deep SVDD network weights from the encoder weights of the pretraining autoencoder. ')
        checkpoint = '../results/' + config['name'][:-4] + '/version_0/checkpoints/last.ckpt'
        print(checkpoint)
        model = torch.load(checkpoint)
        net_dict = model_encoder.state_dict()
        ae_net_dict = {k: v for k, v in model['state_dict'].items() if k in net_dict}
        net_dict.update(ae_net_dict)
        model_encoder.load_state_dict(net_dict)
        print('Finish loading...')

    trainFactory = TrainFactorySVDD()
    modelTrainer = trainFactory.getTrainer(model=model_encoder, dataModule=dataModule, **config['Train'])
    logger = trainFactory.getLogger(config['name'], config['Logger'])

    #generate the trainer
    trainer = Trainer(
            callbacks=trainFactory.getCallbacks(config.get('Callbacks')),
            auto_lr_find=False,
            auto_scale_batch_size= False,
            max_epochs=config['Train']['epochs'],
            accelerator=accelerator,
            # gpus=1,
            logger=logger,
            num_sanity_val_steps=0, #set to 0 to skip the sanity check
            log_every_n_steps=5,
            fast_dev_run=fast_dev_run,
            #limit_val_batches=0, #if you dont wanna run validation
            default_root_dir="../results")

    return trainer, modelTrainer, dataModule
