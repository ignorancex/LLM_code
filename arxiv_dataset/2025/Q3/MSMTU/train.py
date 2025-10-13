import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import torch

from msmtu.datamodules import LULCDataModuleOptim
from msmtu.pl_modules import LitBRITS

import wandb
import os

from pprint import pprint

from msmtu.utils import save_predictions

# ------------
# Set up default hyperparameters
# ------------
default_config = {
    'output_dir': '/home/jrodriguez/MSMTU/outputs/msmtu/brits/default/test',
    'model': {
        'rnn_hid_size': 100,
        'impute_weight': 0.25,
        'label_weight': 0.75,
        'data_dim': 7,
        'seq_len': 12,
        'lr': 3e-3,
        'wd': 0,
        'optimizer': 'adam',
        'scheduler': 'cosine'
    },
    'data': {
        'json_dir': '/mnt/homeGPU/jrodriguez/lifewatch/data/modis_500_3/',
        'batch_size': 2048,
        'data_dim': 7,
        'test_imp': False
    },
    'trainer': {
        'max_epochs': 200,
        'fast_dev_run': 2,
        # 'overfit_batches': 2,
        # 'detect_anomaly': True,
        'accumulate_grad_batches': 1,
        # 'track_grad_norm': 2,
        # 'val_check_interval': 0.25,
        'gradient_clip_val': 0.95,
        'accelerator': 'gpu',
        'devices':1
    },
    'wandb': {
        'project': 'msmtu',
        'name': 'test',
        'save_dir': '/home/jrodriguez/MSMTU/outputs/msmtu/brits/default/test/',
        'log_model': True
    },
    'callbacks': {
        'model_checkpoint': {
            'save_top_k': 1,
            'monitor': "val/rmse_epoch",
            'mode': "min",
            'dirpath': '/home/jrodriguez/MSMTU/outputs/msmtu/brits/default/test/checkpoints',
            'filename': "brits-epoch={epoch}-val_rmse_epoch={val/rmse_epoch:.3f}",
            'auto_insert_metric_name': False,
            'save_last': True,  # Save the last checkpoint in case we want to continue training later
        }
    }
}


def train(project: str, run_name: str, config: dict):
    print('Run config:')
    print(default_config)

    # ------------
    # Create Data Module
    # ------------

    data_module = LULCDataModuleOptim(**config['data'])
    _, idx_to_class = data_module.get_dicts()

    # ------------
    # Create Model
    # ------------

    if config['model']['pre_trained_path'] is  None:
        # Create model from scratch
        model = LitBRITS(
            **config['model'],
            idx_to_class=idx_to_class,
            ancillary_dim=len(data_module.ancillary_data))
    else:
        model = LitBRITS.load_from_checkpoint(config['model']['pre_trained_path'], **config['model'])
        model.change_last_layer(idx_to_class=idx_to_class)

    # ------------
    # Log Metrics using Wandb
    # ------------

    if config['trainer']['fast_dev_run']:
        logger = False
    else:
        logger = False
        """
        logger = WandbLogger(**config['wandb'])
        logger.watch(model, log="gradients", log_freq=200)  # log gradients
        print(f'wandb.run.dir: {wandb.run.dir}')
        """

    # ------------
    # Training
    # ------------
    # saves top-K checkpoints based on "val_loss" metric
    checkpoint_callback = ModelCheckpoint(**config['callbacks']['model_checkpoint'])

    # Create trainer
    trainer = pl.Trainer(logger=logger, **config['trainer'], callbacks=[checkpoint_callback])

    print('Starting training!')
    trainer.fit(model, datamodule=data_module)  # Train the model

    # ------------
    # Optional: test model and save predictions
    # ------------
    if 'test_labels_path' in list(config['data'].keys()) and not config['trainer']['fast_dev_run']:
        print('Starting testing!')
        ckpt_path = trainer.checkpoint_callback.best_model_path
        print(f"Best ckpt path: {ckpt_path}")
        trainer.test(model, datamodule=data_module, verbose=True, ckpt_path=ckpt_path)  # Test with the best model

        csv_path = os.path.join(config['output_dir'], 'predictions.csv')

        print(f'Saving predictions to {csv_path}')
        predictions = trainer.predict(model, datamodule=data_module, ckpt_path=ckpt_path)

        save_predictions(predictions, csv_path, idx_to_class)


if __name__ == '__main__':
    # ------------
    # Set project and entity for wandb logging
    # ------------
    level = 'level_1'
    project = f'BRITS_LULC_SIPNA {level}'
    name = 'scratch 1213 topo'

    default_config['data']['json_dir'] = '/home/jrodriguez/data/MSMTU/ts_files'
    default_config['data'][
        'train_labels_path'] = f'/home/jrodriguez/data/MSMTU/labels/{level}/partitions/train.csv'
    default_config['data'][
        'val_labels_path'] = f'/home/jrodriguez/data/MSMTU/labels/{level}/partitions/validation.csv'
    default_config['data'][
        'test_labels_path'] = f'/home/jrodriguez/data/MSMTU/labels/{level}/partitions/test.csv'
    default_config['data'][
        'ancillary_path'] = '/home/jrodriguez/data/MSMTU/ancillary/topo_climatic.csv'

    default_config['data']['ancillary_data'] = ['longitude', 'latitude', 'altitude', 'slope', 'evapotranspiration']

    default_config['model']['rnn_hid_size'] = 100
    default_config['model']['pre_trained_path'] = None # './pre_trained/timespec4lulc.ckpt'

    default_config['trainer']['fast_dev_run'] = False
    default_config['trainer']['max_epochs'] = 1

    train(project=project, run_name=name, config=default_config)
