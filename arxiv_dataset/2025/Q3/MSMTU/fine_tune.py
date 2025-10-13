import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from msmtu.datamodules import LULCDataModule, LULCDataModuleOptim
from msmtu.lightningmodules import LitBRITS

import wandb
import os

from pprint import pprint

from msmtu.utils import save_predictions

# ------------
# Set up default hyperparameters
# ------------
default_config = {
    'model': {
        'rnn_hid_size': 100,
        'impute_weight': 0.25,
        'label_weight': 0.75,
        'data_dim': 7,
        'seq_len': 12,
        'lr': 3e-3,
        'wd': 0,
        'optimizer': 'adam',
        'scheduler': 'cosine',
        'soft_labels': True,
        'class_weight': None
    },
    'data': {
        'data_dir': '/mnt/homeGPU/jrodriguez/lifewatch/data/SIPNA/Andalucia/soft_data/train_n1_05',
        # /mnt/homeGPU/jrodriguez/lifewatch/data/modis_test
        'batch_size': 2048,
        # 'train_val_test_split': (1.0, 0.0, 0.0),
        'data_dim': 7,
        'test_imp': False
    },
    'trainer': {
        'max_epochs': 150,
        # 'fast_dev_run': 2,
        # 'overfit_batches': 2,
        # 'detect_anomaly': True,
        'accumulate_grad_batches': 1,
        # 'track_grad_norm': 2,
        # 'val_check_interval': 0.25,
        'gradient_clip_val': 0.95
    }
}


def fine_tune(project: str, run_name: str, config: dict, ckpt_path: str):
    print(f'Project: {project}')
    print(f'Run name: {run_name}')
    print('Run config: ')
    print(default_config)

    # ------------
    # Create Data Module
    # ------------

    data_module = LULCDataModuleOptim(**config['data'])
    _, idx_to_class = data_module.get_dicts()
    print(idx_to_class)

    # ------------
    # Load pre-trained Model
    # ------------
    """
    
    checkpoint = torch.load(ckpt_path)  'state_dict'
    
    model = LitBRITS(**config['model'], idx_to_class=idx_to_class)"""

    if 'class_weight' in list(default_config['model'].keys()):
        class_weight = default_config['model']['class_weight']
        del default_config['model']['class_weight']
    else:
        class_weight = None

    print(f'Loading checkpoint: {ckpt_path}')
    model = LitBRITS.load_from_checkpoint(ckpt_path, **config['model'])

    # ------------
    # Create new classification layer
    # ------------

    print('-------------------')
    print('\nModified config:')
    print(default_config)

    print(f'Class weight before change: {class_weight}')
    model.change_last_layer(idx_to_class=idx_to_class, class_weight=class_weight)

    # ------------
    # Log Metrics using Wandb
    # ------------

    if config['trainer']['fast_dev_run']:
        logger = False
    else:
        logger = WandbLogger(project=project, name=run_name, save_dir='./logger', log_model=True)
        logger.watch(model, log="gradients", log_freq=200)  # log gradients
        print(f'wandb.run.dir: {wandb.run.dir}')

    # ------------
    # Training
    # ------------
    # saves top-K checkpoints based on "val_loss" metric
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val/rmse_epoch",
        mode="min",
        dirpath=None,
        filename="brits-epoch={epoch}-val_rmse_epoch={val/rmse_epoch:.3f}",
        auto_insert_metric_name=False,
        save_last=True,  # Save the last checkpoint in case we want to continue training later
    )

    # Create trainer
    trainer = pl.Trainer(logger=logger, **config['trainer'], accelerator='auto', devices=[1],
                         callbacks=[checkpoint_callback])

    trainer.fit(model, datamodule=data_module)  # Train the model

    # ------------
    # Test model if test set is provided
    # ------------
    if 'test_labels_path' in list(config['data'].keys()) and config['data']['test_labels_path'] is not None:
        print('TESTING')
        trainer.test(model, datamodule=data_module, verbose=True, ckpt_path='best')  # Test with the best model

        computed_cm = model.test_cm.compute()

        pprint('Test confusion matrix')
        pprint(computed_cm)

        csv_path = os.path.join(wandb.run.dir, 'predictions.csv')

        print(f'Saving predictions to {csv_path}')
        predictions = trainer.predict(model, datamodule=data_module, ckpt_path='best')

        save_predictions(predictions, csv_path, idx_to_class)


if __name__ == '__main__':
    # ------------
    # Set project and entity for wandb logging
    # ------------
    ckpt_path_l0 = './logger/run_val_split/2we22b7f/checkpoints/brits-epoch=09-val_f1_macro=1.000.ckpt'
    ckpt_path_l1 = './logger/run_val_split/3thoxj0i/checkpoints/brits-epoch=11-val_f1_macro=0.998.ckpt'
    ckpt_path_l2 = './logger/run_val_split/2k7fijb6/checkpoints/brits-epoch=08-val_f1_macro=0.993.ckpt'
    ckpt_path_l3 = './logger/run_val_split/2e4f124j/checkpoints/brits-epoch=16-val_f1_macro=0.995.ckpt'
    ckpt_path_l4 = './logger/run_val_split/9kz7ymmj/checkpoints/brits-epoch=09-val_f1_macro=0.994.ckpt'
    ckpt_path_l5 = './logger/BRITS_LULC_training/1lqykbz6/checkpoints/brits-epoch=166-val_f1_macro=0.963.ckpt'

    ckpt_path_l1 = './logger/brits-epoch=18-val_f1_macro=0.997.ckpt'
    ckpt_path_l5_att = './logger/brits-epoch=34-val_f1_macro=0.963.ckpt'
    ckpt_path_l5_att_100 = './logger/brits-epoch=15-val_f1_macro=0.955.ckpt'

    ckpt_path_l5_100 = './fine_tuned/brits-epoch=32-val_f1_macro=0.967.ckpt'

    ckpt_path = ckpt_path_l5_100

    project = 'BRITS_LULC_SIPNA N1'
    name = 'fine-tuned soft mse 1213'

    default_config['data']['data_dir'] = '/home/jrodriguez/data/SIPNA/13/all_1213'
    default_config['data']['train_labels_path'] = '/home/jrodriguez/data/SIPNA/13/labels/N1/N1_1213/training_df_N1.csv'
    default_config['data']['val_labels_path'] = '/home/jrodriguez/data/SIPNA/13/labels/N1/N1_1213/val_df_N1.csv'
    default_config['data']['test_labels_path'] = '/home/jrodriguez/data/SIPNA/13/labels/N1/N1_1213/test_df_N1.csv'
    default_config['model']['soft_labels'] = True
    default_config['model']['apply_attention'] = False
    default_config['model']['class_weight'] = None
    default_config['model']['rnn_hid_size'] = 100

    default_config['trainer']['fast_dev_run'] = False
    default_config['trainer']['max_epochs'] = 200
    """torch.tensor(
        [2.6196, 0.4413, 13.4818, 0.3319, 6.2597, 2.7221, 0.7822, 1.0132,
         0.5565, 0.8280, 4.2134, 4.3356])"""
    """torch.tensor([3.3197, 18.2348, 22.5624, 27.2901, 69.1979,  0.3287,  9.9252,  0.2479,
         4.6868,  2.0217,  0.5848,  0.7591,  0.4172,  0.6204,  3.1327,  3.2343])"""

    # ------------
    # Start fine-tuning
    # ------------
    fine_tune(
        project=project,
        run_name=name,
        config=default_config,
        ckpt_path=ckpt_path
    )
