import pytorch_lightning as pl
import numpy as np
from pytorch_lightning.loggers import WandbLogger
from pprint import pprint
import os
import wandb

from msmtu.datamodules import LULCDataModuleOptim
from msmtu.pl_modules import LitBRITS

from msmtu.utils import save_predictions


def test(project: str, run_name: str, config: dict, checkpoint_path: str, log: bool = True):
    # ------------
    # Create Model
    # ------------
    model = LitBRITS.load_from_checkpoint(checkpoint_path, **config['model'])
    # model = LitBRITS(**config['model'])

    # ------------
    # Create Data Module
    # ------------
    data_module = LULCDataModuleOptim(**config['data'])
    data_module.prepare_data()
    data_module.setup(stage=None)

    test_targets = data_module.test_set.targets

    unique, counts = np.unique(test_targets, return_counts=True)
    print('Test set support per class')
    pprint(dict(zip(unique, counts)))

    print('--------------------------')

    # ------------
    # Log Metrics using Wandb if required
    # ------------
    if log:
        logger = WandbLogger(project=project, name=run_name, save_dir='./logger')
    else:
        logger = False

    # ------------
    # Training
    # ------------
    trainer = pl.Trainer(logger=logger, accelerator='auto', devices=[0])  # Create trainer

    """trainer.test(model, datamodule=data_module, verbose=True)  # Test the model

    computed_cm = model.test_cm.compute()

    pprint('Test confusion matrix')
    pprint(computed_cm)"""

    print('Computing predictions')
    predictions = trainer.predict(model, datamodule=data_module)

    save_predictions(predictions, os.path.join('./', '111213_att_sep_q72dhyw.csv'), model.hparams.idx_to_class)


if __name__=='__main__':

    ckpt_path = "/mnt/homeGPU/jrodriguez/lifewatch/BRITS_lightning/models/misunderstood-aardvark-3/1dj99m1t/checkpoints/epoch=49-step=715898.ckpt"
    # ckpt_path = "/mnt/homeGPU/jrodriguez/lifewatch/BRITS_lightning/models/dutiful-forest-5/b61e7s8p/checkpoints/epoch_49_step_89500.ckpt"
    ckpt_path = './logger/BRITS_LULC_training/2s2ch58b/checkpoints/brits-epoch=19-val/f1_macro=0.959.ckpt'
    ckpt_path = './logger/BRITS_LULC_training/1lqykbz6/checkpoints/brits-epoch=166-val_f1_macro=0.963.ckpt'



    ckpt_path_l0 = './logger/run_val_split/2we22b7f/checkpoints/brits-epoch=09-val_f1_macro=1.000.ckpt'
    ckpt_path_l1 = './logger/run_val_split/3thoxj0i/checkpoints/brits-epoch=11-val_f1_macro=0.998.ckpt'
    ckpt_path_l2 = './logger/run_val_split/2k7fijb6/checkpoints/brits-epoch=08-val_f1_macro=0.993.ckpt'
    ckpt_path_l3 = './logger/run_val_split/2e4f124j/checkpoints/brits-epoch=16-val_f1_macro=0.995.ckpt'
    ckpt_path_l4 = './logger/run_val_split/9kz7ymmj/checkpoints/brits-epoch=09-val_f1_macro=0.994.ckpt'
    ckpt_path_l5 = './logger/BRITS_LULC_training/1lqykbz6/checkpoints/brits-epoch=166-val_f1_macro=0.963.ckpt'

    ckpt_path_l1 = './logger/brits-epoch=18-val_f1_macro=0.997.ckpt'
    ckpt_path_fine_05 = './logger/fine_tune threshold 0.5/2pr3zz07/checkpoints/brits-epoch=76-val_f1_macro=0.778.ckpt'
    ckpt_path_fine_06 = './logger/fine_tune threshold 0.6/1448wner/checkpoints/brits-epoch=81-val_f1_macro=0.814.ckpt'
    ckpt_path_fine_07 = './logger/fine_tune threshold 0.7/2xg3lt05/checkpoints/brits-epoch=82-val_f1_macro=0.856.ckpt'
    ckpt_path_fine_08 = './logger/fine_tune threshold 0.8/3mg6t9un/checkpoints/brits-epoch=78-val_f1_macro=0.866.ckpt'
    ckpt_path_fine_09 = './logger/fine_tune threshold 0.9/t7cjvkpm/checkpoints/brits-epoch=88-val_f1_macro=0.887.ckpt'
    ckpt_path_fine_1 = './logger/fine_tune threshold 1/3tuxu5ld/checkpoints/brits-epoch=88-val_f1_macro=0.931.ckpt'

    ckpt_path_fine_05 = './logger/fine_tune threshold 0.5/'
    ckpt_path_fine_06 = './logger/fine_tune threshold 0.6/'
    ckpt_path_fine_07 = './logger/fine_tune threshold 0.7/'
    ckpt_path_fine_08 = './logger/fine_tune threshold 0.8/'
    ckpt_path_fine_09 = './logger/fine_tune threshold 0.9/27doc5px/checkpoints/brits-epoch=73-val_f1_macro=0.890.ckpt'
    ckpt_path_fine_1 = './logger/fine_tune threshold 1/2ixf7fie/checkpoints/brits-epoch=82-val_f1_macro=0.929.ckpt'

    # FILTERING BY OUR CONSTRAINT (<= 3 missing months in band 6)
    ckpt_path_fine_05_soft = './logger/fine_tune threshold 0.5 soft-labels/53q7wmw3/checkpoints/brits-epoch=71-val_f1_macro=0.820.ckpt'
    ckpt_path_fine_05_hard = './logger/fine_tune threshold 0.5 NO soft-labels/8ym8vx18/checkpoints/brits-epoch=86-val_f1_macro=0.820.ckpt'
    ckpt_path_fine_1 = './logger/fine_tune threshold 1/1rodb7fs/checkpoints/brits-epoch=77-val_f1_macro=0.963.ckpt'

    # Attention models
    ckpt_path_fine_05_attention = './logger/scratch threshold 0.5/3zokqsi0/checkpoints/brits-epoch=46-val_f1_macro=0.796.ckpt'

    # L1 out of 2240
    ckpt_path_05_l1 = './logger/fine_tune L5 threshold/0.5 attention/3m32i2b6/checkpoints/brits-epoch=43-val_f1_macro=0.837.ckpt'
    ckpt_path_06_l1 = './logger/fine_tune L5 threshold/0.6 attention/ys2fgzmh/checkpoints/brits-epoch=62-val_f1_macro=0.873.ckpt'
    ckpt_path_07_l1 = './logger/fine_tune L5 threshold/0.7 attention/3plrzre4/checkpoints/brits-epoch=65-val_f1_macro=0.836.ckpt'
    ckpt_path_08_l1 = './logger/fine_tune L5 threshold/0.8 attention/2crcoyjp/checkpoints/brits-epoch=80-val_f1_macro=0.942.ckpt'
    ckpt_path_09_l1 = './logger/fine_tune L5 threshold/0.9 attention/2fafs07b/checkpoints/brits-epoch=23-val_f1_macro=0.991.ckpt'

    # L3 out of 2240
    ckpt_path_05_l3 = './logger/fine_tune L5 threshold/0.5 attention/6fdxd7p5/checkpoints/brits-epoch=78-val_f1_macro=0.445.ckpt'
    ckpt_path_06_l3 = './logger/fine_tune L5 threshold/0.6 attention/3jx2fwd9/checkpoints/brits-epoch=72-val_f1_macro=0.533.ckpt'
    ckpt_path_07_l3 = './logger/fine_tune L5 threshold/0.7 attention/32r6zlss/checkpoints/brits-epoch=63-val_f1_macro=0.587.ckpt'
    ckpt_path_08_l3 = './logger/fine_tune L5 threshold/0.8 attention/2sq32bvv/checkpoints/brits-epoch=79-val_f1_macro=0.618.ckpt'
    ckpt_path_09_l3 = './logger/fine_tune L5 threshold/0.9 attention/25ydbe2i/checkpoints/brits-epoch=45-val_f1_macro=0.993.ckpt'

    """cktp_path_hard_025 = './logger/fine-tuned 100 hard XE/15rhg5vj/checkpoints/brits-epoch=95-val_f1_macro=0.821.ckpt'
    cktp_path_soft_025 = './logger/fine-tuned 100 soft XE/402choub/checkpoints/brits-epoch=74-val_f1_macro=0.821.ckpt'
    cktp_path_soft_mse_025 = './logger/fine-tuned 100 soft MSE/2phl9mg5/checkpoints/brits-epoch=83-val_f1_macro=0.815.ckpt'

    cktp_path_hard_05 = './logger/fine-tuned 100 hard XE/2i0113ya/checkpoints/brits-epoch=93-val_f1_macro=0.840.ckpt'
    cktp_path_soft_05 = './logger/fine-tuned 100 soft XE/2g1s9tzu/checkpoints/brits-epoch=80-val_f1_macro=0.833.ckpt'
    cktp_path_soft_mse_05 = './logger/fine-tuned 100 soft MSE/1z4zxxvw/checkpoints/brits-epoch=86-val_f1_macro=0.840.ckpt'

    cktp_path_hard_075 = './logger/fine-tuned 100 hard XE/3s5s600q/checkpoints/brits-epoch=69-val_f1_macro=0.916.ckpt'
    cktp_path_soft_075 = './logger/fine-tuned 100 soft XE/8mrxjg90/checkpoints/brits-epoch=92-val_f1_macro=0.919.ckpt'
    cktp_path_soft_mse_075 = './logger/fine-tuned 100 soft MSE/2kdoium6/checkpoints/brits-epoch=91-val_f1_macro=0.917.ckpt'"""

    cktp_path_hard_025 = './logger/fine-tuned 100 hard XE no L2/168yrh5x/checkpoints/brits-epoch=54-val_f1_macro=0.819.ckpt'
    cktp_path_soft_025 = './logger/fine-tuned 100 soft XE no L2/2m7oidx4/checkpoints/brits-epoch=54-val_f1_macro=0.825.ckpt'
    cktp_path_soft_mse_025 = './logger/fine-tuned 100 soft MSE no L2/2bc7xib9/checkpoints/brits-epoch=74-val_f1_macro=0.823.ckpt'
    cktp_path_soft_cc_025 = './logger/fine-tuned 100 soft CC no L2/25ed98lm/checkpoints/brits-epoch=64-val_f1_macro=0.800.ckpt'

    cktp_path_hard_05 = './logger/fine-tuned 100 soft XE no L2/2ogb45da/checkpoints/brits-epoch=62-val_f1_macro=0.843.ckpt'
    cktp_path_soft_05 = './logger/fine-tuned 100 soft XE no L2/70p0n4qp/checkpoints/brits-epoch=82-val_f1_macro=0.841.ckpt'
    cktp_path_soft_mse_05 = './logger/'

    cktp_path_hard_075 = './logger/fine-tuned 100 soft XE no L2/304w0kvc/checkpoints/brits-epoch=91-val_f1_macro=0.916.ckpt'
    cktp_path_soft_075 = './logger/fine-tuned 100 soft XE no L2/3exztb3f/checkpoints/brits-epoch=46-val_f1_macro=0.917.ckpt'
    cktp_path_soft_mse_075 = './logger/fine-tuned 100 soft MSE no L2/o4k5ml6p/checkpoints/brits-epoch=90-val_f1_macro=0.918.ckpt'

    cktp_path_scratch_hard_025 = './logger/scratch 100 hard XE no L2/1wf5qtpd/checkpoints/brits-epoch=93-val_f1_macro=0.814.ckpt'
    cktp_path_scratch_soft_025 = './logger/scratch 100 soft XE no L2/1nqeqpsj/checkpoints/brits-epoch=89-val_f1_macro=0.817.ckpt'
    cktp_path_scratch_soft_mse_025 = './logger/scratch 100 soft MSE no L2/315a3cf6/checkpoints/brits-epoch=85-val_f1_macro=0.815.ckpt'
    cktp_path_scratch_soft_cc_025 = './logger/scratch 100 soft CC no L2/20equ91n/checkpoints/brits-epoch=69-val_f1_macro=0.795.ckpt'

    # Scratch no att L1 SIPNA
    ckpt_path_scratch_hard_xe_L1 = './logger/scratch 100 hard no L2 no Att/3w50bgf9/checkpoints/brits-epoch=71-val_f1_macro=0.826.ckpt'
    ckpt_path_scratch_soft_xe_L1 = './logger/scratch 100 soft XE no L2 no Att/1kmvtzyb/checkpoints/brits-epoch=66-val_f1_macro=0.825.ckpt'
    ckpt_path_scratch_soft_cc_L1 = './logger/scratch 100 soft CC no L2 no Att/3gf2ma6l/checkpoints/brits-epoch=82-val_f1_macro=0.824.ckpt'
    ckpt_path_scratch_soft_mse_L1 = './logger/scratch 100 soft MSE no L2 no Att/1dfpydbu/checkpoints/brits-epoch=77-val_f1_macro=0.823.ckpt'

    # Fine-tune no att L1 SIPNA
    ckpt_path_fine_hard_xe_L1 = './logger/fine-tuned 100 hard XE no L2/efor7ag8/checkpoints/brits-epoch=56-val_f1_macro=0.826.ckpt'
    ckpt_path_fine_soft_xe_L1 = './logger/fine-tuned 100 soft XE no L2/14mvnojm/checkpoints/brits-epoch=72-val_f1_macro=0.830.ckpt'
    ckpt_path_fine_soft_mse_L1 = './logger/fine-tuned 100 soft MSE no L2/1an2424m/checkpoints/brits-epoch=68-val_f1_macro=0.828.ckpt'

    # Scratch L2 SIPNA
    ckpt_path_scratch_soft_mse_L2 = './logger/scratch 100 soft MSE no L2 no Att/2o2tcpeh/checkpoints/brits-epoch=74-val_f1_macro=0.518.ckpt'
    ckpt_path_scratch_soft_xe_L2 = './logger/scratch 100 soft XE no L2 no Att/1nxearu9/checkpoints/brits-epoch=87-val_f1_macro=0.528.ckpt'
    ckpt_path_scratch_soft_cc_L2 = './logger/scratch 100 soft CC no L2 no Att/33osavpo/checkpoints/brits-epoch=74-val_f1_macro=0.489.ckpt'
    ckpt_path_scratch_hard_xe_L2 = './logger/scratch 100 hard XE no L2 no Att/2exv9l8n/checkpoints/brits-epoch=85-val_f1_macro=0.528.ckpt'

    # Fine-tuned L2 SIPNA
    ckpt_path_fine_soft_mse_L2 = './logger/fine-tuned 100 soft MSE no L2/226ljfbr/checkpoints/brits-epoch=76-val_f1_macro=0.522.ckpt'
    ckpt_path_fine_soft_xe_L2 = './logger/fine-tuned 100 soft XE no L2/e5y52gt8/checkpoints/brits-epoch=70-val_f1_macro=0.527.ckpt'
    ckpt_path_fine_hard_xe_L2 = './logger/fine-tuned 100 soft XE no L2/1oh0tkcs/checkpoints/brits-epoch=74-val_f1_macro=0.536.ckpt'

    # Aggregations N2
    ckpt_path_l2_agg_1 = './logger/fine-tuned 100 hard baseline/2eklc4o2/checkpoints/brits-epoch=111-val_f1_macro=0.628.ckpt'
    # 111213
    ckpt_path_111213 = './logger/scratch soft mse 111213/2qy05ng3/checkpoints/last.ckpt'
    ckpt_path_13 = './logger/scratch soft mse 13/2j0s00xy/checkpoints/last.ckpt'

    ckpt_path_111213_att_sep = './logger/scratch soft mse 111213 att sep/q72dhywp/checkpoints/brits-epoch=179-val_rmse_epoch=0.117.ckpt'
    # Select the desired model
    ckpt_path = ckpt_path_111213_att_sep

    from fine_tune import default_config
    # from train import default_config

    default_config['data']['data_dir'] = '/home/jrodriguez/data/SIPNA/13/all_111213'
    default_config['data']['train_labels_path'] = '/home/jrodriguez/data/SIPNA/13/labels/N1/N1_111213/training_df_N1.csv'
    default_config['data']['val_labels_path'] = '/home/jrodriguez/data/SIPNA/13/labels/N1/N1_111213/val_df_N1.csv'
    default_config['data']['test_labels_path'] = '/home/jrodriguez/data/SIPNA/13/labels/N1/N1_111213/test_df_N1.csv'
    default_config['model']['soft_labels'] = True
    default_config['model']['apply_attention'] = True

    project = 'BRITS_LULC_L1_SIPNA_test'
    run_name = 'Fine-tuned/0.5 attention test/threshold 0.5'
    test(project=project, run_name=run_name, config=default_config, checkpoint_path=ckpt_path, log=False)
