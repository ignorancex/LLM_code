""" Trains a convnet for the shapes task """
import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Optional, Dict, List

ROOT_PROJECT = str(Path(os.path.realpath(__file__)).parent.parent.parent.parent)
sys.path[0] = ROOT_PROJECT

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor

from utils.utils_save import get_storage_root
from weighted_retraining.weighted_retraining import utils
from weighted_retraining.weighted_retraining.metrics import METRIC_LOSSES
from weighted_retraining.weighted_retraining.shapes.shapes_data import WeightedNumpyDataset
from weighted_retraining.weighted_retraining.shapes.shapes_model import ShapesVAE
from weighted_retraining.weighted_retraining.utils import print_flush

MAX_LEN = 5


def shape_get_path(k, predict_target: bool, hdims: List[int] = None, latent_dim: int = 2,
                   metric_loss: Optional[str] = None, metric_loss_kw: Dict[str, Any] = None):
    """ Get path of directory where models will be stored

    Args:
        latent_dim: dimension of the latent space
        k: weight parameter
        predict_target: whether generative model also predicts target value
        hdims: latent dims of target MLP predictor
        metric_loss: metric loss used to structure the embedding
        metric_loss_kw: kwargs for `metric_loss` (see `METRIC_LOSSES`)

    Returns:
        Path to result dir
    """
    res_path = os.path.join(get_storage_root(), f'logs/train/shapes/shapes-k-{k}')
    exp_spec = f'id'
    if latent_dim != 2:
        exp_spec += f'_z-dim-{latent_dim}'
    if predict_target:
        assert hdims is not None
        exp_spec += '_predy-' + '-'.join(map(str, hdims))
    if metric_loss is not None:
        exp_spec += '-' + METRIC_LOSSES[metric_loss]['exp_metric_id'](**metric_loss_kw)
    res_path = os.path.join(res_path, exp_spec)
    print('res_path', res_path)
    return res_path


def main():
    # Create arg parser
    parser = argparse.ArgumentParser()
    parser = ShapesVAE.add_model_specific_args(parser)
    parser = WeightedNumpyDataset.add_model_specific_args(parser)
    parser = utils.DataWeighter.add_weight_args(parser)
    utils.add_default_trainer_args(parser, default_root="")

    # Parse arguments
    hparams = parser.parse_args()

    hparams.root_dir = shape_get_path(k=hparams.rank_weight_k, predict_target=hparams.predict_target,
                                      hdims=hparams.target_predictor_hdims, metric_loss=hparams.metric_loss,
                                      metric_loss_kw=hparams.metric_loss_kw, latent_dim=hparams.latent_dim)
    print_flush(' '.join(sys.argv[1:]))
    print_flush(hparams.root_dir)

    pl.seed_everything(hparams.seed)

    # Create data
    datamodule = WeightedNumpyDataset(hparams, utils.DataWeighter(hparams))

    # Load model
    model = ShapesVAE(hparams)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        period=max(1, hparams.max_epochs // 20),
        monitor="loss/val", save_top_k=1,
        save_last=True, mode='min'
    )

    if hparams.load_from_checkpoint is not None:
        model = ShapesVAE.load_from_checkpoint(hparams.load_from_checkpoint)
        utils.update_hparams(hparams, model)
        trainer = pl.Trainer(gpus=[hparams.cuda] if hparams.cuda is not None else 0,
                             default_root_dir=hparams.root_dir,
                             max_epochs=hparams.max_epochs,
                             callbacks=[checkpoint_callback, LearningRateMonitor(logging_interval='step')],
                             resume_from_checkpoint=hparams.load_from_checkpoint)

        print(f'Load from checkpoint')
    else:
        # Main trainer
        trainer = pl.Trainer(
            gpus=[hparams.cuda] if hparams.cuda is not None else 0,
            default_root_dir=hparams.root_dir,
            max_epochs=hparams.max_epochs,
            checkpoint_callback=True,
            callbacks=[checkpoint_callback, LearningRateMonitor(logging_interval='step')],
            terminate_on_nan=True,
            progress_bar_refresh_rate=100
        )

    # Fit
    trainer.fit(model, datamodule=datamodule)

    print(f"Training finished; end of script: rename {checkpoint_callback.best_model_path}")

    shutil.copyfile(checkpoint_callback.best_model_path, os.path.join(
        os.path.dirname(checkpoint_callback.best_model_path), 'best.ckpt'
    ))


if __name__ == "__main__":
    main()
