from data import build_data
from model import build_model
from configs import build_config
import os

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

if __name__ == "__main__":
    seed_everything(0, workers=True)
    cfg, args = build_config()
    files = list(os.listdir(cfg.CKPT))
    if len(files)==1:
        cfg.CKPT = cfg.CKPT + files[0]
    else:
        exit()
    dataset = build_data(cfg)
    model = build_model(cfg)
    logger = WandbLogger(project=args.wdb_project,name=args.wdb_name,log_model=False,save_dir='outputs/',offline=True)
    trainer = Trainer(
        gpus=cfg.NUM_GPUS, 
        accelerator="gpu",
        strategy=DDPStrategy(find_unused_parameters=False),
        callbacks=[
            LearningRateMonitor(logging_interval='step'), 
            ModelCheckpoint()
        ],
        benchmark=False, 
        deterministic=True,
        max_epochs=cfg.SOLVER.MAX_EPOCHS,
        default_root_dir=cfg.OUTPUT_DIR,
        check_val_every_n_epoch=cfg.CHECK_VAL_EVERY_N_EPOCH,
        num_sanity_val_steps=0,
        logger=logger
    )
    trainer.validate(model, datamodule=dataset, 
        ckpt_path=cfg.CKPT if hasattr(cfg, "CKPT") else None)