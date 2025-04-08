from data import build_data
from model import build_model
from configs import build_config
from pytorch_lightning.loggers import WandbLogger

from pytorch_lightning import Trainer, seed_everything

if __name__ == "__main__":
    seed_everything(42, workers=True)
    cfg, args = build_config()
    data = build_data(cfg)
    model = build_model(cfg)
    logger = WandbLogger(project=args.wdb_project,name=args.wdb_name,log_model=False,save_dir='outputs/',offline=True)
    trainer = Trainer(
        gpus=1, 
        accelerator="gpu",
        benchmark=False, 
        deterministic=True,
        logger=logger
    )
    trainer.test(model, datamodule=data)