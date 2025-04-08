from data import build_data
from model import build_model
from configs import build_config

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

if __name__ == "__main__":
    cfg, args = build_config()
    seed_everything(args.seed, workers=True)
    dataset = build_data(cfg)
    model = build_model(cfg)
    checkpoint_callback = ModelCheckpoint(
        dirpath=f'/data/zclfe/cvpr_comp/LOVEU-CVPR22-AQTC/outputs/cvpr_loveu2023/{args.wdb_name}/',
        filename = '{epoch}-{EncodedAssistQADataModule recall@1:.3f}',
        save_top_k=1,
        monitor= "EncodedAssistQADataModule recall@1",
        mode='max'
    )
    early_stop_callback = EarlyStopping(
        monitor="EncodedAssistQADataModule recall@1",  # 监控的指标名称
        min_delta=0.001,     # 最小变化量，用于确定是否发生改进
        patience=10,          # 在没有改进时等待的轮数
        verbose=True,        # 是否打印提示信息
        mode='max'           # 监控模式，可以是 'min' 或 'max'
        )
    logger = WandbLogger(project=args.wdb_project,name=args.wdb_name,log_model=not args.wdb_offline,save_dir='outputs/',offline=args.wdb_offline)
    trainer = Trainer(
        gpus=cfg.NUM_GPUS, 
        accelerator="gpu",
        strategy=DDPStrategy(find_unused_parameters=False),
        callbacks=[
            LearningRateMonitor(logging_interval='step'), 
            checkpoint_callback,
            early_stop_callback
        ],
        benchmark=False, 
        deterministic=True,
        max_epochs=cfg.SOLVER.MAX_EPOCHS,
        default_root_dir=cfg.OUTPUT_DIR,
        check_val_every_n_epoch=cfg.CHECK_VAL_EVERY_N_EPOCH,
        num_sanity_val_steps=0,
        log_every_n_steps=5,
        logger=logger
    )
    trainer.fit(model, datamodule=dataset, 
        ckpt_path=cfg.CKPT if hasattr(cfg, "CKPT") else None)
    