import argparse
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from preprocess.dataset_train import TrainDataset, TrainDataModule
from preprocess.dataset_cqg import CQGDataset, CQGDataModule
from preprocess.augmentations import basic_transforms
from train_pl import SegmentationModel
from train_cqg import CQGSegmentationModel

def parse_args():
    parser = argparse.ArgumentParser(description="Training configuration")
    parser.add_argument('--backbone', type=str, default="efficientnet-b2", help="Model backbone")
    parser.add_argument('--batch_size', type=int, default=2, help="Batch size for training")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--epochs', type=int, default=15, help="Number of training epochs")
    parser.add_argument('--augs', type=str, choices=['Baseline', 'Augs', 'CQG'], default='Baseline', help="Augmentation strategy")
    parser.add_argument('--name', type=str, default='example', help="Experiment name")
    return parser.parse_args()

def main():
    args = parse_args()

    # Preparing dataset
    if args.augs == 'CQG':
        data_module = CQGDataModule(batch_size=args.batch_size)
    else:
        data_module = TrainDataModule(batch_size=args.batch_size, augs=args.augs)

    # Defining model
    if args.augs == 'CQG':
        model_wrapper = CQGSegmentationModel(
            backbone=args.backbone, 
            lr=args.lr, 
            weight_geo=0.3, 
            weight_aug=0.7, 
            weight_diff=1.0
        )
    else:
        model_wrapper = SegmentationModel(backbone=args.backbone, lr=args.lr)

    # Setup trainer
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss", 
        dirpath="checkpoints",
        filename=args.name,
        save_top_k=1,
        mode="min"
    )

    csv_logger = CSVLogger(save_dir="checkpoints/", name=args.name)
    trainer = Trainer(
        accelerator="gpu",   
        devices=1,        
        max_epochs=args.epochs,
        log_every_n_steps=0,
        logger=csv_logger,
        callbacks=[checkpoint_callback],
    )

    # Run training
    trainer.fit(model_wrapper, datamodule=data_module)

if __name__ == "__main__":
    main()
