import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, DistributedSampler
from torchvision import datasets, transforms
import torch.cuda.amp as amp

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import albumentations as A
from albumentations.pytorch import ToTensorV2

from convkan import ConvKAN, LayerNorm2D
from kan_convs import FastKANConv2DLayer

import argparse
import os
from datetime import timedelta
import numpy as np 

from sklearn.model_selection import train_test_split

os.environ['NCCL_BLOCKING_WAIT'] = '0'
# torch.cuda.empty_cache()
def setup():
    dist.init_process_group(backend='nccl', timeout=timedelta(seconds=7200000))
    # torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup():
    dist.destroy_process_group()


class AlexNetKAN(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        
        self.features = nn.Sequential(
            FastKANConv2DLayer(3, 24, kernel_size=11, stride=4, padding=0),
            LayerNorm2D(24),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            FastKANConv2DLayer(24, 64, kernel_size=5, padding=2, groups=2),
            LayerNorm2D(64),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            FastKANConv2DLayer(64, 96, kernel_size=3, padding=1),
            
            FastKANConv2DLayer(96, 96, kernel_size=3, padding=1, groups=2),
            
            FastKANConv2DLayer(96, 64, kernel_size=3, padding=1, groups=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(64 * 6 * 6, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        valid_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        save_every: int,
        snapshot_path: str,
        accumulation_steps :int = 2
    ) -> None:
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.model = model.to(self.gpu_id)
        self.train_data = train_data
        self.valid_data = valid_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.epochs_run = 0
        self.snapshot_path = snapshot_path

        self.model = DDP(self.model, device_ids=[self.gpu_id])
        self.loss_fn = nn.CrossEntropyLoss()
        self.hist = []

        self.accumulation_steps = accumulation_steps
        self.current_step= 0
        self.scaler = amp.GradScaler()

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=100,
            eta_min=1e-6  # Minimum learning rate
        )
        
        if os.path.exists(snapshot_path):
            print("Loading snapshot")
            self._load_snapshot()


    def _load_snapshot(self):
        loc = f"cuda:{self.gpu_id}"
        snapshot = torch.load(self.snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot['MODEL_STATE'])
        self.optimizer.load_state_dict(snapshot['OPTIMIZER_STATE'])
        self.scheduler.load_state_dict(snapshot['SCHEDULER_STATE'])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        self.hist = snapshot["HIST"]
        print(f"Snapshot loaded at epoch : {self.epochs_run}")

    def _save_snapshot(self, epoch):
        snapshot = {
            "MODEL_STATE": self.model.state_dict(),
            "OPTIMIZER_STATE": self.optimizer.state_dict(),
            "SCHEDULER_STATE": self.scheduler.state_dict(),
            "EPOCHS_RUN": epoch,
            "HIST": self.hist
        }
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def _run_batch_and_get_loss(self, source, targets):
        with amp.autocast(dtype=torch.float16):
            loss = self.get_loss((source, targets))
            loss = loss / self.accumulation_steps
        # torch.cuda.synchronize()
        self.scaler.scale(loss).backward()

        if (self.current_step + 1) % (self.accumulation_steps) == 0:
            self.scaler.unscale_(self.optimizer)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

        self.current_step+=1
        return loss * self.accumulation_steps

    def _run_epoch_and_get_loss(self, epoch):
        self.model.train()
        self.current_step = 0

        b_sz = len(next(iter(self.train_data))[0])

        print(f"[GPU{self.gpu_id}] Epoch {epoch} | "
              f"Batchsize: {b_sz} | Steps: {len(self.train_data)} | "
              f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")

        self.train_data.sampler.set_epoch(epoch)
        train_loss = []
        for source, targets in self.train_data:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            loss = self._run_batch_and_get_loss(source, targets)
            train_loss.append(loss)

        self.scheduler.step()

        return train_loss

    @staticmethod
    def get_accuracy(labels, preds):
        preds = torch.argmax(preds, dim=1)
        acc = (labels==preds).sum()/len(labels)
        return acc

    def get_loss(self, batch):
        features,labels = batch
        preds = self.model(features)
        loss = self.loss_fn(preds, labels)
        return loss

    def validate(self, batch ):
        feature, labels = batch
        loss = self.get_loss(batch)
        pred = self.model(feature)
        acc = self.get_accuracy(labels, pred)
        return {'valid_loss' : loss , 'valid_acc' : acc}
    
    def average_validation(self, out):
        loss = torch.stack([l['valid_loss'] for l in out]).mean()
        acc = torch.stack([l['valid_acc'] for l in out]).mean()
        return {'valid_loss': loss.item() , 'valid_acc': acc.item()}

    @torch.no_grad()
    def validate_and_get_metrics(self):
        self.model.eval()
        out = []
        for source, targets in self.valid_data:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            out.append(self.validate((source, targets)))
        return self.average_validation(out)

    @staticmethod
    def log_epoch( e, epoch, res):
        print('[{} / {}] epoch/s, LR: {:.6f}, training loss: {:.4f}, validation loss: {:.4f}, validation accuracy: {:.4f} '
              .format(e+1, epoch,
                      res['learning_rate'],
                      res['train_loss'],
                      res['valid_loss'],                
                      res['valid_acc']
                     )
              )

    def train(self, max_epochs: int):
        for epoch in range(self.epochs_run, max_epochs):
            train_loss = self._run_epoch_and_get_loss(epoch)
            log_dict = self.validate_and_get_metrics()
            log_dict['train_loss'] = torch.stack(train_loss).mean().item()
            log_dict['learning_rate'] = self.optimizer.param_groups[0]['lr']

            # Step the scheduler at the end of each epoch
            self.scheduler.step()

            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)

            if self.gpu_id == 0:
                self.hist.append(log_dict)
                self.log_epoch(epoch, max_epochs, log_dict)



def load_data():
    _transforms = transforms.Compose([
        transforms.RandomResizedCrop(
            size=(227,227),
            scale=(0.08, 1.0),
            ratio=(3/4,4/3)
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),

        transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.1
        ),
        
        # Lighting noise - simulates variations in lighting conditions
        transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.3),
        
        transforms.RandomRotation(
            degrees=32,
            fill=0,  # black padding for safe rotation
            interpolation=transforms.InterpolationMode.BILINEAR
        ),
        transforms.RandAugment(
            num_ops=2,
            magnitude=9
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet stats by default
            std=[0.229, 0.224, 0.225]
        ),
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(256),  # Resize the shorter side to 256
        transforms.CenterCrop(227),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    train_ds = datasets.ImageNet(
                            root="./imagenet/train/", 
                            split='train', 
                            transform=_transforms,
                        )

    val_ds = datasets.ImageNet(
                            root="./imagenet/val/", 
                            split='val', 
                            transform=val_transforms,
                        )
    # targets = train_ds.targets
    # _, test_train = train_test_split(np.arange(len(targets)), test_size = 0.05, stratify = targets, random_state=42 )
    # train_ds = Subset(train_ds, test_train)
    #
    # valid_targets = val_ds.targets
    # _, test_val = train_test_split(np.arange(len(valid_targets)), test_size = 0.05, stratify = valid_targets, random_state=42)
    # val_ds = Subset(val_ds, test_val)

    print(f" testing dataset size: {len(train_ds)}")
    print(f" testing validation size: {len(val_ds)}")

    return train_ds, val_ds

def prepare_data(dataset, batch_size):
    return DataLoader(
        dataset,
        batch_size= batch_size,
        # num_workers=0,
        shuffle=False,
        pin_memory=True,
        sampler=DistributedSampler(dataset)
    )

def load_model():
    model = AlexNetKAN()
    opt = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
    return model, opt

def trainer_agent(epochs:int, save_every:int, snapshot_path:str):
    setup()
    train_data, val_data = load_data()

    batch_size = 300 
    train_dl = prepare_data(train_data, batch_size)
    val_dl = prepare_data(val_data, batch_size * 2)

    model, opt = load_model()

    trainer = Trainer(
                model,
                train_dl,
                val_dl,
                opt,
                save_every,
                snapshot_path
    )
    trainer.train(epochs)
    cleanup()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s','--save', type=int, help="frequency to save")
    parser.add_argument("-p","--path",help="Path to store snapshot")
    parser.add_argument("-e", "--epochs", type=int, help="number of epochs")
    args = parser.parse_args()

    trainer_agent(args.epochs, args.save, args.path)

if __name__ == "__main__":
    import time 
    start = time.time()
    main()
    end = time.time()
    print(f"time taken to train CNN: {end-start}")

