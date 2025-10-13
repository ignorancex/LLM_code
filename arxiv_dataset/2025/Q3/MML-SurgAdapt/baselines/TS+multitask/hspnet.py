
import torch
from torch.cuda.amp import autocast  # type: ignore
from torchvision import transforms
from torchvision.transforms import RandAugment, RandomErasing, InterpolationMode
import torch.nn as nn

from dataset import build_dataset
from log import logger
from model import load_clip_model, HSPNet, Resnet, ViT, CLIP_for_train
from utils import ModelEma, get_ema_co
from typing import Optional, List, Tuple, Dict

from config import cfg  # isort:skip

class HSPNetTrainer():

    def __init__(self) -> None:
        super().__init__()

        clip_model, _ = load_clip_model()
        # image_size = clip_model.visual.input_resolution
        image_size = cfg.image_size

        train_preprocess = transforms.Compose([
            transforms.Resize((image_size,image_size),
                              interpolation=InterpolationMode.BICUBIC),
            #transforms.RandomHorizontalFlip(p=0.5),
            #customAugment(),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                 (0.26862954, 0.26130258, 0.27577711)),
            # RandomErasing(p=0.5, scale=(0.02, 0.1)),
            # RandomErasing(p=0.5, scale=(0.02, 0.1)),
        ])
        val_preprocess = transforms.Compose([
            transforms.Resize((image_size,image_size),
                              interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                 (0.26862954, 0.26130258, 0.27577711)),
        ])

        train_loader, val_loader, test_loader = build_dataset(train_preprocess,
                                                    val_preprocess)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        classnames = val_loader.dataset.labels()
        print(len(classnames))
        print(cfg.num_classes)
        assert (len(classnames) == cfg.num_classes)

        if cfg.model == 'CLIP':
            self.model = CLIP_for_train(classnames, clip_model)
        elif cfg.model == 'Resnet':
            self.model = Resnet(classnames, clip_model)
        self.classnames = classnames
        print(self.model)
        #logger.info("Turning off gradients in the text encoder")
        # for name, param in self.model.named_parameters():
        #     if "text_encoder" in name or "image_encoder" in name:
        #         param.requires_grad_(False)

        # for name,param in self.model.named_parameters():
        #     if param.requires_grad:
        #         print(name)

        self.model.cuda()
        ema_co = get_ema_co()
        logger.info(f"EMA CO: {ema_co}")
        self.ema = ModelEma(self.model, ema_co)  # 0.9997^641=0.82
        if 'cholec80' in cfg.checkpoint:
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = nn.BCEWithLogitsLoss()
        print(self.criterion)
    
    def train(self, input, target, criterion, epoch, epoch_i) -> torch.Tensor:
        image = input
        image = image.cuda()
        with autocast():  # mixed precision
            output = self.model(
                image).float()  # sigmoid will be done in loss !
        loss = self.criterion(output,target)

        return loss
    