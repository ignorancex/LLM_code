import torch
from torch.cuda.amp import autocast  # type: ignore
from torchvision import transforms
from torchvision.transforms import RandAugment, RandomErasing, InterpolationMode

from dataset import build_dataset
from log import logger
from model import load_clip_model, MMLSurgAdapt, Resnet, CrossModel, ViT, CLIP_for_train, VLPL, HSPNet
from surgvlp import SurgAVLP, CBertViT
from utils import ModelEma, get_ema_co
from typing import Optional, List, Tuple, Dict
import torch.nn as nn
from config import cfg  # isort:skip

class MMLSurgAdaptTrainer():

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
            #RandomErasing(p=0.5, scale=(0.02, 0.1)),
            #RandomErasing(p=0.5, scale=(0.02, 0.1)),
        ])
        val_preprocess = transforms.Compose([
            transforms.Resize((image_size,image_size),
                              interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                 (0.26862954, 0.26130258, 0.27577711)),
        ])
        if cfg.perform_init:
            if cfg.val_sp:
                train_loader, val_loader, val_sp_loader, test_loader, init_train_loader, init_val_loader = build_dataset(train_preprocess,
                                                        val_preprocess)
            else:
                train_loader, val_loader, test_loader, init_train_loader, init_val_loader = build_dataset(train_preprocess,
                                                        val_preprocess)
        else:
            if cfg.val_sp:
                train_loader, val_loader, val_sp_loader, test_loader = build_dataset(train_preprocess,
                                                        val_preprocess)
            else:
                train_loader, val_loader, test_loader = build_dataset(train_preprocess,
                                                        val_preprocess)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        if cfg.val_sp:
            self.val_sp_loader = val_sp_loader

        if cfg.perform_init:
            self.init_train_loader = init_train_loader
            self.init_val_loader = init_val_loader

        classnames = val_loader.dataset.labels()
        assert (len(classnames) == cfg.num_classes)

        if cfg.backbone == 'SurgVLP':
            logger.info("Using SurgVLP weights")
            self.model = SurgAVLP(clip_model,classnames,cfg.bert_path,cfg.vlp_weights)
        else:
            if cfg.model == 'SurgAdapt':
                self.model = MMLSurgAdapt(classnames, clip_model)
            elif cfg.model == 'HSPNet':
                self.model = HSPNet(classnames,clip_model)
            elif cfg.model == 'VLPL':
                self.model = VLPL(classnames,clip_model)
            elif cfg.model == 'Resnet':
                self.model = Resnet(classnames,clip_model)
            elif cfg.model == 'CLIP':
                self.model = CLIP_for_train(classnames,clip_model)
            else:
                raise NameError
            #self.model = CBertViT(clip_model,classnames,cfg.bert_path)
        print(self.model)
        self.classnames = classnames
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
    
    def train(self, input, target, criterion, epoch, epoch_i) -> torch.Tensor:
        image = input
        image = image.cuda()
        with autocast():  # mixed precision
            output = self.model(
                image).float()  # sigmoid will be done in loss !
            
        loss, labels = criterion(output, target, epoch)
        return loss
    

class customAugment(RandAugment):
    
    def __init__(self, num_ops: int = 2, magnitude: int = 8, num_magnitude_bins: int = 31, interpolation: InterpolationMode = InterpolationMode.NEAREST, fill: Optional[List[float]] = None, ) -> None:
        super().__init__()

    def _augmentation_space(self, num_bins: int, image_size: Tuple[int, int]) -> Dict[str, Tuple[torch.Tensor, bool]]:
        return {
            # op_name: (magnitudes, signed)
            "Identity": (torch.tensor(0.0), False),
            "ShearX": (torch.linspace(0.0, 0.3, num_bins), True),
            "ShearY": (torch.linspace(0.0, 0.3, num_bins), True),
            "TranslateX": (torch.linspace(0.0, 150.0 / 331.0 * image_size[1], num_bins), True),
            "TranslateY": (torch.linspace(0.0, 150.0 / 331.0 * image_size[0], num_bins), True),
            "Rotate": (torch.linspace(0.0, 30.0, num_bins), True),
            "Brightness": (torch.linspace(0.0, 0.9, num_bins), True),
            "Color": (torch.linspace(0.0, 0.9, num_bins), True),
            "Contrast": (torch.linspace(0.0, 0.9, num_bins), True),
            "Sharpness": (torch.linspace(0.0, 0.9, num_bins), True),
            #"Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 4)).round().int(), False),
            #"Solarize": (torch.linspace(255.0, 0.0, num_bins), False),
            "AutoContrast": (torch.tensor(0.0), False),
            "Equalize": (torch.tensor(0.0), False),
        }