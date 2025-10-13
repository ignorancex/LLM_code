import os
import argparse
import random
import numpy as np
import torch
import torch.optim as opt
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataset import FullDataset
from DGSUNet import DGSUNet
from loss import structure_loss

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main(args):
    dataset = FullDataset(args.train_image_path, args.train_mask_path, 352, 518, mode='train')
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    device = torch.device("cuda")
    model = DGSUNet(args.dino_model_name, args.dino_hub_dir, args.sam_config_file, args.sam_ckpt_path)
    model.to(device)
    optim = opt.AdamW([{"params": model.parameters(), "initia_lr": args.lr}], lr=args.lr,
                      weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optim, args.epoch, eta_min=1.0e-7)
    os.makedirs(args.save_path, exist_ok=True)
    for epoch in range(args.epoch):
        for i, batch in enumerate(dataloader):
            x1 = batch['image1']
            x2 = batch['image2']
            # print(x1.shape)
            # print(x2.shape)
            target = batch['label']
            x1 = x1.to(device)
            x2 = x2.to(device)
            target = target.to(device)
            optim.zero_grad()
            pred0, pred1, pred2 = model(x2,x1)
            loss0 = structure_loss(pred0, target)
            loss1 = structure_loss(pred1, target)
            loss2 = structure_loss(pred2, target)
            loss = 0.25*loss0 + 0.5*loss1 + loss2
            loss.backward()
            optim.step()
            if i % 50 == 0:
                print("epoch:{}-{}: loss:{}".format(epoch + 1, i + 1, loss.item()))

        scheduler.step()
        if (epoch + 1) % 5 == 0 or (epoch + 1) == args.epoch:
            torch.save(model.state_dict(), os.path.join(args.save_path, 'DGSUNet-%d.pth' % (epoch + 1)))
            print('[Saving Snapshot:]', os.path.join(args.save_path, 'DGSUNet-%d.pth' % (epoch + 1)))


# def seed_torch(seed=1024):
# 	random.seed(seed)
# 	os.environ['PYTHONHASHSEED'] = str(seed)
# 	np.random.seed(seed)
# 	torch.manual_seed(seed)
# 	torch.cuda.manual_seed(seed)
# 	torch.cuda.manual_seed_all(seed)
# 	torch.backends.cudnn.benchmark = False
# 	torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    # seed_torch(1024)
    args = argparse.ArgumentParser()
    args.dino_model_name = "dinov2_vitl14"
    args.dino_hub_dir = "facebookresearch/dinov2"
    args.sam_config_file = r"G:\MyProjectCode\SAM2DINO-Seg\sam2_configs\sam2.1_hiera_l.yaml"
    args.sam_ckpt_path = r"G:\MyProjectCode\SAM2DINO-Seg\checkpoints\sam2.1_hiera_large.pt"
    args.train_image_path = r"H:\Salient\DUTS\DUTS-TR\DUTS-TR-Image/"
    args.train_mask_path = r"H:\Salient\DUTS\DUTS-TR\DUTS-TR-Mask/"
    args.save_path = r"G:\MyProjectCode\SAM2DINO-Seg\checkpoints"
    args.epoch = 50
    args.lr = 0.001
    args.batch_size = 4
    args.weight_decay = 5e-4
    main(args)