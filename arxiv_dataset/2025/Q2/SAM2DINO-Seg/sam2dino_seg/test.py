import argparse
import os
import torch
import imageio
import numpy as np
import torch.nn.functional as F
from DGSUNet import DGSUNet
from dataset import TestDataset

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser()
args = parser.parse_args()
args.checkpoint = '//home/xym/MyPaper/SAM2DINO-Seg/checkpoints_one_wloss/DGSUNet-50.pth'
args.test_image_path = '//home/xym/MyPaper/SAM2DINO-Seg/data/ECSSD/images/'
args.test_gt_path = '//home/xym/MyPaper/SAM2DINO-Seg/data/ECSSD/ground_truth_mask/'
args.save_path = '//home/xym/MyPaper/SAM2DINO-Seg/result/ECSSD_one_wloss/'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_loader = TestDataset(args.test_image_path, args.test_gt_path, 352,518)
model = DGSUNet().to(device)
model.load_state_dict(torch.load(args.checkpoint), strict=True)
model.eval()
model.cuda()
os.makedirs(args.save_path, exist_ok=True)
for i in range(test_loader.size):
    with torch.no_grad():
        image1, image2, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        image1 = image1.to(device)
        image2 = image2.to(device)
        res1, res2, res = model(image2, image1)
        # fix: duplicate sigmoid
        # res = torch.sigmoid(res)
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu()
        res = res.numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        res = (res * 255).astype(np.uint8)
        print("Saving " + name)
        imageio.imsave(os.path.join(args.save_path, name[:-4] + ".png"), res)
