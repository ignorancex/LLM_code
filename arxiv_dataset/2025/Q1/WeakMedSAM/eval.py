from torch.utils.data import DataLoader
import torch
import argparse
from tqdm import tqdm
from utils.metrics import Metric
from unet.unet_model import UNet
from torch.nn.functional import one_hot
import importlib
import os
from torch.cuda.amp import autocast


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--data_module", type=str, required=True)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_classes", type=int)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--gpus", type=str, required=True)
    args = parser.parse_args()
    print(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    model = UNet(n_channels=1, n_classes=args.num_classes, bilinear=True)
    model.load_state_dict(torch.load(args.ckpt))
    model = torch.nn.DataParallel(model).cuda()
    model.eval()

    data_module = importlib.import_module(f"{args.data_module}.dataset")
    _, _, test_dataset = data_module.get_seg_dataset(
        args.data_path,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=4,
    )

    metrics = Metric()

    for datapack in tqdm(test_loader, total=len(test_loader)):

        imgs = datapack["img"].cuda()
        segs = datapack["seg"].cuda()

        with autocast():
            with torch.no_grad():
                x = model(imgs)

        pred = (
            one_hot(
                x.argmax(dim=1),
                args.num_classes,
            )
            .permute(0, 3, 1, 2)
            .float()
        )

        metrics.add(pred[:, 1:], segs)

    print(metrics)


if __name__ == "__main__":
    main()
