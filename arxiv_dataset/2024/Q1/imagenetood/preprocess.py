import sys
sys.path.append(".") 
from torch.utils.data import DataLoader
from dataset import ImageNet_Subset
import torch
from models import model_list
import argparse
import os
import numpy as np
from tqdm import tqdm
from functools import partial
from torchvision.models import ResNet50_Weights


if __name__ == "__main__":
    print("started")
    parser = argparse.ArgumentParser(description="", allow_abbrev=False)
    parser.add_argument("--imagenet_path", type=str, required=True)
    parser.add_argument("--subset_file", type=str, required=True)
    parser.add_argument("--result_path", type=str, default='.')
    args = parser.parse_args()

    if not os.path.exists(args.result_path):
        os.mkdir(args.result_path)

    print("loading model")
    models = [
        model_list[f"torchvision-model-{i}"](1000) for i in range(13)
    ]

    transform = ResNet50_Weights.IMAGENET1K_V1.transforms()

    temp_features = [{} for i in range(13)]
    def generic_hook(model, input, output, features):
        assert len(input) == 1
        features["resnet50"] = input[0]

    for i in range(4):
        models[i].classifier.register_forward_hook(partial(generic_hook, features=temp_features[i]))
        models[i].cuda()
        models[i].eval()
    for i in range(4, 13):
        models[i].fc.register_forward_hook(partial(generic_hook, features=temp_features[i]))
        models[i].cuda()
        models[i].eval()
    
    print("loading data")
    train_dataset = ImageNet_Subset(
        imagenet_path=args.imagenet_path, subset_file=args.subset_file, transform=transform
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=256, num_workers=8, shuffle=False
    )
    train_features = [[] for _ in range(13)]
    train_logits = [[] for _ in range(13)]
    train_y = []
    # requires at least 64GB of ram to store the imagenet embeddings
    with torch.no_grad():
        for x, y in tqdm(train_dataloader):
            x = x.cuda()
            for i in range(13):
                train_logits[i].append(models[i](x).detach().cpu())
                train_features[i].append(
                    temp_features[i]["resnet50"].detach().cpu())
            train_y += y.cpu().detach().numpy().tolist()

    for i in range(13):
        train_features[i] = torch.cat(train_features[i])
        train_logits[i] = torch.cat(train_logits[i]).numpy()
    train_y = np.array(train_y)
    for i in range(13):
        torch.save(train_logits[i],
                        os.path.join(args.result_path, f"train_logits-{i}.pt"))
        torch.save(train_features[i],
                        os.path.join(args.result_path, f"train_features-{i}.pt"))
    np.save(os.path.join(args.result_path, "train_y.npy"), train_y)


   