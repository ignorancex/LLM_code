import sys
sys.path.append(".") 
from torch.utils.data import DataLoader
import torch
from models import model_list
import argparse
import os
from algorithms import logit_only, training_based
from sklearn.metrics import roc_auc_score
import argparse
import numpy as np
from tqdm import tqdm
import pickle
from functools import partial
from torchvision.models import ResNet50_Weights
from dataset import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="", allow_abbrev=False)
    parser.add_argument("--preprocess_path", type=str, default='.')
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--root_path", type=str, required=True)
    parser.add_argument("--subset_file", type=str, required=False)
    parser.add_argument("--result_file", type=str, required=True)
    parser.add_argument("--semantic", type=int, default=0)
    args = parser.parse_args()

    load_path = args.preprocess_path

    print("loading model")
    models = [
        model_list[f"torchvision-model-{i}"](1000) for i in range(13)
    ]

    transform = ResNet50_Weights.IMAGENET1K_V1.transforms()

    temp_features = [{} for i in range(13)]
    def generic_hook(model, input, output, features):
        assert len(input) == 1
        features["resnet50"] = input[0]

    for i in tqdm(range(4)):
        models[i].classifier.register_forward_hook(partial(generic_hook, features=temp_features[i]))
        models[i].cuda()
        models[i].eval()
    for i in tqdm(range(4, 13)):
        models[i].fc.register_forward_hook(partial(generic_hook, features=temp_features[i]))
        models[i].cuda()
        models[i].eval()

    print("loading training")
    train_features = []
    train_logits = []
    for i in tqdm(range(13)):
        train_features.append(torch.load(os.path.join(load_path,f"train_features-{i}.pt")))
        train_logits.append(torch.load(os.path.join(load_path,f"train_logits-{i}.pt")))
    train_y = np.load(os.path.join(load_path,"train_y.npy"))
    
    print("loading data")
    eval_features = [[] for _ in range(13)]
    eval_logits = [[] for _ in range(13)]
    label = [[] for _ in range(13)]
    
    dataset = args.dataset
    if dataset == 'ImageNet': # imagenet format
        dataset = ImageNet_Format(path=args.root_path, transform=transform)
    elif dataset == 'ImageNetOOD': # subset of imagenet format (imagenetood)
        dataset = ImageNetOOD(imagenet_path = args.root_path, subset_file = args.subset_file, transform=transform)
    elif dataset == 'OpenImageO':
        dataset = Generic_Subset(path=args.root_path, subset_file=args.subset_file, transform=transform)
    elif dataset == 'ImageNetOOD_standalone':
        dataset = ImageNetOOD_standalone(path=args.root_path, transform=transform)
    else:
        assert False, "not a valid dataset"
    out_dataloader = DataLoader(
        dataset, batch_size=128, num_workers=8, shuffle=False)
    with torch.no_grad():
        for x, y in tqdm(out_dataloader):
            x = x.cuda()
            y = y.cuda()
            for i in range(13):
                pred = models[i](x)
                eval_logits[i].append(pred.detach().cpu())
                eval_features[i].append(
                    temp_features[i]["resnet50"].detach().cpu())
                if args.semantic == 0:
                    label[i] += (torch.argmax(pred, dim=1) == y.cuda()).tolist()
                else:
                    label[i] += [0]*x.shape[0]

    for i in range(13):
        eval_features[i] = torch.cat(eval_features[i])
        eval_logits[i] = torch.cat(eval_logits[i])

    data = []
    for i in range(4):
        data.append((train_features[i], train_logits[i], eval_features[i], eval_logits[i], models[i].classifier))
    for i in range(4, 13):
        data.append((train_features[i], train_logits[i], eval_features[i], eval_logits[i], models[i].fc))

    result = [{} for _ in range(13)]

    for i, (t_feat, t_log, e_feat, e_log, last_layer) in enumerate(data):
        result[i]['Maximum Cosine'] = logit_only["Maximum Cosine"](e_feat, last_layer)
    
    for i, (t_feat, t_log, e_feat, e_log, last_layer) in enumerate(data):
        result[i]['ash_b'] = training_based["ash"](e_feat, last_layer, 90, version='b')

    for i, (t_feat, t_log, e_feat, e_log, _ ) in enumerate(data):
        result[i]['msp'] = logit_only['Maximum Softmax Probability'](e_log)

    for i, (t_feat, t_log, e_feat, e_log, _ ) in enumerate(data):
        result[i]['ml'] = logit_only['Maximum Logits'](e_log)

    for i, (t_feat, t_log, e_feat, e_log, _) in enumerate(data):
        result[i]['energy'] = logit_only['Energy'](e_log)

    for i, (t_feat, t_log, e_feat, e_log, _) in enumerate(data):
        result[i]['mahalanobis'] = training_based["Mahalanobis"](t_feat, train_y, e_feat, 1000) 

    for i, (t_feat, t_log, e_feat, e_log, last_layer) in enumerate(data):
        result[i]['vim'] = training_based["ViM"](t_feat, e_feat, 1000, last_layer)
   
    for i, (t_feat, t_log, e_feat, e_log, last_layer) in enumerate(data):
        result[i]['knn'] = training_based["KNN"](t_feat.cpu().numpy(), e_feat.cpu().numpy())

    for i, (t_feat, t_log, e_feat, e_log, last_layer) in enumerate(data):
        result[i]['react'] = training_based["ReAct"](t_feat, e_feat, 90, last_layer)
    
    for i in range(13):
        result[i]['label'] = label[i]

    with open(args.result_file, "wb") as f:
        pickle.dump(result, f, protocol = pickle.HIGHEST_PROTOCOL)