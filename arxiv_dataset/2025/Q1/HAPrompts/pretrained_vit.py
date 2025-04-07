"""
load PyTorch ViT pretrained on ImageNet
compare its predictions error structure with CLIP zero-shot predictions
"""
import os
import torch
import pickle
import argparse
import torchvision
import numpy as np
from tqdm import tqdm
from torchvision.models import vit_b_16, vit_b_32, vit_l_16, vit_l_32 
from torchvision.models import ViT_B_16_Weights, ViT_B_32_Weights, ViT_L_16_Weights, ViT_L_32_Weights

from utils.directory import load_config


"""
#############################################################
need to download original ImageNet-1k validation set and 
development tool kit manually to <dataset-home-dir> specified
by ./data_paths.yml

download links provided by PyTorch docs:
https://pytorch.org/vision/stable/generated/torchvision.datasets.ImageNet.html?highlight=imagenet#torchvision.datasets.ImageNet
#############################################################
"""



def get_name(name_tag):
    return name_tag.replace('_', ' ').lower().split('@')[0]


def get_offset(name_tag):
    return name_tag.split('@')[1]


def name_preprocessing(classnames):
    classnames = [get_name(name) for name in classnames]
    return  classnames


def is_match(my_name, vit_name):
    vit_name = vit_name.lower()
    vit_name = vit_name.replace('/', 'or').replace(' ', '-')
    my_name = my_name.replace(' ', '-')
    return my_name == vit_name


def get_mapping_from_vit_to_my(pytorch_dataset, my_dataset):
    """
        pytorch vit predictions follow default pytorch ImageNet numeric labels (ordered class WordNet synset offset)
        I need to sort my classes' according to their respective WordNet offset to match with vit numeric labels
    """

    vit_name_label = [(n, k) for k, n in enumerate(pytorch_dataset.classes)]
    my_name_offset_label = [(get_name(name_tag), get_offset(name_tag), k) for k, name_tag in enumerate(my_dataset.classes)]

    # we removed the following 2 classes from original ImageNet: 
    # sunglass (n04355933), projectile (n04008634)
    # their respective synset definition and image examples are inconsistent

    # sort my (classname,offset,numeric label) tuple according to offset
    my_name_offset_label = sorted(my_name_offset_label, reverse=False, key=lambda x:x[1])

    # matching my sorted list of class labels with pytorch imagenet labels (naturally ordered by synset offset)
    skip_offset = 0
    i = 0
    vit_pred_to_my = dict()

    while i < len(my_name_offset_label):
        
        # pytorch imagenet.classes offers a tuple of alternative names per class
        vit_name_tup = vit_name_label[i+skip_offset][0]

        # check for ['sunglass', 'projectile']
        skip_flag = False
        for vit_name in vit_name_tup:
            if is_match('sunglass', vit_name):
                skip_flag = True
            elif is_match('projectile', vit_name):
                skip_flag = True
            if skip_flag: break
        
        if skip_flag: 
            skip_offset += 1
        else:
            my_numeric_label = my_name_offset_label[i][2]
            vit_numeric_label = vit_name_label[i+skip_offset][1]
            vit_pred_to_my[vit_numeric_label] = my_numeric_label

            # update to match next pair of labels
            i += 1

    return vit_pred_to_my



def vit_inference(model, my_test_loader, vit_pred_to_my_map):
    """
        set the logits of removed 2 classes to -100.0 to remove their predictions
    """
    model.cuda()
    model.eval()
    vit_pred = [] # vit predictions (need mapping to my labels)
    gt = [] # my ground truth

    with torch.no_grad():
        for k, (images, targets) in enumerate(tqdm(my_test_loader, total=len(my_test_loader))):
            images = images.cuda()
            logits = model(images)

            # remove projectile - 744
            # remove sunglass - 836
            logits[:, 744] = -100.0
            logits[:, 836] = -100.0
            pred = logits.argmax(dim=1)

            # collect gt & pred
            gt.append(targets.numpy())
            vit_pred.append(pred)
    
    gt = np.concatenate(gt)
    vit_pred = torch.cat(vit_pred, dim=0)
    vit_pred = vit_pred.detach().cpu().numpy()
    my_pred = np.array([vit_pred_to_my_map[p] for p in vit_pred])
    return gt, my_pred



def vit_inference_index_select(model, my_test_loader, vit_pred_to_my_map):
    """
        set the logits of removed 2 classes to -100.0 to remove their predictions
    """
    model.cuda()
    model.eval()
    my_pred = [] # vit predictions (need mapping to my labels)
    gt = [] # my ground truth

    my_pred_to_vit = {v:k for k, v in vit_pred_to_my_map.items()}
    mapped_index = [my_pred_to_vit[i] for i in range(len(my_test_loader.dataset.classes))]
    mapped_index = torch.LongTensor(mapped_index).cuda()


    with torch.no_grad():
        for k, (images, targets) in enumerate(tqdm(my_test_loader, total=len(my_test_loader))):
            images = images.cuda()
            logits = model(images)

            # re-arrange column of logits - mapping vit labels to my labels
            # only selecting 998/1000 classes hence removing sunglass and projectile
            logits = torch.index_select(logits, 1, mapped_index)

            pred = logits.argmax(dim=1)

            # collect gt & pred
            gt.append(targets.numpy())
            my_pred.append(pred)
    
    gt = np.concatenate(gt)
    my_pred = torch.cat(my_pred, dim=0)
    my_pred = my_pred.detach().cpu().numpy()
    return gt, my_pred





if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='ViT-B-16', choices=[
        'ViT-B-16', 'ViT-B-32',
        'ViT-L-16', 'ViT-L-32',
    ])

    opts = parser.parse_args()

    # inference on val/test set
    partition = 'test'
    model_name = opts.model # 'ViT-B-16', 'ViT-B-32', 'ViT-L-16', 'ViT-L-32', 
    model_dict = {
        'ViT-B-16': vit_b_16,
        'ViT-B-32': vit_b_32,
        'ViT-L-16': vit_l_16,
        'ViT-L-32': vit_l_32,
    }
    model_transforms_dict = {
        'ViT-B-16': ViT_B_16_Weights.IMAGENET1K_V1.transforms(),
        'ViT-B-32': ViT_B_32_Weights.IMAGENET1K_V1.transforms(),
        'ViT-L-16': ViT_L_16_Weights.IMAGENET1K_V1.transforms(),
        'ViT-L-32': ViT_L_32_Weights.IMAGENET1K_V1.transforms(),
    }

    # default imagenet dir
    data_dir = load_config('./data_paths.yml')['dataset-home-dir']
    
    # pytorch vit image transforms
    # img_transforms = ViT_L_32_Weights.IMAGENET1K_V1.transforms()
    img_transforms = model_transforms_dict[model_name]
    
    # pytorch ImageNet 
    print(f'loading pytorch imagenet val dataset for index matching')
    pytorch_dataset = torchvision.datasets.ImageNet(root=data_dir, split='val', transform=img_transforms)
    print(f'finish loading pytorch imagenet val dataset')

    # get pytorch imagenet (classname, numeric label) pairs
    default_name_idx = [(n,k) for k, n in enumerate(pytorch_dataset.classes)]

    # load my ImageNet dataset (different class ordering -> different numeric label)
    my_data_dir = load_config('./data_paths.yml')['imagenet']
    my_test_dir = os.path.join(my_data_dir, partition)

    # evaluation pytorch vit on my ImageNet test set
    my_test_set = torchvision.datasets.ImageFolder(
        root=my_test_dir,
        transform=img_transforms,
    )
    my_test_loader = torch.utils.data.DataLoader(
        my_test_set,
        num_workers=2,
        batch_size=16,
        shuffle=False,
        drop_last=False,
        pin_memory=False,
    )

    # load vit model
    # more info: https://pytorch.org/vision/stable/models/vision_transformer.html 
    model = model_dict[model_name](weights='IMAGENET1K_V1')

    # get mapping from vit label to my label
    vit_pred_to_my = get_mapping_from_vit_to_my(pytorch_dataset, my_test_set)

    # check if the mapping is 1-to-1
    keys = []
    vals = []
    for k, v in vit_pred_to_my.items():
        keys.append(k)
        vals.append(v)
    print(f'# of unique keys: {len(np.unique(keys))}')
    print(f'# of unique vals: {len(np.unique(vals))}')

    # vit inference on ImageNet
    gt, pred = vit_inference_index_select(model, my_test_loader, vit_pred_to_my)

    # evaluation
    top1 = 100*np.sum(pred==gt)/len(gt)
    print(f'my test set top1: {top1:.2f}%')


    # save prediction results
    save_dir = f'./results-{partition}/imagenet'
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    fname = f"imagenet_pretrained_{model_name.replace('-','_')}.pkl"
    save_path = os.path.join(save_dir, fname)
    with open(save_path, 'wb') as f:
        pickle.dump({'gt': gt, 'pred': pred}, f)
    print(f'saving inference results at {save_path}')