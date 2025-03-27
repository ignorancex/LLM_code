import torch.nn as nn
import torch
import sys, os
import random
import csv
import pandas as pd
import numpy as np
from shutil import copy
import matplotlib.pyplot as plt
from copy import deepcopy
from omegaconf import OmegaConf
import shutil
import pickle
import random
from PIL import Image
from tqdm import tqdm
from torchvision.datasets.folder import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision
import pdb
from collections import defaultdict

from util.func import get_patch_size
from hcompnet.model import HComPNet, get_network
from util.log import Log
from util.args import get_args, save_args, get_optimizer_nn
from util.data import get_dataloaders
from util.func import init_weights_xavier
from util.node import Node
from util.phylo_utils import construct_phylo_tree, construct_discretized_phylo_tree
from util.func import get_patch_size


def get_topk_cub_nodewise(net, root, projectloader, k, epoch, device, args):   

    list_csvfile_topk = []
    list_node_wise_df = []
    dict_node_wise_df = {}

    if isinstance(projectloader.sampler, torch.utils.data.RandomSampler):
        raise Exception('Dataset should not be in shuffle')
    # Make sure the model is in evaluation mode
    net.eval()
    
    # IMPORTANT: dataloader should NOT be in shuffle, because imgs will not be shuffled, indexing wont be right
    name2label = projectloader.dataset.class_to_idx
    label2name = {label:name for name, label in name2label.items()}
    # Show progress on progress bar
    project_iter = tqdm(enumerate(projectloader),
                        total=len(projectloader),
                        desc='Collecting top-k Prototypes CUB parts',
                        mininterval=50.,
                        ncols=0)
    imgs = projectloader.dataset.imgs

    patchsize, skip = get_patch_size(args)

    scores_per_prototype = dict() # maps node.name -> proto_idx -> leaf_name -> list(topk)
    for node in root.nodes_with_children():
        scores_per_prototype[node.name] = defaultdict(lambda: defaultdict(list))

        # Iterate through the projection set
    for i, (xs, orig_y) in project_iter:
        xs= xs.to(device)

        # coarse_label = ys.item()
        leaf_label = orig_y.item()
        leaf_name = label2name[leaf_label]

        with torch.no_grad():
            # Use the model to classify this batch of input data
            _, pfs_dict, pooled_dict, _ = net(xs)

            for node in root.nodes_with_children():

                if leaf_name not in node.leaf_descendents:
                    continue

                classification_weights = getattr(net.module, '_'+node.name+'_classification').weight

                child_name = node.closest_descendent_for(leaf_name).name
                coarse_label = node.children_to_labels[child_name]
                coarse_label2name = {label: name for name, label in node.children_to_labels.items()}
                
                pfs = pfs_dict[node.name]
                pooled = pooled_dict[node.name]
                pooled = pooled.squeeze(0) 
                pfs = pfs.squeeze(0) 
                for p in range(pooled.shape[0]):

                    if (classification_weights[coarse_label, p].item() > 1e-3):
                        scores_per_prototype[node.name][p][leaf_label].append((i, pooled[p].item(), pfs[p,:,:]))
    
    csvfolderpath = os.path.join(args.log_dir, f'node_wise_top{k}')
    os.makedirs(csvfolderpath, exist_ok=True)

    for node in root.nodes_with_children():
        proto_img_coordinates = []
        proto_img_coordinates_df = []
        csvfilepath = os.path.join(csvfolderpath, f'{node.name}_prototypes_top{k}_{str(epoch)}.csv')
        print('csv filepath:', csvfilepath)
        too_small = set()
        protoype_iter = tqdm(enumerate(scores_per_prototype[node.name].keys()), total=len(list(scores_per_prototype[node.name].keys())),mininterval=5.,ncols=0,desc='Collecting top-k patch coordinates CUB')
        with open(csvfilepath, "w", newline='') as csvfile:
            print("Writing CSV file with top k image patches..", flush=True)
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(["node", "child", "leaf", "prototype", \
                             "img name", "h_min_224", "h_max_224", \
                             "w_min_224", "w_max_224", "scores"])
            for _, prototype in protoype_iter:
                for leaf_label in scores_per_prototype[node.name][prototype]:
                    leaf_descendent_name = label2name[leaf_label]
                    child_name = node.closest_descendent_for(leaf_descendent_name).name
                    leaf_descendent_name = leaf_descendent_name[4:7] # taking only the number from class name
                    df = pd.DataFrame(scores_per_prototype[node.name][prototype][leaf_label], columns=['img_id', 'scores', 'latent_activation'])
                    topk = df.nlargest(k, 'scores')
                    for index, row in topk.iterrows():
                        imgid = int(row['img_id'])
                        imgname = imgs[imgid][0]
                        with torch.no_grad():
                            if row['scores'] < 0.1:
                                too_small.add(p)
                                
                            location_h, location_h_idx = torch.max(row['latent_activation'], dim=0)
                            _, location_w_idx = torch.max(location_h, dim=0)
                            location = (location_h_idx[location_w_idx].item(), location_w_idx.item())
                            h_coor_min, h_coor_max, w_coor_min, w_coor_max = get_img_coordinates(args.image_size, row['latent_activation'].unsqueeze(0).shape, \
                                                                                                 patchsize, skip, location[0], location[1])
                            
                            proto_img_coordinates.append([node.name, child_name, leaf_descendent_name, \
                                                          prototype, imgname, h_coor_min, h_coor_max, \
                                                          w_coor_min, w_coor_max, row['scores']])
                            proto_img_coordinates_df.append([node.name, child_name, leaf_descendent_name, \
                                                          prototype, imgname, h_coor_min, h_coor_max, \
                                                          w_coor_min, w_coor_max, row['scores'], row['latent_activation']])
                # write intermediate results in case of large dataset
                if len(proto_img_coordinates) > 10000:
                    writer.writerows(proto_img_coordinates)
                    proto_img_coordinates = []
            print("Warning: image patches included in topk, but similarity < 0.1! This might unfairly reduce the purity metric because prototype has less than k similar image patches. You could consider reducing k for prototypes", too_small, flush=True)
    
            writer.writerows(proto_img_coordinates) 
        # do something about this
        df = pd.DataFrame(proto_img_coordinates_df, columns=["node", "child", "leaf", "prototype", \
                                                         "img name", "h_min_224", "h_max_224", \
                                                         "w_min_224", "w_max_224", "scores", 'latent_activation'])
        list_csvfile_topk.append(csvfilepath)
        list_node_wise_df.append(df)
        dict_node_wise_df[node.name] = df
    # return df
    return list_csvfile_topk, list_node_wise_df, dict_node_wise_df

def eval_prototypes_cub_parts_csv_nodewise_maxmin(node, csvfile, parts_loc_path, parts_name_path, imgs_id_path, epoch, args, desc_threshold=0, log=None):
    patchsize, _ = get_patch_size(args)
    imgresize = float(args.image_size)
    path_to_id = dict()
    id_to_path = dict()
    with open(imgs_id_path) as f:
        for line in f:
            id, path = line.split('\n')[0].split(' ')
            path_to_id[path]=id
            id_to_path[id]=path

    img_to_part_xy_vis = dict()
    with open(parts_loc_path) as f:
        for line in f:
            img, partid, x, y, vis = line.split('\n')[0].split(' ')
            vis = str(int(float(vis)))
            x =float(x)
            y =float(y)
            if x > 1.06 or y > 1.05:
                raise Exception('Provide normalized coordinated for part loc')
            if img not in img_to_part_xy_vis.keys():
                img_to_part_xy_vis[img]=dict()
            if vis == '1':
                img_to_part_xy_vis[img][partid]=(x,y)

    parts_id_to_name = dict()
    parts_name_to_id = dict()
    with open (parts_name_path) as f:
        for line in f:
            id, name = line.split('\n')[0].split(' ',1)
            parts_id_to_name[id]=name
            parts_name_to_id[name]=id

    # merge left and right cub parts
    duplicate_part_ids = []
    with open (parts_name_path) as f:
        for line in f:
            id, name = line.split('\n')[0].split(' ',1)
            if 'left' in name:
                new_name = name.replace('left', 'right')
                
                duplicate_part_ids.append((id, parts_name_to_id[new_name]))
           
    proto_parts_presences = dict()
    child_name_to_protos = defaultdict(set)
    
    with open (csvfile, newline='') as f:
        filereader = csv.reader(f, delimiter=',')
        next(filereader) #skip header
        for (node_name, child_name, leaf_descendant_name, prototype, imgname, h_min_224, h_max_224, w_min_224, w_max_224, scores) in filereader:
            child_name_to_protos[child_name].add(prototype)
            
            if prototype not in proto_parts_presences.keys():
                proto_parts_presences[prototype]=dict()
            if leaf_descendant_name not in proto_parts_presences[prototype].keys():
                proto_parts_presences[prototype][leaf_descendant_name]=dict()
            p = prototype
            img = Image.open(imgname)
            imgname = imgname.replace('\\', '/')
            imgnamec, imgnamef = imgname.split('/')[-2:]
            if 'normal_' in imgnamef:
                imgnamef = imgnamef.split('normal_')[-1]
            imgname = imgnamec+'/'+imgnamef
            img_id = path_to_id[imgname]
            img_orig_width, img_orig_height = img.size
            h_min_224, h_max_224, w_min_224, w_max_224 = float(h_min_224), float(h_max_224), float(w_min_224), float(w_max_224)
            
            
            diffh = h_max_224 - h_min_224
            diffw = w_max_224 - w_min_224
            if diffh > patchsize: #patch size too big, we take the center. otherwise the bigger the patch, the higher the purity.
                correction = diffh-patchsize
                h_min_224 = h_min_224 + correction//2.
                h_max_224 = h_max_224 - correction//2.
            if diffw > patchsize:
                correction = diffw-patchsize
                w_min_224 = w_min_224 + correction//2.
                w_max_224 = w_max_224 - correction//2.

            normalized_h_min = (h_min_224/imgresize) 
            normalized_h_max = (h_max_224/imgresize) 
            normalized_w_min = (w_min_224/imgresize) 
            normalized_w_max = (w_max_224/imgresize) 
                        
            part_dict_img = img_to_part_xy_vis[img_id]
            for part in part_dict_img.keys():
                x,y = part_dict_img[part]                
                part_in_patch = 0 
                if y >= normalized_h_min and y <= normalized_h_max:
                    if x >= normalized_w_min and x <= normalized_w_max:
                        part_in_patch = 1
                if part not in proto_parts_presences[p][leaf_descendant_name].keys():
                    proto_parts_presences[p][leaf_descendant_name][part]=[]
                    
                proto_parts_presences[p][leaf_descendant_name][part].append(part_in_patch)
            
            for pair in duplicate_part_ids:
                if pair[0] in part_dict_img.keys():
                    if pair[1] in part_dict_img.keys():
                        presence0 = proto_parts_presences[p][leaf_descendant_name][pair[0]][-1]
                        presence1 = proto_parts_presences[p][leaf_descendant_name][pair[1]][-1]
                        if presence0 > presence1: 
                            proto_parts_presences[p][leaf_descendant_name][pair[1]][-1] = presence0

                        del proto_parts_presences[p][leaf_descendant_name][pair[0]]
                    else:

                        if pair[1] not in proto_parts_presences[p][leaf_descendant_name].keys():
                            proto_parts_presences[p][leaf_descendant_name][pair[1]]=[]
                        proto_parts_presences[p][leaf_descendant_name][pair[1]].append(proto_parts_presences[p][leaf_descendant_name][pair[0]][-1])
                        del proto_parts_presences[p][leaf_descendant_name][pair[0]]
                        
    print("Number of prototypes in parts_presences: ", len(proto_parts_presences.keys()), flush=True)
    
    prototypes_part_related = 0
    max_presence_purity = dict()
    max_presence_purity_part = dict()
    max_presence_purity_sum = dict()

    most_often_present_purity = dict()
    part_most_present = dict()

    # for each of a proto taking the least occurence with respect to topk from each descendant
    proto_parts_presences_copy = dict()
    for proto in proto_parts_presences.keys():
        proto_parts_presences_copy[proto] = dict()
        for part in part_dict_img.keys():
            proto_parts_presences_copy[proto][part] = None
            for leaf_descendant_name in proto_parts_presences[proto].keys():

                # to avoid the keyvalue error
                if part not in proto_parts_presences[proto][leaf_descendant_name]:
                    continue
                
                if proto_parts_presences_copy[proto][part] is None:
                    proto_parts_presences_copy[proto][part] = proto_parts_presences[proto][leaf_descendant_name][part]
                else:
                    if np.array(proto_parts_presences_copy[proto][part]).sum() > np.array(proto_parts_presences[proto][leaf_descendant_name][part]).sum():
                        proto_parts_presences_copy[proto][part] = proto_parts_presences[proto][leaf_descendant_name][part]
            if proto_parts_presences_copy[proto][part] is None:
                # meaning the part did not occur in any of the descendant
                del proto_parts_presences_copy[proto][part]
    proto_parts_presences = proto_parts_presences_copy
    
    for proto in proto_parts_presences.keys():
        
        max_presence_purity[proto]= 0.
        part_most_present[proto] = ('0',0)
        most_often_present_purity[proto] = 0.

        # CUB parts 7,8 and 9 are  duplicate (right and left). additional check that these should not occur (already fixed earlier in this function)
        if ('7' in proto_parts_presences[proto].keys() or '8' in proto_parts_presences[proto].keys() or '9' in proto_parts_presences[proto].keys()):
            print("unused part in keys! ", proto, proto_parts_presences[proto].keys(), proto_parts_presences[proto], flush=True)
            raise ValueError()
        
        for part in proto_parts_presences[proto].keys():
            presence_purity = np.mean(proto_parts_presences[proto][part])
            sum_occurs = np.array(proto_parts_presences[proto][part]).sum()
        
            # evaluate whether the purity of this prototype for this part is higher than for other parts
            if presence_purity > max_presence_purity[proto]:
                max_presence_purity[proto]=presence_purity
                max_presence_purity_part[proto]=parts_id_to_name[part]
                max_presence_purity_sum[proto] = sum_occurs
            elif presence_purity == max_presence_purity[proto]:
                if presence_purity == 0.:
                    max_presence_purity[proto]=presence_purity
                    max_presence_purity_part[proto]=parts_id_to_name[part]
                    max_presence_purity_sum[proto] = sum_occurs
                elif sum_occurs > max_presence_purity_sum[proto]:
                    max_presence_purity[proto]=presence_purity
                    max_presence_purity_part[proto]=parts_id_to_name[part]
                    max_presence_purity_sum[proto] = sum_occurs
        
            if sum_occurs > part_most_present[proto][1]:
                part_most_present[proto] = (part, sum_occurs)
                most_often_present_purity[proto]=presence_purity         
        if max_presence_purity[proto] > 0.5:
            prototypes_part_related += 1 
        
            
    print(f"Part-related (purity>0.5): {prototypes_part_related}", flush=True)
    purity_of_child = {}
    for child_name in child_name_to_protos:
        purity_of_child[child_name] = np.mean([max_presence_purity[p] for p in child_name_to_protos[child_name]])
        std = np.std([max_presence_purity[p] for p in child_name_to_protos[child_name]])
        num_descendants = node.get_node(child_name).num_leaf_descendents()
        print('Node:', node.name, '| Child:', child_name, '| Purity:', purity_of_child[child_name], '| Num desc:', num_descendants, '| Num protos:', len(child_name_to_protos[child_name]))
        for p in child_name_to_protos[child_name]:
            print('\tProto:', p, max_presence_purity[p])

    if log:
        log.log_values('log_epoch_overview', "p_cub_"+str(epoch), "mean purity (averaged over all prototypes, corresponding to purest part)", "std purity", "mean purity (averaged over all prototypes, corresponding to part with most often overlap)", "std purity", "# prototypes in csv", "#part-related prototypes (purity > 0.5)","","")

        log.log_values('log_epoch_overview', "p_cub_"+str(epoch), np.mean(list(max_presence_purity.values())), np.std(list(max_presence_purity.values())), np.mean(list(most_often_present_purity.values())), np.std(list(most_often_present_purity.values())), len(list(proto_parts_presences.keys())), prototypes_part_related, "", "")

    overall_node_purity = np.mean([max_presence_purity[p] for child_name in child_name_to_protos for p in child_name_to_protos[child_name]])
    return overall_node_purity, max_presence_purity