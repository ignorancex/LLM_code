# Imports
#pip install transformers
#pip install ftfy regex tqdm
#pip install git+https://github.com/openai/CLIP.git

import argparse
import os
import sys
import numpy as np
import torch
from torch import autograd
import torchvision 
import torchvision.datasets as datasets  
import torchvision.transforms as transforms  
from torchvision.transforms import (CenterCrop, Compose, Normalize, Resize,
                                    ToTensor)
from torch import optim  
from torch import nn  
from torch.utils.data import DataLoader, Dataset  
from tqdm import tqdm
import random
#import clip  
from torchvision.datasets import CIFAR100
from torchvision.datasets import ImageNet
from torchvision.datasets import CocoCaptions
from tqdm import tqdm
from robustness.robustness.tools.breeds_helpers import ClassHierarchy
from robustness.robustness.tools.breeds_helpers import setup_breeds
from create_coarse_sentences import get_caption_tree, get_caption_tree4, get_caption_tree6, get_caption_tree6_lemmas, get_caption_tree7, get_caption_tree8
from create_coarse_sentences import Node, expand_caption, expand_caption2, plotly_plot_tree

#from create_coarse_sentences_debug import get_caption_tree, get_caption_tree4, Node, expand_caption, plotly_plot_tree

#from lora.lib.CLIP.clip import *
from hilaCAM_lora.lib.CLIP.clip import *
#import hilaCAM_CLIP.clip as clip
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import cv2
#from gradcam.CLIP_explainability import get_image_text_relevance
from gradcam.CLIP_explainability import interpret, show_image_relevance, get_image_text_relevance

#import datasets

learning_rate = 0.001
batch_size = 64
num_epochs = 3
device = "cuda:0" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser(description='Linear probe CLIP training on ImageNet')

parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=1, type=int, metavar='N',
                    help='number of total epochs to run')

parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=32, type=int,
                    metavar='N',
                    help='mini-batch size (default: 32), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')

parser.add_argument('--checkpoint_file', default='', type=str,
                    help='filename to save checkpoint to')


parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://localhost:2375', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for ei ther single node or '
                         'multi node data parallel training')
                                     
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')
parser.add_argument('--num-cycles', default=10, type=int)




#imagenet_train_dataset = ImageNet(root="/mnt5/nir/dataset/ImageNet", split='train', transform=preprocess)
    
#imagenet_test_dataset = ImageNet(root="/mnt5/nir/dataset/ImageNet", split='val', transform=preprocess)                     

class CocoCaptionsDataset(CocoCaptions):
  
  def __init__(self, root, annFile, transform):
    super().__init__(root, annFile, transform)

  def _load_image(self, id: int) -> Image.Image:
    path = self.coco.loadImgs(id)[0]["file_name"]
    full_path = os.path.join(self.root, path)
    return Image.open(full_path).convert("RGB") , full_path

    
  def __getitem__(self, index):

    id = self.ids[index]
    image , full_path = self._load_image(id)
    #target contains 5 captions per image - use only the first caption
    target = self._load_target(id)[0]

    if self.transforms is not None:
      image, target = self.transforms(image, target)

    return full_path, image, target
    
  
class SubsetDataset(ImageNet):
    def __init__(self, transform, good_labels, split='train', root="/mnt5/nir/dataset/ImageNet"):
    
         super().__init__(root=root, split=split, transform=transform)
         self.__remove__(good_labels)

         #self.imagenet = ImageNet(root=root, split=split, transform=transform)
                                   
         #self.data = self.imagenet.data
         #self.data = self.imagenet.imgs
         #self.targets = self.imagenet.targets
         #self.final_data, self.final_targets = self.__remove__(good_labels)
      
    #def __getitem__(self, index):
    #    data, target = self.final_data[index], self.final_targets[index]
    #    return data, target, index

    #def __len__(self):
    #    return len(self.final_data)

    def __convert_list_to_tuple__(self, lst):
      return (lst[0], int(lst[1]))

    def __remove__(self, good_labels):

        mask = np.isin(self.targets, good_labels)
        #index_list = np.asarray(self.targets in good_labels).nonzero()
        #index_list = np.nonzero(mask)
        #print(f'{[i for i in index_list[0:5]]}')

        #print(f'self.targets[0:5]: {self.targets[0:5]}')
        #print(f'self.samples[0:5]: {self.samples[0:5]}')
        #print(f'self.targets[0:5]: {self.targets[0:5]}')
        self.targets = np.array(self.targets)[mask]
        #print(f'self.targets[0:5]: {self.targets[0:5]}')
        #self.targets = list(map(tuple, self.targets))
        #self.targets = [self.targets[i] for i in index_list]
        #self.samples = [self.samples[i] for i in index_list]
        
        #print(f'self.samples[0:5]: {self.samples[0:5]}')
        #print(f'self.samples[0:5][1]: {self.samples[0:5][1]}')
        self.samples = np.array(self.samples)[mask]
        #print(f'self.samples[0:5]: {self.samples[0:5]}')
        #print(f'self.samples[0:5][1]: {self.samples[0:5][1]}')
        self.samples = list(map(lambda l : (l[0], int(l[1])), self.samples))
        #self.samples = list(map(tuple, self.samples))
        #print(f'self.samples[0:5]: {self.samples[0:5]}')
        #print(f'self.samples[0:5][1]: {self.samples[0:5][1]}')
        #self.samples = [(t[0], int(t[1])) for t in self.samples]
        #print(f'self.samples[0:5]: {self.samples[0:5]}')
        #print(f'self.samples[0:5][1]: {self.samples[0:5][1]}')
        #exit(0)
        

        #data = self.data[mask]
        #targets = self.targets[mask]
        

        #return data, targets


def to_array(tensor):
    """Convert torch.Tensor to np.ndarray.
    Parameters
    ----------
    tensor : torch.Tensor
        Tensor of shape `(1, 3, *, *)` representing one sample batch of images.
    Returns
    -------
    arr : np.ndarray
        Array of shape `(*, *, 3)` representing an image that can be plotted
        directly.
    """
    tensor_ = tensor.squeeze()

    unnormalize_transform = Compose([Normalize(mean=[0, 0, 0],
                                               std=[1 / 0.26862954, 1 / 0.26130258, 1 / 0.27577711]),
                                     Normalize(mean=[-0.48145466, -0.4578275, -0.40821073],
                                               std=[1, 1, 1])])
    arr_ = unnormalize_transform(tensor_)
    arr = arr_.permute(1, 2, 0).detach().cpu().numpy()

    return arr

class DecisionTreeClipModel(nn.Module):
  def __init__(self, lora=-1, checkpoint_file=""):
    super(DecisionTreeClipModel, self).__init__()
    
    vit_name = 'ViT-B/32'
    #self.load_model(base_name=vit_name, weight_name=checkpoint_file)
    #load with lora lib
    self.clip_model, self.preprocess = self.load_model(base_name=vit_name, lora_r=lora)
    #load with hilaCAM
    #self.clip_model, self.preprocess = clip.load(vit_name, device=device, jit=False)
    self.clip_model = self.clip_model.to(device=device)
    #self.freeze_visual_head()
    self.clip_model = self.clip_model.float()
    #self.clip_model = clip_model.float()
    #self.clip_model = clip_model.double()
    self.sample_num = 0

    """   
    for param in self.clip_model.parameters():
        grad_exist = False
        if param.requires_grad:
          grad_exist = True
          print('param requires_grad')
    if not grad_exist:
      print('no param requires_grad')
    
    exit(0)
    """

    #self.hier = hier
    #self.root_id = list(hier.get_nodes_at_level(0))[0]
    #root_leaves_reachable = [hier.LEAF_ID_TO_NUM[leaf] for leaf in hier.leaves_reachable(self.root_id)]
    #name = hier.HIER_NODE_NAME[self.root_id]
    #print(f'root_id: {self.root_id}, name: {name}, #leaves_reachable: {len(root_leaves_reachable)}')
    #exit(0)

    #add encoded prompts hier to the nodes
    #self.decendants_prompts_dict = dict()
    #self.descendants_dict = dict()
    #self.leaves_reachable_dict = dict()
    #self.decendants_prompts_dict, self.descendants_dict, self.leaves_reachable_dict = self.GetDecendantsPromptsDict(self.root_id, self.decendants_prompts_dict, self.descendants_dict, self.leaves_reachable_dict)
    #self.encoded_prompts = self.get_encoded_prompts()
    self.encoded_prompts = None

    #self.good_labels = []
    #self.label_path_dict = dict()
    #self.calc_label_path_dict()


    #freeze self.clip_model params
    #for param in self.clip_model.parameters():
    #    param.requires_grad = False
    #    param.data = param.data.float()
    #    if param.grad:
    #      param.grad.data = param.grad.data.float() 

    #unfreeze image encoder params (keep text encoder frozen)
    #for param in self.clip_model.visual.parameters():
    #    param.requires_grad = True

    self.softmax = nn.Softmax(dim=1)

    #self.clip_model.eval()
    #self.fc = nn.Linear(input_size, num_classes)

  def freeze_visual_head(self):
    #Freeze all visual encoder layers
    for name, param in self.clip_model.named_parameters():
      if 'visual' in name:
        print(f'visual name: {name}, param.requires_grad: {param.requires_grad}')
        param.requires_grad = False

  def load_model(self, base_name="ViT-B/32", lora_r=-1, weight_name=""):
    clip_model, preprocess = clip.load(base_name, jit=False, lora=lora_r)
    #clip_model = clip_model.cuda()
    clip_model = clip_model.to(device=device)
    clip_model = clip_model.float()
    
    for param in clip_model.parameters():
        param.data = param.data.float()
        if param.grad:
          param.grad.data = param.grad.data.float() 

    if lora_r <= 0:
      #clip.model.convert_weights(self.clip_model)
      model.convert_weights(clip_model)

    if weight_name:
      clip_model.load_state_dict(torch.load(weight_name)['state_dict'])
        
    input_resolution = clip_model.visual.input_resolution
    context_length = clip_model.context_length
    vocab_size = clip_model.vocab_size

    print("=========")
    print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in clip_model.parameters()]):,}")
    print("Input resolution:", input_resolution)
    print("Context length:", context_length)
    print("Vocab size:", vocab_size)
    print("Image Preprocessing:", preprocess)
    print("=========")

    return clip_model, preprocess

  def calc_label_path_dict(self):
    
    num_imagenet_classes = len(self.hier.LEAF_IDS)

    for class_num in range(num_imagenet_classes):
      path, class_num = self.GetPathToLeaf(class_num)
      if path:
        self.good_labels.append(class_num)
        #to get label path from node id path
        #label_path = [hier.HIER_NODE_ID_TO_NUM[id] for id in path]
        self.label_path_dict[class_num] = path
  

  def get_encoded_prompts(self, node_names, req_grad=False):
    #make sure prompts are in the same order of HIER_NODE_NUM(which is the node label)
    #num_nodes = len(all_nodes)
    #node_names = [node.prompt for node in all_nodes ]

    prompts = [f'a photo of {name}' for name in node_names]

    tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(device=device)
    #if req_grad:
    #  tokenized_prompts = tokenized_prompts.float()
    #  tokenized_prompts.requires_grad = True
    #tokenized_prompts.double()
    #print(f'tokenized_prompts.type(): {tokenized_prompts.type()}')
    
    with torch.no_grad():
        encoded_prompts = self.clip_model.encode_text(tokenized_prompts).to(device=device)
        encoded_prompts /= encoded_prompts.norm(dim=-1, keepdim=True)   

    #encoded_prompts = encoded_prompts.double()
    #move to gpu only in forward
    #encoded_prompts = encoded_prompts.to('cpu')
    return encoded_prompts


  #def interpret_prediction(self, x, caption):
  #  self.classify_example(x, caption, print_path=True, break_on_incorrect=True)


  def get_caption_tree_figure(self, caption, file_name="tree_fig.png"):
    root, true_label_path, all_tree_nodes, edges, node_texts = get_caption_tree8(caption)
    return self.get_caption_tree_figure(all_tree_nodes, edges, caption, node_texts, file_name)

  def get_caption_tree_figure(self, all_tree_nodes, edges, caption, node_texts, file_name="tree_fig.png"):
    
    file_path = plotly_plot_tree(all_tree_nodes, edges, caption, node_texts, file_name)
    tree_fig = Image.open(file_path)
    return tree_fig

  def get_image_figure(self, image, permute=True):
    if permute:
      image = np.transpose(image[0].cpu().detach().numpy(), (1, 2, 0))
    
    image *= (1.0/image.max())
    image=(image+1)/2
    
    pil_image=Image.fromarray((image * 255).astype(np.uint8))
    return pil_image

  def get_caption_tree_and_image_figure(self, image, all_tree_nodes, edges, caption, node_texts, image_dir = "classification_samples", tree_file_name="tree_fig.png"):
    img_fig = self.get_image_figure(image)
    tree_fig = self.get_caption_tree_figure(all_tree_nodes, edges, caption, node_texts, tree_file_name)

    if not os.path.exists(image_dir):
      os.mkdir(image_dir)

    merged_img = get_concat_h_blank(img_fig, tree_fig)

    merged_file = f"caption_tree.png"
    merged_path = image_dir + "/" + merged_file
    merged_path = f'{image_dir}{merged_file}'
    print(f'get_caption_tree_and_image_figure - merged_path: {merged_path}')

    merged_img.save(merged_path)
    

  def get_grad_cam(self, x, img_path):
    input_grad = x.grad.data.cpu()
    img_arr = x.detach().clone().cpu()
    #x.shape: torch.Size([1, 3, 224, 224]), input_grad.shape: torch.Size([1, 3, 224, 224])
    #print(f'\nget_grad_cam x.shape: {x.shape}, input_grad.shape: {input_grad.shape}\n')

    ######
    pil_img = Image.open(img_path)

    transform = Compose([
                         #Resize(224),
                         #CenterCrop(224),
                         ToTensor(),
                         #Normalize(mean=[0.485, 0.456, 0.406],
                         #          std=[0.229, 0.224, 0.225])
                         ])

    
    img_tensor_ = transform(pil_img)
    #img_tensor_ = ToTensor(pil_img)
    img_tensor = img_tensor_.unsqueeze(0)
    #img_arr = img_tensor.clone()
    print(f'\nimg_tensor.shape: {img_tensor.shape}')
    ######

    # pool the gradients across the channels
    pooled_gradients = torch.mean(input_grad, dim=[0, 2, 3])
    #pooled_gradients.shape: torch.Size([3])
    #print(f'pooled_gradients.shape: {pooled_gradients.shape}')

    # weight the channels by corresponding gradients
    for i in range(3):
      #img_arr[:, :, i] *= pooled_gradients[i]
      img_arr[:, i, :, :] *= pooled_gradients[i]
      img_tensor[:, i, :, :] *= pooled_gradients[i]

    # average the channels of the image
    heatmap = torch.mean(img_arr, dim=1).squeeze()
    #heatmap = torch.mean(img_tensor, dim=1).squeeze()
    #heatmap.shape: torch.Size([224, 224])
    #print(f'heatmap.shape: {heatmap.shape}')

    # relu on top of the heatmap
    # expression (2) in https://arxiv.org/pdf/1610.02391.pdf
    heatmap = np.maximum(heatmap, 0)
    # normalize the heatmap
    heatmap /= torch.max(heatmap)

    dpi = 400
    fig = plt.figure(figsize=(6, 6), dpi=dpi)
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.matshow(heatmap.squeeze())

    #TODO: move to a function
    dir_path = "heatmap"

    if not os.path.exists(dir_path):
      os.mkdir(dir_path)

    #sample_num = 1
    file_name = f"heatmap_{self.sample_num}.png"
    save_path = dir_path + "/" + file_name
    ###

    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')

    
    img = cv2.imread(img_path)
    img = cv2.resize(img, (heatmap.shape[1], heatmap.shape[0]))
    print(f'img.shape: \n{img.shape}')
    #heatmap = cv2.resize(heatmap.numpy(), (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    file_name = f"resized_heatmap_{self.sample_num}.png"
    save_path = dir_path + "/" + file_name
    cv2.imwrite(save_path, heatmap)

    superimposed_img = heatmap * 0.4 + img
    #superimposed_img = heatmap * 0.4 + img_arr.squeeze().permute(1,2,0).numpy()

    #TODO: move to a function
    dir_path = "gradcam"

    if not os.path.exists(dir_path):
      os.mkdir(dir_path)

    #sample_num = 1
    file_name = f"gradcam_{self.sample_num}.png"
    gradcam_save_path = dir_path + "/" + file_name

    file_name = f"img_{self.sample_num}.png"
    img_save_path = dir_path + "/" + file_name

    ###

    cv2.imwrite(gradcam_save_path, superimposed_img)
    pil_img.save(img_save_path)
    #cv2.imwrite(img_save_path, pil_img)

  def get_img_patches(self, img, num_patches):

    patches_dim = num_patches ** 0.5
    patch_size = int(img.shape[-1] / patches_dim)

    kc = img.shape[1] #channels kernel size
    kh, kw = patch_size, patch_size  # height, width kernel size
    dc = kc #channels stride
    dh, dw = patch_size, patch_size  # height, width stride
    
    img_patches = img.clone()

    # Pad to multiples of 32
    #x = F.pad(x, (x.size(2)%kw // 2, x.size(2)%kw // 2,
    #          x.size(1)%kh // 2, x.size(1)%kh // 2,
    #          x.size(0)%kc // 2, x.size(0)%kc // 2))

    img_patches = img_patches.unfold(1, kc, dc).unfold(2, kh, dh).unfold(3, kw, dw)
    unfold_shape = img_patches.size()
    img_patches = img_patches.contiguous().view(-1, kc, kh, kw)
    #print(f'img_patches.shape: {img_patches.shape}, unfold_shape: {unfold_shape}')
    #img_patches.shape: torch.Size([49, 3, 32, 32]), unfold_shape: torch.Size([1, 1, 7, 7, 3, 32, 32])
    #patches_orig.shape: torch.Size([1, 3, 224, 224])

    return img_patches, unfold_shape


  def reshape_patches(self, img_patches, unfold_shape):

    # Reshape back
    patches_orig = img_patches.view(unfold_shape)
    output_c = unfold_shape[1] * unfold_shape[4]
    output_h = unfold_shape[2] * unfold_shape[5]
    output_w = unfold_shape[3] * unfold_shape[6]
    patches_orig = patches_orig.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
    patches_orig = patches_orig.view(1, output_c, output_h, output_w)

    #print(f'patches_orig.shape: {patches_orig.shape}')
    #patches_orig.shape: torch.Size([1, 3, 224, 224])

    return patches_orig

  def get_relative_relevancy_image(self, img, cam_image):
    total_num_patches = cam_image.shape[0]
    img_patches, unfold_shape = self.get_img_patches(img, total_num_patches)

    if (img_patches.max() - img_patches.min() != 0):
        img_patches = (img_patches - img_patches.min()) / (img_patches.max() - img_patches.min())
    

    #print(f'cam_image.min(): {cam_image.min()}, cam_image.max(): {cam_image.max()}\n')
    cam_image = (cam_image - cam_image.min()) / (cam_image.max() - cam_image.min()) 
    #print(f'cam_image.min(): {cam_image.min()}, cam_image.max(): {cam_image.max()}\n')
               
    #print(f'before scale img_patches.min(): {img_patches.min()}, img_patches.max(): {img_patches.max()}\n')


    #patch_indices = list(range(len(cam_image)))
    #img_patches[patch_indices, : , : , : ] = img_patches[patch_indices, : , : , : ]*cam_image[patch_indices]
    #[top_patches_indices, : , : , : ] = img_patches[top_patches_indices, : , : , : ]

    #print(f'img_patches.shape: {img_patches.shape},  cam_image.shape: {cam_image.shape}before torch.mul')
    #print(f'\nimg_patches before torch.mul: \n{img_patches[0]}')
    for i, scale in enumerate(cam_image):
        #img_patches[i] = img_patches[i]*scale
        img_patches[i] = img_patches[i]*scale
    #print(f'\n\nimg_patches after torch.mul: \n{img_patches[0]}')

    #print(f'after scale img_patches.min(): {img_patches.min()}, img_patches.max(): {img_patches.max()}\n')


    restored_img = self.reshape_patches(img_patches, unfold_shape)
    #print(f'restored_img.min(): {restored_img.min()}, restored_img.max(): {restored_img.max()}\n')


    restored_img = restored_img.squeeze(0)

    if (restored_img.max() - restored_img.min() != 0):
        restored_img = (restored_img - restored_img.min()) / (restored_img.max() - restored_img.min())
    
    #print(f'restored_img.min(): {restored_img.min()}, restored_img.max(): {restored_img.max()}\n')

    restored_img = restored_img.cpu().permute(1, 2, 0)

    return restored_img



  def perturbation_image(self, img, text_inputs, cam_image, pert_steps, is_positive_pert=False, folder_path="" , suffix="", model_str=""):
    with torch.no_grad():
        
        cam_shaded_img_vis = self.get_relative_relevancy_image(img, cam_image)
        fig, axs = plt.subplots()
        axs.imshow(cam_shaded_img_vis)
        axs.axis('off')
        axs.set_title('DiRe relevancy', fontsize=8)
        plt.savefig(f'{folder_path}_3VL_shaded_relevancy_{suffix}.png')
        plt.close()

        if is_positive_pert:
            cam_image = cam_image * (-1)
        
        total_num_patches = cam_image.shape[0]
        #print(f'total_num_patches: {total_num_patches}')

        ##############################################
        ##############################################

                
        img_patches, unfold_shape = self.get_img_patches(img, total_num_patches)
        
        if (img_patches.max() - img_patches.min() != 0):
            img_patches = (img_patches - img_patches.min()) / (img_patches.max() - img_patches.min())
    

        positive_text_fig = plt.figure(figsize=(9, 9), dpi=200)
        negative_text_fig = plt.figure(figsize=(9, 9), dpi=200)
        masked_img_fig = plt.figure(figsize=(9, 9), dpi=200)
        figures = [("pos_text", positive_text_fig), ("neg_text", negative_text_fig)]
        #figures = [("neg_text", negative_text_fig)]
        #for step_idx, num_top_patches in enumerate(num_tokens):
        for step_idx, step in enumerate(pert_steps):
            # find top step boxes
            num_top_patches = int((1 - step) * total_num_patches)
            #print(f'step_idx: {step_idx}, step: {step}, num_top_patches: {num_top_patches}')
            _, top_patches_indices = cam_image.topk(k=num_top_patches, dim=-1)
            #print(f'top_patches_indices: {top_patches_indices}')
            top_patches_indices = top_patches_indices.cpu().data.numpy()
            top_patches_indices.sort()

            
            masked_img = torch.zeros_like(img_patches)

           
            masked_img[top_patches_indices, : , : , : ] = img_patches[top_patches_indices, : , : , : ]
            masked_img = self.reshape_patches(masked_img, unfold_shape)

            #masked_img = masked_img.reshape_as(img)
            masked_img = masked_img.squeeze(0)
            if (masked_img.max() - masked_img.min() != 0):
                masked_img = (masked_img - masked_img.min()) / (masked_img.max() - masked_img.min())
            #img.shape: torch.Size([1, 3, 224, 224]), img_patches.shape: torch.Size([3, 49, 1024]), masked_img.shape: torch.Size([3, 224, 224])
            #print(f'img.shape: {img.shape}, img_patches.shape: {img_patches.shape}, masked_img.shape: {masked_img.shape}')
            masked_img = masked_img.cpu().permute(1, 2, 0)
            #masked_img = np.uint8(255 * masked_img)


            ax_mask = masked_img_fig.add_subplot(3, 3, step_idx+1)
            ax_mask.axis('off')
            #ax.set_title(f"{pos_neg_str}_{int(100 - 100*num_top_patches/num_tokens[0])}% removed")
            ax_mask.set_title(f"Image with {100*step}% patches removed")
            ax_mask.title.set_fontsize(8)
            #pil_img = Image.open(img_file_name)
            ax_mask.imshow(masked_img)


            
            with torch.set_grad_enabled(True):
                R_text, R_image = interpret(model=self.clip_model, image=img, texts=text_inputs, device=device, top_patches_indices=top_patches_indices)
            
            batch_size = text_inputs.shape[0]
            for i, (pos_neg_str, fig) in enumerate(figures):
                
                image_relevance = torch.zeros_like(cam_image)
                
                for rel_index, patch_index in enumerate(top_patches_indices):
                    image_relevance[patch_index] = R_image[i][rel_index]
                
                #img_file_name = show_image_relevance(R_image[i], img, img, idx=step_idx, file_name_prefix=f"VL_checklist_VG_{pos_neg_str}")
                img_file_name = show_image_relevance(image_relevance, img, img, idx=step_idx, file_name_prefix=f"VL_checklist_VG_{pos_neg_str}")


                ax = fig.add_subplot(3, 3, step_idx+1)
                ax.axis('off')
                #ax.set_title(f"{pos_neg_str}_{int(100 - 100*num_top_patches/num_tokens[0])}% removed")
                ax.set_title(f"{pos_neg_str} {100*step}% removed")
                ax.title.set_fontsize(8)
                pil_img = Image.open(img_file_name)
                ax.imshow(pil_img)

                
        for pos_neg_str, fig in figures: 
            fig.savefig(f"{folder_path}_3VL_{pos_neg_str}_perturb_relevance_{suffix}.jpg", dpi=200, bbox_inches='tight')
        
        masked_img_fig.savefig(f"{folder_path}_3VL_masked_image_{suffix}.jpg", dpi=200, bbox_inches='tight')
        plt.close()

  def classify_example(self, img_path, x, caption, image_num, print_all=False, print_incorrect=True):
    correct_caption = "wooden bench"
    predicted_caption = "metal bench"
    orig_image = Image.open(img_path)
    img = self.preprocess(orig_image).unsqueeze(0).to(device)

    image_dir = f'COCO_vis/wooden_metal_bench_image_num_{image_num}'
    os.makedirs(image_dir, exist_ok=True)

    image_dir += '/'

    fig, axs = plt.subplots()
    axs.imshow(orig_image)
    axs.axis('off')
    plt.savefig(f'{image_dir}orig_image.png')
    plt.close()

    self.get_HilaCAM_visualizations(orig_image, img, img_path ,image_dir, texts=[correct_caption, predicted_caption, 'white bench', 'cement bench'], suffix=f"{self.sample_num}_predicted")
    exit(0)

    root, true_label_path, all_tree_nodes, edges, node_texts = get_caption_tree6_lemmas(caption)

    with torch.set_grad_enabled(True):
      probs_matrix, _ = self.get_probs_matrix(x, true_label_path, all_tree_nodes, req_grad=True)
      prob_values, predictions = probs_matrix.topk(1)
    #print(f'\nprob_values: {prob_values}\npredictions: {predictions}\n')
    #print(f'\ntrue_label_path: {true_label_path}\n')
    
    if print_all or print_incorrect:
      #print('\nprediction path:')
      #print('predicted caption (probability) - correct/incorrect\n')
      str_to_print = '\nprediction path:\n'
      str_to_print += 'predicted caption (probability) - correct/incorrect\n\n'
      total_correct = True
      
      for i, predicted_node_num in enumerate(predictions):
        is_correct = (predicted_node_num == true_label_path[i])
        correct_str = "correct" if is_correct else "incorrect"
        predicted_caption = all_tree_nodes[predicted_node_num].prompt
        
        #print(f'{predicted_caption} ({100 * prob_values[i].item():.2f}%) - {correct_str}')
        str_to_print += f'{predicted_caption} ({100 * prob_values[i].item():.2f}%) - {correct_str}\n'

        if print_all:
           print(str_to_print)

        if not is_correct:
          total_correct = False

          original_stdout = sys.stdout # Save a reference to the original standard output

          image_dir = f'COCO_vis/image_num_{image_num}'
          os.makedirs(image_dir, exist_ok=True)

          image_dir += '/'

          with open(f'{image_dir}expanded_captions_{self.sample_num}.txt', 'w') as f:
              sys.stdout = f # Change the standard output to the file we created.

              if print_incorrect and (not print_all):
                print(str_to_print)
            
              correct_node_num = true_label_path[i]
              correct_caption = all_tree_nodes[correct_node_num].prompt
              correct_prob = probs_matrix[i][correct_node_num].item()

              #print(f'path_index: {i}, all_tree_nodes[correct_node_num].node_num: {all_tree_nodes[correct_node_num].node_num}, correct_node_num: {correct_node_num}')
              #print(f'all_tree_nodes[correct_node_num].prompt is: {all_tree_nodes[correct_node_num].prompt}')

              print(f'correct caption is: \'{correct_caption}\' - with probability: ({100 * correct_prob:.2f}%)')
              #expanded_captions = expand_caption(predicted_caption, correct_caption)
              expanded_captions = expand_caption2(predicted_caption, correct_caption)
              #expanded_captions = expand_caption2(caption, predicted_caption, correct_caption)
              encoded_prompts = self.get_encoded_prompts(expanded_captions)
              similarity = self.get_cosine_similarity(x,encoded_prompts)
              probs = similarity.softmax(dim=-1)
              #print(f'\n\nexpanded_captions: \n{expanded_captions}')
              #print(f'\n\nencoded_prompts: {encoded_prompts}')
              #print(f'\n\nsimilarity: {similarity}')
              #print(f'\n\nprobs: {probs}')
              #print(f'\n\nencoded_prompts.shape: {encoded_prompts.shape}')
              #print(f'\n\nsimilarity.shape: {similarity.shape}')
              #print(f'\n\nprobs.shape: {probs.shape}')

              probs = probs.flatten()
              #print(f'\n\nprobs.shape: {probs.shape}, probs: {probs}')
              #print(f'\n\nlen(expanded_captions): {len(expanded_captions)}, \nexpanded_captions: {expanded_captions}')
              #print(f'\n\probs.flatten().shape: {probs.flatten().shape}')

              correct_token = expanded_captions[-1]
              predicted_token = expanded_captions[-2]
             
              print(f'\nexpand node prediction: ')
              expanded_texts = [f'a photo of {txt}' for txt in expanded_captions]
              #for idx, capt in enumerate(expanded_captions):
              for idx, capt in enumerate(expanded_texts):
                #print(f'caption: \'{capt}\', prob: ({100 * probs[idx].item():.2f}%)')
                print(f'\'{capt}\', prob: ({100 * probs[idx].item():.2f}%)')
                           
              sys.stdout = original_stdout # Reset the standard output to its original value
  
          
          fig, axs = plt.subplots()
          axs.imshow(orig_image)
          axs.axis('off')
          plt.savefig(f'{image_dir}orig_image.png')
          plt.close()


          #self.train() 
          with torch.set_grad_enabled(True):
            #HilaCAM
            #texts = [predicted_caption, correct_caption]
            #texts = expanded_captions
            #get_image_text_relevance(img_path, texts, self.clip_model, self.preprocess)

            #texts = [correct_caption, predicted_caption]
            
            
            
            #####################
            self.get_HilaCAM_visualizations(orig_image, img, img_path ,image_dir, texts=[correct_caption, predicted_caption], suffix=f"{self.sample_num}_predicted")
            
            for i, capt in enumerate(expanded_texts[0:-1]):
              self.get_HilaCAM_visualizations(orig_image, img, img_path, image_dir, texts=[expanded_texts[-1],capt], suffix=f"{self.sample_num}_expanded_{i}")

            ####################
            self.sample_num += 1

            """
            clip_3VL_R_text, clip_3VL_R_image = get_image_text_relevance(img_path, texts, self.clip_model, self.preprocess, file_name_prefix="COCO_")

            clip_3VL_positive_cam_image = clip_3VL_R_image[0]
            clip_3VL_negative_cam_image = clip_3VL_R_image[1]

            clip_3VL_pos_relative_relevancy_vis = self.get_relative_relevancy_image(img, clip_3VL_positive_cam_image)
            clip_3VL_neg_relative_relevancy_vis = self.get_relative_relevancy_image(img, clip_3VL_negative_cam_image)

            clip_3VL_relative_relevancy_fig = plt.figure(figsize=(6, 6), dpi=200)

            ax1_rel = clip_3VL_relative_relevancy_fig.add_subplot(1, 2, 1)
            ax1_rel.imshow(clip_3VL_pos_relative_relevancy_vis)
            ax1_rel.axis('off')
            ax1_rel.set_title(texts[0], fontsize=8)
            ax2_rel = clip_3VL_relative_relevancy_fig.add_subplot(1, 2, 2)
            ax2_rel.imshow(clip_3VL_neg_relative_relevancy_vis)
            ax2_rel.axis('off')
            ax2_rel.set_title(texts[1], fontsize=8)
            plt.tight_layout()
            plt.savefig(f'{image_dir}_3VL_pos_neg_relative_relevancy_{self.sample_num}.png', bbox_inches='tight')
            plt.close()

            #visualize 3VL positive negative relevancy maps
            clip_3VL_pos_cam_img_file_name = show_image_relevance(clip_3VL_positive_cam_image, img, orig_image=orig_image, idx=2)
            clip_3VL_neg_cam_img_file_name = show_image_relevance(clip_3VL_negative_cam_image, img, orig_image=orig_image, idx=3)


            clip_3VL_pos_image = Image.open(clip_3VL_pos_cam_img_file_name)
            clip_3VL_neg_image = Image.open(clip_3VL_neg_cam_img_file_name)

            #fig = plt.figure(figsize=(6, 6), dpi=200)
            fig = plt.figure()
                   
            ax1 = fig.add_subplot(1, 2, 1)
            ax1.imshow(clip_3VL_pos_image)
            ax1.axis('off')
            ax1.set_title(f'{texts[0]}', fontsize=8)
            ax2 = fig.add_subplot(1, 2, 2)
            ax2.imshow(clip_3VL_neg_image)
            ax2.axis('off')
            ax2.set_title(f'{texts[1]}', fontsize=8)
            plt.tight_layout()
            plt.savefig(f'{image_dir}_3VL_pos_neg_images_{self.sample_num}.png', bbox_inches='tight')
            plt.close()


            #visualize 3VL diff relevance map
            clip_3VL_cam_image = clip_3VL_positive_cam_image - clip_3VL_negative_cam_image
            #cam_image = negative_cam_image *(-1)

            clip_3VL_cam_image = (clip_3VL_cam_image - clip_3VL_cam_image.min()) / (clip_3VL_cam_image.max() - clip_3VL_cam_image.min())

            #visualize diff relevance after norm
            clip_3VL_cam_img_file_name = show_image_relevance(clip_3VL_cam_image, img, orig_image=orig_image, idx=1)

            clip_3VL_image_diff_cam = Image.open(clip_3VL_cam_img_file_name)
                    
            fig, axs = plt.subplots()
            axs.imshow(clip_3VL_image_diff_cam)
            axs.axis('off')
            plt.savefig(f'{image_dir}_3VL_image_diff_cam_{self.sample_num}.png')
            plt.close()

            tokenized_text = clip.tokenize(texts).to(device)
            

            #self.perturbation_image(img_path, text_inputs, cam_image, pert_steps, is_positive_pert=False, folder_path="" , model_str=""):
            pert_steps = [0, 0.25, 0.5, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
            self.perturbation_image(img, tokenized_text, clip_3VL_cam_image, pert_steps, is_positive_pert=False, folder_path=image_dir , model_str="")   
            """
            #self.sample_num += 1

      if not total_correct:
        self.get_caption_tree_and_image_figure(x, all_tree_nodes, edges, caption, node_texts, image_dir)
        #exit(0)
            
            
      
      #if not total_correct:
      #  exit(0)
      #break

    return predictions, torch.tensor(true_label_path, device=device)

  def get_HilaCAM_visualizations(self, orig_image, img, img_path, image_dir, texts, suffix=f"_predicted"):
    
    clip_3VL_R_text, clip_3VL_R_image = get_image_text_relevance(img_path, texts, self.clip_model, self.preprocess, file_name_prefix="COCO_")

    clip_3VL_positive_cam_image = clip_3VL_R_image[0]
    clip_3VL_negative_cam_image = clip_3VL_R_image[1]
    clip_3VL_white_cam_image = clip_3VL_R_image[2]
    clip_3VL_cement_cam_image = clip_3VL_R_image[3]


    clip_3VL_pos_relative_relevancy_vis = self.get_relative_relevancy_image(img, clip_3VL_positive_cam_image)
    clip_3VL_neg_relative_relevancy_vis = self.get_relative_relevancy_image(img, clip_3VL_negative_cam_image)
    clip_3VL_white_relative_relevancy_vis = self.get_relative_relevancy_image(img, clip_3VL_white_cam_image)
    clip_3VL_cement_relative_relevancy_vis = self.get_relative_relevancy_image(img, clip_3VL_cement_cam_image)


    clip_3VL_relative_relevancy_fig = plt.figure(figsize=(6, 6), dpi=200)

    ax1_rel = clip_3VL_relative_relevancy_fig.add_subplot(2, 2, 1)
    ax1_rel.imshow(clip_3VL_pos_relative_relevancy_vis)
    ax1_rel.axis('off')
    ax1_rel.set_title(texts[0], fontsize=8)
    ax2_rel = clip_3VL_relative_relevancy_fig.add_subplot(2, 2, 2)
    ax2_rel.imshow(clip_3VL_neg_relative_relevancy_vis)
    ax2_rel.axis('off')
    ax2_rel.set_title(texts[1], fontsize=8)
    ax3_rel = clip_3VL_relative_relevancy_fig.add_subplot(2, 2, 3)
    ax3_rel.imshow(clip_3VL_white_relative_relevancy_vis)
    ax3_rel.axis('off')
    ax3_rel.set_title(texts[2], fontsize=8)
    ax4_rel = clip_3VL_relative_relevancy_fig.add_subplot(2, 2, 4)
    ax4_rel.imshow(clip_3VL_cement_relative_relevancy_vis)
    ax4_rel.axis('off')
    ax4_rel.set_title(texts[3], fontsize=8)

    plt.tight_layout()
    path = f'{image_dir}_3VL_wooden_metal_white_cement_bench_relative_relevancy_{suffix}.png'
    print(f'path: {path}')
    plt.savefig(path, bbox_inches='tight')
    plt.close()

    #visualize 3VL positive negative relevancy maps
    clip_3VL_pos_cam_img_file_name = show_image_relevance(clip_3VL_positive_cam_image, img, orig_image=orig_image, idx=2)
    clip_3VL_neg_cam_img_file_name = show_image_relevance(clip_3VL_negative_cam_image, img, orig_image=orig_image, idx=3)


    clip_3VL_pos_image = Image.open(clip_3VL_pos_cam_img_file_name)
    clip_3VL_neg_image = Image.open(clip_3VL_neg_cam_img_file_name)

    #fig = plt.figure(figsize=(6, 6), dpi=200)
    fig = plt.figure()
            
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(clip_3VL_pos_image)
    ax1.axis('off')
    ax1.set_title(f'{texts[0]}', fontsize=8)
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(clip_3VL_neg_image)
    ax2.axis('off')
    ax2.set_title(f'{texts[1]}', fontsize=8)
    plt.tight_layout()
    path = f'{image_dir}_3VL_pos_neg_wooden_metal_bench_{suffix}.png'
    print(f'path: {path}')
    plt.savefig(path, bbox_inches='tight')
    plt.close()


    #visualize 3VL diff relevance map
    clip_3VL_DiRe_cam_image = clip_3VL_positive_cam_image - clip_3VL_negative_cam_image
    #cam_image = negative_cam_image *(-1)

    clip_3VL_DiRe_cam_image = (clip_3VL_DiRe_cam_image - clip_3VL_DiRe_cam_image.min()) / (clip_3VL_DiRe_cam_image.max() - clip_3VL_DiRe_cam_image.min())

    #visualize diff relevance after norm
    clip_3VL_DiRe_img_file_name = show_image_relevance(clip_3VL_DiRe_cam_image, img, orig_image=orig_image, idx=1)

    clip_3VL_image_DiRe = Image.open(clip_3VL_DiRe_img_file_name)
            
    fig, axs = plt.subplots()
    axs.imshow(clip_3VL_image_DiRe)
    axs.axis('off')
    path = f'{image_dir}_3VL_DiRe_wooden_metal_bench_{suffix}.png'
    print(f'path: {path}')
    plt.savefig(path)
    plt.close()

    clip_3VL_DiRe_relative_relevancy_vis = self.get_relative_relevancy_image(img, clip_3VL_DiRe_cam_image)

    fig, axs = plt.subplots()
    axs.imshow(clip_3VL_DiRe_relative_relevancy_vis)
    axs.axis('off')
    path = f'{image_dir}_3VL_DiRe_relative_relevancy_wooden_metal_bench_{suffix}.png'
    print(f'path: {path}')
    plt.savefig(path)
    plt.close()


    tokenized_text = clip.tokenize(texts).to(device)
            

    #self.perturbation_image(img_path, text_inputs, cam_image, pert_steps, is_positive_pert=False, folder_path="" , model_str=""):
    pert_steps = [0, 0.25, 0.5, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
    # self.perturbation_image(img, tokenized_text, clip_3VL_cam_image, pert_steps, is_positive_pert=False, folder_path=image_dir , suffix=suffix, model_str="")   



  """
  def classify_example(self, x, print_path=False):
    curr_node = self.root_id
    #node_path_list = []

    decendants = self.descendants_dict[curr_node]
    if print_path:
      print('prediction path:')
      print('prediction (probability)\n')

    while (decendants):
    
      encoded_prompts = self.decendants_prompts_dict[curr_node]
      similarity = self.get_cosine_similarity(x,encoded_prompts)
      probs = similarity.softmax(dim=-1)
      #print(f'similarity: {similarity}, probs: {probs}')
      value, index = probs[0].topk(1)
      curr_node = decendants[index]
      if print_path:
        print(f'{self.hier.HIER_NODE_NAME[curr_node].split(", ")[0]} ({100 * value.item():.2f}%)')
        #node_path_list.append((curr_node,value))
      decendants = self.descendants_dict[curr_node]
    
    return self.hier.LEAF_ID_TO_NUM[curr_node]
    """

  def get_cosine_similarity(self, x, encoded_prompts):
    images = x.to(device=device)
    image_features = self.clip_model.encode_image(images)
    
    #print(f'get_cosine_similarity image_features.type(): {image_features.type()}, image_features: \n\n{image_features}')
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    encoded_prompts = encoded_prompts.to(device=device)
    similarity = 100.0 * image_features @ encoded_prompts.t() #batch_size X num nodes
    #print(f'get_cosine_similarity similarity: \n\n{similarity}')

    #encoded_prompts = encoded_prompts.to(device='cpu')
    return similarity


  def get_probs_matrix(self, images, true_label_path, all_tree_nodes, req_grad=False):
    
    node_names = [node.prompt for node in all_tree_nodes ]
    self.encoded_prompts = self.get_encoded_prompts(node_names,req_grad)
    #print(f'\nencoded_prompts.grad_fn: {self.encoded_prompts.grad_fn}\n')

    similarity = self.get_cosine_similarity(images,self.encoded_prompts)
    #print(f'\nsimilarity.grad_fn: {similarity.grad_fn}\n')
    exponent_similarity = similarity.exp()
    #print(f'\nexponent_similarity.grad_fn: {exponent_similarity.grad_fn}\n')

    #duplicate each row of exponent_similarity by len of its coresponding path
    duplicated_exponent_similarity = self.duplicate_matrix_rows(exponent_similarity,true_label_path)
    #print(f'\nduplicated_exponent_similarity.grad_fn: {duplicated_exponent_similarity.grad_fn}\n')

    #print(f'\n\n\nlen(unrolled_node_ids): {len(unrolled_node_ids)}')
    #print(f'exponent_similarity.shape: {exponent_similarity.shape}')
    #print(f'duplicated_exponent_similarity.shape: {duplicated_exponent_similarity.shape}')

    #exit(0)

    #DT_CLIP_model(images,true_label_path, all_tree_nodes)

    path_mask_matrix = self.get_path_mask_matrix(duplicated_exponent_similarity, true_label_path, all_tree_nodes)
    #print(f'\npath_mask_matrix.grad_fn: {path_mask_matrix.grad_fn}\n')

    softmax_numerator = duplicated_exponent_similarity*path_mask_matrix
    #print(f'\nsoftmax_numerator.grad_fn: {softmax_numerator.grad_fn}\n')
    
    softmax_denominator = softmax_numerator.sum(dim=1,keepdim=True)
    #print(f'\nsoftmax_denominator.grad_fn: {softmax_denominator.grad_fn}\n')
    probs = softmax_numerator / softmax_denominator

    return probs, path_mask_matrix

  def forward(self, images, true_label_path, all_tree_nodes):

    probs_matrix, path_mask_matrix = self.get_probs_matrix(images, true_label_path, all_tree_nodes)
    #print(f'\nprobs_matrix.grad_fn: {probs_matrix.grad_fn}\n')
    #print(f'\npath_mask_matrix.grad_fn: {path_mask_matrix.grad_fn}\n')

    #log of zero probabilities will return -inf but will not be used 
    #as only the log probability of the ground truth is used in NLL Loss
    #RuntimeError: Function 'LogBackward0' returned nan values in its 0th output.
    #log_probs = probs.log()
    probs_matrix[path_mask_matrix.bool()] = probs_matrix[path_mask_matrix.bool()].log()
    #print(f'\nprobs_matrix.grad_fn: {probs_matrix.grad_fn}\n')

    #return log_probs
    return probs_matrix


  def duplicate_matrix_rows(self, x, true_label_path):
    
    #res = torch.cat([x[idx].repeat(len(self.label_path_dict[l.item()]),1) for idx, l in enumerate(labels)] )
    path_len = len(true_label_path)
    res = x.repeat(path_len,1)
    return res
    """
    >>> t = torch.tensor([[1, 2, 3], [4, 4, 4]])                                                        
    >>> t
    tensor([[1, 2, 3],
        [4, 4, 4]])
    >>> s = torch.cat([t[i].repeat(5,1) for i in range(len(t))] )
    >>> s
    tensor([[1, 2, 3],
        [1, 2, 3],
        [1, 2, 3],
        [1, 2, 3],
        [1, 2, 3],
        [4, 4, 4],
        [4, 4, 4],
        [4, 4, 4],
        [4, 4, 4],
        [4, 4, 4]])
    """

  def exponent_matrix(self, x):
    res = torch.exp(x)
    return res 

    """
    >>> t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9] ])
    >>> t
    tensor([[1, 2, 3],
           [4, 5, 6],
           [7, 8, 9]])
    >>> ex = torch.exp(t)
    >>> ex
    tensor([[2.7183e+00, 7.3891e+00, 2.0086e+01],
            [5.4598e+01, 1.4841e+02, 4.0343e+02],
            [1.0966e+03, 2.9810e+03, 8.1031e+03]])  
    """

  def get_path_mask_matrix(self, similarity_matrix, true_label_path, all_tree_nodes):
    
    mask = torch.zeros_like(similarity_matrix)
    
    for idx, node_num in enumerate(true_label_path):
      true_label_node = all_tree_nodes[node_num]
      parent_node = true_label_node.parent
      siblings = parent_node.children
      row_mask = list(siblings.keys())
      mask[idx][row_mask] = 1

    #for idx, node_id in enumerate(unrolled_node_ids):
    #  siblings = self.get_siblings_node_ids(node_id)
    #  row_mask = [self.hier.HIER_NODE_ID_TO_NUM[s] for s in siblings]
    #  mask[idx][row_mask] = 1
    
    #TODO: just mask similarity_matrix with similarity_matrix[idx][~row_mask] = 0 ??

    return mask

    """
    >>> t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9] ])
    >>> t
    tensor([[1, 2, 3],
           [4, 5, 6],
           [7, 8, 9]])
    >>> p = torch.zeros_like(t)
    >>> p
    tensor([[0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]]) 
    >>> row_mask = torch.tensor([0, 2])
    >>> row_mask
    tensor([0, 2])
    >>> p[0][row_mask] = 1
    >>> p
    tensor([[1, 0, 1],
            [0, 0, 0],
            [0, 0, 0]])
    """

  def get_probabilities_matrix(self, masked_exponents_matrix):
    
    denominator = masked_exponents_matrix.sum(dim=1,keepdim=True)
    res = masked_exponents_matrix / denominator
    
    return res    

    """
    >>> t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9] ])
    >>> t
    tensor([[1, 2, 3],
           [4, 5, 6],
           [7, 8, 9]])
    >>> su = t.sum(dim=1,keepdim=True)
    >>> su
    tensor([[ 6],
            [15],
            [24]])
    >>> su.shape
    torch.Size([3, 1])
    >>> res = t / su
    >>> res
    tensor([[0.1667, 0.3333, 0.5000],
            [0.2667, 0.3333, 0.4000],
            [0.2917, 0.3333, 0.3750]])

    """

  def get_log_probabilities_matrix(self, probabilities_matrix):
    
    #zero probabilities will return -inf but will not be used 
    #as only the log probability of the ground truth is used in NLL Loss
    res = probabilities_matrix.log()
    
    return res    

    """
    >>> t = torch.tensor([[0, 2, 3], [4, 5, 6], [7, 8, 9] ])
    >>> t
    tensor([[0, 2, 3],
            [4, 5, 6],
            [7, 8, 9]])
    >>> res = t.log()
    >>> res
    tensor([[  -inf, 0.6931, 1.0986],
            [1.3863, 1.6094, 1.7918],
            [1.9459, 2.0794, 2.1972]])

    """

  def get_unrolled_node_id_labels(self, labels):
    
    #print(f'self.label_path_dict: \n\n{self.label_path_dict}')
    #paths_lst = [self.label_path_dict[l.item()] for l in labels ]
    #print(f'\n\n\npaths_lst: \n\n{paths_lst}')
    #numpy_paths_lst = [np.array(self.label_path_dict[l.item()]) for l in labels ]
    #print(f'\n\n\nnumpy_paths_lst: \n\n{numpy_paths_lst}')
    
    unrolled_labels = np.concatenate([np.array(self.label_path_dict[l.item()]) for l in labels ])
    #print(f'\n\n\nunrolled_labels: \n\n{unrolled_labels}')
    unrolled_labels = list(unrolled_labels)
    #print(f'\n\n\nunrolled_labels: \n\n{unrolled_labels}')

    #unrolled_labels = torch.cat([torch.tensor(self.label_path_dict[l.item()]) for l in labels ])
    total_labels_length = 0
    for l in labels:
      total_labels_length += len(self.label_path_dict[l.item()])
    #print(f'len(unrolled_labels): {len(unrolled_labels)}, total labels_path length: {total_labels_length}')
    #exit(0)
    return unrolled_labels    

    """
    >>> t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9] ])
    >>> t
    tensor([[1, 2, 3],
           [4, 5, 6],
           [7, 8, 9]])
    >>> labels = torch.cat([t[i] for i in range(len(t)) ])
    >>> labels
    tensor([1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> labels.shape
    torch.Size([9])
    """

  def make_prediction(self, x, tot_prob, curr_node_id, final_preds):
   
    descendants = list(self.hier.get_direct_descendants(curr_node_id))

    #check if leaf node 
    if not descendants:
        label = self.hier.LEAF_ID_TO_NUM[curr_node_id]
        #name = self.hier.HIER_NODE_NAME[curr_node_id]
        final_preds[:, label] = tot_prob        
        return final_preds

    encoded_prompts = self.decendants_prompts_dict[curr_node_id]
    #print(f'x.type(): {x.type()} encoded_prompts.type(): {encoded_prompts.type()}')
    similarity = 100.0 * x @ encoded_prompts.t() #batch_size X num descendants
    curr_prob = self.softmax(similarity)
    curr_prob = similarity.softmax(dim=-1)
    #print(f'curr_prob: {curr_prob}')
    
    for idx, next_node_id in enumerate(descendants):
        next_node_prob = curr_prob[:, idx]
        #print(f'idx: {idx}, next_node_prob: {next_node_prob}')
        next_tot_prob = next_node_prob * tot_prob
        #print(f'next_tot_prob: {next_tot_prob}')
        final_preds = self.make_prediction(x, next_tot_prob, next_node_id, final_preds)

    return final_preds

"""
  def forward_full_tree_path(self, x):
    images = x.to(device=device)

    image_features = self.clip_model.encode_image(images)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    image_features = image_features.double()
    #print(f'image_features.type(): {image_features.type()}')

    batch_size = image_features.shape[0]
    #initial_label_prob = torch.ones(batch_size,1).to(device=device)
    initial_label_prob = torch.ones(batch_size, dtype=torch.double).to(device=device)
    preds = torch.zeros((batch_size,self.num_classes), dtype=torch.double)
    final_preds = self.make_prediction(image_features, initial_label_prob, self.root_id, preds)
    #no need to do softmax, performed inside CrossEntropyLoss criterion
    #final_preds = self.softmax(final_preds)
    return final_preds.to(device=device) 
"""

def get_concat_h_blank(im1, im2, color=(0, 0, 0)):
    #dst = Image.new('RGB', (im1.width + im2.width, max(im1.height, im2.height)), color)
    dst = Image.new('RGB', (im1.width + im2.width, max(im1.height, im2.height)))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def get_concat_v_blank(im1, im2, color=(0, 0, 0)):
    dst = Image.new('RGB', (max(im1.width, im2.width), im1.height + im2.height), color)
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst


def DT_Contrastive_CLIP_train(DT_CLIP_model, num_epochs, data_loader, tree_criterion, img_criterion, txt_criterion, optimizer, save_frequency=5, checkpoint_name='checkpoint_DT_CLIP_epoch#'):
  DT_CLIP_model.train()
  loaded_epoch = 0
  #loss = None
  

  #PATH = '/mnt5/nir/CLIP/interpret/DT_CLIP_CC3M_checkpoint_epoch_0020.pt.tar'
  #PATH = 'DT_CLIP_COCO_checkpoint_epoch_0001.pt.tar'
  #PATH = 'DT_LoRA_CLIP_COCO_checkpoint_epoch_0007.pt.tar'
  #PATH = 'DT_Contrastive_CLIP_LoRA_COCO_checkpoint_epoch_0005.pt.tar'
  #PATH = 'DT_0.8_Contrastive_0.2_CLIP_LoRA_COCO_checkpoint_epoch_0005.pt.tar'
  #PATH = 'DT_0.5_flan_t5_LoRA_CLIP_contrast_0.5_checkpoint_epoch_0014.pt.tar'
  PATH = 'DT_0.5_hila_tree4_CLIP_contrast_0.5_checkpoint_epoch0003.pt.tar'

  
  checkpoint = torch.load(PATH)
  DT_CLIP_model.clip_model.load_state_dict(checkpoint['state_dict'])
  print(f'loaded checkpoint: {PATH}')
  optimizer.load_state_dict(checkpoint['optimizer'])
  loaded_epoch = checkpoint['completed_epoch']
  ##loss = checkpoint['loss']
  print(f'DT_Contrastive_CLIP_train: loaded epoch: {loaded_epoch}')

  for epoch in range(loaded_epoch,num_epochs):
    print(f'start DT_Contrastive_CLIP_train epoch#: {epoch+1}')

    losses = []
    total_examples = 0
    
    for batch_idx, (full_path, images, captions) in enumerate(tqdm(data_loader)):

      #with autograd.detect_anomaly():
      
      rand_sample_index = random.randint(0, len(images)-1)
      images = images.to(device=device)
      rand_image = images[rand_sample_index].unsqueeze(0)
      #captions = captions.to(device=device)
      #print(f'captions: {captions}')
      #print(f'captions[0]: {captions[0]}')
      #print(f'captions[0][0]: {captions[0][0]}')
      root, true_label_path, all_tree_nodes, edges ,node_texts = get_caption_tree8(captions[rand_sample_index])
      
      log_probabilities = DT_CLIP_model(rand_image,true_label_path, all_tree_nodes)
      node_num_labels = torch.tensor(true_label_path,device=device)

      tree_loss = tree_criterion(log_probabilities, node_num_labels)     

      #contrastive loss
      prompts = [f'a photo of {cap}' for cap in captions]
      
      tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(device=device)
      #tokenized_prompts.shape: torch.Size([128, 77])

      logits_per_image, logits_per_text = DT_CLIP_model.clip_model(images, tokenized_prompts)
      ground_truth = torch.arange(len(images), dtype=torch.long).to(device=device)

      loss_img = img_criterion(logits_per_image, ground_truth)
      loss_txt = txt_criterion(logits_per_text, ground_truth)
      contrastive_loss = (loss_img + loss_txt)/2

      #tree_loss_weight = 0.2
      #tree_loss_weight = 0.8
      tree_loss_weight = 0.5
      total_loss = tree_loss*tree_loss_weight + (1-tree_loss_weight)*contrastive_loss

      losses.append(total_loss.item())
      num_examples = len(images)
      
      total_examples += num_examples
      
      #print(f'batch loss: {total_loss.item()/num_examples}, curr_loss: {total_loss.item()}, num_examples: {num_examples}')

      optimizer.zero_grad()
      total_loss.backward()
      optimizer.step()
      
    
    completed_epoch = epoch + 1
    print(f'epoch loss: {sum(losses) / total_examples}')


    if (completed_epoch == num_epochs or 
                (
                    save_frequency > 0 and completed_epoch % save_frequency == 0
                )):
      
        # Save only clip_model checkpoints
        checkpoint_dict = {
              "completed_epoch": completed_epoch,
              "state_dict": DT_CLIP_model.clip_model.state_dict(),
              "optimizer": optimizer.state_dict(),
        }  

        filename = checkpoint_name + f"{completed_epoch:04d}.pt.tar"
        print(f'saving model state to filename: {filename}')
        torch.save(checkpoint_dict, filename)




def check_accuracy(loader, DT_CLIP_model):
    num_correct = 0
    num_samples = 0
    DT_CLIP_model.eval()

    image_cnt = 0
    # image_num = 82 # desired num to start from
    image_num = 3202 # desired num to start from
    with torch.no_grad():
        #for batch in tqdm(loader):
        
        for full_path, images, captions in tqdm(loader):
            
            # print(f'image_cnt: {image_cnt}, caption: {captions[0]}')
            # image_cnt += 1
            # continue
            

            if image_cnt < image_num:
              print(f'image_cnt: {image_cnt}, image_num: {image_num}')
              image_cnt += 1
              continue

            images = images.to(device=device)
            #print(f'\nfull_path: {full_path}\n')
            #print(f'images: \n{images}')
            #im = Image.open(r"C:\Users\System-Pc\Desktop\home.png")
            #im = Image.open(images[0])
            #im.show()
            #plt.imshow(images[0].cpu().permute(1,2,0))
            #img = np.transpose(images[0].cpu().detach().numpy(), (1, 2, 0))
            #img *= (1.0/img.max())
            #img=(img+1)/2
            #plt.imshow(img)
            #plt.show()
            #pil_image=Image.fromarray((img * 255).astype(np.uint8))
            #pil_image.show()
            #pil_image.save("/mnt5/nir/CLIP/interpret/tmp.jpg")
            #exit(0)
            print(f'image_cnt: {image_cnt}')
            with torch.set_grad_enabled(True):
              predictions, labels = DT_CLIP_model.classify_example(full_path[0], images,captions[0], image_cnt, print_all=False, print_incorrect=True)

            image_cnt += 1                        
            num_correct += (predictions == labels).sum()
            num_samples += labels.size(0)
            #num_samples += 1
            #print(f'num_correct: {num_correct}, num_samples: {num_samples}, accuracy: {100*num_correct/num_samples:.2f}')

    DT_CLIP_model.train()
    return num_correct/num_samples


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])


def load_and_save_clip_model_state_dict(DT_CLIP_model):
  PATH = 'DT_LoRA_CLIP_COCO_checkpoint_epoch_0005.pt.tar'
    
  checkpoint = torch.load(PATH)
  DT_CLIP_model.load_state_dict(checkpoint['state_dict'])
  print(f'loaded DT_CLIP_model checkpoint: {PATH}')

 
  # Save only clip_model checkpoints
  checkpoint_dict = {
    "completed_epoch": checkpoint["completed_epoch"],
    "state_dict": DT_CLIP_model.clip_model.state_dict(),
    "optimizer": checkpoint["optimizer"],
  }  

  filename = 'LoRA_CLIP_COCO_checkpoint_epoch_0005.pt.tar'
  print(f'saving DT_CLIP_model.clip_model state to filename: {filename}')
  torch.save(checkpoint_dict, filename)

def seed_everything(seed: int = 42):
  
  print(f'seed_everythin with seed: {seed}')        
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = True

def main():

    args = parser.parse_args()
    root_dir = os.path.dirname(os.path.realpath(__file__))    
    cur_dir = os.path.realpath(os.curdir)

    seed_everything()
    # seed_everything(1)
    #seed_everything(2)
    #seed_everything(3)
    #seed_everything(4)
    #seed_everything(5)
    #seed_everything(6)
    #seed_everything(7) #a boy standing in the grass (until/among/over/after/out/from the grass). all have same gradCAM
    #seed_everything(8) #a black cat is sitting in a bathroom
    #seed_everything(10)
    #seed_everything(11)
    #seed_everything(12)
    #seed_everything(13)
    #seed_everything(14)
    #seed_everything(15)# does and doing
    # seed_everything(17)



    batch_size = args.batch_size
    #batch_size = 1
    num_epochs = args.epochs
    print(f'num_epochs: {args.epochs}')
    print(f'batch_size: {batch_size}')

    # Load the model
    #clip_model, preprocess = clip.load('ViT-B/32', device=device)

    #info_dir = './BREEDS-Benchmarks/imagenet_class_hierarchy/modified'
    #hier = ClassHierarchy(info_dir)

    #num_classes = len(imagenet_test_dataset.classes)
    #num_imagenet_classes = len(hier.LEAF_IDS)
    #DT_CLIP_model = DecisionTreeClipModel(clip_model)
    DT_CLIP_model = DecisionTreeClipModel(lora=1)
    DT_CLIP_model = DT_CLIP_model.to(device)

    #load_and_save_clip_model_state_dict(DT_CLIP_model)
    #exit(0)
    
    #num_classes = len(DT_CLIP_model.good_labels)
    #print(f'num_classes: {num_classes}')

    #exit(0)

  
    #train_dataset = CocoCaptions(root="/mnt5/yoavkurtz/datasets/coco2017/train2017", annFile = "/mnt5/yoavkurtz/datasets/coco2017/annotations/captions_train2017.json", transform=DT_CLIP_model.preprocess)
    #test_dataset = CocoCaptions(root="/mnt5/yoavkurtz/datasets/coco2017/val2017", annFile = "/mnt5/yoavkurtz/datasets/coco2017/annotations/captions_val2017.json", transform=DT_CLIP_model.preprocess)
    train_dataset = CocoCaptionsDataset(root="/mnt5/yoavkurtz/datasets/coco2017/train2017", annFile = "/mnt5/yoavkurtz/datasets/coco2017/annotations/captions_train2017.json", transform=DT_CLIP_model.preprocess)
    test_dataset = CocoCaptionsDataset(root="/mnt5/yoavkurtz/datasets/coco2017/val2017", annFile = "/mnt5/yoavkurtz/datasets/coco2017/annotations/captions_val2017.json", transform=DT_CLIP_model.preprocess)


    #print('Number of train_dataset samples: ', len(train_dataset))
    #img, target = train_dataset[3] # load 4th sample

    #print("Image Size: ", img.size())
    #print(f'target caption: {target}')

    #print('Number of test_dataset samples: ', len(test_dataset))
    #img, target = test_dataset[3] # load 4th sample

    #print("Image Size: ", img.size())
    #print(f'target caption: {target}')
    
    #train_dataset = ImageCaptionDatasetCLIP(dataset='cc3m', root="/mnt5/nir/dataset/CC3M/cc3m", transform=preprocess, good_labels=DT_CLIP_model.good_labels, split='train')
    #test_dataset = SubsetDataset(transform=preprocess, good_labels=DT_CLIP_model.good_labels, split='val', root="/mnt5/nir/dataset/ImageNet")

    #train_dataset = SubsetDataset(transform=preprocess, good_labels=DT_CLIP_model.good_labels, split='train', root="/mnt5/nir/dataset/ImageNet")
    #test_dataset = SubsetDataset(transform=preprocess, good_labels=DT_CLIP_model.good_labels, split='val', root="/mnt5/nir/dataset/ImageNet")

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)

    #imagenet_train_dataset = ImageNet(root="/mnt5/nir/dataset/ImageNet", split='train', transform=preprocess)
    #print(f'imagenet_train_dataset[1]: {imagenet_train_dataset[1]}')
    #exit(0)
    #imagenet_test_dataset = ImageNet(root="/mnt5/nir/dataset/ImageNet", split='val', transform=preprocess)                     


    #train_loader = DataLoader(dataset=imagenet_train_dataset, batch_size=batch_size, shuffle=True)
    #test_loader = DataLoader(dataset=imagenet_test_dataset, batch_size=batch_size, shuffle=True)
    #test_loader = DataLoader(dataset=imagenet_test_dataset, batch_size=1, shuffle=True)

    

    #calc path to leaf node 0 ("tench, Tinca tinca") a kind of fish
    #node_id = hier.LEAF_NUM_TO_ID[0]
    #path = hier.traverse(nodes=[node_id], direction='up', depth=100)
    #path.reverse()
    #for idx, n in enumerate(path):
    #  print(f'idx: {idx}, ID: {n}, node_num: {hier.HIER_NODE_ID_TO_NUM[n]}, name: {hier.HIER_NODE_NAME[n]}')


    #for batch_idx, (images, labels) in enumerate(tqdm(test_loader)):
    #  DT_CLIP_model.interpret_prediction(images)
    #  exit(0)

    #num_epochs = 3
    save_frequency = 1

    # Loss and optimizer
    #decision tree loss - will sample one image from each batch
    tree_criterion = nn.NLLLoss()

    # Contrastive Loss 
    img_criterion = nn.CrossEntropyLoss()
    txt_criterion = nn.CrossEntropyLoss()

    learning_rate = 3e-6
    #TODO: try lower learning_rate 
    #learning_rate = 1e-6
    weight_decay = 0.1

    params = [p for p in DT_CLIP_model.parameters() if p.requires_grad]
    #num_params = sum([np.prod(p.size()) for p in params])
    #num_params: 87849216
    #print(f'num_params: {num_params}')
    #exit(0)
    optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)
    #optimizer = optim.Adam(params, lr=learning_rate)

    # optionally resume from a checkpoint
    start_epoch = 0
    checkpoint_path = args.checkpoint_file 
    if not checkpoint_path:
        checkpoint_path = 'DT_CLIP_CC3M_checkpoint_epoch_'

    #training loop
    #print(f'starting training loop on train dataset')
    #DT_CLIP_CC3M_train(DT_CLIP_model, num_epochs, train_loader, criterion, optimizer, save_frequency, checkpoint_path)
    
    #print(f'starting training loop on test set')
    #DT_Contrastive_CLIP_train(DT_CLIP_model, num_epochs, train_loader, tree_criterion, img_criterion, txt_criterion, optimizer, save_frequency, checkpoint_path)

    #print(f'starting training set check_accuracy')
    #train_accuracy = check_accuracy(train_loader, DT_CLIP_model)
    #print(f"Accuracy on training set: {train_accuracy*100:.2f}%")

    #PATH = 'DT_CLIP_COCO_checkpoint_epoch_0004.pt.tar'
    #PATH = 'DT_LoRA_CLIP_COCO_checkpoint_epoch_0005.pt.tar'
    #PATH = 'DT_LoRA_CLIP_state_dict_COCO_checkpoint_epoch_0005.pt.tar'
    #PATH = 'DT_0.5_hila_tree4_CLIP_contrast_0.5_checkpoint_epoch0003.pt.tar'
    
    #checkpoint trained with get_caption_tree6(flan T5) LoRA 0.5 * DT loos  + 0.5 * contrastive loss (tree_loss_weight = 0.5)
    PATH = '/mnt5/nir/CLIP/interpret/DT_0.5_flan_t5_LoRA_CLIP_contrast_0.5_checkpoint_epoch_0012.pt.tar'

    
    checkpoint = torch.load(PATH, map_location=torch.device(device))
    DT_CLIP_model.clip_model.load_state_dict(checkpoint['state_dict'])
    print(f'loaded checkpoint: {PATH}')
    print(f'starting test set check_accuracy')
    test_accuracy = check_accuracy(test_loader, DT_CLIP_model)
    
    print(f"Accuracy on test set: {test_accuracy*100:.2f}%")
    


if __name__ == "__main__":
    main()
    # print('\nprediction path:')
    # print('predicted caption (probability) - correct/incorrect\n')
    # print('a white dog (99.68%) - correct')
    # print('a white dog standing on top (52.54%) - correct')
    # print('a white dog standing on top of a metal bench (53.57%) - incorrect')
    # print('\ncorrect caption is: \'a white dog standing on top of a wooden bench\' - with probability: (39.65%)')
    # print('\nexpand node prediction:')
    # print('\'a photo of a hydrogen bench\', prob: (1.97%)')
    # print('\'a photo of a metal bench\', prob: (34.45%)')
    # print('\'a photo of a wooden bench\', prob: (14.48%)')
    # print('\'a photo of a varnish bench\', prob: (2.98%)')
    # print('\'a photo of a cement bench\', prob: (6.15%)')
    # print('\'a photo of a blue bench\', prob: (3.15%)')
    # print('\'a photo of a white bench\', prob: (23.04%)')
    # print('\'a photo of a black bench\', prob: (4.90%)\n\n')

   
