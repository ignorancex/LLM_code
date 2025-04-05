# Imports
#pip install transformers
#pip install ftfy regex tqdm
#pip install git+https://github.com/openai/CLIP.git

import argparse
import os
import math
import numpy as np
from collections import Counter
import random
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
#import clip  
from torchvision.datasets import CIFAR100
from torchvision.datasets import ImageNet
from torchvision.datasets import CocoCaptions
from tqdm import tqdm
from robustness.robustness.tools.breeds_helpers import ClassHierarchy
from robustness.robustness.tools.breeds_helpers import setup_breeds

#os.environ['TRANSFORMERS_CACHE'] = '/mnt5/nir/transformers/cache/'
#Disabling transformers parallelism to avoid deadlock with Dataloader parallelism
#os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import json
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import string
#from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, T5Tokenizer, T5ForConditionalGeneration
import random
import spacy
from nltk.corpus import wordnet as wn
import igraph
from igraph import Graph, EdgeSeq, plot
import plotly.graph_objects as go

#from create_coarse_sentences import get_caption_tree, get_caption_tree2, get_caption_tree3, get_caption_tree4, get_caption_tree6, Node, expand_caption, plotly_plot_tree
#from create_coarse_sentences import *
from create_coarse_sentences import Node, get_t5_opposite, find_word_difference, expand_caption, get_box_text, plotly_plot_tree
from create_coarse_sentences import get_caption_tree6, get_caption_tree6_lemmas, get_caption_tree6_shuffled_nouns_adjectives, get_caption_tree6_shuffle_all, get_caption_tree6_shuffle_random_all_branches, get_caption_tree6_with_noun_adj_shuffle
from create_coarse_sentences import get_caption_tree4, get_caption_tree6_1, get_caption_tree6_2, get_caption_tree8, get_caption_tree6_shuffle_all_branches, get_caption_tree6_shuffle_nouns_all_branches, get_caption_tree6_with_random_shuffle, get_caption_tree6_with_all_shuffles
from create_coarse_sentences import get_caption_tree6_gen_si, get_caption_tree6_depth1, get_caption_tree6_depth2, get_caption_tree6_depth3

#train CLIP with LoRA
from lora.lib.CLIP.clip import *

#CLIP with LoRA and hilaCAM - adds gradient hooks (use only for inference. no need to train with it as it requires more memory)
#from hilaCAM_lora.lib.CLIP.clip import *

from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import cv2
from gradcam.CLIP_explainability import get_image_text_relevance
#import datasets
import sys
sys.path.insert(0, '/mnt5/nir/CLIP/')
from vision_language_models_are_bows.model_zoo import get_model


prepositions_list = ['about', 'above', 'across', 'after', 'against', 'along', 'among', 'around',
                      'at', 'before',	'behind',	'below', 'between', 'beyond', 'but', 'by', 'concerning',
                      'despite', 'down', 'during', 'except', 'following', 'for', 'from', 'in',
                      'including', 'into', 'like', 'near', 'of', 'off', 'on', 'onto', 'out',
                      'over',	'past', 'plus', 'since', 'throughout', 'to', 'towards', 'under',
                      'until', 'up', 'upon', 'up to', 'with', 'within',	'without'
                    ]

nlp = spacy.load("en_core_web_sm")


learning_rate = 0.001
batch_size = 64
num_epochs = 3
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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


def seed_everything(seed: int = 42):
  
  print(f'seed_everythin with seed: {seed}')        
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = True

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
    self.clip_model, self.preprocess = self.load_model(base_name=vit_name, lora_r=lora)
    #self.freeze_visual_head()
    self.clip_model = self.clip_model.float()
    #self.clip_model = clip_model.float()
    #self.clip_model = clip_model.double()
    self.sample_num = 0
    self.lora = lora
    self.correct_positive_words = []
    self.correct_negative_words = []
    self.correct_pos = []
    self.incorrect_positive_words = []
    self.incorrect_negative_words = []
    self.incorrect_pos = []
    self.correct_captions = []
    self.incorrect_captions = []


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
  

  def get_encoded_prompts(self, node_names):
    #make sure prompts are in the same order of HIER_NODE_NUM(which is the node label)
    #num_nodes = len(all_nodes)
    #node_names = [node.prompt for node in all_nodes ]

    prompts = [f'a photo of {name}' for name in node_names]

    tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(device=device)
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
    root, true_label_path, all_tree_nodes, edges, node_texts = get_caption_tree6_shuffle_all(caption)
    
    return self.get_caption_tree_figure(all_tree_nodes, edges, caption, node_texts, file_name)

  def get_caption_tree_figure(self, all_tree_nodes, edges, caption, node_texts, file_name="tree_fig.png"):
    #TODO: remove
    #root, true_label_path, all_tree_nodes, edges, node_texts = get_caption_tree2(caption)
    #root, true_label_path, all_tree_nodes, edges, node_texts = get_caption_tree(caption)

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
    #img_fig = self.get_image_figure(image)
    img_fig = Image.open(image)
    #TODO: remove
    #root, true_label_path, all_tree_nodes, edges, node_texts = get_caption_tree(caption)
    tree_fig = self.get_caption_tree_figure(all_tree_nodes, edges, caption, node_texts, tree_file_name)

    if not os.path.exists(image_dir):
      os.mkdir(image_dir)

    merged_img = get_concat_h_blank(img_fig, tree_fig)

    merged_file = f"sample_{self.sample_num}.png"
    merged_path = image_dir + "/" + merged_file

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

  def collect_word_cloud_stats(self, true_label_path, all_tree_nodes, predictions):
    ############################
    #collect words to create word clouds


    for i, predicted_node_num in enumerate(predictions):
        is_correct = (predicted_node_num == true_label_path[i])
        predicted_node = all_tree_nodes[predicted_node_num]
        predicted_caption = predicted_node.prompt
        if not is_correct:           
          correct_node_num = true_label_path[i]
          correct_caption = all_tree_nodes[correct_node_num].prompt

          incorrect_str = f'\ncorrect_caption: {correct_caption}\npredicted_caption: {predicted_caption}\n'
          self.incorrect_captions.append(incorrect_str)

          correct_word, incorrect_word, pos = find_word_difference(correct_caption, predicted_caption)
          if pos != -1:
            self.incorrect_positive_words.append(correct_word)
            self.incorrect_negative_words.append(incorrect_word)
            self.incorrect_pos.append(pos)

        else:
          correct_caption = predicted_caption
          
          #get incorrect captions
          parent_node = predicted_node.parent
          siblings = parent_node.children #includes correct node
          if len(siblings) > 1:
            #print(f'predicted_node_num: {predicted_node_num}, predicted_node_num.item(): {predicted_node_num.item()} \ncorrect_caption: {correct_caption}, \nsiblings:\n')
            #for k, v in siblings.items():
            #  print(f'({k}: {v.prompt}), k != predicted_node_num.item(): {k != predicted_node_num.item()}')
            siblings = [value for key, value in siblings.items() if key != predicted_node_num.item()]
            incorrect_captions = [sib.prompt for sib in siblings]
            #print(f'siblings: {siblings}')
            #print(f'correct_caption: {correct_caption}')
            #print(f'incorrect_captions: {incorrect_captions}')
            
            if incorrect_captions:
              correct_str = f'\ncorrect_caption: {correct_caption}\nincorrect_captions: {incorrect_captions}\n'
              self.correct_captions.append(correct_str) 
              for incorrect in incorrect_captions:
                #correct word is different for each negative caption 
                if correct_caption != incorrect:
                  correct_word, incorrect_word, pos = find_word_difference(correct_caption, incorrect)
                  if pos != -1:
                    self.correct_negative_words.append(incorrect_word)
                    self.correct_positive_words.append(correct_word)
                    self.correct_pos.append(pos)


    #end of collect words to create word clouds
    ############################

  def generate_word_clouds(self):
    
    #text = ' '.join(text).lower()
    incorrect_positive_text = ' '.join(self.incorrect_positive_words)
    incorrect_negative_text = ' '.join(self.incorrect_negative_words)
    incorrect_pos_text = ' '.join(self.incorrect_pos)
    
    correct_positive_text = ' '.join(self.correct_positive_words)
    correct_negative_text = ' '.join(self.correct_negative_words)
    correct_pos_text = ' '.join(self.correct_pos)
    
    #save texts
    with open("Statistics_NegCLIP_COCO_correct_classification_text.json", "w") as fp:
        json.dump(self.correct_captions, fp)

    with open("Statistics_NegCLIP_COCO_incorrect_classification_text.json", "w") as fp:
        json.dump(self.incorrect_captions, fp)

    with open("wordcloud_NegCLIP_COCO_incorrect_positive_text.json", "w") as fp:
        json.dump(incorrect_positive_text, fp)

    with open("wordcloud_NegCLIP_COCO_incorrect_negative_text.json", "w") as fp:
        json.dump(incorrect_negative_text, fp)

    with open("wordcloud_NegCLIP_COCO_incorrect_pos_text.json", "w") as fp:
        json.dump(incorrect_pos_text, fp)

    with open("wordcloud_NegCLIP_COCO_correct_positive_text.json", "w") as fp:
        json.dump(correct_positive_text, fp)

    with open("wordcloud_NegCLIP_COCO_correct_negative_text.json", "w") as fp:
        json.dump(correct_negative_text, fp)

    with open("wordcloud_NegCLIP_COCO_correct_pos_text.json", "w") as fp:
        json.dump(correct_pos_text, fp)

    #create the wordcloud object
    #incorrect_positive_wordcloud = WordCloud(stopwords = STOPWORDS, background_color="white", collocations=True).generate(text)

    incorrect_positive_wordcloud = WordCloud(background_color="white", 
      collocations=True).generate(incorrect_positive_text)
    
    incorrect_negative_wordcloud = WordCloud(background_color="white", 
      collocations=True).generate(incorrect_negative_text)

    incorrect_pos_wordcloud = WordCloud(background_color="white", 
      collocations=True).generate(incorrect_pos_text)

    correct_positive_wordcloud = WordCloud(background_color="white", 
      collocations=True).generate(correct_positive_text)
    
    correct_negative_wordcloud = WordCloud(background_color="white", 
      collocations=True).generate(correct_negative_text)

    correct_pos_wordcloud = WordCloud(background_color="white", 
      collocations=True).generate(correct_pos_text)

    

    fig = plt.figure(figsize=(6, 6), dpi=200)

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(incorrect_positive_wordcloud)
    ax1.axis('off')
    ax1.set_title('incorrect classification positive text', fontsize=8)
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(incorrect_negative_wordcloud)
    ax2.axis('off')
    ax2.set_title('incorrect classification negative text', fontsize=8)
    plt.tight_layout()
    plt.savefig('wordcloud_CLIP_COCO_incorrect_classification.png')
    plt.close()

    fig = plt.figure(figsize=(6, 6), dpi=200)
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(correct_positive_wordcloud)
    ax1.axis('off')
    ax1.set_title('correct classification positive text', fontsize=8)
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(correct_negative_wordcloud)
    ax2.axis('off')
    ax2.set_title('correct classification negative text', fontsize=8)
    plt.tight_layout()
    plt.savefig('wordcloud_CLIP_COCO_correct_classification.png')
    plt.close()    
    
    fig = plt.figure(figsize=(6, 6), dpi=200)
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(correct_negative_wordcloud)
    ax1.axis('off')
    ax1.set_title('correct classification negative text', fontsize=8)
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(incorrect_negative_wordcloud)
    ax2.axis('off')
    ax2.set_title('incorrect classification negative text', fontsize=8)
    plt.tight_layout()
    plt.savefig('wordcloud_CLIP_COCO_negative_text_image.png')
    plt.close()  

    fig = plt.figure(figsize=(6, 6), dpi=200)
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(correct_positive_wordcloud)
    ax1.axis('off')
    ax1.set_title('correct classification positive text', fontsize=8)
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(incorrect_positive_wordcloud)
    ax2.axis('off')
    ax2.set_title('incorrect classification positive text', fontsize=8)
    plt.tight_layout()
    plt.savefig('wordcloud_CLIP_COCO_positive_text_image.png')
    plt.close() 

    fig = plt.figure(figsize=(6, 6), dpi=200)
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(correct_pos_wordcloud)
    ax1.axis('off')
    ax1.set_title('correct classification part of speech', fontsize=8)
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(incorrect_pos_wordcloud)
    ax2.axis('off')
    ax2.set_title('incorrect classification part of speech', fontsize=8)
    plt.tight_layout()
    plt.savefig('wordcloud_CLIP_COCO_part_of_speech_image.png')
    plt.close()    

    #create a figure for each category with 2 image together correct and incorrect

  def classify_example(self, img_path, x, caption, print_all=False, print_incorrect=True):
    
    root, true_label_path, all_tree_nodes, edges, node_texts = get_caption_tree6_lemmas(caption)
    #root, true_label_path, all_tree_nodes, edges, node_texts = get_caption_tree6_shuffle_all(caption)
    #root, true_label_path, all_tree_nodes, edges, node_texts = get_caption_tree6_shuffle_nouns_all_branches(caption)
    #root, true_label_path, all_tree_nodes, edges, node_texts = get_caption_tree6_shuffled_nouns_adjectives(caption)
    #root, true_label_path, all_tree_nodes, edges, node_texts = get_caption_tree3(caption)
    #root, true_label_path, all_tree_nodes, edges, node_texts = get_caption_tree2(caption)

    self.get_caption_tree_and_image_figure(img_path, all_tree_nodes, edges, caption, node_texts)
    exit(0)
    
    probs_matrix, _ = self.get_probs_matrix(x, true_label_path, all_tree_nodes)
    prob_values, predictions = probs_matrix.topk(1)
    #print(f'\nprob_values: {prob_values}\npredictions: {predictions}\n')
    #print(f'\ntrue_label_path: {true_label_path}\n')
    
    self.collect_word_cloud_stats(true_label_path, all_tree_nodes, predictions)

    if print_all or print_incorrect:
      #print('\nprediction path:')
      #print('predicted caption (probability) - correct/incorrect\n')
      str_to_print = '\nprediction path:\n'
      str_to_print += 'predicted caption (probability) - correct/incorrect\n\n'
      for i, predicted_node_num in enumerate(predictions):
        is_correct = (predicted_node_num == true_label_path[i])
        correct_str = "correct" if is_correct else "incorrect"
        predicted_caption = all_tree_nodes[predicted_node_num].prompt
        
        #print(f'{predicted_caption} ({100 * prob_values[i].item():.2f}%) - {correct_str}')
        str_to_print += f'{predicted_caption} ({100 * prob_values[i].item():.2f}%) - {correct_str}\n'

        if print_all:
           print(str_to_print)

        if not is_correct:
          if print_incorrect and (not print_all):
            print(str_to_print)
            
          correct_node_num = true_label_path[i]
          correct_caption = all_tree_nodes[correct_node_num].prompt
          correct_prob = probs_matrix[i][correct_node_num].item()

          #print(f'path_index: {i}, all_tree_nodes[correct_node_num].node_num: {all_tree_nodes[correct_node_num].node_num}, correct_node_num: {correct_node_num}')
          #print(f'all_tree_nodes[correct_node_num].prompt is: {all_tree_nodes[correct_node_num].prompt}')

          print(f'correct caption is: {correct_caption} - with probability: ({100 * correct_prob:.2f}%)')
          expanded_captions = expand_caption(predicted_caption, correct_caption)
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
  
          print(f'\nexpand node prediction: ')
          for idx, capt in enumerate(expanded_captions):
            print(f'caption: \'{capt}\', prob: ({100 * probs[idx].item():.2f}%)')

          #self.train() 
          with torch.set_grad_enabled(True):
            #HilaCAM
            #texts = [predicted_caption, correct_caption]
            #get_image_text_relevance(img_path, texts, self.clip_model, self.preprocess)
            x = x.to(device=device)
            x.requires_grad = True
            loss_fn = nn.NLLLoss(reduction='none')
            ###################
          
            log_probabilities = self.forward(x,true_label_path, all_tree_nodes)
            node_num_labels = torch.tensor(true_label_path,device=device)

            loss = loss_fn(log_probabilities, node_num_labels)
            #print(f'i: {i}, log_probabilities.shape: {log_probabilities.shape}, node_num_labels.shape: {node_num_labels.shape}, loss.shape: {loss.shape}')
            #incorrect_loss = loss[i, correct_node_num]
            incorrect_loss = loss[i]
            incorrect_loss.backward()
            #input_grad = x.grad.data.cpu().numpy()
            input_grad = x.grad.data
            #input_grad.shape: torch.Size([1, 3, 224, 224])
            #print(f'input_grad.shape: {input_grad.shape}')
            grad_arr = torch.abs(input_grad).mean(dim=1).permute(1, 2, 0)
            grad_arr /= grad_arr.quantile(0.98)
            grad_arr = torch.clamp(grad_arr, 0, 1)
            grad_arr = grad_arr.cpu().numpy()
            #input_grad_scaled = 
            x.requires_grad = False
            x_arr = to_array(x)
            #x_arr = x.squeeze()
            #x_arr = x_arr.cpu().permute(1, 2, 0).detach().numpy()
            grad_img_arr = x_arr * grad_arr
            #grad_img_arr = x * grad_arr
            #grad_img_arr = x * input_grad
            grad_pil_img = self.get_image_figure(grad_img_arr, permute=False)
            #grad_pil_img = self.get_image_figure(grad_img_arr)
          

            dir_path = "gradients_samples"

            if not os.path.exists(dir_path):
              os.mkdir(dir_path)

            #sample_num = 1
            file_name = f"gradient_{self.sample_num}.png"
            save_path = dir_path + "/" + file_name

            grad_pil_img.save(save_path)
            #self.eval()
            ###################
            self.get_grad_cam(x, img_path)
            #self.get_caption_tree_and_image_figure(x, all_tree_nodes, edges, caption, node_texts)
            self.get_caption_tree_and_image_figure(img_path, all_tree_nodes, edges, caption, node_texts)
            
            self.sample_num += 1
            exit(0)
            break

    return predictions, torch.tensor(true_label_path, device=device)

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


  def get_probs_matrix(self, images, true_label_path, all_tree_nodes):
    
    node_names = [node.prompt for node in all_tree_nodes ]
    self.encoded_prompts = self.get_encoded_prompts(node_names)
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


def get_tree_loss(model, tree_criterion, images, captions, rand_sample_index, tree_func):
  
  rand_image = images[rand_sample_index].unsqueeze(0)

  with torch.no_grad(): 
        
        root, true_label_path, all_tree_nodes, edges ,node_texts = tree_func(captions[rand_sample_index])

        #rare, but can happen that node_texts are empty ( node_texts = [''] ) (no noun chunks)
        max_tries = 3
        num_tries = 0
        used_indexes = []
        while not any(node_texts):
          print(f'node_texts is: {node_texts}. num_tries: {num_tries}, max_tries: {max_tries}')
          if num_tries < max_tries:
            used_indexes.append(rand_sample_index)
            print(f'try using another image-text pair')
            num_tries += 1
            idx_list = [*range(len(images))]
            rand_sample_index = random.choice([idx for idx in idx_list if idx not in used_indexes])
            root, true_label_path, all_tree_nodes, edges ,node_texts = tree_func(captions[rand_sample_index])

          else:
            return None

  log_probabilities = model(rand_image,true_label_path, all_tree_nodes)
     
  #nn.NLLLoss and thus also nn.CrossEntropyLoss don’t support float16 tensors on the CPU
  log_probabilities = log_probabilities.to(device)
  log_probabilities = log_probabilities.float()
  node_num_labels = torch.tensor(true_label_path,device=device)

  #node_num_labels = torch.tensor(true_label_path, dtype=torch.long, device=device)

  tree_loss = tree_criterion(log_probabilities, node_num_labels)
  return tree_loss, rand_sample_index


def DT_Contrastive_CLIP_train(DT_CLIP_model, num_epochs, data_loader, tree_criterion, img_criterion, txt_criterion, optimizer, save_freq=5, update_weights_freq=1, checkpoint_name='checkpoint_DT_CLIP_epoch#'):
  DT_CLIP_model.train()
  loaded_epoch = 0
  #loss = None
  

  #PATH = '/mnt5/nir/CLIP/interpret/DT_CLIP_CC3M_checkpoint_epoch_0020.pt.tar'
  #PATH = 'DT_CLIP_COCO_checkpoint_epoch_0001.pt.tar'
  #PATH = 'DT_LoRA_CLIP_COCO_checkpoint_epoch_0007.pt.tar'
  #PATH = 'DT_Contrastive_CLIP_LoRA_COCO_checkpoint_epoch_0005.pt.tar'
  #PATH = 'DT_0.8_Contrastive_0.2_CLIP_LoRA_COCO_checkpoint_epoch_0005.pt.tar'
  #PATH = 'DT_0.5_flan_t5_LoRA_CLIP_contrast_0.5_checkpoint_epoch_0014.pt.tar'

  #checkpoint trained with LoRA, hilaCAM and caption tree 8 (flan t5 opposites if exist and t5 mask fill otherwise)
  #PATH = '/mnt5/nir/CLIP/interpret/DT_0.5_t5_caption8_LoRA_contrast_checkpoint_epoch_0002.pt.tar'

  #checkpoint trained with LR 1e-6, LoRA, hilaCAM and caption tree 8 (flan t5 opposites if exist and t5 mask fill otherwise)
  #PATH = '/mnt5/nir/CLIP/interpret/DT_0.5_t5_caption8_LoRA_contrast_LR_1e6_checkpoint_epoch_0002.pt.tar'

  #checkpoint trained with LR 3e-6, LoRA, hilaCAM and caption tree 6.1 (flan t5 opposites if exist, otherwise rand spatial if 'ADP' else do nothing )
  #PATH = '/mnt5/nir/CLIP/interpret/DT_0.5_t5_caption6_1_LoRA_contrast_LR_3e6_checkpoint_epoch_0001.pt.tar'
  
  #checkpoint trained with LR 3e-6, LoRA, hilaCAM and caption tree 6.2 (flan t5 opposites if exist, otherwise rand spatial if 'ADP' else co_hyponym )
  #PATH = '/mnt5/nir/CLIP/interpret/DT_0.5_t5_caption6_2_LoRA_contrast_LR_3e6_checkpoint_epoch_0001.pt.tar'
  
  # PATH = '/mnt5/nir/CLIP/interpret/DT_0.5_t5_caption6_LoRA_4_contrast_LR_3e6_checkpoint_epoch_0015.pt.tar'

  #checkpoint trained with get_caption_tree6(flan T5) LoRA 0.5 * DT loos  + 0.5 * contrastive loss (tree_loss_weight = 0.5)
  # PATH = '/mnt5/nir/CLIP/interpret/DT_0.5_flan_t5_LoRA_CLIP_contrast_0.5_checkpoint_epoch_0012.pt.tar'

  #checkpoint trained with get_caption_tree6(flan T5) with random shuffle LoRA 0.5 * DT loos  + 0.5 * contrastive loss (tree_loss_weight = 0.5)
  # PATH = '/mnt5/nir/CLIP/interpret/COCO_3VL_DT_0.5_tree6_with_random_shuffle_LoRA_1_contrast_LR_3e6_checkpoint_epoch_0012.pt.tar'

  # PATH = '/mnt5/nir/CLIP/interpret/COCO_3VL_DT_0.5_tree6_with_nouns_adjs_shuffle_LoRA_1_contrast_LR_3e6_checkpoint_epoch_0008.pt.tar'

  # PATH = '/mnt5/nir/CLIP/interpret/COCO_DT_0.5_caption6_RB_color_LoRA_1_contrast_LR_3e6_checkpoint_epoch_0016.pt.tar'

  # PATH = '/mnt5/nir/CLIP/interpret/COCO_DT_0.5_caption6_RB_size_LoRA_1_contrast_LR_3e6_checkpoint_epoch_0016.pt.tar'

  
  # checkpoint = torch.load(PATH)
  # DT_CLIP_model.clip_model.load_state_dict(checkpoint['state_dict'])
  # optimizer.load_state_dict(checkpoint['optimizer'])
  # loaded_epoch = checkpoint['completed_epoch']
  # #loss = checkpoint['loss']
  # print(f'DT_Contrastive_CLIP_train: loaded checkpoint from path: {PATH}')
  # print(f'DT_Contrastive_CLIP_train: loaded epoch: {loaded_epoch}')

  print(f'DT_Contrastive_CLIP_train: start from scratch (epoch: {loaded_epoch})')

  # tree_funcs = [get_caption_tree6_depth1, get_caption_tree6_depth2, get_caption_tree6_depth3]
  tree_funcs = [get_caption_tree6, get_caption_tree6_depth1]
  tree_loss_weight = 0.5

  print(f'train with lora_rank: {DT_CLIP_model.lora}, tree_loss_weight: {tree_loss_weight}, tree_funcs: {tree_funcs}')
  # print(f'train with lora_rank: {DT_CLIP_model.lora}, tree_loss_weight: {tree_loss_weight}, get_caption_tree6_shuffle_all_branches: flan t5 opposites if exist otherwise co-hyponym.')
  # print(f'train with lora_rank: {DT_CLIP_model.lora}, tree_loss_weight: {tree_loss_weight}, get_caption_tree6: RB color -> flan t5 opposites if exist otherwise co-hyponym.')
#   print(f'train with lora_rank: {DT_CLIP_model.lora}, tree_loss_weight: {tree_loss_weight}, get_caption_tree6_shuffle_all_branches: rule based -> flan t5 opposites if exist otherwise co-hyponym. add all shuffuled caption after each branch')
  # print(f'train with lora_rank: {DT_CLIP_model.lora}, tree_loss_weight: {tree_loss_weight}, get_caption_tree6_with_random_shuffle: flan t5 opposites if exist otherwise co-hyponym. with random shuffle')
  # print(f'train with lora_rank: {DT_CLIP_model.lora}, tree_loss_weight: {tree_loss_weight}, get_caption_tree6_with_noun_adj_shuffle: flan t5 opposites if exist otherwise co-hyponym. with nouns and adjectives shuffle')
  # print(f'train with lora_rank: {DT_CLIP_model.lora}, tree_loss_weight: {tree_loss_weight}, get_caption_tree6_with_all_shuffles: flan t5 opposites if exist otherwise co-hyponym. with all shuffles')

   
  for epoch in range(loaded_epoch,num_epochs):
    print(f'start DT_Contrastive_CLIP_train epoch#: {epoch+1}')

    losses = []
    total_examples = 0
    
    for batch_idx, (full_path, images, captions) in enumerate(tqdm(data_loader)):

      #with autograd.detect_anomaly():
      
      rand_sample_index = random.randint(0, len(images)-1)

      images = images.to(device=device)
      # rand_image = images[rand_sample_index].unsqueeze(0)
      
      curr_tree_loss = None
      tree_loss = 0
      
      for func in tree_funcs:
        curr_tree_loss, rand_sample_index = get_tree_loss(DT_CLIP_model, tree_criterion, images, captions, rand_sample_index, func)
        
        if curr_tree_loss is None:
          print(f'{func} node_texts is not valid. break')
          break
        
        tree_loss = tree_loss + curr_tree_loss

        
      if curr_tree_loss is None:
        print(f'node_texts is not valid. continue to next batch')
        continue

      
      tree_loss = tree_loss/len(tree_funcs)

      #contrastive loss
      prompts = [f'a photo of {cap}' for cap in captions]
      
      tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(device=device)
      #tokenized_prompts.shape: torch.Size([128, 77])

      logits_per_image, logits_per_text = DT_CLIP_model.clip_model(images, tokenized_prompts)
      ground_truth = torch.arange(len(images), dtype=torch.long).to(device=device)

      #nn.NLLLoss and thus also nn.CrossEntropyLoss don’t support float16 tensors on the CPU
      logits_per_image = logits_per_image.to(device)
      logits_per_image = logits_per_image.float()
      logits_per_text = logits_per_text.to(device)
      logits_per_text = logits_per_text.float()

      loss_img = img_criterion(logits_per_image, ground_truth)
      loss_txt = txt_criterion(logits_per_text, ground_truth)
      contrastive_loss = (loss_img + loss_txt)/2

      #tree_loss_weight = 0.2
      #tree_loss_weight = 0.8
      #tree_loss_weight = 0.5
      
      total_loss = tree_loss*tree_loss_weight + (1-tree_loss_weight)*contrastive_loss

      losses.append(total_loss.item())
      num_examples = len(images)
      
      total_examples += num_examples
      
      #print(f'batch loss: {total_loss.item()/num_examples}, curr_loss: {total_loss.item()}, num_examples: {num_examples}')

      # normalize loss to account for batch accumulation
      #total_loss = total_loss / update_weights_freq


      # backward pass
      total_loss = total_loss.to(device)
      #total_loss = total_loss.float()
      total_loss.backward() 
      
      # weights update
      if ((batch_idx + 1) % update_weights_freq == 0) or (batch_idx + 1 == len(data_loader)):
          optimizer.step()
          optimizer.zero_grad()
      
    
    completed_epoch = epoch + 1
    print(f'epoch loss: {sum(losses) / total_examples}')


    if (completed_epoch == num_epochs or 
                (
                    save_freq > 0 and completed_epoch % save_freq == 0
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

    with torch.no_grad():
        #for batch in tqdm(loader):
        
        for full_path, images, captions in tqdm(loader):
           
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


            #predictions, labels = DT_CLIP_model.classify_example(full_path[0], images,captions[0], print_all=False, print_incorrect=True)
            predictions, labels = DT_CLIP_model.classify_example(full_path[0], images,captions[0], print_all=False, print_incorrect=False)
            #DT_CLIP_model.generate_word_clouds()
            #exit(0)
                        
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


def gen_word_cloud_from_file(file_name=None, image_file_name=None):


  #################################################################
  #generate part_of_speech WordCloud Image
  #################################################################
  
  with open("wordcloud_NegCLIP_COCO_correct_pos_text.json", "r") as fp:
        correct_pos_text = json.load(fp)
  
  with open("wordcloud_NegCLIP_COCO_incorrect_pos_text.json", "r") as fp:
        incorrect_pos_text = json.load(fp)


  correct_pos_text_lst = correct_pos_text.split()
  incorrect_pos_text_lst = incorrect_pos_text.split()

  correct_pos_count = Counter(correct_pos_text_lst)
  incorrect_pos_count = Counter(incorrect_pos_text_lst)

  correct_pos_count={k: v for k, v in sorted(correct_pos_count.items(),reverse=True, key=lambda item: item[1])}
  incorrect_pos_count={k: v for k, v in sorted(incorrect_pos_count.items(),reverse=True, key=lambda item: item[1])}

  print(f'correct_pos_count: \n:{correct_pos_count}')
  print(f'incorrect_pos_count: \n:{incorrect_pos_count}')

  correct_pos_count = dict(list(correct_pos_count.items())[:4])
  incorrect_pos_count = dict(list(incorrect_pos_count.items())[:4])

  print(f'correct_pos_count: \n:{correct_pos_count}')
  print(f'incorrect_pos_count: \n:{incorrect_pos_count}')

  with open("wordcloud_NegCLIP_COCO_dict_correct_part_of_speech.json", "w") as fp:
        json.dump(correct_pos_count, fp)

  with open("wordcloud_NegCLIP_COCO_dict_incorrect_part_of_speech.json", "w") as fp:
        json.dump(incorrect_pos_count, fp)

  # incorrect_pos_wordcloud = WordCloud(background_color="white", 
  #     collocations=False).generate(incorrect_pos_text)
    
  # correct_pos_wordcloud = WordCloud(background_color="white", 
  #   collocations=False).generate(correct_pos_text)

  # correct_pos_text_dictionary = correct_pos_wordcloud.process_text(correct_pos_text)

  # incorrect_pos_text_dictionary = incorrect_pos_wordcloud.process_text(incorrect_pos_text)

  # correct_word_freq={k: v for k, v in sorted(correct_pos_text_dictionary.items(),reverse=True, key=lambda item: item[1])}
  # incorrect_word_freq={k: v for k, v in sorted(incorrect_pos_text_dictionary.items(),reverse=True, key=lambda item: item[1])}

  # correct_wordcloud = WordCloud(collocations=False,
  #                     background_color='white')

  # incorrect_wordcloud = WordCloud(collocations=False,
  #                    background_color='white')

  # correct_word_freq = dict(list(correct_word_freq.items())[:4])
  # incorrect_word_freq = dict(list(incorrect_word_freq.items())[:4])
  
  # print(f'correct_word_freq: \n:{correct_word_freq}')
  # print(f'incorrect_word_freq: \n:{incorrect_word_freq}')
 
  # correct_wordcloud.generate_from_frequencies(correct_word_freq)
  # incorrect_wordcloud.generate_from_frequencies(correct_word_freq)

  #print('correct_word_freq[0:5]: ')
  #print(list(correct_word_freq.items())[:5])
  #print('correct_word_freq: ')
  #print(list(correct_word_freq.items()))
  

  #print('incorrect_word_freq[0:5]: ')
  #print(list(incorrect_word_freq.items())[:5])
  #print('incorrect_word_freq: ')
  #print(list(incorrect_word_freq.items()))


    
  # fig = plt.figure(figsize=(6, 6), dpi=200)
  # ax1 = fig.add_subplot(1, 2, 1)
  # ax1.imshow(correct_wordcloud)
  # ax1.axis('off')
  # ax1.set_title('correct classification part of speech', fontsize=8)
  # ax2 = fig.add_subplot(1, 2, 2)
  # ax2.imshow(incorrect_wordcloud)
  # ax2.axis('off')
  # ax2.set_title('incorrect classification part of speech', fontsize=8)
  # plt.tight_layout()
  # plt.savefig('wordcloud_NegCLIP_COCO_part_of_speech_image.png')
  # plt.close()    
  

  #################################################################
  #generate part_of_speech WordCloud Image CLIP Vs 3VL
  #################################################################
  """
  with open("wordcloud_CLIP_COCO_incorrect_pos_text.json", "r") as fp:
        CLIP_incorrect_pos_text = json.load(fp)
  
  with open("wordcloud_3VL_COCO_incorrect_pos_text.json", "r") as fp:
        tVL_incorrect_pos_text = json.load(fp)

  
  
  

  with open("wordcloud_CLIP_COCO_dict_incorrect_part_of_speech.json", "r") as fp:
        CLIP_incorrect_pos_dict = json.load(fp)
  
  with open("wordcloud_3VL_COCO_dict_incorrect_part_of_speech.json", "r") as fp:
        tVL_incorrect_pos_dict = json.load(fp)


  total_CLIP = sum(CLIP_incorrect_pos_dict.values())
  total_tVL = sum(tVL_incorrect_pos_dict.values())

  print(f'CLIP_incorrect_pos_dict: {CLIP_incorrect_pos_dict}, total_CLIP: {total_CLIP}')
  print(f'tVL_incorrect_pos_dict: {tVL_incorrect_pos_dict}, total_tVL: {total_tVL}')

  performance_increace_dict = dict()
  CLIP_relative = dict()
  tVL_relative = dict()
  for key in CLIP_incorrect_pos_dict:
    print(f'key: {key}, CLIP: {CLIP_incorrect_pos_dict[key]}, 3VL: {tVL_incorrect_pos_dict[key]}')
    performance_increace_dict[key] = (CLIP_incorrect_pos_dict[key] - tVL_incorrect_pos_dict[key])*100 / CLIP_incorrect_pos_dict[key]
    CLIP_relative[key] = CLIP_incorrect_pos_dict[key] / total_CLIP
    tVL_relative[key] = tVL_incorrect_pos_dict[key] / total_tVL

  #total_CLIP = sum(CLIP_incorrect_pos_dict.values())
  #total_tVL = sum(tVL_incorrect_pos_dict.values())

  total_performance_increace = sum(performance_increace_dict.values())
  performance_increace_dict={k: v for k, v in sorted(performance_increace_dict.items(),reverse=True, key=lambda item: item[1])}

  
  
  # print(f'CLIP_incorrect_pos_dict: {CLIP_incorrect_pos_dict}, total_CLIP: {total_CLIP}')
  # print(f'tVL_incorrect_pos_dict: {tVL_incorrect_pos_dict}, total_tVL: {total_tVL}')
  # print(f'CLIP_relative_incorrect_pos_dict: {CLIP_relative}')
  # print(f'tVL_relative_incorrect_pos_dict: {tVL_relative}')
  # print(f'performance_increace_dict: {performance_increace_dict}, total_performance_increace: {total_performance_increace}')

  performance_increace_wordcloud = WordCloud(collocations=False,
                      background_color='white')

   
  performance_increace_wordcloud.generate_from_frequencies(performance_increace_dict)
  

  fig = plt.figure(figsize=(6, 6), dpi=200)
  ax1 = fig.add_subplot(1, 1, 1)
  ax1.imshow(performance_increace_wordcloud)
  ax1.axis('off')
  #ax1.set_title('3VL part of speech performance increase', fontsize=8)
  # ax2 = fig.add_subplot(1, 2, 2)
  # ax2.imshow(tVL_incorrect_wordcloud)
  # ax2.axis('off')
  # ax2.set_title('3VL incorrect classification part of speech', fontsize=8)
  plt.tight_layout()
  plt.savefig('wordcloud_3VL_part_of_speech_perf_increase_image.png')
  plt.close()    

  
  CLIP_incorrect_pos_wordcloud = WordCloud(background_color="white", 
      collocations=False).generate(CLIP_incorrect_pos_text)
    
  tVL_incorrect_pos_wordcloud = WordCloud(background_color="white", 
    collocations=False).generate(tVL_incorrect_pos_text)

  CLIP_incorrect_pos_text_dictionary = CLIP_incorrect_pos_wordcloud.process_text(CLIP_incorrect_pos_text)

  tVL_incorrect_pos_text_dictionary = tVL_incorrect_pos_wordcloud.process_text(tVL_incorrect_pos_text)

  CLIP_incorrect_word_freq={k: v for k, v in sorted(CLIP_incorrect_pos_text_dictionary.items(),reverse=True, key=lambda item: item[1])}
  tVL_incorrect_word_freq={k: v for k, v in sorted(tVL_incorrect_pos_text_dictionary.items(),reverse=True, key=lambda item: item[1])}

  CLIP_incorrect_wordcloud = WordCloud(collocations=False,
                      background_color='white')

  tVL_incorrect_wordcloud = WordCloud(collocations=False,
                      background_color='white')

  CLIP_word_freq = dict(list(CLIP_incorrect_word_freq.items())[:4])
  tVL_word_freq = dict(list(tVL_incorrect_word_freq.items())[:4])

  with open("wordcloud_CLIP_COCO_dict_incorrect_part_of_speech.json", "w") as fp:
        json.dump(CLIP_word_freq, fp)

  with open("wordcloud_3VL_COCO_dict_incorrect_part_of_speech.json", "w") as fp:
        json.dump(tVL_word_freq, fp)
  
  #print(f'incorrect POS CLIP_word_freq: \n:{CLIP_word_freq}')
  #print(f'incorrect POS 3VL_word_freq: \n:{tVL_word_freq}')
 
  CLIP_incorrect_wordcloud.generate_from_frequencies(CLIP_word_freq)
  tVL_incorrect_wordcloud.generate_from_frequencies(tVL_word_freq)
  """

  #print('correct_word_freq[0:5]: ')
  #print(list(correct_word_freq.items())[:5])
  #print('correct_word_freq: ')
  #print(list(correct_word_freq.items()))
  

  #print('incorrect_word_freq[0:5]: ')
  #print(list(incorrect_word_freq.items())[:5])
  #print('incorrect_word_freq: ')
  #print(list(incorrect_word_freq.items()))


  """  
  fig = plt.figure(figsize=(6, 6), dpi=200)
  ax1 = fig.add_subplot(1, 2, 1)
  ax1.imshow(CLIP_incorrect_wordcloud)
  ax1.axis('off')
  ax1.set_title('CLIP incorrect classification part of speech', fontsize=8)
  ax2 = fig.add_subplot(1, 2, 2)
  ax2.imshow(tVL_incorrect_wordcloud)
  ax2.axis('off')
  ax2.set_title('3VL incorrect classification part of speech', fontsize=8)
  plt.tight_layout()
  plt.savefig('wordcloud_3VL_VS_CLIP_COCO_incorrect_part_of_speech_image.png')
  plt.close()    
  """
  
  #################################################################
  #concat positive and negative texts for incorrect  and correct classification
  #################################################################
  """
  with open("wordcloud_3VL_COCO_incorrect_positive_text.json", "r") as fp:
        incorrect_positive_text = json.load(fp)

  with open("wordcloud_3VL_COCO_incorrect_negative_text.json", "r") as fp:
      incorrect_negative_text = json.load(fp)


  incorrect_positive_text_lst = incorrect_positive_text.split()
  incorrect_negative_text_lst = incorrect_negative_text.split()

  incorrect_positive_count = Counter(incorrect_positive_text_lst)
  incorrect_negative_count = Counter(incorrect_negative_text_lst)

  incorrect_positive_count={k: v for k, v in sorted(incorrect_positive_count.items(),reverse=True, key=lambda item: item[1])}
  incorrect_negative_count={k: v for k, v in sorted(incorrect_negative_count.items(),reverse=True, key=lambda item: item[1])}

  with open("wordcloud_3VL_COCO_dict_incorrect_positive.json", "w") as fp:
        json.dump(incorrect_positive_count, fp)

  with open("wordcloud_3VL_COCO_dict_incorrect_negative.json", "w") as fp:
        json.dump(incorrect_negative_count, fp)
  """
  #incorrect_positive_negative_text_lst = [f'{positive_word}_{negative_word}' for positive_word, negative_word in zip(incorrect_positive_text_lst, incorrect_negative_text_lst)]

  #incorrect_positive_negative_text = ' '.join(incorrect_positive_negative_text_lst) 

  #with open("wordcloud_CLIP_COCO_incorrect_positive_negative_text.json", "w") as fp:
  #      json.dump(incorrect_positive_negative_text, fp)  
  
  #do not use correct classification text. correct text can have several negative texts so we can't zip together positive and negative
  #correct text can have several negative texts so we can't zip together positive and negative
  
  """
  with open("wordcloud_CLIP_COCO_correct_positive_text.json", "r") as fp:
        correct_positive_text = json.load(fp)

  with open("wordcloud_CLIP_COCO_correct_negative_text.json", "r") as fp:
      correct_negative_text = json.load(fp)


  correct_positive_text_lst = correct_positive_text.split()
  correct_negative_text_lst = correct_negative_text.split()

  correct_positive_count = Counter(correct_positive_text_lst)
  correct_negative_count = Counter(correct_negative_text_lst)

  correct_positive_count={k: v for k, v in sorted(correct_positive_count.items(),reverse=True, key=lambda item: item[1])}
  correct_negative_count={k: v for k, v in sorted(correct_negative_count.items(),reverse=True, key=lambda item: item[1])}

  with open("wordcloud_CLIP_COCO_dict_correct_positive.json", "w") as fp:
        json.dump(correct_positive_count, fp)

  with open("wordcloud_CLIP_COCO_dict_correct_negative.json", "w") as fp:
        json.dump(correct_negative_count, fp)
  
  #correct_positive_negative_text_lst = [f'{positive_word}_{negative_word}' for positive_word, negative_word in zip(correct_positive_text_lst, correct_negative_text_lst)]

  #correct_positive_negative_text = ' '.join(correct_positive_negative_text_lst) 

  #do not use correct classification text. correct text can have several negative texts so we can't zip together positive and negative
  #correct text can have several negative texts so we can't zip together positive and negative

  #with open("wordcloud_3VL_COCO_correct_positive_negative_text.json", "w") as fp:
  #      json.dump(correct_positive_negative_text, fp)  
  """
  #############################################################
  #generate incorrect positive_negative WordCloud Image
  ##############################################################

  """
  incorrect_positive_negative_wordcloud = WordCloud(background_color="white", 
      collocations=False).generate(incorrect_positive_negative_text)
    
  #correct_positive_negative_wordcloud = WordCloud(background_color="white", 
  #  collocations=True).generate(correct_positive_negative_text)

    
  fig = plt.figure(figsize=(6, 6), dpi=200)
  #ax1 = fig.add_subplot(1, 2, 1)
  #ax1 = fig.add_subplot(1, 1, 1)
  #ax1.imshow(correct_positive_negative_wordcloud)
  #ax1.axis('off')
  #ax1.set_title('correct classification positive_negative words', fontsize=8)
  #ax2 = fig.add_subplot(1, 2, 2)
  ax2 = fig.add_subplot(1, 1, 1)
  ax2.imshow(incorrect_positive_negative_wordcloud)
  ax2.axis('off')
  ax2.set_title('incorrect classification positive_negative words', fontsize=8)
  plt.tight_layout()
  plt.savefig('wordcloud_CLIP_COCO_incorrect_classification_pos_neg_image.png')
  plt.close()    
  """

  #############################################################
  #generate incorrect positive_negative WordCloud Image
  ##############################################################

  """
  with open("wordcloud_CLIP_COCO_incorrect_positive_negative_text.json", "r") as fp:
        CLIP_incorrect_positive_negative_text = json.load(fp)

  with open("wordcloud_3VL_COCO_incorrect_positive_negative_text.json", "r") as fp:
      tVL_incorrect_positive_negative_text = json.load(fp)


  CLIP_incorrect_positive_negative_lst = CLIP_incorrect_positive_negative_text.split()
  tVL_incorrect_positive_negative_lst = tVL_incorrect_positive_negative_text.split()

  CLIP_incorrect_positive_negative_count = Counter(CLIP_incorrect_positive_negative_lst)
  tVL_incorrect_positive_negative_count = Counter(tVL_incorrect_positive_negative_lst)

  CLIP_incorrect_positive_negative_count={k: v for k, v in sorted(CLIP_incorrect_positive_negative_count.items(),reverse=True, key=lambda item: item[1])}
  tVL_incorrect_positive_negative_count={k: v for k, v in sorted(tVL_incorrect_positive_negative_count.items(),reverse=True, key=lambda item: item[1])}

  with open("wordcloud_CLIP_COCO_dict_incorrect_positive_negative.json", "w") as fp:
        json.dump(CLIP_incorrect_positive_negative_count, fp)

  with open("wordcloud_3VL_COCO_dict_incorrect_positive_negative.json", "w") as fp:
        json.dump(tVL_incorrect_positive_negative_count, fp)


  
  CLIP_incorrect_positive_negative = WordCloud(background_color="white", 
      collocations=False).generate(CLIP_incorrect_positive_negative_text)
    
  tVL_incorrect_positive_negative = WordCloud(background_color="white", 
      collocations=False).generate(tVL_incorrect_positive_negative_text)
    
  fig = plt.figure(figsize=(6, 6), dpi=200)
  ax1 = fig.add_subplot(1, 2, 1)
  ax1.set_title('CLIP incorrect classification \npositive_negative words', fontsize=8)
  ax1.imshow(CLIP_incorrect_positive_negative)
  ax1.axis('off')
  ax2 = fig.add_subplot(1, 2, 2)
  ax2.set_title('3VL incorrect classification \npositive_negative words', fontsize=8)
  ax2.imshow(tVL_incorrect_positive_negative)
  ax2.axis('off')
  plt.tight_layout()
  plt.savefig('wordcloud_COCO_incorrect_classification_pos_neg_image.png')
  plt.close()    
  """

  #############################################################################
  #calc incorrect positive_negative divided by total number of 
  #postive word to get failure rate
  ##############################################################################
  with open("wordcloud_CLIP_COCO_dict_correct_positive.json", "rb") as fp:
        CLIP_correct_positive_count = json.load(fp)

  with open("wordcloud_CLIP_COCO_dict_incorrect_positive.json", "rb") as fp:
        CLIP_incorrect_positive_count = json.load(fp)

  with open("wordcloud_3VL_COCO_dict_correct_positive.json", "rb") as fp:
        tVL_correct_positive_count = json.load(fp)

  with open("wordcloud_3VL_COCO_dict_incorrect_positive.json", "rb") as fp:
        tVL_incorrect_positive_count = json.load(fp)

  with open("wordcloud_CLIP_COCO_dict_incorrect_positive_negative.json", "rb") as fp:
        CLIP_incorrect_positive_negative_count = json.load(fp)

  with open("wordcloud_3VL_COCO_dict_incorrect_positive_negative.json", "rb") as fp:
        tVL_incorrect_positive_negative_count = json.load(fp)
        

  tVL_positive_negative_fail_rate = dict()
  for key in tVL_incorrect_positive_negative_count:
    positive_word_key = key.split("_")[0]
    #CLIP_incorrect_positive_negative_count[key]
    #print(f'key: {key}, positive_word: {positive_word}')
    positive_word_count = tVL_correct_positive_count.get(positive_word_key,0) + tVL_incorrect_positive_count.get(positive_word_key,0)
    tVL_positive_negative_fail_rate[key] = 100*tVL_incorrect_positive_negative_count[key] / positive_word_count


  tVL_positive_negative_fail_rate={k: v for k, v in sorted(tVL_positive_negative_fail_rate.items(),reverse=True, key=lambda item: item[1])}

  print(f'tVL_positive_negative_fail_rate.items()[0:50]: {list(tVL_positive_negative_fail_rate.items())[0:50]}')

  with open("wordcloud_3VL_COCO_dict_positive_negative_fail_rate.json", "w") as fp:
        json.dump(tVL_positive_negative_fail_rate, fp)

  #############################################################
  #print incorrect positive_negative frequencies (CLIP VS 3VL)
  ##############################################################

  """
  with open("wordcloud_CLIP_COCO_incorrect_positive_negative_text.json", "r") as fp:
        CLIP_incorrect_positive_negative_text = json.load(fp)

  with open("wordcloud_3VL_COCO_incorrect_positive_negative_text.json", "r") as fp:
      tVL_incorrect_positive_negative_text = json.load(fp)


  CLIP_incorrect_positive_negative = WordCloud(background_color="white", 
      collocations=False).generate(CLIP_incorrect_positive_negative_text)
    
  tVL_incorrect_positive_negative = WordCloud(background_color="white", 
      collocations=False).generate(tVL_incorrect_positive_negative_text)

  

  CLIP_text_dictionary = CLIP_incorrect_positive_negative.process_text(CLIP_incorrect_positive_negative_text)

  tVL_text_dictionary = tVL_incorrect_positive_negative.process_text(tVL_incorrect_positive_negative_text)

  CLIP_word_freq={k: v for k, v in sorted(CLIP_text_dictionary.items(),reverse=True, key=lambda item: item[1])}
  tVL_word_freq={k: v for k, v in sorted(tVL_text_dictionary.items(),reverse=True, key=lambda item: item[1])}

  with open("wordcloud_CLIP_COCO_dict_incorrect_positive_negative.json", "w") as fp:
        json.dump(CLIP_word_freq, fp)

  with open("wordcloud_3VL_COCO_dict_incorrect_positive_negative.json", "w") as fp:
        json.dump(tVL_word_freq, fp)

  CLIP_incorrect_wordcloud = WordCloud(collocations=False,
                      background_color='white')

  tVL_incorrect_wordcloud = WordCloud(collocations=False,
                      background_color='white')

  """
  #CLIP_word_freq = dict(list(CLIP_incorrect_word_freq.items())[:4])
  #tVL_word_freq = dict(list(tVL_incorrect_word_freq.items())[:4])
  
  #print(f'incorrect POS CLIP_word_freq: \n:{CLIP_word_freq}')
  #print(f'incorrect POS 3VL_word_freq: \n:{tVL_word_freq}')
 
  #CLIP_incorrect_wordcloud.generate_from_frequencies(CLIP_word_freq)
  #tVL_incorrect_wordcloud.generate_from_frequencies(tVL_word_freq)
  

  #num_top_freq = 50
  #print(f'get top {num_top_freq} frequency incorrect positive_negative word pairs')
  #print('\nCLIP base:')
  #print(list(CLIP_word_freq.items())[:num_top_freq])
  #print('\n3VL:')
  #print(list(tVL_word_freq.items())[:num_top_freq])
  

  #print('incorrect_word_freq[0:5]: ')
  #print(list(incorrect_word_freq.items())[:5])
  #print('incorrect_word_freq: ')
  #print(list(incorrect_word_freq.items()))
  
  
  #############################################################
  #print incorrect positive frequencies (CLIP VS 3VL)
  ##############################################################

  """ 
  with open("wordcloud_CLIP_COCO_incorrect_positive_text.json", "r") as fp:
        CLIP_incorrect_positive_text = json.load(fp)

  with open("wordcloud_3VL_COCO_incorrect_positive_text.json", "r") as fp:
      tVL_incorrect_positive_text = json.load(fp)


  CLIP_incorrect_positive_WordCloud = WordCloud(background_color="white", 
      collocations=False).generate(CLIP_incorrect_positive_text)
    
  tVL_incorrect_positive_WordCloud = WordCloud(background_color="white", 
      collocations=False).generate(tVL_incorrect_positive_text)

  

  CLIP_text_dictionary = CLIP_incorrect_positive_WordCloud.process_text(CLIP_incorrect_positive_text)

  tVL_text_dictionary = tVL_incorrect_positive_WordCloud.process_text(tVL_incorrect_positive_text)

  CLIP_word_freq={k: v for k, v in sorted(CLIP_text_dictionary.items(),reverse=True, key=lambda item: item[1])}
  tVL_word_freq={k: v for k, v in sorted(tVL_text_dictionary.items(),reverse=True, key=lambda item: item[1])}


  with open("wordcloud_CLIP_COCO_dict_incorrect_positive.json", "w") as fp:
        json.dump(CLIP_word_freq, fp)

  with open("wordcloud_3VL_COCO_dict_incorrect_positive.json", "w") as fp:
        json.dump(tVL_word_freq, fp)

  CLIP_incorrect_wordcloud = WordCloud(collocations=False,
                      background_color='white')

  tVL_incorrect_wordcloud = WordCloud(collocations=False,
                      background_color='white')

  
  #CLIP_word_freq = dict(list(CLIP_incorrect_word_freq.items())[:4])
  #tVL_word_freq = dict(list(tVL_incorrect_word_freq.items())[:4])
  
  #print(f'incorrect POS CLIP_word_freq: \n:{CLIP_word_freq}')
  #print(f'incorrect POS 3VL_word_freq: \n:{tVL_word_freq}')
 
  #CLIP_incorrect_wordcloud.generate_from_frequencies(CLIP_word_freq)
  #tVL_incorrect_wordcloud.generate_from_frequencies(tVL_word_freq)
  

  num_top_freq = 50
  print(f'get top {num_top_freq} frequency for incorrect positive words')
  print('\nCLIP base:')
  print(list(CLIP_word_freq.items())[:num_top_freq])
  print('\n3VL:')
  print(list(tVL_word_freq.items())[:num_top_freq])
  
  """
  #print('incorrect_word_freq[0:5]: ')
  #print(list(incorrect_word_freq.items())[:5])
  #print('incorrect_word_freq: ')
  #print(list(incorrect_word_freq.items()))

  #############################################################
  ##############################################################
  #############################################################
  #print correct positive frequencies (CLIP VS 3VL)
  ##############################################################
  """
  with open("wordcloud_CLIP_COCO_correct_positive_text.json", "r") as fp:
        CLIP_correct_positive_text = json.load(fp)

  with open("wordcloud_3VL_COCO_correct_positive_text.json", "r") as fp:
      tVL_correct_positive_text = json.load(fp)


  CLIP_correct_positive_WordCloud = WordCloud(background_color="white", 
      collocations=False).generate(CLIP_correct_positive_text)
    
  tVL_correct_positive_WordCloud = WordCloud(background_color="white", 
      collocations=False).generate(tVL_correct_positive_text)

  

  CLIP_text_dictionary = CLIP_correct_positive_WordCloud.process_text(CLIP_correct_positive_text)

  tVL_text_dictionary = tVL_correct_positive_WordCloud.process_text(tVL_correct_positive_text)

  CLIP_word_freq={k: v for k, v in sorted(CLIP_text_dictionary.items(),reverse=True, key=lambda item: item[1])}
  tVL_word_freq={k: v for k, v in sorted(tVL_text_dictionary.items(),reverse=True, key=lambda item: item[1])}


  #with open("wordcloud_CLIP_COCO_dict_correct_positive.json", "w") as fp:
  #      json.dump(CLIP_word_freq, fp)

  #with open("wordcloud_3VL_COCO_dict_correct_positive.json", "w") as fp:
  #      json.dump(tVL_word_freq, fp)


  CLIP_incorrect_wordcloud = WordCloud(collocations=False,
                      background_color='white')

  tVL_incorrect_wordcloud = WordCloud(collocations=False,
                      background_color='white')

  
  #CLIP_word_freq = dict(list(CLIP_incorrect_word_freq.items())[:4])
  #tVL_word_freq = dict(list(tVL_incorrect_word_freq.items())[:4])
  
  #print(f'incorrect POS CLIP_word_freq: \n:{CLIP_word_freq}')
  #print(f'incorrect POS 3VL_word_freq: \n:{tVL_word_freq}')
 
  #CLIP_incorrect_wordcloud.generate_from_frequencies(CLIP_word_freq)
  #tVL_incorrect_wordcloud.generate_from_frequencies(tVL_word_freq)
  

  num_top_freq = 50
  print(f'get top {num_top_freq} frequency for correct positive words')
  print('\nCLIP base:')
  print(list(CLIP_word_freq.items())[:num_top_freq])
  print('\n3VL:')
  print(list(tVL_word_freq.items())[:num_top_freq])
  

  #print('incorrect_word_freq[0:5]: ')
  #print(list(incorrect_word_freq.items())[:5])
  #print('incorrect_word_freq: ')
  #print(list(incorrect_word_freq.items()))
  """
  #############################################################
  ##############################################################
  


def main():

    args = parser.parse_args()
    root_dir = os.path.dirname(os.path.realpath(__file__))    
    cur_dir = os.path.realpath(os.curdir)

    batch_size = args.batch_size
    #batch_size = 1
    num_epochs = args.epochs
    print(f'num_epochs: {args.epochs}')
    print(f'batch_size: {batch_size}')

    seed_everything()
    #seed_everything(1)
    #seed_everything(2)
    #seed_everything(3)

    #TODO: change to lora=4
    lora=1
    #lora=4
    
    #lora = -1#no LoRA
    DT_CLIP_model = DecisionTreeClipModel(lora=lora)
    

    #train_dataset = CocoCaptions(root="/mnt5/yoavkurtz/datasets/coco2017/train2017", annFile = "/mnt5/yoavkurtz/datasets/coco2017/annotations/captions_train2017.json", transform=DT_CLIP_model.preprocess)
    #test_dataset = CocoCaptions(root="/mnt5/yoavkurtz/datasets/coco2017/val2017", annFile = "/mnt5/yoavkurtz/datasets/coco2017/annotations/captions_val2017.json", transform=DT_CLIP_model.preprocess)
    train_dataset = CocoCaptionsDataset(root="/mnt5/yoavkurtz/datasets/coco2017/train2017", annFile = "/mnt5/yoavkurtz/datasets/coco2017/annotations/captions_train2017.json", transform=DT_CLIP_model.preprocess)
    # train_dataset = CocoCaptionsDataset(root="/mnt5/yoavkurtz/datasets/coco2017/train2017", annFile = "/mnt5/nir/CLIP/DATASET/coco_train_captions.jsonl", transform=DT_CLIP_model.preprocess)

    test_dataset = CocoCaptionsDataset(root="/mnt5/yoavkurtz/datasets/coco2017/val2017", annFile = "/mnt5/yoavkurtz/datasets/coco2017/annotations/captions_val2017.json", transform=DT_CLIP_model.preprocess)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)

    
    save_frequency = 1

    # Loss and optimizer
    #decision tree loss - will sample one image from each batch
    tree_criterion = nn.NLLLoss()

    # Contrastive Loss 
    img_criterion = nn.CrossEntropyLoss()
    txt_criterion = nn.CrossEntropyLoss()

    #update_weights_freq = 4
    update_weights_freq = 1

    learning_rate = 3e-6
    #TODO: try lower learning_rate 
    #learning_rate = 1e-6
    weight_decay = 0.1

    #increase learning_rate linearly by the factor of update_weights_freq
    #learning_rate = learning_rate*update_weights_freq

    #increase learning_rate linearly by a factor of square root of update_weights_freq
    squrt_update_freq = math.sqrt(update_weights_freq)
    print(f'learning_rate: {learning_rate}, update_weights_freq: {update_weights_freq}, squrt_update_freq: {squrt_update_freq}')
    learning_rate = learning_rate*squrt_update_freq
    print(f'learning_rate: {learning_rate}')


    params = [p for p in DT_CLIP_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)
    #optimizer = optim.Adam(params, lr=learning_rate)

    # optionally resume from a checkpoint
    start_epoch = 0
    checkpoint_path = args.checkpoint_file 
    if not checkpoint_path:
        checkpoint_path = 'DT_CLIP_COCO_checkpoint_epoch_'

    

    #training loop
    print(f'starting training loop on train dataset')
    DT_Contrastive_CLIP_train(DT_CLIP_model, num_epochs, train_loader, tree_criterion, img_criterion, txt_criterion, optimizer, save_frequency, update_weights_freq, checkpoint_path)
    
    #PATH = 'DT_CLIP_COCO_checkpoint_epoch_0004.pt.tar'
    #PATH = 'DT_LoRA_CLIP_COCO_checkpoint_epoch_0005.pt.tar'
    #PATH = 'DT_LoRA_CLIP_state_dict_COCO_checkpoint_epoch_0005.pt.tar'
    #PATH = 'DT_Contrastive_CLIP_LoRA_COCO_checkpoint_epoch_0005.pt.tar'
    #PATH = 'DT_0.8_Contrastive_0.2_CLIP_LoRA_COCO_checkpoint_epoch_0005.pt.tar'
    
    #checkpoint trained with get_caption_tree6(flan T5) LoRA 0.5 * DT loos  + 0.5 * contrastive loss (tree_loss_weight = 0.5)
    PATH = '/mnt5/nir/CLIP/interpret/DT_0.5_flan_t5_LoRA_CLIP_contrast_0.5_checkpoint_epoch_0012.pt.tar'

    
    checkpoint = torch.load(PATH, map_location=torch.device(device))
    #DT_CLIP_model.clip_model.load_state_dict(checkpoint['state_dict'])
    #print(f'loaded checkpoint: {PATH}')
    
    #CLIPWrapperModel, preprocess = get_model(model_name="NegCLIP", device=device, root_dir='/mnt5/nir/CLIP/vision_language_models_are_bows/~/.cache')
    #DT_CLIP_model.clip_model = CLIPWrapperModel.model

    #print('running with NegCLIP')


    #print(f'starting test set check_accuracy')
    #test_accuracy = check_accuracy(test_loader, DT_CLIP_model)
    
    #print(f"Accuracy on test set: {test_accuracy*100:.2f}%")
    #DT_CLIP_model.generate_word_clouds()



"""
def get_co_hyponym(word):
  co_hyponyms = get_co_hyponym_list(word)
  #print(f'\nword: {word}, co_hyponyms: \n\n{co_hyponyms}\n')

  if not co_hyponyms:
    #print(f'no co_hyponyms found for word: {word}\n')
    return ''

  
  #seed_everything()
  return random.choice(co_hyponyms)

def get_co_hyponym_list(word):
  #print(f'get_co_hyponym word: {word}','\n')
  co_hyponyms = []

  for syn in wn.synsets(word):
    #print(f'syn: {syn}','\n')
    hypernym_lst = syn.hypernyms()
    #print(f'hypernym_lst: {hypernym_lst}','\n')

    if hypernym_lst:
      hypernym = hypernym_lst[0]
      hyponyms = hypernym.hyponyms()
      #print(f'hyponyms: {hyponyms}','\n')

      for hypo in hyponyms:
        hypo_names = [l.name() for l in hypo.lemmas()]
        #print(f'hypo_names: {hypo_names}','\n')
        co_hyponyms.extend(hypo_names)
  
  
  co_hyponyms = set(co_hyponyms)
  co_hyponyms.discard(word)
  co_hyponyms = list(co_hyponyms)
  #sort the list to make it deterministic (with random.seed())
  co_hyponyms.sort() 
  #print(f'\nco_hyponyms: {co_hyponyms}\n')

  return co_hyponyms


def get_rand_adposition(word):
  adpositions = get_rand_adposition_list(word)
  
  #seed_everything()
  return random.choice(adpositions)



def get_rand_adposition_list(word):
  adpositions = prepositions_list.copy()
  adpositions = set(adpositions)
  adpositions.discard(word)
  adpositions = list(adpositions)
  
  if adpositions:
    adpositions.sort()
  
  return adpositions


def get_t5_model_and_tokenizer():
  #tokenizer = T5Tokenizer.from_pretrained("t5-3b", model_max_length=30)
  #model = T5ForConditionalGeneration.from_pretrained("t5-3b", device_map="auto")
  
  model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
  tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")

  return model, tokenizer

def get_t5_opposite(word):
  
  #if not model:
  #  model, tokenizer = get_t5_model_and_tokenizer()
  model, tokenizer = get_t5_model_and_tokenizer()
  prompt = f"find an opposite for the word: {word}"
  print(f'get_t5_opposite. prompt: {prompt}')
  inputs = tokenizer(prompt, return_tensors="pt")
  
  outputs = model.generate(**inputs, max_new_tokens=3, num_beams = 5, num_return_sequences=3)
  decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
  print(f'get_t5_opposite. decoded_outputs: {decoded_outputs}')
  return decoded_outputs

def get_t5_rand_word2(token_text, token_pos):
  if not token_pos in ['NOUN', 'VERB', 'ADJ', 'ADP']:
    print(f'get_t5_rand_word2 token_pos ({token_pos}) of token: {token_text} is not in [NOUN, VERB, ADJ, ADP]')
    return ''
  
  lower_txt = token_text.lower() 
  opposites = get_t5_opposite(lower_txt)
  print(f'get_t5_rand_word2 lower_txt: {lower_txt}, opposites: {opposites}')

  #if token_text has an opposite return it
  if not lower_txt in opposites[0].lower() and opposites[0].lower() != 'na':
    return opposites[0]

  if 'un' in opposites[0].lower() or 'non' in opposites[0].lower():
    return opposites[0]

  #if token_text is an adposition which have no opposite, return random adposition
  if token_pos == 'ADP':
    return get_rand_adposition(token_text)

  if len(opposites) > 1:
    for opp in opposites[1:]:
      if not lower_txt in opp.lower() and opp.lower() != 'na':
        return opp

  #TODO: return hare similar word from T5 ? or fill masked word ?
  return get_co_hyponym(token_text)

def get_t5_new_sentences2(orig_text, pos_to_replace=['NOUN', 'ADJ']):
  
  orig_text_tokens_lst = [token.text for token in orig_text]
  print(f'get_t5_new_sentences2 orig_text.text: {orig_text.text}')
  new_sentences = []
  token_index = 0
  #new_text_lst = []
  for token in orig_text:
    
    if token.pos_ in pos_to_replace:
      
      rand_word = get_t5_rand_word2(token.text, token.pos_)
      
      if rand_word != '':
        
        new_text_lst = orig_text_tokens_lst.copy()
        new_text_lst[token_index] = rand_word
        
        new_text = ' '.join(new_text_lst)
        
        new_sentences.append(new_text)
    
    token_index += 1

def create_path_node(curr_index, paernt_node, text, true_label_path, edges, node_texts, all_tree_nodes):
  
  true_label_path.append(curr_index)

  return create_node(curr_index, paernt_node, text, edges, node_texts, all_tree_nodes)


def create_node(curr_index, paernt_node, text, edges, node_texts, all_tree_nodes):
  
  edges.append([paernt_node.node_num, curr_index])
  next_node = Node(node_num=curr_index, prompt=text, parent=paernt_node)

  node_texts.append(get_box_text(text))
  all_tree_nodes.append(next_node)
    
  paernt_node.children[curr_index] = next_node

  return next_node

def get_text_to_modify(prev_noun_chunk, curr_noun_chunk, full_doc):
  
  #returns a list of tuples (doc_mod, prefix, suffix) 
  #doc_mod - the part of the doc to modify 
  #prefix and suffix strings to add before and after the prompt without modification
  

  res = []
  
  if not prev_noun_chunk:
    #this is the first chunk - return the doc from the first index without prefix or suffix
    res.append((full_doc[0:curr_noun_chunk.end], "", ""))
    return res
  
  #use the prefix from the first token to the end of prev_noun_chunk and concatenate 'and' to it
  prefix = full_doc[0:prev_noun_chunk.end].text + ' '
  #modify curr_noun_chunk, and do not add suffix
  res.append((curr_noun_chunk, prefix + 'and ', ''))

  #if prev_noun_chunk and curr_noun_chunk have verbs and adpositions connecting between them
  #modify verbs and adpositions, and add prev_noun_chunk and curr_noun_chunk as prefix and suffix
  if prev_noun_chunk.end != curr_noun_chunk.start:
    suffix = ' ' + curr_noun_chunk.text
    doc_mod = full_doc[prev_noun_chunk.end:curr_noun_chunk.start]
    res.append((doc_mod, prefix, suffix))   
  
  return res

def get_caption_tree6(original_caption):
  #print(f'\noriginal_sentence: {original_caption}\n')
  original_caption = original_caption.translate(str.maketrans('', '', string.punctuation))
  original_caption = original_caption.lower()
  doc = nlp(original_caption)
 
  node_index = 0
  root = Node(node_num=node_index, prompt='entity')
 
  node_index += 1
  true_label_path = []
  all_tree_nodes = []
  node_texts = []
  edges = []
  all_tree_nodes.append(root)
  
  prev_text = ""
  add_prev_text = False
  node_texts.append("")
  next_path_node = root
  prev_path_node = root
  #curr_path_node = None
  curr_doc = None
  prev_chunk_end_index = -1
  prev_noun_chunk = None
  print(f'get_caption_tree6: original_caption: {original_caption}')
  last_chunk_end_index = 0
  #print(f'original_caption : {original_caption}')
  for chunk in doc.noun_chunks:
    print(f'get_caption_tree6: chunk.text: {chunk.text}')
    text_to_modify_lst = get_text_to_modify(prev_noun_chunk, chunk, doc)
    prev_noun_chunk = chunk
    for (doc_mod, prefix, suffix) in text_to_modify_lst:
      print(f'get_caption_tree6: doc_mod.text: {doc_mod.text}')
      prev_path_node = next_path_node
      
      text = prefix + doc_mod.text + suffix
      #print(text)
      next_path_node = create_path_node(node_index, prev_path_node, text, true_label_path, edges, node_texts, all_tree_nodes)
      node_index += 1
   
      #new_noun_phrases = get_new_noun_phrases(doc_mod)
      new_noun_phrases = get_t5_new_sentences2(doc_mod, pos_to_replace=['NOUN', 'ADJ', 'VERB', 'ADP'])
      print(f'new_noun_phrases: {new_noun_phrases}')
      #get_t5_new_sentences(orig_doc, pos_to_replace=['NOUN', 'ADJ', 'VERB', 'ADP'])
      for mod_text in new_noun_phrases:
        new_text = prefix + mod_text + suffix
      
        tmp_node = create_node(node_index, prev_path_node, new_text, edges, node_texts, all_tree_nodes)
        node_index += 1
       
  #add original sentence only if not added before with last noun_chunk
  if prev_noun_chunk.end != len(doc):
    
    prev_path_node = next_path_node
    
    next_path_node = create_path_node(node_index, prev_path_node, original_caption, true_label_path, edges, node_texts, all_tree_nodes)
    node_index += 1
  
    #new_sentences = get_new_verb_phrases(doc)
    #new_sentences = get_new_noun_phrases(doc[prev_noun_chunk.end:])
    new_sentences = get_t5_new_sentences2(doc[prev_noun_chunk.end:], pos_to_replace=['NOUN', 'ADJ', 'VERB', 'ADP'])
    print(f'new_sentences: {new_sentences}')
    for new_sentence in new_sentences:
      text = doc[0:prev_noun_chunk.end].text + " " + new_sentence
      print(f'text: {text}')
      tmp_node = create_node(node_index, prev_path_node, text, edges, node_texts, all_tree_nodes)
      node_index += 1
  
 
  return root, true_label_path, all_tree_nodes, edges ,node_texts
  """

if __name__ == "__main__":
    caption = "A sandwich and sauce on a white plate."
    """
    get_t5_opposite. prompt: find an opposite for the word: sandwich
    get_t5_opposite. decoded_outputs: ['the the.', 'the thes', 'the theX']
    get_t5_rand_word2 lower_txt: sandwich, opposites: ['the the.', 'the thes', 'the theX']
    get_t5_new_sentences2 new_sentences: ['a the the.']
    """
    #root, true_label_path, all_tree_nodes, edges, node_texts = get_caption_tree6(caption)
    #root, true_label_path, all_tree_nodes, edges, node_texts = get_caption_tree4(caption)
    #print(node_texts)
    #token_text = 'sandwich'
    #token_text = 'sauce'
    #lower_txt = token_text.lower() 
    #opposites = get_t5_opposite(lower_txt)
    #get_t5_rand_word2 lower_txt: sauce, opposites: ['. thes', '. theX', '. thea']
    #(f'get_t5_rand_word2 lower_txt: {lower_txt}, opposites: {opposites}')
    """
    get_t5_opposite. prompt: find an opposite for the word: sandwich
    get_t5_opposite. decoded_outputs: ['burger', 'sand', 'tuna']
    get_t5_rand_word2 lower_txt: sandwich, opposites: ['burger', 'sand', 'tuna']
    """
    #gen_word_cloud_from_file()
    #print_top_word_cloud_frequencies()
    main()