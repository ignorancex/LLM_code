# Imports
#pip install transformers
#pip install ftfy regex tqdm
#pip install git+https://github.com/openai/CLIP.git

import argparse
import os
import numpy as np
import torch
from torch import autograd
import torchvision 
import torchvision.datasets as datasets  
import torchvision.transforms as transforms  
from torch import optim  
from torch import nn  
from torch.utils.data import DataLoader, Dataset  
from tqdm import tqdm
import clip  
from torchvision.datasets import CIFAR100
from torchvision.datasets import ImageNet
from tqdm import tqdm
from robustness.robustness.tools.breeds_helpers import ClassHierarchy
from robustness.robustness.tools.breeds_helpers import setup_breeds

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


class DecisionTreeClipModel(nn.Module):
  def __init__(self, clip_model, hier):
    super(DecisionTreeClipModel, self).__init__()
    
    self.clip_model = clip_model.float()
    #self.clip_model = clip_model.double()
    self.hier = hier
    self.root_id = list(hier.get_nodes_at_level(0))[0]
    #root_leaves_reachable = [hier.LEAF_ID_TO_NUM[leaf] for leaf in hier.leaves_reachable(self.root_id)]
    #name = hier.HIER_NODE_NAME[self.root_id]
    #print(f'root_id: {self.root_id}, name: {name}, #leaves_reachable: {len(root_leaves_reachable)}')
    #exit(0)

    #add encoded prompts hier to the nodes
    self.decendants_prompts_dict = dict()
    self.descendants_dict = dict()
    self.leaves_reachable_dict = dict()
    self.decendants_prompts_dict, self.descendants_dict, self.leaves_reachable_dict = self.GetDecendantsPromptsDict(self.root_id, self.decendants_prompts_dict, self.descendants_dict, self.leaves_reachable_dict)
    self.encoded_prompts = self.get_encoded_prompts()

    self.good_labels = []
    self.label_path_dict = dict()
    self.calc_label_path_dict()


    #freeze self.clip_model params
    for param in self.clip_model.parameters():
        param.requires_grad = False
        param.data = param.data.float()
        if param.grad:
          param.grad.data = param.grad.data.float() 

    #unfreeze image encoder params (keep text encoder frozen)
    for param in self.clip_model.visual.parameters():
        param.requires_grad = True

    self.softmax = nn.Softmax(dim=1)

    #self.clip_model.eval()
    #self.fc = nn.Linear(input_size, num_classes)



  def calc_label_path_dict(self):
    
    num_imagenet_classes = len(self.hier.LEAF_IDS)

    for class_num in range(num_imagenet_classes):
      path, class_num = self.GetPathToLeaf(class_num)
      if path:
        self.good_labels.append(class_num)
        #to get label path from node id path
        #label_path = [hier.HIER_NODE_ID_TO_NUM[id] for id in path]
        self.label_path_dict[class_num] = path
  

  def get_encoded_prompts(self):
    #make sure prompts are in the same order of HIER_NODE_NUM(which is the node label)
    num_nodes = len(list(self.hier.HIER_NODE_NUM_TO_NAME.keys()))
    node_names = [self.hier.HIER_NODE_NUM_TO_NAME[i].split(", ")[0] for i in range(num_nodes) ]

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

  #TODO: add hierarcy level param to the function can create encoded_prompts for all hierarcy level nodes
  def get_living_encoded_prompts(self, clip_model):
    node_names = ['living thing', 'non-living thing']

    prompts = [f'a photo of {name}' for name in node_names]

    tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(device)
    
    with torch.no_grad():
        encoded_prompts = clip_model.encode_text(tokenized_prompts).to(device=device)
        encoded_prompts /= encoded_prompts.norm(dim=-1, keepdim=True)   

    #encoded_prompts = encoded_prompts.double()
    #move to gpu only in forward
    encoded_prompts = encoded_prompts.to('cpu')
    return encoded_prompts

  def GetPathToLeaf(self, class_num):
    #print(f'\nclass_num: {class_num}')
    node_id = self.hier.LEAF_NUM_TO_ID[class_num]
    try:
      path = self.hier.traverse(nodes=[node_id], direction='up', depth=100)
      path.reverse()
      path.pop(0) #remove 'entity' root node as it is shared for all paths
      #for idx, n in enumerate(path):
        #print(f'idx: {idx}, ID: {n}, node_num: {self.hier.HIER_NODE_ID_TO_NUM[n]}, name: {self.hier.HIER_NODE_NAME[n]}')
      return path, class_num 
    except:
      #print("\nGetPathToLeaf: class_num does not exist in BREEDs graph\n")
      return None, -1

  def get_parent(self, node_id):
    path = self.hier.traverse(nodes=[node_id], direction='up', depth=1)
    #path[0] is the node itself
    parent = path[1]
    #print(f'get_parent for node_id: {node_id}, name: {self.hier.HIER_NODE_NAME[node_id]}')
    #print(f'path: {path}')
    #print(f'parent node_id: {parent}, name: {self.hier.HIER_NODE_NAME[parent]}')
    return parent

  def get_siblings_node_ids(self, node_id):
    parent_node_id = self.get_parent(node_id)
    siblings = list(self.hier.get_direct_descendants(parent_node_id))
    #print(f'get_siblings_node_ids for node_id: {node_id}, name: {self.hier.HIER_NODE_NAME[node_id]}')
    #print(f'siblings: {siblings}, name: {self.hier.HIER_NODE_NAME[node_id]}')
    #print(f'siblings names: ')
    #for s in siblings:
      #print(f'{self.hier.HIER_NODE_NAME[s]}')
    return siblings



  def GetDecendantsPromptsDict(self, curr_node_id, prompts_dict, descendants_dict, leaves_reachable_dict):
    
    descendants = list(self.hier.get_direct_descendants(curr_node_id))
    descendants_dict[curr_node_id] = descendants
    leaves_reachable_dict[curr_node_id] = [self.hier.LEAF_ID_TO_NUM[leaf] for leaf in self.hier.leaves_reachable(curr_node_id)]

    #check if leaf node 
    if not descendants:
        prompts_dict[curr_node_id] = None
        return prompts_dict, descendants_dict, leaves_reachable_dict

    descendant_names = [self.hier.HIER_NODE_NAME[d].split(", ")[0] for d in descendants]
    prompts = [f'a photo of {name}' for name in descendant_names]
    #for idx, p in enumerate(prompts):
    #  print(f'idx: {idx}, prompt: {p}')

    tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(device)
    
    with torch.no_grad():
        encoded_prompts = self.clip_model.encode_text(tokenized_prompts).to(device=device)
        encoded_prompts /= encoded_prompts.norm(dim=-1, keepdim=True)
        #if curr_node_id == self.root_id:
        #  print(f'root encoded_prompts: \n{encoded_prompts}')
        #prompts_dict[curr_node_id] =  encoded_prompts.double()
        #move to gpu only in forward function
        encoded_prompts = encoded_prompts.to('cpu')
        prompts_dict[curr_node_id] = encoded_prompts  

    for idx, next_node_id in enumerate(descendants):
        prompts_dict, descendants_dict, leaves_reachable_dict = self.GetDecendantsPromptsDict(next_node_id, prompts_dict, descendants_dict, leaves_reachable_dict)

    return prompts_dict, descendants_dict, leaves_reachable_dict

  def GetNodeBatch(self, node_id, batch):
    """
    returns modified batch the match this node
    1. keep only examples that are reachable from this node
    2. convert image net labels(0-999) to node labels(0-#descendants-1 )
    """
    
    #make sure this is not a leaf node   
    descendants = self.descendants_dict[node_id]
    if not descendants:
      print(f'GetNodeLabels got leaf node_id: {node_id}, class: {self.hier.LEAF_ID_TO_NUM[node_id]}, name: {self.hier.HIER_NODE_NAME[node_id]}')
      return None

    #first check which examples are reachable from this node
    images, labels = batch
    #print(f'labels[0]: {labels[0]}, images[0]: {images[0]}')
    #print(f'images[0]: {images[0]}')
    zipped_batch = list(zip(images, labels))
    #print(f'len(zipped_batch): {len(zipped_batch)}')
    #print(f'zipped_batch[0]: {zipped_batch[0]}')
    #print(f'zipped_batch: {zipped_batch}')
    
    valid_labels = self.leaves_reachable_dict[node_id]
    #res_batch = [x[1] for x in batch]
    valid_samples = [x for x in zipped_batch if x[1] in valid_labels]
    #valid_samples = [x for x in zipped_batch ]
    #print(f'res_batch[0]: {res_batch[0]}')
    #print(f'len(res_batch): {len(res_batch)}')
    valid_batch = list(zip(*valid_samples))
    new_images, imagenet_labels = valid_batch
    #print(f'new_images[0]: {new_images[0]}')
    #print(f'imagenet_labels[0]: {imagenet_labels[0]}, images[0]: {images[0]}')
    new_images = list(new_images)
    new_images = torch.stack(new_images)
    #print(f'new_images[0]: {new_images[0]}')
    #exit(0)
    imagenet_labels = list(imagenet_labels)
    imagenet_labels = torch.tensor(imagenet_labels)
    #print(f'imagenet_labels[0]: {imagenet_labels[0]}, images[0]: {images[0]}')
    

    #convert imagenet label to the matching child node index
    res_labels = torch.empty_like(imagenet_labels)
    for label_idx, label in enumerate(imagenet_labels):
      for node_idx, child_node_id in enumerate(descendants):
        if label in self.leaves_reachable_dict[child_node_id]:
          res_labels[label_idx] = node_idx
          break

    #print(f'res_labels[0:10]: {res_labels[0:10]}')
    #print(f'len(images): {len(images)}, len(res_labels): {len(res_labels)}')

    #exit(0)
    return new_images, res_labels

  def interpret_prediction(self, x):
    self.classify_example(x,print_path=True)


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


  def forward(self, images, labels, unrolled_node_ids):

    similarity = self.get_cosine_similarity(images,self.encoded_prompts)
    exponent_similarity = similarity.exp()

    #duplicate each row of exponent_similarity by len of its coresponding path
    duplicated_exponent_similarity = self.duplicate_matrix_rows(exponent_similarity,labels)
    #print(f'\n\n\nlen(unrolled_node_ids): {len(unrolled_node_ids)}')
    #print(f'exponent_similarity.shape: {exponent_similarity.shape}')
    #print(f'duplicated_exponent_similarity.shape: {duplicated_exponent_similarity.shape}')

    #exit(0)

    path_mask_matrix = self.get_path_mask_matrix(duplicated_exponent_similarity, unrolled_node_ids)
    softmax_numerator = duplicated_exponent_similarity*path_mask_matrix
    
    softmax_denominator = softmax_numerator.sum(dim=1,keepdim=True)
    probs = softmax_numerator / softmax_denominator

    #log of zero probabilities will return -inf but will not be used 
    #as only the log probability of the ground truth is used in NLL Loss
    #RuntimeError: Function 'LogBackward0' returned nan values in its 0th output.
    #log_probs = probs.log()
    probs[path_mask_matrix.bool()] = probs[path_mask_matrix.bool()].log()

    #return log_probs
    return probs

   

  def predict_living(self, hier, image_features, clip_model):
    
    encoded_prompts = self.living_prompts
    similarity = 100.0 * image_features @ encoded_prompts.t() #batch_size X num descendants
    return similarity
    #curr_prob = self.softmax(similarity)

    #print(f'encoded_prompts.shape: {encoded_prompts.shape:}, img_embeds.shape: {img_embeds.shape:}')
    #print(f'similarity.shape: {similarity.shape:}, predicted_labels.shape: {predicted_labels.shape:}')
    #predicted_labels = predicted_labels.cpu()
    #predicted_labels = predicted_labels.numpy()

    #exit(0)

    #return predicted_labels

  """
  def make_prediction(self, x, tot_prob, curr_node_id, final_preds):
   
    descendants = list(self.hier.get_direct_descendants(curr_node_id))
    #print(f'tot_prob.shape: {tot_prob.shape}')

    #check if leaf node 
    if not descendants:
        label = self.hier.LEAF_ID_TO_NUM[curr_node_id]
        name = self.hier.HIER_NODE_NAME[curr_node_id]
        #print(f'node {name} is leaf. class num: {label}')
        #print(f'returned prob for label: {label}. tot_prob.shape: {tot_prob.shape}')
        #print(f'returned prob for label: {label}. tot_prob[0:10]: {tot_prob[0:10]}')
        #print(f'final_preds.shape: {final_preds.shape}')
        final_preds[:, label] = tot_prob
        
        #print(f'final_preds[[0:5,:]]: {final_preds[0:5,:]}')
        return final_preds
        #return label, tot_prob

    #TODO: keep encoded_prompts in each node
    descendant_names = [self.hier.HIER_NODE_NAME[d].split(", ")[0] for d in descendants]
    prompts = [f'a photo of {name}' for name in descendant_names]

    tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(device)
    
    with torch.no_grad():
        encoded_prompts = self.clip_model.encode_text(tokenized_prompts).to(device=device)
        encoded_prompts /= encoded_prompts.norm(dim=-1, keepdim=True)   

    #encoded_prompts = encoded_prompts.double()
    #similarity = encoded_prompts @ x.t()
    #print(f'x.type(): {x.type()} encoded_prompts.type(): {encoded_prompts.type()}')
    similarity = x @ encoded_prompts.t() #batch_size X num descendants
    curr_prob = self.softmax(similarity)
    #print(f'curr_prob.shape: {curr_prob.shape}')
    for idx, next_node_id in enumerate(descendants):
        next_node_prob = curr_prob[:, idx]
        #print(f'next_node_prob.shape: {next_node_prob.shape}')
        next_tot_prob = next_node_prob * tot_prob
        #print(f'next_tot_prob.shape: {next_tot_prob.shape}')
        #label, prob = make_prediction( x, next_tot_prob, next_node_id, final_preds, clip_model)
        final_preds = self.make_prediction(x, next_tot_prob, next_node_id, final_preds)
        #print(f'returned prob for label: {label}. prob.shape: {prob.shape}')

    return final_preds
    """    

  def duplicate_matrix_rows(self, x, labels):
    
    res = torch.cat([x[idx].repeat(len(self.label_path_dict[l.item()]),1) for idx, l in enumerate(labels)] )
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

  def get_path_mask_matrix(self, similarity_matrix, unrolled_node_ids):
    
    mask = torch.zeros_like(similarity_matrix)
    
    for idx, node_id in enumerate(unrolled_node_ids):
      siblings = self.get_siblings_node_ids(node_id)
      row_mask = [self.hier.HIER_NODE_ID_TO_NUM[s] for s in siblings]
      mask[idx][row_mask] = 1
    
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

def DT_CLIP_train(DT_CLIP_model, num_epochs, data_loader, criterion, optimizer, save_frequency=5, checkpoint_name='checkpoint_DT_CLIP_epoch#'):
  DT_CLIP_model.train()
  loaded_epoch = 0
  #loss = None
  
  #torch.save({
  #          'epoch': loaded_epoch,
  #          'model_state_dict': model.state_dict(),
  #          'optimizer_state_dict': optimizer.state_dict(),
  #          'loss': loss
  #        }, PATH)

  #PATH = '/disk5/nir/clip_cap_venv/clip_venv/interpret/DT_CLIP_checkpoint_epoch_0003.pt.tar'
  PATH = '/mnt5/nir/CLIP/interpret/DT_CLIP_checkpoint_epoch_0020.pt.tar'
  
  checkpoint = torch.load(PATH)
  DT_CLIP_model.load_state_dict(checkpoint['state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer'])
  loaded_epoch = checkpoint['completed_epoch']
  #loss = checkpoint['loss']
  print(f'DT_CLIP_train: loaded epoch: {loaded_epoch}')

  for epoch in range(loaded_epoch,num_epochs):
    print(f'start DT_CLIP_train epoch#: {epoch+1}')

    losses = []
    total_examples = 0

    for batch_idx, (images, labels) in enumerate(tqdm(data_loader)):

      #with autograd.detect_anomaly():
        
      images = images.to(device=device)
      #print(f'labels: {labels}')
      #labels = list(labels)
      #print(f'labels: {labels}')
      #labels = torch.tensor(labels).to(device=device)
      labels = labels.to(device=device)
      
      unrolled_node_id_labels = DT_CLIP_model.get_unrolled_node_id_labels(labels)
      unrolled_labels = [DT_CLIP_model.hier.HIER_NODE_ID_TO_NUM[id] for id in unrolled_node_id_labels]

      log_probabilities = DT_CLIP_model(images,labels,unrolled_node_id_labels)
      unrolled_labels = torch.tensor(unrolled_labels,device=device)
      #print(f'labels: \n{labels}')
      #print(f'scores: \n{scores}')
      loss = criterion(log_probabilities, unrolled_labels)
      
      losses.append(loss.item())
      num_examples = len(unrolled_labels)
      
      total_examples += num_examples
      
      #print(f'batch loss: {loss.item()/num_examples}, curr_loss: {loss.item()}, num_examples: {num_examples}')

      optimizer.zero_grad()

      #should have better performance than zero_grad()
      #for param in DT_CLIP_model.parameters():
      #for param in optimizer.parameters():
      #  param.grad = None

      loss.backward()
      optimizer.step()
      
      ##check grads for NaN values
      """
      total_grad_norm = 0
      parameters = [p for p in DT_CLIP_model.parameters() if p.grad is not None and p.requires_grad]
      if len(parameters) > 0:
   
        for p in parameters:
            param_grad = p.grad.detach().data
            param_grad_norm = p.grad.detach().data.norm(2)
            total_grad_norm += param_grad_norm.item() ** 2
            print(f'param_grad: {param_grad}, param_grad_norm: {param_grad_norm}, total_grad_norm: {total_grad_norm}')
        total_grad_norm = total_norm ** 0.5
      print(f'total_norm: {total_norm}')
      """
      ##check grads for NaN values 
    
    completed_epoch = epoch + 1
    print(f'epoch loss: {sum(losses) / total_examples}')


    if (completed_epoch == num_epochs or 
                (
                    save_frequency > 0 and completed_epoch % save_frequency == 0
                )):
      
        # Saving checkpoints.
        checkpoint_dict = {
              "completed_epoch": completed_epoch,
              "state_dict": DT_CLIP_model.state_dict(),
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
        for images, labels in tqdm(loader):
            
            #root_node_id = DT_CLIP_model.root_id
            #images, labels = DT_CLIP_model.GetNodeBatch(root_node_id, batch)

            images = images.to(device=device)
            labels = labels.to(device=device)

            #scores = DT_CLIP_model(images)

            #level = 1 # Could be any number smaller than max 
            #superclasses = DT_CLIP_model.hier.get_nodes_at_level(level)
            #Superclasses at level 1:
            #(0: living thing, animate thing), (1: non-living thing)
            #living_thing_node_id = list(superclasses)[0]
            #living_label_ids = [DT_CLIP_model.hier.LEAF_ID_TO_NUM[leaf] for leaf in DT_CLIP_model.hier.leaves_reachable(living_thing_node_id)]
            #living_labels = [0 if y in living_label_ids else 1 for y in labels]
            #living_labels = torch.tensor(living_labels).to(device=device)
            
            #_, predictions = scores.max(1)
            #num_correct += (predictions == labels).sum()
            prediction = DT_CLIP_model.classify_example(images)
            label = labels[0].data
            #print(f'prediction: {prediction}, label: {label}')
            if prediction == label:
              num_correct += 1
            #num_correct += (predictions == living_labels).sum()
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


def main():

    args = parser.parse_args()
    root_dir = os.path.dirname(os.path.realpath(__file__))    
    cur_dir = os.path.realpath(os.curdir)

    batch_size = args.batch_size
    num_epochs = args.epochs
    print(f'num_epochs: {args.epochs}')
    print(f'args.batch_size: {args.batch_size}')

    # Load the model
    clip_model, preprocess = clip.load('ViT-B/32', device=device)

    info_dir = './BREEDS-Benchmarks/imagenet_class_hierarchy/modified'
    hier = ClassHierarchy(info_dir)

    #num_classes = len(imagenet_test_dataset.classes)
    num_imagenet_classes = len(hier.LEAF_IDS)
    DT_CLIP_model = DecisionTreeClipModel(clip_model, hier)
    
    num_classes = len(DT_CLIP_model.good_labels)
    print(f'num_classes: {num_classes}')

    #exit(0)

  
    train_dataset = SubsetDataset(transform=preprocess, good_labels=DT_CLIP_model.good_labels, split='train', root="/mnt5/nir/dataset/ImageNet")
    test_dataset = SubsetDataset(transform=preprocess, good_labels=DT_CLIP_model.good_labels, split='val', root="/mnt5/nir/dataset/ImageNet")

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
    #criterion = nn.CrossEntropyLoss()
    criterion = nn.NLLLoss()

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
        checkpoint_path = 'DT_CLIP_checkpoint_epoch_'

    #training loop
    #print(f'starting training loop on train dataset')
    #DT_CLIP_train(DT_CLIP_model, num_epochs, train_loader, criterion, optimizer, save_frequency, checkpoint_path)
    
    #print(f'starting training loop on test set')
    #DT_CLIP_train(DT_CLIP_model, num_epochs, test_loader, criterion, optimizer, save_frequency, checkpoint_path)


    #print(f'starting training set check_accuracy')
    #train_accuracy = check_accuracy(train_loader, DT_CLIP_model)
    #print(f"Accuracy on training set: {train_accuracy*100:.2f}%")

    #PATH = '/mnt5/nir/CLIP/interpret/DT_CLIP_Living_test_set_checkpoint_epoch_0001.pt.tar'
    PATH = '/mnt5/nir/CLIP/interpret/DT_CLIP_checkpoint_epoch_0015.pt.tar'

    checkpoint = torch.load(PATH)
    DT_CLIP_model.load_state_dict(checkpoint['state_dict'])
    print(f'loaded checkpoint: {PATH}')
    print(f'starting test set check_accuracy')
    test_accuracy = check_accuracy(test_loader, DT_CLIP_model)
    
    print(f"Accuracy on test set: {test_accuracy*100:.2f}%")
    


if __name__ == "__main__":
    main()
