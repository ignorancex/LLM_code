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
from torchvision.transforms import (CenterCrop, Compose, Normalize, Resize,
                                    ToTensor)
from torch import optim  
from torch import nn  
from torch.utils.data import DataLoader, Dataset  
from tqdm import tqdm
import clip  
from torchvision.datasets import CocoCaptions
#from lora.lib.CLIP.clip import *
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import cv2
from gradcam.CLIP_explainability import get_image_text_relevance
#import datasets

learning_rate = 0.001
batch_size = 64
num_epochs = 3
device = "cuda:0" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser(description='CLIP COCO LoRA finetune')

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
    
  

def CLIP_COCO_train(model, num_epochs, data_loader, img_criterion, txt_criterion, optimizer, save_frequency=5, checkpoint_name='CLIP_LoRA_COCO_checkpoint_epoch_'):
  model.train()
  loaded_epoch = 0
  #loss = None
  
  #torch.save({
  #          'epoch': loaded_epoch,
  #          'model_state_dict': model.state_dict(),
  #          'optimizer_state_dict': optimizer.state_dict(),
  #          'loss': loss
  #        }, PATH)

  
  #PATH = 'CLIP_COCO_checkpoint_epoch_0003.pt.tar'
  
  #checkpoint = torch.load(PATH)
  #model.load_state_dict(checkpoint['state_dict'])
  #optimizer.load_state_dict(checkpoint['optimizer'])
  #loaded_epoch = checkpoint['completed_epoch']
  ##loss = checkpoint['loss']
  
  print(f'CLIP_COCO_train: loaded epoch: {loaded_epoch}')

  for epoch in range(loaded_epoch,num_epochs):
    print(f'start CLIP_COCO_train epoch#: {epoch+1}')

    losses = []
    total_examples = 0
    
    for batch_idx, (full_path, images, captions) in enumerate(tqdm(data_loader)):

      #with autograd.detect_anomaly():
        
      images = images.to(device=device)
      
      #print(f'captions: {captions}')
      #print(f'captions[0]: {captions[0]}')
      #print(f'captions[0][0]: {captions[0][0]}')
      
      prompts = [f'a photo of {cap}' for cap in captions]
      #print(f'\ncaptions: {captions}\n')
      #print(f'\nprompts: {prompts}\n')


      tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(device=device)
      #tokenized_prompts.shape: torch.Size([128, 77])
      #print(f'\ntokenized_prompts.shape: {tokenized_prompts.shape}')

      #tokenized_captions = torch.cat([clip.tokenize(p) for p in prompts]).to(device=device)

      logits_per_image, logits_per_text = model(images, tokenized_prompts)
      ground_truth = torch.arange(len(images), dtype=torch.long).to(device=device)

      loss_img = img_criterion(logits_per_image, ground_truth)
      loss_txt = txt_criterion(logits_per_text, ground_truth)

      total_loss = (loss_img + loss_txt)/2
      
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
      
        # Saving checkpoints.
        checkpoint_dict = {
              "completed_epoch": completed_epoch,
              "state_dict": model.state_dict(),
              "optimizer": optimizer.state_dict(),
        }

        filename = checkpoint_name + f"{completed_epoch:04d}.pt.tar"
        print(f'saving model state to filename: {filename}')
        torch.save(checkpoint_dict, filename)



def check_accuracy(loader, clip_model):
    num_correct = 0
    num_samples = 0
    clip_model.eval()

    with torch.no_grad():
        #for batch in tqdm(loader):
        
        for full_path, images, captions in tqdm(loader):
           
            images = images.to(device=device)
            
            #########################
            #########################

            prompts = [f'a photo of {cap}' for cap in captions]
            tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(device=device)
      
            logits_per_image, logits_per_text = clip_model(images, tokenized_prompts)
            #probs = logits_per_image.softmax(dim=-1).cpu().numpy()
            probs = logits_per_image.softmax(dim=-1)
            labels = torch.arange(len(images), dtype=torch.long).to(device=device)
            
            _, predictions = probs.max(1)

            #########################
            #########################
                       
            num_correct += (predictions == labels).sum()
            num_samples += labels.size(0)
            #num_samples += 1
            #print(f'num_correct: {num_correct}, num_samples: {num_samples}, accuracy: {100*num_correct/num_samples:.2f}')

    clip_model.train()
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


def load_model(base_name="ViT-B/32", lora_r=-1, weight_name=""):
  clip_model, preprocess = clip.load(base_name, jit=False, lora=lora_r)
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


def main():

    args = parser.parse_args()
    root_dir = os.path.dirname(os.path.realpath(__file__))    
    cur_dir = os.path.realpath(os.curdir)

    batch_size = args.batch_size
    #batch_size = 1
    num_epochs = args.epochs
    print(f'num_epochs: {args.epochs}')
    print(f'batch_size: {batch_size}')

    #Load the model
    vit_name = 'ViT-B/32'
    clip_model, preprocess = clip.load(vit_name, device=device)
    clip_model = clip_model.float()

    #freeze self.clip_model params
    for param in clip_model.parameters():
        param.requires_grad = False
        param.data = param.data.float()
        if param.grad:
          param.grad.data = param.grad.data.float() 

    #unfreeze image encoder params (keep text encoder frozen)
    for param in clip_model.visual.parameters():
        param.requires_grad = True
        
    train_dataset = CocoCaptionsDataset(root="/mnt5/yoavkurtz/datasets/coco2017/train2017", annFile = "/mnt5/yoavkurtz/datasets/coco2017/annotations/captions_train2017.json", transform=preprocess)
    test_dataset = CocoCaptionsDataset(root="/mnt5/yoavkurtz/datasets/coco2017/val2017", annFile = "/mnt5/yoavkurtz/datasets/coco2017/annotations/captions_val2017.json", transform=preprocess)
   
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    save_frequency = 1

    # Loss and optimizer
    img_criterion = nn.CrossEntropyLoss()
    txt_criterion = nn.CrossEntropyLoss()
    #criterion = nn.NLLLoss()

    learning_rate = 3e-6
    #learning_rate = 1e-6
    weight_decay = 0.1

    params = [p for p in clip_model.parameters() if p.requires_grad]
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
        checkpoint_path = 'CLIP_COCO_checkpoint_epoch_'

    #training loop
    print(f'starting training loop on train dataset')
    CLIP_COCO_train(clip_model, num_epochs, train_loader, img_criterion, txt_criterion, optimizer, save_frequency, checkpoint_path)

    
    #PATH = 'CLIP_COCO_checkpoint_epoch_0001.pt.tar'
    
    #checkpoint = torch.load(PATH, map_location=torch.device(device))
    #clip_model.load_state_dict(checkpoint['state_dict'])
    #print(f'loaded checkpoint: {PATH}')
    #print(f'starting test set check_accuracy')
    #test_accuracy = check_accuracy(test_loader, clip_model)
    
    #print(f"Accuracy on test set: {test_accuracy*100:.2f}%")
    


if __name__ == "__main__":
    main()
