# Imports
#pip install transformers
#pip install ftfy regex tqdm
#pip install git+https://github.com/openai/CLIP.git

import os
import torch
import torchvision 
import torchvision.datasets as datasets  
import torchvision.transforms as transforms  
from torch import optim  
from torch import nn  
from torch.utils.data import DataLoader  
from tqdm import tqdm
import clip  
from torchvision.datasets import CIFAR100
from torchvision.datasets import ImageNet
from tqdm import tqdm

learning_rate = 0.001
batch_size = 64
num_epochs = 3
device = "cuda:0" if torch.cuda.is_available() else "cpu"

class LinearProbeClipModel(nn.Module):
  #def __init__(self, clip_model, input_size, num_classes):
  def __init__(self, input_size, num_classes):
        super(LinearProbeClipModel, self).__init__()
        #self.clip_model = clip_model
        #self.clip_model.eval()
        self.fc = nn.Linear(input_size, num_classes)
        
        #self.prompt_features = torch.nn.Parameter(torch.randn_like(text_features))
        
        #self.prompt_features.requires_grad = True   
        
        # freeze CLIP model params and only train prompt.
        #for param in self.clip_model.parameters():
        #  param.requires_grad = False

  def forward(self, images_embeddings):
        
        #images_embeddings /= images_embeddings.norm(dim=-1, keepdim=True)
        return self.fc(images_embeddings)

def linear_probe_train(linear_probe_model, clip_model, num_epochs, data_loader, criterion, optimizer, save_frequency=5, checkpoint_name='checkpoint_linear_probe_epoch#'):
  
  loaded_epoch = 0
  #loss = None
  
  #torch.save({
  #          'epoch': loaded_epoch,
  #          'model_state_dict': model.state_dict(),
  #          'optimizer_state_dict': optimizer.state_dict(),
  #          'loss': loss
  #        }, PATH)

  PATH = '/disk5/nir/clip_cap_venv/clip_venv/interpret/checkpoint_imgnet_linear_probe_epoch_0106.pt.tar'
  checkpoint = torch.load(PATH)
  linear_probe_model.load_state_dict(checkpoint['state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer'])
  #loaded_epoch = checkpoint['epoch'] + 1
  loaded_epoch = checkpoint['completed_epoch']
  #loss = checkpoint['loss']
  print(f'linear_probe_train: loaded epoch: {loaded_epoch}')

  for epoch in range(loaded_epoch,num_epochs):
    print(f'start linear_probe_train epoch#: {epoch+1}')
    for batch_idx, (images, labels) in enumerate(tqdm(data_loader)):
      images = images.to(device=device)
      labels = labels.to(device=device)

      image_features = clip_model.encode_image(images)
      image_features /= image_features.norm(dim=-1, keepdim=True)
      image_features.float()
      image_features = image_features.detach()

      scores = linear_probe_model(image_features)
      loss = criterion(scores, labels)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
    
    completed_epoch = epoch + 1

    if (completed_epoch == num_epochs or 
                (
                    save_frequency > 0 and completed_epoch % save_frequency == 0
                )):
      
        # Saving checkpoints.
        checkpoint_dict = {
              "completed_epoch": completed_epoch,
              "state_dict": linear_probe_model.state_dict(),
              "optimizer": optimizer.state_dict(),
        } 

        filename = checkpoint_name + f"{completed_epoch:04d}.pt.tar"
        print(f'saving model state to filename: {filename}')
        torch.save(checkpoint_dict, filename)
      

def check_accuracy(loader, linear_probe_model, clip_model):
    num_correct = 0
    num_samples = 0
    linear_probe_model.eval()
    clip_model.eval()
      
    with torch.no_grad():
        for images, labels in tqdm(loader):
            
            images = images.to(device=device)
            labels = labels.to(device=device)

            image_features = clip_model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            image_features.float()
            image_features = image_features.detach()

            scores = linear_probe_model(image_features)

            #scores = linear_probe_model(image_features).softmax(dim=-1)
            
            #values, indices = scores[0].topk(5)
            
            _, predictions = scores.max(1)
            num_correct += (predictions == labels).sum()
            num_samples += predictions.size(0)

    linear_probe_model.train()
    return num_correct/num_samples


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])


def main():

    
    # Load the model
    clip_model, preprocess = clip.load('ViT-B/32', device)

    # Download the dataset
    #train_dataset = CIFAR100(root=os.path.expanduser("dataset/"), download=True, train=True, transform=preprocess)
    #test_dataset = CIFAR100(root=os.path.expanduser("dataset/"), download=True, train=False, transform=preprocess)
    train_dataset = CIFAR100(root='/disk1/dataset/cifar100/', download=False, train=True, transform=preprocess)
    test_dataset = CIFAR100(root='/disk1/dataset/cifar100/', download=False, train=False, transform=preprocess)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    #val_set = datasets.ImageNet('/disk1/dataset/ImageNet/val/', split='val', transform=preprocess)
    
    #test_set = datasets.ImageNet('/disk1/dataset/ImageNet/test/', split='test', transform=preprocess)
                                   

    #train_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=True)
    #test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

    prompt_clip_model = PromptClipModel(clip_model, train_dataset.classes, is_single_token=True)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(prompt_clip_model.parameters(), lr=learning_rate)

    #training loop
    prompt_train(prompt_clip_model, num_epochs, train_loader, criterion, optimizer)

    print(f'start training set check_accuracy')
    train_accuracy = check_accuracy(train_loader, prompt_clip_model)
    print(f'start test set check_accuracy')
    test_accuracy = check_accuracy(test_loader, prompt_clip_model)

    print(f"Accuracy on training set: {train_accuracy*100:.2f}")
    print(f"Accuracy on test set: {test_accuracy*100:.2f}")


if __name__ == "__main__":
    main()
