from custom_collate import collate_mismatch
import dataset_mismatch  
import json 
import os 
import torch
import torchvision
import torch.nn as nn
import numpy as np 
from torch.utils.data import DataLoader
import argparse
import io
import clip_classifier
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import time 
import clip 
from sklearn.metrics import f1_score
import dataset_mismatch_domain_prompt
from sklearn.metrics import confusion_matrix, classification_report

parser = argparse.ArgumentParser(description='Training using the precomputed embeddings')


parser.add_argument('--visual_news_root', type=str, default='',
                    help='location to the root folder of the visualnews dataset')
parser.add_argument('--news_clip_root', type=str, default='',
                    help='location to the root folder of the clip dataset')               
parser.add_argument('--exp_folder', type=str, default='',
                    help='path to the folder to log the output and save the models')
                    
###### model details ########                    
parser.add_argument('--pdrop', type=float, default=0.05,
                    help='dropout probability')


##### Training details #####
parser.add_argument('--batch_size', type=int, default=64,
                    help='dimension of domains embeddings') 
parser.add_argument('--num_workers', type=int, default=6,
                    help='number of data loaders workers') 
parser.add_argument('--epochs', type=int, default = 30,
                    help='number of epochs to run')
parser.add_argument('--start_epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts) (default: 0)')
parser.add_argument('--log_interval', type=int, default=200,
                    help='how many batches')

                   
parser.add_argument('--resume', type=str, default = '', help='path to model')
parser.add_argument('--optimizer', type=str, default='adam',
                    help='which optimizer to use')
parser.add_argument('--lr_clip', type=float, default=1e-4,
                    help='learning rate of the clip model')
parser.add_argument('--lr_classifier', type=float, default=1e-4,
                    help='learning rate of the clip model')                    
parser.add_argument('--sgd_momentum', type=float, default=0.9,
                    help='momentum when using sgd')                      
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--wdecay', default=1.2e-6, type=float,
                        help='weight decay pow (default: -5)')

args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

#### load input files ####



#### settings of the model ####
model_settings = {'pdrop': args.pdrop}
base_clip, preprocess = clip.load("ViT-B/32", device="cuda")
classifier_clip = clip_classifier.ClipClassifier(model_settings,base_clip)
classifier_clip.cuda()

#### load Datasets and DataLoader ####
data_path_test= json.load(open(""))
test_dataset = dataset_mismatch_domain_prompt.NewsClip(data_path_test,args.visual_news_root, args.news_clip_root, 'train', preprocess)
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn = collate_mismatch,num_workers=args.num_workers,  pin_memory=True)


#resume training
stored_loss = 100000000   
stored_acc = 0

classifier_list = ['classifier.weight', 'classifier.bias']
classifier_params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] in classifier_list, classifier_clip.named_parameters()))))
base_params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] not in classifier_list, classifier_clip.named_parameters()))))

 
#define loss function
criterion = nn.BCEWithLogitsLoss().cuda()

params = list(classifier_clip.parameters())
total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in params if x.size())
print('Args:', args)
print('Model total parameters:', total_params)

device = "cuda:0"
def evaluate(loader):
    print("EVALUATE!!!!")
    total_loss = 0
    correct = 0
    total_num = 0
    classifier_clip.eval()

    total_preds = []
    total_truth = []

    for (idx, batch) in enumerate(loader):
        labels = batch[0].to(device)
        imgs = batch[1].to(device)
        captions = batch[2].to(device)
        caption_u = batch[3]
        domain  = batch[4]


        with torch.no_grad():
            #forward 
            output = classifier_clip(imgs,captions,caption_u,domain) 
            #compute loss     
            loss = criterion(output, torch.unsqueeze(labels.float(), 1))

            loss =loss.squeeze()
            loss = loss #* polsim
            loss = loss.mean()

            total_loss += loss.item()
            #compute correct predictions 
            pred = torch.sigmoid(output) >= 0.5
            truth = torch.unsqueeze(labels,1) >= 0.5
            pred_m = pred.cpu()
            truth_m = truth.cpu()
            total_preds.append(np.array(pred_m))
            total_truth.append(np.array(truth_m))
            # print(truth)
            correct += pred.eq(truth).sum().item()
            total_num += labels.size(0) 
    avg_loss = total_loss/len(loader)    
    acc = (correct/total_num)*100
    return avg_loss, acc , total_preds , total_truth
def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        if p.grad is not None: p.grad.data = p.grad.data.float() 
    
    

# Load the best saved model if exists and run for the test data.
if os.path.isfile(os.path.join(args.exp_folder, 'best_model_acc.pth.tar')):
    checkpoint = torch.load(os.path.join(args.exp_folder, 'best_model_acc.pth.tar'))
    classifier_clip.load_state_dict(checkpoint['state_dict'])
    
    clip.model.convert_weights(classifier_clip)
    
    val_loss, val_acc,total_predictions , total_truths= evaluate(test_dataloader)
    tot_preds= []
    for ele in total_predictions :
        for e in ele:
            tot_preds.append(int(e))

    tot_truths= []
    for ele in total_truths :
        for e in ele :
            tot_truths.append(int(e))    

    print("Confusion Matrix!" , confusion_matrix(tot_truths, tot_preds))

    print('-' * 89)

    print('| end of epoch {:3d} | time: {:5.2f}s | val loss {:5.2f} | '
            'val acc {:8.2f}'.format(
                30, (time.time() - 0), val_loss , val_acc))
    print('-' * 89)