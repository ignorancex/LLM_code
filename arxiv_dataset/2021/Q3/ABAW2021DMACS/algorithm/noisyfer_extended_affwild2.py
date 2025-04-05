 # -*- coding:utf-8 -*-
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from model.cnn import  resModel
import numpy as np
from common.utils import accuracy
import os
from algorithm.loss import CCTloss

import sklearn.metrics as sm


def statistic(target, predict):
    precision = sm.precision_score(target, predict, average="macro",  zero_division=1)
    recall = sm.recall_score(target, predict, average="macro", zero_division=1)
    F1_score = sm.f1_score(target, predict, average="macro", zero_division=1)
    return precision, recall, F1_score


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def val_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)

    top1 = pred[:, 0].cpu().numpy()
    target_np = target.view(-1).cpu().numpy()

    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    
    cm = 0
    
    #print(target_np.shape, top1.shape)
    if top1.size == 0 and target_np.size == 0:
       cm = 0
       precision, recall, F1_score = -1, -1, -1
    else:
       #cm = sm.confusion_matrix(target_np, top1, labels=range(7),normalize='all') #'true,'pred'
       precision, recall, F1_score = statistic(target_np, top1)
    return res, cm, precision, recall, F1_score
    
def switch_expression(expression_argument): #convert back from Affecnet labels to Affwild2 labels
    switcher = {
          0:0,
          1:4,
          2:5,
          3:6 ,
          4:3, 5:2, 6:1,
    }
    return switcher.get(expression_argument, 0)    
    
    
class noisyfer:
    def __init__(self, args, train_dataset, device, input_channel, num_classes):

        # Hyper Parameters
        self.batch_size = args.batch_size
        learning_rate = args.lr
        
        # Adjust learning rate and betas for Adam Optimizer
        mom1 = 0.9
        mom2 = 0.1
        self.device = device
        self.num_iter_per_epoch = args.num_iter_per_epoch
        self.print_freq = args.print_freq
        #self.co_lambda = args.co_lambda
        self.n_epoch = args.n_epoch
        self.train_dataset = train_dataset
        self.co_lambda_max = args.co_lambda_max
        self.beta = args.beta
        
        print('\n beta ',args.beta)
        
        self.num_classes  = args.num_classes
        self.max_epochs =args.n_epoch

        if args.model_type == "cnn":
            self.model1 = CNN(input_channel=input_channel, n_outputs=num_classes)
            self.model2 = CNN(input_channel=input_channel, n_outputs=num_classes)
        elif args.model_type == "mlp":
            self.model1 = MLPNet()
            self.model2 = MLPNet()
        elif args.model_type=="res":
            self.model1 = resModel(args)     
            self.model2 = resModel(args)
            self.model3 = resModel(args)

        self.model1.to(device)
        self.model2.to(device)
        self.model3.to(device)
        filter_list = ['module.classifier.weight', 'module.classifier.bias']
        base_parameters_model1 = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] not in filter_list, self.model1.named_parameters()))))
        base_parameters_model2 = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] not in filter_list, self.model2.named_parameters()))))
        base_parameters_model3 = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] not in filter_list, self.model3.named_parameters()))))
      
        self.optimizer = torch.optim.Adam([{'params': base_parameters_model1}, {'params': list(self.model1.module.classifier.parameters()), 'lr': learning_rate}], lr=1e-3)
        self.optimizer.add_param_group({'params': base_parameters_model2, 'lr':1e-3})
        self.optimizer.add_param_group({'params': list(self.model2.module.classifier.parameters()), 'lr':learning_rate})    
        self.optimizer.add_param_group({'params': base_parameters_model3, 'lr':1e-3})
        self.optimizer.add_param_group({'params': list(self.model3.module.classifier.parameters()), 'lr':learning_rate})                                         
                                             
        print('\n Initial learning rate is:')
        for param_group in self.optimizer.param_groups:
            print(  param_group['lr'])                              
        
        if args.resume:
           pretrained = torch.load(args.resume)
           
           pretrained_state_dict1 = pretrained['model_1']   
           pretrained_state_dict2 = pretrained['model_2']   
           pretrained_state_dict3 = pretrained['model_3']   
          
           model1_state_dict =  self.model1.state_dict()
           model2_state_dict =  self.model2.state_dict()
           model3_state_dict =  self.model3.state_dict()
           loaded_keys = 0
           total_keys = 0
           for key in pretrained_state_dict1: 
               #print(key)   
               if  ((key=='module.fcx.weight')|(key=='module.fcx.bias')):
                   print(key)
                   pass
               else:    
                   model1_state_dict[key] = pretrained_state_dict1[key]
                   model2_state_dict[key] = pretrained_state_dict2[key]
                   model3_state_dict[key] = pretrained_state_dict3[key]
                   total_keys+=1
                   if key in model1_state_dict and key in model2_state_dict and  key in model3_state_dict:
                      loaded_keys+=1
           print("Loaded params num:", loaded_keys)
           print("Total params num:", total_keys)
           self.model1.load_state_dict(model1_state_dict) 
           self.model2.load_state_dict(model2_state_dict)
           self.model3.load_state_dict(model3_state_dict)
           
           print('All 3 models loaded from ',args.resume)
           
        else:
           print('\n No checkpoint found.\n')         
        
        self.loss_fn = CCTloss
        
        self.m1_statedict =  self.model1.state_dict()
        self.m2_statedict =  self.model2.state_dict()
        self.m3_statedict =  self.model3.state_dict()
        self.o_statedict = self.optimizer.state_dict()  

        self.adjust_lr = args.adjust_lr
    
    
        
    # Evaluate the Model
    def evaluate(self, test_loader):
        print('Evaluating ...')
        self.model1.eval()  # Change model to 'eval' mode.
        self.model2.eval()  # Change model to 'eval' mode
        self.model3.eval()
        correct1 = 0
        total1 = 0
        correct2 = 0
        total2 = 0
        correct3 = 0
        total3 = 0
        correct  = 0
        cm = 0 
        f1 = AverageMeter()
        top1 = AverageMeter()
        with torch.no_grad():
            for images, labels, _ in test_loader:
                images = (images).to(self.device)
                logits1 = self.model1(images)
                outputs1 = F.softmax(logits1, dim=1)
                _, pred1 = torch.max(outputs1.data, 1)
                total1 += labels.size(0)
                correct1 += (pred1.cpu() == labels).sum()

            
                #for images, labels, _ in test_loader:
                #images = (images).to(self.device)
                logits2 = self.model2(images)                
                outputs2 = F.softmax(logits2, dim=1)
                _, pred2 = torch.max(outputs2.data, 1)
                total2 += labels.size(0)
                correct2 += (pred2.cpu() == labels).sum()
                
                logits3 = self.model3(images)
                outputs3 = F.softmax(logits3, dim=1)
                _, pred3 = torch.max(outputs3.data, 1)
                total3 += labels.size(0)
                correct3 += (pred3.cpu() == labels).sum()
                
                avg_output = 0.33*(outputs1.data + outputs2.data+ outputs3.data) 
                _, avg_pred = torch.max(avg_output, 1)
                correct += (avg_pred.cpu() == labels).sum()
                
                
                avg_prec, expr_cm, precision, recall, F1_score  = val_accuracy(avg_output, labels.to(self.device), topk=(1,))       
                top1.update(avg_prec[0], images.size(0))
                f1.update(F1_score, images.size(0))
                cm += expr_cm
                
            acc1 = 100 * float(correct1) / float(total1)
            acc2 = 100 * float(correct2) / float(total2)
            acc3 = 100 * float(correct3) / float(total3)
            acc = 100 * float(correct) / float(total2)
        return acc1, acc2,acc3, acc, top1.avg, f1.avg, cm
    
    
    # Evaluate the Model
    def test_results_generation(self, test_loader, dest_file='ExprChallenge_21/test_predictions.csv'):
        print('Evaluating ...')
        self.model1.eval()  
        self.model2.eval()  
        self.model3.eval()
        correct1 = 0
        total1 = 0
        correct2 = 0
        total2 = 0
        correct3 = 0
        total3 = 0
        correct  = 0
        cm = 0 
        f1 = AverageMeter()
        top1 = AverageMeter()
        
        preds = np.empty(0, dtype=int)
        filenames = []
        
        with torch.no_grad():
            for i ,(images, imgPath,_) in enumerate(test_loader):
                images = (images).to(self.device)
                logits1 = self.model1(images)
                outputs1 = F.softmax(logits1, dim=1)
                _, pred1 = torch.max(outputs1.data, 1)
                               
                logits2 = self.model2(images)                
                outputs2 = F.softmax(logits2, dim=1)
                _, pred2 = torch.max(outputs2.data, 1)
                
                logits3 = self.model3(images)
                outputs3 = F.softmax(logits3, dim=1)
                _, pred3 = torch.max(outputs3.data, 1)
                
                avg_output = 0.33*(outputs1.data + outputs2.data+ outputs3.data) 
                
                print(i)
                _, pred = torch.max(avg_output,dim=1)
                
                preds = np.concatenate((preds,pred.cpu().numpy()))
                filenames = filenames + list(imgPath)
                
                
        print( len(filenames),filenames[:2])#, np.array(filenames).size, preds.size )
        table = np.array([[d.replace("'",""), switch_expression(int(c))] for d, c in zip(filenames, preds)])
        #print(table)
        np.savetxt(dest_file, table, delimiter='\n', fmt="%50s,%s")
           
    def save_model(self, epoch, avg, prec1,f1):
        torch.save({' epoch':  epoch,
                    'model_1': self.m1_statedict,
                    'model_2': self.m2_statedict,
                    'model_3': self.m3_statedict, 
                    'optimizer':self.o_statedict,},                          
                     os.path.join('checkpoints/affwild2/', "epoch_"+str(epoch)+'_prec_f1_'+str(prec1)+'_'+str(f1)+"_avg_"+str(avg)+".pth")) 
        print('Models saved '+os.path.join('checkpoints/affwild2/', "epoch_"+str(epoch)+'_prec_f1_'+str(prec1)+'_'+str(f1)+"_avg_"+str(avg)+".pth")) 
    
    
    
               
    # Train the Model
    def train(self, train_loader, epoch):
        print('Training ...')
        self.model1.train()  # Change model to 'train' mode.
        self.model2.train()  # Change model to 'train' mode
        self.model3.train()
        
        
        if epoch > 0:
           self.adjust_learning_rate(self.optimizer, epoch)
       
        train_total = 0
        train_correct = 0
        train_total2 = 0
        train_correct2 = 0
        train_total3 = 0
        train_correct3 = 0
        pure_ratio_1_list = []
        pure_ratio_2_list = []

        for i, (images, labels, indexes) in enumerate(train_loader):
            ind = indexes.cpu().numpy().transpose()
        
            if i > self.num_iter_per_epoch:
                break
                
            #labels1, labels2, labels3 = self.dynamic_target_variablity(labels, epoch)

            images = images.to(self.device)
            labels = labels.to(self.device)
            #labels1, labels2, labels3 = labels1.to(self.device), labels2.to(self.device), labels3.to(self.device)
            # Forward + Backward + Optimize
            
            logits1 = self.model1(images)
            prec1 = accuracy(logits1, labels, topk=(1,))
            train_total += 1
            train_correct += prec1

            logits2 = self.model2.forward(images)
            
            prec2 = accuracy(logits2, labels, topk=(1,))

            train_total2 += 1
            train_correct2 += prec2
            
            
            logits3 = self.model3(images)        
            
            
            prec3 = accuracy(logits3, labels, topk=(1,))
            train_total3 += 1
            train_correct3 += prec3
            
             
            avg_prec = accuracy(0.33*(logits1+logits2+logits3), labels, topk=(1,))
            
            loss = self.loss_fn(logits1, logits2, logits3, labels, self.co_lambda_max,  self.beta, epoch, self.n_epoch)
            

            self.optimizer.zero_grad()
            loss.backward()
            
            self.optimizer.step()
            
            if (i + 1) % self.print_freq == 0:
                print(
                    'Epoch [%d/%d], Iter [%d/%d] Training Accuracy1: %.4F, Training Accuracy2: %.4f, Training Accuracy3: %.4f,Avg Accuracy: %.4f, Loss1: %.4f, %%'
                    % (epoch + 1, self.n_epoch, i + 1, len(self.train_dataset) // self.batch_size, prec1, prec2,prec3,avg_prec,  loss.data.item() ))

        train_acc1 = float(train_correct) / float(train_total)
        train_acc2 = float(train_correct2) / float(train_total2)
        train_acc3 = float(train_correct3) / float(train_total2)
        return train_acc1, train_acc2, train_acc3 , avg_prec
    
    def adjust_learning_rate(self, optimizer, epoch):
        print('\n******************************\n\tAdjusted learning rate: '+str(epoch) +'\n')    
        for param_group in optimizer.param_groups:
           param_group['lr'] *= 0.95
           print(param_group['lr'])              
        print('******************************')
    
