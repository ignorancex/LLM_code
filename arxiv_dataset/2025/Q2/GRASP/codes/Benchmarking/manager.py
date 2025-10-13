import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import time

class graph_manager():
    def __init__(self, model,device,model_name='GRASP'):
        self.model = model
        self.device = device
        self.model_name = model_name
        if device == "cuda":
            self.model.cuda()
    
    def train(self, train_val_loader, optimizer, epochs, weights, scheduler_flag=False): # train_val_loader: {'train': train_loader, 'val': val_loader}
        self.model.to(self.device)
        self.model.eval()
        #self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optimizer
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)
        self.train_loss = []
        self.train_acc = []
        self.val_loss = []
        self.val_acc = []
        self.best_val_acc = 0
        self.best_epoch = 0
        self.best_model_wts = copy.deepcopy(self.model.state_dict())
        #self.weights_loss = weights

        for epoch in range(epochs):
            print('Epoch {}/{}'.format(epoch, epochs), flush=True)
            print('-' * 25, flush=True)
            self.logits = {'train':[], 'val':[]}
            self.labels = {'train':[], 'val':[]}
            self.preds = {'train':[], 'val':[]}
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.    
            
                for g, labels, slide in train_val_loader[phase]:
                    g = g.to(self.device)
                    labels = labels.to(self.device)

                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        if self.model_name in ['GRASP', 'ZoomMIL', 'GRASP_D', 'GRASP_1']:
                            outputs, attention = self.model(g, g.ndata['x'])
                            size_data = g.ndata['x'].size(0)
                        elif self.model_name in ['H2MIL', 'PatchGCN', 'DGCN', 'HiGT']:
                            outputs, attention = self.model(g, g.x)
                            size_data = g.x.size(0)
                        else:
                            raise NotImplementedError(f"Yo bro, this {self.model_name} has not been implemented!")
                        #print(outputs)
                        _, preds = torch.max(outputs, 1)
                        #print(outputs, preds, labels)
                        if weights == None:
                            loss = F.cross_entropy(outputs, labels)
                        else:
                            loss = F.cross_entropy(outputs, labels, weight=weights)
                        # backward + optimize only if in training phase
                        #print(outputs)
                        self.logits[phase].append(outputs.clone().detach().squeeze().cpu())
                        self.labels[phase].extend(labels)
                        self.preds[phase].extend(preds.detach())

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()
                    
                    # statistics
                    running_loss += loss.item() * size_data
                    #print(running_loss)
                    running_corrects += torch.sum(torch.tensor(self.preds[phase] == self.labels[phase]))

                if phase == 'train':
                    if scheduler_flag:
                        self.scheduler.step()
                    self.train_loss.append(running_loss / len(train_val_loader[phase].dataset))
                    self.train_acc.append(running_corrects.double() / len(train_val_loader[phase].dataset))
                    print('Train Loss: {:.4f}'.format(running_loss / len(train_val_loader[phase].dataset)), flush=True)
                    #print('Train Acc: {:.4f}'.format(running_corrects.double() / len(train_val_loader[phase].dataset)))
                if phase == 'val':
                    self.val_loss.append(running_loss / len(train_val_loader[phase].dataset))
                    self.val_acc.append(running_corrects.double() / len(train_val_loader[phase].dataset))
                    print('Val Loss: {:.4f}'.format(running_loss / len(train_val_loader[phase].dataset)), flush=True)
                    #print('Val Acc: {:.4f}'.format(running_corrects.double() / len(train_val_loader[phase].dataset)))
                    
                #print(self.logits[phase])
                #self.logits[phase] = self.logits[phase]
                self.labels[phase] = torch.tensor(self.labels[phase])
                self.preds[phase] = torch.tensor(self.preds[phase])
                print(phase + ' Acc: {:.4f}'.format(torch.sum(self.preds[phase] == self.labels[phase])/len(train_val_loader[phase].dataset)), flush = True)
            if phase == 'val' and torch.sum(self.preds[phase] == self.labels[phase])/len(train_val_loader[phase].dataset) > self.best_val_acc:
                self.best_val_acc = torch.sum(self.preds[phase] == self.labels[phase]).cpu()/len(train_val_loader[phase].dataset)
                self.best_epoch = epoch
                self.best_model_wts = copy.deepcopy(self.model.state_dict())
                self.best_optimizer =  copy.deepcopy(self.optimizer)
                #torch.save(self.best_model_wts, 'best_model.pth')
        self.model.load_state_dict(self.best_model_wts)
        return self.model, self.best_optimizer, self.best_val_acc, self.best_epoch
    
    def test(self, test_loader, best_model):
        best_model.to(self.device)
        best_model.eval()
        self.logits = []
        self.slides = []
        self.labels = []
        self.preds = []
        self.attention = []
        self.criterion = nn.CrossEntropyLoss()
        start = time.time() 
        for g, labels, slide in test_loader:
            g = g.to(self.device)
            labels = labels.to(self.device)
            self.slides.append(slide)
            self.labels.extend(labels.cpu())
            print(f"Memory Allocated: {torch.cuda.memory_allocated(self.device) / 1024**2:.2f} MB")
            print(f"Memory Reserved: {torch.cuda.memory_reserved(self.device) / 1024**2:.2f} MB")
            if self.model_name in ['GRASP', 'ZoomMIL', 'GRASP_D', 'GRASP_1']:
                outputs, attention = best_model(g, g.ndata['x'])
                #print(g)
            elif self.model_name in ['H2MIL', 'PatchGCN', 'DGCN', 'HiGT']:
                outputs, attention = best_model(g, g.x)
            print(f"After Memory Allocated: {torch.cuda.memory_allocated(self.device) / 1024**2:.2f} MB")
            print(f"After Memory Reserved: {torch.cuda.memory_reserved(self.device) / 1024**2:.2f} MB")
            #torch.cuda.empty_cache()
            self.logits.append(outputs.clone().detach().squeeze().cpu())
            self.attention.append(attention.clone().detach().cpu())
            _, preds = torch.max(outputs, 1)
            self.preds.extend(preds.cpu())
        duration = time.time() - start
        avg_time = float(duration/len(test_loader))
        self.attention = torch.cat(self.attention, 0)
        #self.logits = torch.cat(self.logits, 0)
        self.labels = torch.tensor(self.labels)
        self.preds = torch.tensor(self.preds)
        print('-' * 25)
        print('Test Acc: {:.4f}'.format(torch.sum(self.preds == self.labels)/len(test_loader.dataset)), flush=True)
        output = {'logits': self.logits, 'slides': self.slides, 'labels': self.labels, 'preds': self.preds, 'attention': self.attention, 'average_time': avg_time}
        return output


class MovingAvg:
    def __init__(self, network, ema=False, sma_start_iter=20):
        self.network = network
        self.network_sma = copy.deepcopy(network)
        self.sma_start_iter = sma_start_iter
        self.global_iter = 0
        self.sma_count = 0
        self.ema = ema

    def update_sma(self):
        self.global_iter += 1
        if self.global_iter >= self.sma_start_iter and self.ema:
            # if False:
            self.sma_count += 1
            for param_q, param_k in zip(
                self.network.parameters(), self.network_sma.parameters()
            ):
                param_k.data = (param_k.data * self.sma_count + param_q.data) / (
                    1.0 + self.sma_count
                )
        else:
            for param_q, param_k in zip(
                self.network.parameters(), self.network_sma.parameters()
            ):
                param_k.data = param_q.data

class graph_manager_ma():
    def __init__(self, model,device,model_name='GRASP'):
        self.model = MovingAvg(model, ema=True) 
        self.device = device
        self.model_name = model_name
        if device == "cuda:0":
            self.model.cuda()
    
    def train(self, train_val_loader, optimizer, epochs, weights, scheduler_flag=False): # train_val_loader: {'train': train_loader, 'val': val_loader}
        self.model.network.to(self.device)
        self.model.network_sma.to(self.device)
        self.model.network.eval()
        self.model.network_sma.eval()
        #self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optimizer
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)
        self.train_loss = []
        self.train_acc = []
        self.val_loss = []
        self.val_acc = []
        self.best_val_acc = 0
        self.best_epoch = 0
        self.best_model_wts = copy.deepcopy(self.model.network_sma.state_dict())
        #self.weights_loss = weights

        for epoch in range(epochs):
            print('Epoch {}/{}'.format(epoch, epochs), flush=True)
            print('-' * 25, flush=True)
            self.logits = {'train':[], 'val':[]}
            self.labels = {'train':[], 'val':[]}
            self.preds = {'train':[], 'val':[]}
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.network.train()  # Set model to training mode
                    self.model.network_sma.train()
                else:
                    self.model.network.eval()   # Set model to evaluate mode
                    self.model.network_sma.eval()

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.    
            
                for g, labels, slide in train_val_loader[phase]:
                    g = g.to(self.device)
                    labels = labels.to(self.device)

                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        if self.model_name in ['GRASP', 'ZoomMIL', 'GRASP_D', 'GRASP_dropout']:
                            outputs, attention = self.model.network(g, g.ndata['x'])
                            size_data = g.ndata['x'].size(0)
                        elif self.model_name in ['H2MIL', 'PatchGCN', 'DGCN', 'HiGT']:
                            outputs, attention = self.model.network(g, g.x)
                            size_data = g.x.size(0)
                        else:
                            raise NotImplementedError(f"Yo bro, this {self.model_name} has not been implemented!")
                        #print(outputs)
                        _, preds = torch.max(outputs, 1)
                        #print(outputs, preds)
                        if weights == None:
                            loss = F.cross_entropy(outputs, labels)
                        else:
                            loss = F.cross_entropy(outputs, labels, weight=weights)
                        # backward + optimize only if in training phase
                        #print(outputs)
                        self.logits[phase].append(outputs.clone().detach().squeeze().cpu())
                        self.labels[phase].extend(labels)
                        self.preds[phase].extend(preds.detach())

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()
                            self.model.update_sma()
                    
                    # statistics
                    running_loss += loss.item() * size_data
                    running_corrects += torch.sum(torch.tensor(self.preds[phase] == self.labels[phase]))

                if phase == 'train':
                    if scheduler_flag:
                        self.scheduler.step()
                    self.train_loss.append(running_loss / len(train_val_loader[phase].dataset))
                    self.train_acc.append(running_corrects.double() / len(train_val_loader[phase].dataset))
                    print('Train Loss: {:.4f}'.format(running_loss / len(train_val_loader[phase].dataset)), flush=True)
                    #print('Train Acc: {:.4f}'.format(running_corrects.double() / len(train_val_loader[phase].dataset)))
                if phase == 'val':
                    self.val_loss.append(running_loss / len(train_val_loader[phase].dataset))
                    self.val_acc.append(running_corrects.double() / len(train_val_loader[phase].dataset))
                    print('Val Loss: {:.4f}'.format(running_loss / len(train_val_loader[phase].dataset)), flush=True)
                    #print('Val Acc: {:.4f}'.format(running_corrects.double() / len(train_val_loader[phase].dataset)))
                    
                #print(self.logits[phase])
                #self.logits[phase] = self.logits[phase]
                self.labels[phase] = torch.tensor(self.labels[phase])
                self.preds[phase] = torch.tensor(self.preds[phase])
                print(phase + ' Acc: {:.4f}'.format(torch.sum(self.preds[phase] == self.labels[phase])/len(train_val_loader[phase].dataset)), flush = True)
            if phase == 'val' and torch.sum(self.preds[phase] == self.labels[phase])/len(train_val_loader[phase].dataset) > self.best_val_acc:
                self.best_val_acc = torch.sum(self.preds[phase] == self.labels[phase]).cpu()/len(train_val_loader[phase].dataset)
                self.best_epoch = epoch
                self.best_model_wts = copy.deepcopy(self.model.network_sma.state_dict())
                self.best_optimizer =  copy.deepcopy(self.optimizer)
                #torch.save(self.best_model_wts, 'best_model.pth')
        self.model.network_sma.load_state_dict(self.best_model_wts)
        return self.model.network_sma, self.best_optimizer, self.best_val_acc, self.best_epoch
    
    def test(self, test_loader, best_model):
        best_model.to(self.device)
        best_model.eval()
        self.logits = []
        self.slides = []
        self.labels = []
        self.preds = []
        self.attention = []
        self.criterion = nn.CrossEntropyLoss()
        start = time.time() 
        for g, labels, slide in test_loader:
            g = g.to(self.device)
            labels = labels.to(self.device)
            self.slides.append(slide)
            self.labels.extend(labels.cpu())
            if self.model_name in ['GRASP', 'ZoomMIL', 'GRASP_D', 'GRASP_dropout']:
                outputs, attention = best_model(g, g.ndata['x'])
            elif self.model_name in ['H2MIL', 'PatchGCN', 'DGCN', 'HiGT']:
                outputs, attention = best_model(g, g.x)
            self.logits.append(outputs.clone().detach().squeeze().cpu())
            self.attention.append(attention.clone().detach().cpu())
            _, preds = torch.max(outputs, 1)
            self.preds.extend(preds.cpu())
        duration = time.time() - start
        avg_time = float(duration/len(test_loader))
        self.attention = torch.cat(self.attention, 0)
        #self.logits = torch.cat(self.logits, 0)
        self.labels = torch.tensor(self.labels)
        self.preds = torch.tensor(self.preds)
        print('-' * 25)
        print('Test Acc: {:.4f}'.format(torch.sum(self.preds == self.labels)/len(test_loader.dataset)), flush=True)
        output = {'logits': self.logits, 'slides': self.slides, 'labels': self.labels, 'preds': self.preds, 'attention': self.attention, 'average_time': avg_time}
        return output

