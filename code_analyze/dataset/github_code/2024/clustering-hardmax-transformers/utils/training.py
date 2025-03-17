import torch
from torch import nn
from tqdm import tqdm
from utils.models import encoder, hardmaxDynamics, decoder
import pickle

class Trainer():
    """
    Given an optimizer, we write a training loop for minimizing a loss functional (on the training set)
    Moreover, we validate our results (on the test set).
    We need the following hyperparameters:
        model: the neural network architecture, defined as a nn.Module()
        optim: the optimizer
        criterion: the loss functional
        device: cpu or gpu
        hm_loss: if True, calculates also the loss with the hardmaxModel (every 10 epochs)
        save_dir: directory where the trained model is saved
        grad_clip: upper bound on the value of the gradients, facilitates training
    """
    def __init__(self, model, optim, criterion, device, hm_loss = False, save_dir = None, grad_clip = 5):
        self.model = model
        self.optim = optim
        self.criterion = criterion
        self.hm_loss = hm_loss
        self.device = device
        self.save_dir = save_dir
        self.grad_clip = grad_clip
    
    def train(self, trainloader, testloader, num_epochs, d, save_every = 10):
        """
        Run the training and testing loop.
        The trainloader and testloader include the batched samples in the training and test set, respectively.
        The loop is run num_epochs times. 
        d is the encoder dimension.
        The trained model is saved every save_every = 10 epochs. 
        """
        history = {
            'train_loss': [],
            'train_loss_hm': [],
            'train_acc': [],
            'test_loss': [],
            'test_loss_hm': [],
            'test_acc': [],
            'epochs': num_epochs
        }   
        epochloop = tqdm(range(num_epochs), position=0, desc='Training', leave=True)
        for e in epochloop:
            #################
            # training mode #
            #################
            self.model.train()
            train_loss = 0
            train_loss_hm = 0
            train_acc = 0
            for id, (feature, target) in enumerate(trainloader):
                # add epoch meta info
                epochloop.set_postfix_str(f'Training batch {id}/{len(trainloader)}')
                # move to device
                feature, target = feature.to(self.device), target.to(self.device)
                # reset optimizer
                self.optim.zero_grad()
                # forward pass
                out = self.model(feature)
                # accuracy
                predicted = torch.tensor([1 if i == True else 0 for i in out > 0.5], device=self.device)
                equals = predicted == target
                acc = torch.mean(equals.type(torch.FloatTensor))
                train_acc += acc.item()
                # loss
                loss = self.criterion(out.squeeze(), target.float())
                train_loss += loss.item()
                loss.backward()
                #calculate hardmax loss every 10 epochs
                if self.hm_loss and e % 10 == 0:     
                    # loss of the HardMax Model
                    E = self.model.encoder.weight
                    v = self.model.decoder.weight
                    b = self.model.decoder.bias
                    alpha = self.model.attention.alpha
                    depth = self.model.depth
                    Z0 = encoder(E,feature)
                    Zf = hardmaxDynamics(alpha,depth,Z0)
                    pred = decoder(v,b,Zf)
                    pred = torch.from_numpy(pred).squeeze()
                    loss_hm = self.criterion(pred.float(), target.float())
                    train_loss_hm += loss_hm.item()
                # clip gradient
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                # update optimizer
                self.optim.step()
                # Restrict \alpha to stay positive
                self.model.attention.alpha.clamp(0)
                # free some memory
                del feature, target, predicted
            history['train_loss'].append(train_loss / len(trainloader))
            history['train_acc'].append(train_acc / len(trainloader))
            if self.hm_loss:
                history['train_loss_hm'].append(train_loss_hm / len(trainloader))
            ####################
            # test mode #
            ####################
            self.model.eval()
            test_loss = 0
            test_loss_hm = 0
            test_acc = 0
            with torch.no_grad():
                for id, (feature, target) in enumerate(testloader):
                    # add epoch meta info
                    epochloop.set_postfix_str(f'Test batch {id}/{len(testloader)}')
                    # move to device
                    feature, target = feature.to(self.device), target.to(self.device)
                    # reset optimizer
                    self.optim.zero_grad()
                    # forward pass
                    out = self.model(feature)
                    # accuracy
                    predicted = torch.tensor([1 if i == True else 0 for i in out > 0.5], device=self.device)
                    equals = predicted == target
                    acc = torch.mean(equals.type(torch.FloatTensor))
                    test_acc += acc.item()
                    # loss
                    loss = self.criterion(out.squeeze(), target.float())
                    test_loss += loss.item()
                    #calculate hardmax loss every 10 epochs
                    if self.hm_loss and e % 10 == 0:
                        # loss of the HardMax Model
                        E = self.model.encoder.weight
                        v = self.model.decoder.weight
                        b = self.model.decoder.bias
                        alpha = self.model.attention.alpha
                        depth = self.model.depth
                        Z0 = encoder(E,feature)
                        Zf = hardmaxDynamics(alpha,depth,Z0)
                        pred = decoder(v,b,Zf)
                        pred = torch.from_numpy(pred).squeeze()
                        loss_hm = self.criterion(pred.float(), target.float())
                        test_loss_hm += loss_hm.item()
                    # free some memory
                    del feature, target, predicted
                history['test_loss'].append(test_loss / len(testloader))
                history['test_acc'].append(test_acc / len(testloader))
            if self.hm_loss:
                history['test_loss_hm'].append(test_loss_hm / len(testloader))
            # reset model mode
            self.model.train()
            # add epoch meta info
            epochloop.set_postfix_str(f'Test Loss: {test_loss / len(testloader):.3f} | Test Acc: {test_acc / len(testloader):.3f}')
            # print epoch
            epochloop.write(f'Epoch {e+1}/{num_epochs} | Train Loss: {train_loss / len(trainloader):.3f} Train Acc: {train_acc / len(trainloader):.3f} | Test Loss: {test_loss / len(testloader):.3f} Test Acc: {test_acc / len(testloader):.3f}')
            epochloop.update()
            #save model periodically
            if e>0 and e % save_every == 0:
                torch.save(self.model, self.save_dir + '/model_d='  + str(d) + '_epoch=' + str(e) + '.pth')
            #save model at final iteration
            if e == num_epochs - 1 and e % save_every != 0: 
                torch.save(self.model, self.save_dir + '/model_d='  + str(d) + '_epoch=' + str(e) + '.pth')
        #save history data
        with open(self.save_dir + '/history_d=' + str(d) + '.pkl', 'wb') as f:
            pickle.dump(history, f)