import torch.nn as nn
import torch
from . import NetUtils
from util.Config import obj2dic

'''
    This file contains a template for all input/output and networks
'''
class TrainerPart(nn.Module):
    def __init__(self,network,loss):
        super(TrainerPart,self).__init__()
        self.network=network
        self.loss=loss
    
    def forward(self,x,returnLoss=False):
        input = x["input"]["img"]
        if returnLoss:
            target = x["target"]["num"]
            y = self.network(input)
            # print('y',y)
            loss = self.loss(y,target)
            return y, loss
        else:
            return self.network(input)

# parent class
class Trainer(nn.Module):

    # set netowrk and lossfunc
    def _createNetwork(self):
        pass

    def __init__(self,param):
        super(Trainer,self).__init__()
        self.param=param
        self._createNetwork()
        self.moduleNet = TrainerPart(self.network,self.lossfunc)


    def setConfig(self,config,device):
        self.config=config
        self.device=device
        self.opt = NetUtils.getOpt(self.network.parameters(),config)
        self.learnScheduler = NetUtils.getSch(self.opt, config)
        if config.cuda.parallel:
            self.network = nn.DataParallel(self.moduleNet,device_ids=config.cuda.use_gpu)
        self.network = self.network.to(device)
        self.moduleNet = self.moduleNet.to(device)
        self.lossfunc = self.lossfunc.to(device)
        self.max_norm = self.config.trainParam.clipNorm

    def _convert(self,x,containTarget=False):
        x["input"]["img"] = x["input"]["img"].to(self.device).float()
        if containTarget:
            #x["target"]["ratio"] = x["target"]["ratio"].to(self.device)
            #x["target"]["label"] = x["target"]["label"].to(self.device)
            x["target"]["num"] = x["target"]["num"].to(self.device).float()
        return x

    def forward(self,x, returnLoss=False):
        return self.moduleNet(self._convert(x,returnLoss),returnLoss)

    def getLR(self):
        return self.opt.param_groups[0]['lr']

    def trainData(self,x):

        self.opt.zero_grad()
        y, loss = self.moduleNet(self._convert(x,True),True)

        loss.backward()
        if self.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_norm)
        self.opt.step()

        loss = {"loss":loss}

        return loss, y

    def onEpochComplete(self,epochIndex):
        self.learnScheduler.step()

    @torch.no_grad()
    def test(self,x):
        y, loss = self.moduleNet(self._convert(x,True),True)

        v = {}
        v["loss"] = {"loss":loss}
        v["result"] = {'pred_n':y}
        #v["result"]["label"]=y["pred_label"]
        return v