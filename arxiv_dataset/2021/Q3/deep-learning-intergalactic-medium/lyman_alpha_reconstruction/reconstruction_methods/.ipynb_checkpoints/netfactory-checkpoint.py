from .conv_net import Convolutional_Neural_Net
from .log_predictor import log_predictor
from ..data_preprocessing import *
import torch
import torchvision
import os

from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (20,10)
torch.set_default_tensor_type('torch.cuda.FloatTensor')


from datetime import datetime
import numpy as np
from numpy import log, sqrt, exp
from scipy.stats import pearsonr


noiseDict = {
    "none": 0,
    "low": 1/10,
    "mid": 1/5,
    "high": 1/2.5
}
    
def spin(tensor, midpoint):
    return torch.cat((tensor[midpoint-256:],tensor[:midpoint-256]),dim=0)
    
def RMSE(pred, act):
    
    pred = torch.Tensor(pred)
    act = torch.Tensor(act)
    result = torch.sqrt(torch.mean((pred-act)**2))
    print(result)
    return result    

class NetFactory():
    def __init__(self, noise, printAll = True):
        self.net = Convolutional_Neural_Net().cuda()
            
        self.printAll = printAll
        self.comment = ""
        if noise == "low":
            self.fluxValidate = fluxValidate + torch.load("lowValidationNoise.pt")
        elif noise == "mid":
            self.fluxValidate = fluxValidate + torch.load("midValidationNoise.pt")
        elif noise == "high":
            self.fluxValidate = fluxValidate + torch.load("highValidationNoise.pt")
        self.noise = noise
        

    
    
    def randomSample(self, num, noise="low", source="train"):
        #function to sample flux sightlines
        if source == "test":
            tSet = tauTest
            fSet = fluxTest
        elif source == "validate":
            tSet = tauValidate
            fSet = fluxValidate
        elif source == "train":
            tSet = tauTrain
            fSet = fluxTrain
        else:
            print(source)
            print("Not valid")



        indices = torch.randint(low=0,high=tSet.shape[0],size=(num,))
        points = torch.randint(low=0,high=512,size=(num,))
        taus = torch.zeros((num))
        fluxs = torch.zeros(num,512)
        taus = tSet[indices,0,points]
        fluxs = fSet[indices, 0]
        points = points-256
        scatterPoints = torch.zeros(num, 512, dtype=torch.long).cuda()
        scatterPoints = torch.add(scatterPoints, torch.arange(512))
        scatterPoints = torch.add(scatterPoints, points.view(num,1))
        scatterPoints[scatterPoints < 0] += 512
        scatterPoints[scatterPoints > 511] -= 512
        fluxs = fluxs.gather(1, scatterPoints)

        noiseGen =  torch.distributions.normal.Normal(0.0, signalRMS*noiseDict[noise], validate_args=None)

        fluxs = fluxs + noiseGen.sample(fluxs.shape)
        return taus, fluxs
    
    
    def validate(self, noise):
        # Calculates validation loss
        loss_func = torch.nn.MSELoss()
        loss = torch.zeros(512)
        self.net.eval()
        with torch.no_grad():
            for midpoint in range(512):
                fluxs = torch.cat((self.fluxValidate[:,:,midpoint-256:],self.fluxValidate[:,:,:midpoint-256]),2)
                taus = tauValidate[:,0,midpoint]
                prediction = self.net(fluxs)
                loss[midpoint] = loss_func(prediction, taus)
        self.net.train()
        return loss.mean()
    
    
    def run(self, epochs, learningRate):
        # Trains neural network
        noise = self.noise
        now = datetime.now()
 
        print("now =", now)
    
        
        loss_func = torch.nn.MSELoss()
        sampler = self.randomSample
        
        if not (os.path.isdir('runs')):
            os.mkdir('runs')
            
        if not (os.path.isdir('CopyStorage')):
            os.mkdir('runs')
        
        pathString = "runs/ConvNN_default_" + str(epochs) + "_" + str(learningRate) + "_" + noise + "_" + \
                   now.strftime('%Y-%m-%d_%H-%M-%S')
        
        writer = SummaryWriter(pathString)
        
        writer.add_text("Net", str(self.net))
        writer.add_text("Epochs", str(epochs))
        writer.add_text("Learning Rate", str(learningRate))
        writer.add_text("Comment", self.comment)
        
        optimizer = torch.optim.Adam( self.net.parameters(), learningRate, weight_decay=0.0005 )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 50, gamma=1)
        
        
        validationLoss = self.validate(noise)
        writer.add_scalar('Loss/validation', validationLoss, 0)
        
        self.results(noise, 0, writer)
            
        for epoch in range(epochs):
            
            
            tauSet,fluxSet = sampler(num=10000, noise=noise, source="train")
            tauSet = tauSet.float().cuda()
            fluxSet = fluxSet.float().cuda()
            tauSet = torch.reshape(tauSet, (-1,1,1))
            fluxSet = torch.reshape(fluxSet, (-1,1,512))
            
            prediction = self.net(fluxSet)
                
            
            loss = loss_func(prediction, tauSet)     # must be (1. nn output, 2. target)
            optimizer.zero_grad()   # clear gradients for next train
            loss.backward()         # backpropagation, compute gradients
            optimizer.step()        # apply gradients
            scheduler.step()        # scheduler decreases learning rate geometrically every n epochs

            
            
            
            writer.add_scalar('Loss/train', loss, epoch)
                        
            
            
            
            if self.printAll:
                if epoch % 100 == 0:
                    print("Epoch = ", epoch)
            if epoch % 1000 == 999:
                hundredth = plt.figure()
                a = torch.flatten(tauSet)
                b = torch.flatten(prediction)
                plt.title("%dth epoch predictions" %epoch)
                plt.plot(a.cpu().detach().numpy(),"r", label="actual")
                plt.plot(b.cpu().detach().numpy(),"g.", label="prediction")
                plt.show()
                writer.add_figure("Every Thousand Epochs/" + str(epoch), hundredth)
                
                validationLoss = self.validate(noise)
                writer.add_scalar('Loss/validation', validationLoss, epoch)
                self.results(noise, epoch, writer)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss
                    }, "CopyStorage/" + pathString[5:] + "Copy" + str(epoch))


        self.results(noise, epoch, writer)
        
        end = datetime.now()
 
        print("end =", end)
        
        return 1
        
    def results(self, noise, epoch, writer):
        # Saves graphs and statistics (loss for neural network and log_predictor) to tensorboard 
        taus = tauValidate[0,0,...].cuda()
        fluxs = torch.zeros((512,1,512)).cuda()
        for i in range(512):
            fluxs[i] = spin(self.fluxValidate[0,0,...],i)
    
        fluxs = fluxs
        
        
        taus = taus.float()
        fluxs = fluxs.float()
        taus = torch.reshape(taus, (-1,1,1))
        fluxs = torch.reshape(fluxs, (-1,1,512))
        prediction = self.net(fluxs)
        prediction = torch.flatten(prediction)
        taus = torch.flatten(taus)
        logPrediction = log_predictor(torch.flatten(self.fluxValidate[0,0,...]))
                                       
        example = plt.figure()
        
        plt.plot(taus.data.cpu().numpy(), "r", label="actual")
        plt.plot(prediction.data.cpu().numpy(), "g", label="prediction")
        plt.plot(logPrediction, "b", label="log and cubic splines")
        plt.legend()
        plt.show()
        writer.add_figure("One Sightline/" + str(epoch), example)
        
        difference = plt.figure()
        plt.plot(prediction.data.cpu().numpy()-taus.data.cpu().numpy(), "g", label="prediction")
        plt.plot(logPrediction-taus.data.cpu().numpy(), "b", label="log and cubic splines")
        plt.legend()
        plt.title("Difference")
        plt.show()
        writer.add_figure("Difference/" + str(epoch), difference)
        
        fracDifference = plt.figure()
        plt.plot((prediction.data.cpu().numpy()-taus.data.cpu().numpy())/taus.data.cpu().numpy(), "g", label="prediction")
        plt.plot((logPrediction-taus.data.cpu().numpy())/taus.data.cpu().numpy(), "b", label="log and cubic splines")
        plt.legend()
        plt.title("Fractional Difference")
        plt.show()
        writer.add_figure("Fractional Difference/" + str(epoch), fracDifference)
        
        predictionList = np.zeros((512*20))
        with torch.no_grad():
            for midpoint in range(512):
                fluxs = torch.cat((self.fluxValidate[0:20,:,midpoint-256:],self.fluxValidate[0:20,:,:midpoint-256]),2)
                prediction = self.net(fluxs)
                predictionList[midpoint::512] = prediction.flatten().data.cpu().numpy()
        self.net.train()
        
        logPList = np.zeros((512*20))
        for i in range(20):
            logP = log_predictor(torch.flatten(self.fluxValidate[i,0]))
            logPList[i*512:(i+1)*512] = logP
            
        
        a = torch.flatten(tauValidate[0:20,0,:]).cpu().numpy()
        b = predictionList
        logP = logPList
        pearsonCoeff = pearsonr(a,b)
        RMSD = RMSE(b, a)
        writer.add_scalar("NeuralNet/Total/RMSD", RMSD, epoch)
        logRMSD = RMSE(logP, a)
        writer.add_scalar("LogPrediction/Total/RMSD", logRMSD, epoch)
        
        print("Pearson's Coefficient is:")
        print(pearsonCoeff)
        writer.add_scalar("NeuralNet/Total/Pearson", pearsonCoeff[0], epoch)
        print("Compared to log:")
        logPearson = pearsonr(a,logP)
        print(logPearson)
        writer.add_scalar("LogPrediction/Total/Pearson", logPearson[0], epoch)
        
        c = np.zeros(a.shape)
        
        newA = np.where(a>2, a, c)
        newB = np.where(a>2, b, c)
        newLogP = np.where(a>2, logP, c)
        newA = newA[np.nonzero(newA)]
        newB = newB[np.nonzero(newB)]
        highA = np.copy(newA)
        highB = np.copy(newB)
        newLogP = newLogP[np.nonzero(newLogP)]
        RMSD = RMSE(newB, newA)
        writer.add_scalar("NeuralNet/High/RMSD", RMSD, epoch)
        print("Pearson's Coefficient for high numbers")
        pearsonCoeff = pearsonr(newA,newB)
        print(pearsonCoeff)
        writer.add_scalar("NeuralNet/High/Pearson", pearsonCoeff[0], epoch)
        print("Compared to log:")
        logPearson = pearsonr(newA,newLogP)
        print(logPearson)
        writer.add_scalar("LogPrediction/High/Pearson", logPearson[0], epoch)
        logRMSD = RMSE(newLogP, newA)
        writer.add_scalar("LogPrediction/High/RMSD", logRMSD, epoch)
        
        
        print("percent high:")
        print(len(newA)/len(a))
        
        newA = np.where(a<2, a, c)
        newB = np.where(a<2, b, c)
        newLogP = np.where(a<2, logP, c)
        newA = newA[np.nonzero(newA)]
        newB = newB[np.nonzero(newB)]
        newLogP = newLogP[np.nonzero(newLogP)]
        RMSD = RMSE(newB, newA)
        writer.add_scalar("NeuralNet/Low/RMSD", RMSD, epoch)
        print("Pearson's Coefficient for low numbers")
        pearsonCoeff = pearsonr(newA,newB)
        print(pearsonCoeff)
        writer.add_scalar("NeuralNet/Low/Pearson", pearsonCoeff[0], epoch)
        print("Compared to log:")
        logPearson = pearsonr(newA,newLogP)
        print(logPearson)
        writer.add_scalar("LogPrediction/Low/Pearson", logPearson[0], epoch)
        logRMSD = RMSE(newLogP, newA)
        writer.add_scalar("LogPrediction/Low/RMSD", logRMSD, epoch)
        
        print("percent low:")
        print(len(newA)/len(a))
        
        line = np.linspace(0,a.max())
        scatter = plt.figure()
        plt.scatter(a,b)
        plt.title("Scatter plot of Prediction vs Tau")
        plt.ylabel("Prediction")
        plt.xlabel("Tau")
        plt.plot(line, line, "k")
        plt.show()
        writer.add_figure("Scatter/" + str(epoch), scatter)