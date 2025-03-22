
# This script contains functions for the encoder CNN CLASS
# This CLASS allows to create a CNN and perform encoding.

###############################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F

# from Models import Model
from Encoder.Models import Model


###############################################################################

class CNN_encoder(Model.Model):
    
    # Attributes:
        
    # - number_of_cameras
    # - image_channels
    # - image_size
    
    # - dim_filters
    # - number_of_convolutions
    # - kernel
    # - stride
    # - layersDimensions
    # - dimensionsAtEndOfConvolutions
    # - flattenedDimensionsAtEndOfConvolutions
    # - number_of_neurons_middle_FC
    # - number_of_neurons_final_FC
    # - DimensionsAfterConcatenation
    
    # - encoderLayers
    # - middleFCs
    # - finalFCs
        
    
    ###########################################################################
    

    # Initialize the network
    def __init__(self, minValuesOutput, maxValuesOutput):
        super(CNN_encoder, self).__init__()
        
        self.rangeValues = maxValuesOutput - minValuesOutput
        self.minValuesOutput   = minValuesOutput
        self.maxValuesOutput   = maxValuesOutput
        
        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=32,
                               kernel_size=(3, 3),
                               padding=1,
                               stride=1)
        
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(in_channels=32,
                               out_channels=64,
                               kernel_size=(3, 3),
                               padding=1,
                               stride=1)
        
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(in_channels=64,
                               out_channels=64,
                               kernel_size=(3, 3),
                               padding=1,
                               stride=1)
        
        self.bn3 = nn.BatchNorm2d(64)
        
        self.conv4 = nn.Conv2d(in_channels=64,
                               out_channels=128,
                               kernel_size=(3, 3),
                               padding=1,
                               stride=1)
        
        self.bn4 = nn.BatchNorm2d(128)
        
        self.linear1 = nn.Linear(7*7*128, 64)
        self.bn5 = nn.BatchNorm1d(64)
        
        self.linear2 = nn.Linear(64, 6)
        self.linear3 = nn.Linear(9, 1) # 6 + 2 roi_info features
        
        self.pool = nn.MaxPool2d((2, 2))
        
        return
    
    

    ###########################################################################
    # Print function
    def print(self):
        
        return
    
    def forward(self, img, roi_info):
        o1 = F.relu(self.conv1(img))
        o1 = self.pool(self.bn1(o1))
        #print(o1.shape)
        
        o2 = F.relu(self.conv2(o1))
        o2 = self.pool(self.bn2(o2))
        #print(o2.shape)
        
        o3 = F.relu(self.conv3(o2))
        o3 = self.pool(self.bn3(o3))
        #print(o3.shape)
        
        o4 = F.relu(self.conv4(o3))
        o4 = self.pool(self.bn4(o4))
        #print(o4.shape)

        # Keep batch dim and flatten
        conv_out = o4.view(-1, 7*7*128)
        
        l1 = F.relu(self.linear1(conv_out))
        l1 = self.bn5(l1)
        
        l2 = F.relu(self.linear2(l1))
        
        # Concat roi_info along with the processed conv features
        concat = torch.cat((l2, roi_info), 1)
        #print(l1[0])

        # Pay attention to the activation here
        out = self.linear3(concat)
        
        return out
    
    ###########################################################################
    # Loss calculation
    
    @staticmethod
    def CalculateMSELoss(predictedValues, realValues):
        
        MSELoss = F.mse_loss(predictedValues, realValues,size_average=False)
        
        return MSELoss
    
    def CalculateOutputValueDenormalized(self, outputValues, batch_size):
        
        # Repeat min -max value (the range) as many times as the number of batches
        # to proceed with the denormalising
        repeated_range = torch.unsqueeze(self.rangeValues, 0)
        repeated_range = repeated_range.repeat(batch_size,1)
        
        outputValuesDenorm = outputValues*repeated_range + self.minValuesOutput
        
        return outputValuesDenorm
    
    def CalculateDenormalizedError(self, predictedValues, realValues, batch_size):
        
        # Repeat min -max value (the range) as many times as the number of batches
        # to proceed with the denormalising
        repeated_range = torch.unsqueeze(self.rangeValues, 0)
        repeated_range = repeated_range.repeat(batch_size,1)
        
        # Denormalize
        predictedValuesNorm = self.CalculateOutputValueDenormalized(predictedValues, batch_size)
        realValuesNorm      = self.CalculateOutputValueDenormalized(realValues, batch_size)
        
        # Mean absolute error
        denormError = torch.mean(torch.mean(torch.abs(predictedValuesNorm - realValuesNorm)))
        
        return denormError, predictedValuesNorm, realValuesNorm

    def AveragePredictions(self, predictedValues):
        return torch.mean(predictedValues)


    