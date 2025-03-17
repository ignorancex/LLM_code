
# This script contains functions for the encoder CNN CLASS
# This CLASS allows to create a CNN and perform encoding.

###############################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.nn.init as init

import matplotlib.pyplot as plt
from torchvision.utils import make_grid

import numpy as np

from torchsummary import summary
from torchvision.utils import save_image

# from Models import SummaryHolder                   as SH
# from Models import SummaryHolderLossesAcrossEpochs as SHLAE
# from Models import Model
# from Models import Exceptions
from Encoder.Models import SummaryHolder                   as SH
from Encoder.Models import SummaryHolderLossesAcrossEpochs as SHLAE
from Encoder.Models import Model
from Encoder.Models import Exceptions
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
    def __init__(self, image_size, dim_filters, kernel, stride,
                 number_of_neurons_middle_FC, number_of_neurons_final_FC, 
                 number_of_cameras, minValuesOutput, maxValuesOutput):
        super(CNN_encoder, self).__init__()
        
        # ---------------------------------------------------------------------
        # From indications set in the configuration file
        self.number_of_cameras = number_of_cameras
        self.image_channels    = 3*self.number_of_cameras # 1
        self.image_size        = image_size
        self.minValuesOutput   = minValuesOutput
        self.maxValuesOutput   = maxValuesOutput
        self.rangeValues       = maxValuesOutput - minValuesOutput
        
        # ---------------------------------------------------------------------
        # EXTRACTION OF DIMENSIONS/STRUCTURE
        # Extractions for Convolutional layers
        self.ExtractDimFilters(dim_filters) # extract dimensions of convolutional filters
        self.FindNumberOfConvolutions() # find number of convolutional layers
        self.ExtractDimKernels(kernel) # extract 'kernel', i.e., dimensions of kernels of all conv layers
        self.ExtractStrides(stride) # extract 'stride', i.e., strides of all conv layers
        self.FindEncoderDimensions() # extract 'layersDimensions', i.e., dimensions of inputs at end of all conv layers
        self.dimensionsAtEndOfConvolutions = self.layersDimensions[self.number_of_convolutions, :]
        
        self.flattenedDimensionsAtEndOfConvolutions = \
            int(self.dimensionsAtEndOfConvolutions[0]*self.dimensionsAtEndOfConvolutions[1]*self.dim_filters[-1])
        
        # Extractions for middle fully connected layers
        self.ExtractDimMiddleFCLayers(number_of_neurons_middle_FC)
        # Extractions for final fully connected layers
        self.ExtractDimFinalFCLayers(number_of_neurons_final_FC)
        
        # Find dimension after concatenation
        self.DimensionsAfterConcatenation = self.number_of_neurons_middle_FC[-1] + 3*self.number_of_cameras
        
        # ---------------------------------------------------------------------
        # CHECKING STRUCTURE
        self.CheckForExceptionsOnCNNStructure()
        
        # ---------------------------------------------------------------------
        # BUILDING STRUCTURE
        self.BuildCNN()
        
        #self.apply(weight_init)
        
        return
    
    
    ###########################################################################
    # Extractions of dimensions/structure
    
    @staticmethod
    def SeparateNumbersInString(string):
        
        separatedNumbers = [int(f) for f in string.split(',')]
        separatedNumbers = np.asarray(separatedNumbers)
        
        return separatedNumbers
    
    ####
    # Extractions related to convolutions
    
    def ExtractDimFilters(self, dim_filters):
        
        self.dim_filters = CNN_encoder.SeparateNumbersInString(dim_filters)
        
        return
    
    def FindNumberOfConvolutions(self):
        
        self.number_of_convolutions = len(self.dim_filters)
        
        return
    
    def ExtractDimKernels(self, kernel):
        
        self.kernel      = CNN_encoder.SeparateNumbersInString(kernel)
        
        if len(self.kernel) == 1:
            
            self.kernel = self.kernel.repeat(self.number_of_convolutions)
        
        return
    
    def ExtractStrides(self, stride):
        
        self.stride      = CNN_encoder.SeparateNumbersInString(stride)
        
        if len(self.stride) == 1:
            
            self.stride = self.stride.repeat(self.number_of_convolutions)
        
        return
    
    # Function to extract the dimensions of each layer of the ENCODER
    # using the formula:
    # U = (I - K + 2*P)/2 + 1
    # where:
    # - U is the output dimension (= layersDimensions[i, :])
    # - I is the input dimension (curr_size)
    # - K is the kernel dimension (self.kernel)
    # - P is the padding (set to 0)
    # - S is the stride (self.stride)
    # Each of them has 2 parts, for X and Y
    # Input: VAE and its parameters set in Config file.
    # Output: implicitly, dimensions of each layer of the encoder (layersDimensions)
    def FindEncoderDimensions(self):
        
        # Array to contain the dimensions of the encoder
        self.layersDimensions = np.zeros((self.number_of_convolutions + 1, 2))
        
        # We take as first input dimension the dimensions of the images
        curr_size = np.asarray(self.image_size)
        # Input dimension
        self.layersDimensions[0, :] = curr_size
        
        # Finding all the layer dimensions
        for i in range(self.number_of_convolutions):
            
            # Find output dimension
            # U = (I - K + 2*P)/2 + 1,    with P = 0
            curr_size                   = np.floor((curr_size - self.kernel[i])/self.stride[i] + 1)
            
            # Input dimension
            self.layersDimensions[i+1, :] = curr_size
            
            
        return
    
    #####
    # Extractions related to middle fully connected layers
    
    def ExtractDimMiddleFCLayers(self, number_of_neurons_middle_FC):
        
        self.number_of_neurons_middle_FC = CNN_encoder.SeparateNumbersInString(number_of_neurons_middle_FC)
        
        return
    
    #####
    # Extractions related to final fully connected layers
    
    def ExtractDimFinalFCLayers(self, number_of_neurons_final_FC):
        
        self.number_of_neurons_final_FC = CNN_encoder.SeparateNumbersInString(number_of_neurons_final_FC)
        
        return

    #####
    # Checking for wrong CNN structure
    def CheckForExceptionsOnCNNStructure(self):
        
        # Check that the first two dimensions at the end of convolutions did not go below zero
        if self.dimensionsAtEndOfConvolutions[0] < 1 or self.dimensionsAtEndOfConvolutions[1] < 1:
            raise Exceptions.TwoManyConvolutionalLayersException(self.dimensionsAtEndOfConvolutions[0], 
                                                                 self.dimensionsAtEndOfConvolutions[1])
        # Check that flattened dimensions at the end of convolutions is not 
        # smaller than neurons at end of first FC middle layers
        if self.flattenedDimensionsAtEndOfConvolutions < self.number_of_neurons_middle_FC[0]:
            raise Exceptions.FlattenedNumberOfNeuronsLessThanFCMiddleLayersException(self.flattenedDimensionsAtEndOfConvolutions, 
                                                                 self.number_of_neurons_middle_FC[0])
            
        # Check that the dimension after the concatenation with depth and image ratio
        # is smaller than neurons at end of first FC final layers
        if self.DimensionsAfterConcatenation < self.number_of_neurons_final_FC[0]:
            raise Exceptions.FlattenedNumberOfNeuronsLessThanFCMiddleLayersException(self.DimensionsAfterConcatenation, 
                                                                 self.number_of_neurons_final_FC[0])
        return
      
    
    ###########################################################################
    # Functions to BUILD the encoder
            
    # Function to build the ENCODER CONVOLUTIONAL LAYERS
    # The layers are built by using:
    # - the same stride defined in self.stride
    # - the kernel defined in self.kernel
    # - dimension of filters as defined in self.dim_filters
    # - zero padding
    # - putting a LeakyReLU after each covolutional layer
    # Implicit output: self.encoder -> encoder function containing all the 
    #                  encoder layers.
    def DefineEncoder(self):
        
        # Where to put list of layers
        encoderLayers = []
        
        # ENCODER LAYERS
        for i in range(self.number_of_convolutions):
            
            # Number of input channels of current layer
            if i == 0:
                # If this is the first layer, the number of input channels
                # is the number of image channels...
                input_channels = self.image_channels
            else:
                # ... otherwise, we use the number of filters per layer
                input_channels = self.dim_filters[i-1]
            
            # Number of output channels of current layer
            output_channels    = self.dim_filters[i]
            
            # Define the current layer
            
            currentLayer       = nn.Sequential(*[
                                                 nn.Conv2d(in_channels  = input_channels,
                                                           out_channels = output_channels,
                                                           kernel_size  = (self.kernel[i], self.kernel[i]),
                                                           padding      =  0, 
                                                           stride       = (self.stride[i], self.stride[i])),
                                                 nn.BatchNorm2d(output_channels),
                                                 nn.Dropout(p = 0.2),
                                                 nn.ReLU()
                                                 #nn.MaxPool2d()
                                                 ])
            '''
            
            currentLayer       = nn.Sequential(*[
                                                 nn.Conv2d(in_channels  = input_channels,
                                                           out_channels = output_channels,
                                                           kernel_size  = (self.kernel[i], self.kernel[i]),
                                                           padding      =  0, 
                                                           stride       = (1, 1)),
                                                 nn.BatchNorm2d(output_channels),
                                                 nn.Dropout(p = 0.3),
                                                 nn.ReLU(),
                                                 nn.MaxPool2d(kernel_size=3)
                                                 ])
            '''
            
            # Add layer to the list
            encoderLayers.append(currentLayer)
            
        # Final encoder
        self.encoder = nn.Sequential(*encoderLayers)
        
        #self.encoder.apply(weight_init)
            
        return
       
    # Function to build the Middle Fully Connected Layers
    def DefineMiddleFCLayers(self):
        
        middleFCs = []
        
        numberOfLayersMiddleFC = len(self.number_of_neurons_middle_FC)
        
        for i in range(numberOfLayersMiddleFC):
            
            if i == 0:
                inputDimensions = self.flattenedDimensionsAtEndOfConvolutions
            else:
                inputDimensions = self.number_of_neurons_middle_FC[i-1]
                    
            outputDimensions    = self.number_of_neurons_middle_FC[i]
            
            
            if i == numberOfLayersMiddleFC - 1:
            
                # Definition of linear layer
                currentLayer  = nn.Sequential(*[
                                                nn.Linear(inputDimensions, outputDimensions),
                                                nn.BatchNorm1d(outputDimensions),
                                                nn.Dropout(p = 0.3),
                                                nn.ReLU()
                                                ])
                
            else:
                
                currentLayer  = nn.Sequential(*[
                                                nn.Linear(inputDimensions, outputDimensions),
                                                nn.BatchNorm1d(outputDimensions),
                                                nn.Dropout(p = 0.3),
                                                nn.ReLU()
                                                ])
            
            middleFCs.append(currentLayer)
        
        self.middleFCs = nn.Sequential(*middleFCs)
        
        #self.middleFCs.apply(weight_init)
        
        return
    
    # Function to build the Final Fully Connected Layers
    def DefineFinalFCLayers(self):
        
        finalFCs = []
        
        numberOfLayersFinalFC = len(self.number_of_neurons_final_FC)
        
        for i in range(numberOfLayersFinalFC):
            
            if i == 0:
                inputDimensions = self.DimensionsAfterConcatenation
            else:
                inputDimensions = self.number_of_neurons_final_FC[i-1]
                    
            outputDimensions    = self.number_of_neurons_final_FC[i]
            
            if i == numberOfLayersFinalFC - 1:
            
                # Definition of linear layer
                currentLayer  = nn.Sequential(*[
                                                nn.Linear(inputDimensions, outputDimensions),
                                                nn.BatchNorm1d(outputDimensions),
                                                nn.Dropout(p = 0.3),
                                                nn.Sigmoid()
                                                ])
                
            else:
                
                currentLayer  = nn.Sequential(*[
                                                nn.Linear(inputDimensions, outputDimensions),
                                                nn.BatchNorm1d(outputDimensions),
                                                nn.Dropout(p = 0.3),
                                                nn.ReLU()
                                                ])
            
            finalFCs.append(currentLayer)
        
        self.finalFCs = nn.Sequential(*finalFCs)
        
        #self.finalFCs.apply(weight_init)
        
        return
    
    # Build the VAE with the given information
    def BuildCNN(self):
        
        self.DefineEncoder()
        self.DefineMiddleFCLayers()
        self.DefineFinalFCLayers()
        
        return


    ###########################################################################
    # Print function
    def print(self):

        print('Number of image channels:')
        print(self.image_channels)
        print('Filter dimensions:')
        print(self.dim_filters)
        print('Number of filters:')
        print(self.number_of_convolutions)
        print('Base dimensions of kernels:')
        print(self.kernel)
        print('Stride:')
        print(self.stride)
        
        print('Dimensions at input and output of convolutions (except channel):')
        print(self.layersDimensions)
        
        print('Dimension at the end of the convolutions')
        print(self.dimensionsAtEndOfConvolutions)
        
        print('Flattened dimensions at the end of convolutions')
        print(self.flattenedDimensionsAtEndOfConvolutions)
        
        print('Number of neurons in the middle FC layers')
        print(self.number_of_neurons_middle_FC)
        
        print('Dimensions after concatenation')
        print(self.DimensionsAfterConcatenation)
        
        print('Number of neurons in the final FC layers')
        print(self.number_of_neurons_final_FC)
        
        return
    
    ###########################################################################
    # Function for forward call
                                              
    # Flatten the features along a single dimension (plus batch size)
    # OUTPUT of Flatten -> [batch_size; remaining_dimensions]
    # To use at end of ENCODER
    def Flatten(self, input):
        
        return input.view(input.size(0),-1)

    # Encode image to latent space
    # INPUTS:
    # - images
    # - depthAndImageRatios
    # OUTPUTS:
    # - outputMiddleFCLayers
    def Encode(self, images, depthAndImageRatios):
        
        # Go through convolutional layers
        outputConvolutionalLayers          = self.encoder(images)   
        
        # Flatten
        flattenedOutputConvolutionalLayers = self.Flatten(outputConvolutionalLayers)
        # Go through Middle FC layers
        outputMiddleFCLayers               = self.middleFCs(flattenedOutputConvolutionalLayers)
        
        # Concatenate output of Middle FC layers with depth and imageRatios
        concatenation = torch.cat((outputMiddleFCLayers, depthAndImageRatios), dim = 1)
        # Go through Final FC layers
        outputFinalFCLayers                = self.finalFCs(concatenation)
        
        return outputFinalFCLayers
      
    # Forward call of CNN encoder
    # INPUTS:
    # - images: (batch size * number_of_inputs_camera * size_x * size_y)
    #      number_of_inputs_camera = 3*number_of_cameras
    #      size_x = size_y
    # - depthAndImageRatios: (batch size * number_of_inputs_ratios)
    #      number_of_inputs_ratios = 3*number_of_cameras
    # OUTPUTS:
    # - predictedValues
    def forward(self, images, depthAndImageRatios):
        
        # Encoding
        predictedValues = self.Encode(images, depthAndImageRatios)

        return predictedValues
    
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


    