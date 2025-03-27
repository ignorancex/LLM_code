
# This is the File for training the CNN encoder.

###############################################################################
# Importing necessary libraries

import os
import torch
from Models import ConfigurationHolder as CH
from Models import SummaryHolder       as SH
from Models import SummaryHolderLossesAcrossEpochs       as SHLAE
from Models import DataExtractor as DE
from Models import PlotGraphs_utils as PG

from Models import GradientFlowChecker as GFC 

from PIL import Image,ImageDraw

import numpy as np

import matplotlib.pyplot as plt
from torchvision import transforms

###############################################################################
CNN_selection = 1 # 0 = original one, 1 = one with pooling

if CNN_selection == 0:
    from Models import CNN_encoder
elif CNN_selection == 1:
    from Models import CNN_encoder_pooling as CNN_encoder

###############################################################################
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

###############################################################################
# This contains only those value of which we are interested in the MEAN over all epochs
summaryNamesAllEpochs    = ['MSE_loss', 'denorm_err', 'learning_rates']
# This instead, contains values that are only epoch specific
summaryNamesCurrentEpoch = summaryNamesAllEpochs + ['predictions']

###############################################################################
# Retrieving path to current folder and to configuration file.
# Configuration file contains all the information regarding parameters.

# Current folder
pathCurrentFolder     = os.path.dirname(os.path.abspath(__file__))
# Path to configuration file (containing all paramters of nets)
pathConfigurationFile = pathCurrentFolder + '/Configuration/ConfigurationFile.json'
configHolder          = CH.ConfigurationHolder() 
configHolder.LoadConfigFromJSONFile(pathConfigurationFile)
# Path to locations file (containing the paths to the inputs and if to use GPU/CPU)
pathLocationsFile     = pathCurrentFolder + '/Configuration/LocationsFile.json'
locationsHolder       = CH.ConfigurationHolder() 
locationsHolder.LoadConfigFromJSONFile(pathLocationsFile)

# Path to output Folder
outputFolder = locationsHolder.config['output_folder']

# Take out the path to the annotations datasets
pathAnnotationsFileTraining   = locationsHolder.config['annotations_folder_training']
pathAnnotationsFileValidation = locationsHolder.config['annotations_folder_validation']
# Create a configuration holder for the annotations
annotationsHolderTraining     = CH.ConfigurationHolder() 
annotationsHolderTraining.LoadConfigFromJSONFile(pathAnnotationsFileTraining)
annotationsHolderValidation   = CH.ConfigurationHolder() 
annotationsHolderValidation.LoadConfigFromJSONFile(pathAnnotationsFileValidation)

# Take out the path to the images
pathImagesFileTraining   = locationsHolder.config['images_folder_training']
pathImagesFileValidation = locationsHolder.config['images_folder_validation']

###############################################################################
# Creating data holders for training and validation data
# Training
dataExtractorTraining   = DE.DataExtractor(annotationsHolderTraining, pathImagesFileTraining, configHolder)
# Take the mins and maxs of training and pass them to normalize the validation
# trainingMinsAndMaxs = dataExtractorTraining.minAverageDistance, dataExtractorTraining.maxAverageDistance, \
#     dataExtractorTraining.minWidthTop, dataExtractorTraining.maxWidthTop, \
#     dataExtractorTraining.minWidthBottom, dataExtractorTraining.maxWidthBottom
trainingMinsAndMaxs = dataExtractorTraining.minAverageDistance, dataExtractorTraining.maxAverageDistance, \
    dataExtractorTraining.minRatioWidth, dataExtractorTraining.maxRatioWidth, \
        dataExtractorTraining.minRatioHeight, dataExtractorTraining.maxRatioHeight, \
            dataExtractorTraining.minMass, dataExtractorTraining.maxMass

# Validation/Testing
dataExtractorValidation = DE.DataExtractor(annotationsHolderValidation, pathImagesFileValidation, configHolder, trainingMinsAndMaxs)

# Plot one batch of images just to check how they are
oneImageBatchFromDataExtractorTraining = dataExtractorTraining.inputImagesBatched[0,:]
# for i in range(oneImageBatchFromDataExtractorTraining.shape[0]):
#
#     plt.imshow(oneImageBatchFromDataExtractorTraining[i,:].permute(1,2,0))
#     plt.savefig(outputFolder + '/random_IMG_' + str(i))


# Checking correspondance between images and annotations.

# chosen_batch = 5
# chosen_value = 0
#
# inputImagesBatched        = dataExtractorTraining.inputImagesBatched[chosen_batch, chosen_value]
# inputSingleValuesBatched  = dataExtractorTraining.inputSingleValuesBatched[chosen_batch, chosen_value]
# outputSingleValuesBatched = dataExtractorTraining.outputSingleValuesBatched[chosen_batch, chosen_value]
#
# print('Inputs: ' + str(inputSingleValuesBatched))
# print('Outputs: ' + str(outputSingleValuesBatched))

# plt.imshow( inputImagesBatched.permute(1, 2, 0) )



###############################################################################
    
# Initialize CNN and print it
if CNN_selection == 0:
    CNN = CNN_encoder.CNN_encoder(image_size     = configHolder.config['x_size'], 
                                  dim_filters    = configHolder.config['dim_filters'],
                                  kernel         = configHolder.config['kernels_size'],
                                  stride         = configHolder.config['stride'], 
                                  number_of_neurons_middle_FC = configHolder.config['number_of_neurons_middle_FC'],
                                  number_of_neurons_final_FC  = configHolder.config['number_of_neurons_final_FC'], 
                                  number_of_cameras           = configHolder.config['number_of_cameras'],
                                  minValuesOutput      = dataExtractorTraining.minValuesOutput,
                                  maxValuesOutput      = dataExtractorTraining.maxValuesOutput)
elif CNN_selection == 1:
    CNN = CNN_encoder.CNN_encoder(minValuesOutput      = dataExtractorTraining.minValuesOutput,
                                  maxValuesOutput      = dataExtractorTraining.maxValuesOutput)
CNN.print()
    
###############################################################################
# Training loop

learningRate = configHolder.config['learning_rate']

# Summary of training, for all epochs
summaryTrainingAllEpochs    = SHLAE.SummaryHolderLossesAcrossEpochs(summaryNamesAllEpochs)
# Summary of validation, for all epochs
summaryValidationAllEpochs  = SHLAE.SummaryHolderLossesAcrossEpochs(summaryNamesAllEpochs)

for n in range(configHolder.config['epochs']):
    
    ### TRAINING
    
    CNN.train()
    
    print('Epoch number ' + str(n))
    
    # Summary of training losses and values, for current epoch
    summaryCurrentEpochTraining = SH.SummaryHolder(summaryNamesCurrentEpoch)
    
    # Modify learning rate
    if n > 0:
        globalStep   = n
        learningRate = learningRate*np.power(configHolder.config['decay_rate'], 
                                             globalStep/configHolder.config['decay_steps'])
    # Put the current learning rate in the summary
    summaryCurrentEpochTraining.AppendValueInSummary('learning_rates', learningRate)  
    
    list_of_names_training_parameters = list(CNN.named_parameters())
    
    # Looping over the number of batches
    for i in range(dataExtractorTraining.numberOfBatches):
        
        # Take the input for the current batch
        currentInputImagesBatch              = dataExtractorTraining.inputImagesBatched[i,:,:,:,:]
        currentInputSingleValuesBatch        = dataExtractorTraining.inputSingleValuesBatched[i,:,:]
        # Take the output for the current batch
        currentOutputSingleValuesImagesBatch = dataExtractorTraining.outputSingleValuesBatched[i,:,:]
        
        # Define the optimizer
        optimizer = torch.optim.Adam(CNN.parameters(), lr =learningRate, 
                                     weight_decay = configHolder.config['weight_decay'])
        
        # Call to the CNN encoder
        predictedValuesBatch = CNN(currentInputImagesBatch, currentInputSingleValuesBatch)
        
        # Calculate the loss of the CNN
        loss = CNN_encoder.CNN_encoder.CalculateMSELoss(predictedValuesBatch, currentOutputSingleValuesImagesBatch)
        
        # Denormalize predictions
        predictedValuesBatchDenorm = CNN.CalculateOutputValueDenormalized(predictedValuesBatch, 
                                                                          dataExtractorTraining.batch_size)
        
        # Denormalized error (just for plotting and getting real range estimation)
        denormError, predictedValuesDenorm, realValuesDenorm = CNN.CalculateDenormalizedError(predictedValuesBatch, 
                                                     currentOutputSingleValuesImagesBatch, 
                                                     dataExtractorTraining.batch_size)
        
        # Printing all the predictions, with all the information given above
        # if not os.path.exists(outputFolder + "/TRAIN/"):
        #      os.makedirs(outputFolder + "/TRAIN/")
        # filePathSaveSingleImagesWithPreds = outputFolder + "/TRAIN/" + "_batch_" + str(i)
        # PG.PrintImagesWithInputsAndPredictions(currentInputImagesBatch, currentInputSingleValuesBatch,
        #                                 realValuesDenorm, predictedValuesDenorm,
        #                                 currentOutputSingleValuesImagesBatch, predictedValuesBatch,
        #                                 filePathSaveSingleImagesWithPreds)

        # Optimize 
        optimizer.zero_grad()
        loss.backward()       
        
        # GFC.plot_and_save_grad_flow(named_parameters = list_of_names_training_parameters,
        #                             fileName         = outputFolder + '/TRAIN/GRADIENT_KF_' + 'epoch_' + str(n) + '_batch_' + str(i) +'.png')
        
        torch.nn.utils.clip_grad_norm_(CNN.parameters(), configHolder.config['max_grad_norm'])
        optimizer.step()
        
        # Append loss in summary
        summaryCurrentEpochTraining.AppendValueInSummary('MSE_loss', loss.cpu().detach().numpy())
        summaryCurrentEpochTraining.AppendValueInSummary('denorm_err', denormError.cpu().detach().numpy())
        
        # Append prediction in summary
        summaryCurrentEpochTraining.AppendValueInSummary('predictions', predictedValuesBatchDenorm.cpu().detach().numpy())
        
        print('Loss of Training batch: ' + str(loss.item()))
        print('Denorm error of Training batch: ' + str(denormError.item()))
        
        del currentInputImagesBatch, currentInputSingleValuesBatch, currentOutputSingleValuesImagesBatch
        del predictedValuesBatch, predictedValuesBatchDenorm
        del predictedValuesDenorm, realValuesDenorm
        del loss, denormError
        
        if device.type == "cuda":
            torch.cuda.empty_cache()  
        
    ###########################################################################
    # Handle the losses over TRAINING epochs
    # Add the mean of the losses of the current epoch to the overall summary
    summaryTrainingAllEpochs.AppendToOverallSummaryMeanValuesOfCurrentEpochSummary(summaryCurrentEpochTraining)
    # Plot losses
    summaryTrainingAllEpochs.PlotValuesInSummaryAcrossTime(outputFolder = outputFolder, filePrefix = 'TRAIN_PLOT_')
    # Save losses to matlab
    summaryTrainingAllEpochs.BringValuesToMatlab(outputFolder = outputFolder, filePrefix = 'TRAIN_')
    
    ###########################################################################
    # Save to matlab the values over the current epoch
    summaryCurrentEpochTraining.BringValuesToMatlab(outputFolder = outputFolder, 
                                                    dataStructure = 2, batchSize = configHolder.config['batch_size'], 
                                                    filePrefix = 'TRAIN_currEpoch')
    
    ###########################################################################          
    # Save the models
    # torch.save(CNN.state_dict(), outputFolder + '/CNN.torch')
    torch.save(CNN.state_dict(), os.path.join(outputFolder, 'CNN_{}.torch'.format(n)))
        
        
    ### VALIDATION
    
    with torch.no_grad():
        
        CNN.eval()
        
        # Summary of training losses and values, for current epoch
        summaryCurrentEpochValidation = SH.SummaryHolder(summaryNamesCurrentEpoch)
        
        # Looping over the number of batches
        for i in range(dataExtractorValidation.numberOfBatches):
            
            # Take the input for the current batch
            currentInputImagesBatch              = dataExtractorValidation.inputImagesBatched[i,:,:,:,:]
            currentInputSingleValuesBatch        = dataExtractorValidation.inputSingleValuesBatched[i,:,:]
            # Take the output for the current batch
            currentOutputSingleValuesImagesBatch = dataExtractorValidation.outputSingleValuesBatched[i,:,:]
            
            # Call to the CNN encoder
            predictedValuesBatch = CNN(currentInputImagesBatch, currentInputSingleValuesBatch)
            
            # Calculate the loss of the CNN
            loss = CNN_encoder.CNN_encoder.CalculateMSELoss(predictedValuesBatch, currentOutputSingleValuesImagesBatch)
            
            # Denormalize predictions
            predictedValuesBatchDenorm = CNN.CalculateOutputValueDenormalized(predictedValuesBatch, 
                                                                              dataExtractorValidation.batch_size)
            
            # Denormalized error (just for plotting and getting real range estimation)
            denormError, predictedValuesDenorm, realValuesDenorm = CNN.CalculateDenormalizedError(predictedValuesBatch, currentOutputSingleValuesImagesBatch, 
                                                         dataExtractorValidation.batch_size)
            
            # Printing all the predictions, with all the information given above
            # if not os.path.exists(outputFolder + "/VAL/"):
            #      os.makedirs(outputFolder + "/VAL/")
            # filePathSaveSingleImagesWithPreds = outputFolder + "/VAL/" + "_batch_" + str(i)
            # PG.PrintImagesWithInputsAndPredictions(currentInputImagesBatch, currentInputSingleValuesBatch,
            #                                 realValuesDenorm, predictedValuesDenorm,
            #                                 currentOutputSingleValuesImagesBatch, predictedValuesBatch,
            #                                 filePathSaveSingleImagesWithPreds)
            
            # Append loss in summary
            summaryCurrentEpochValidation.AppendValueInSummary('MSE_loss', loss.cpu().detach().numpy())
            summaryCurrentEpochValidation.AppendValueInSummary('denorm_err', denormError.cpu().detach().numpy())
            
            # Append prediction in summary
            summaryCurrentEpochValidation.AppendValueInSummary('predictions', predictedValuesBatchDenorm.cpu().detach().numpy())
            
            print('Loss of Validation batch: ' + str(loss.item()))
            print('Denorm error of Validation batch: ' + str(denormError.item()))
            
            del currentInputImagesBatch, currentInputSingleValuesBatch, currentOutputSingleValuesImagesBatch
            del predictedValuesBatch, predictedValuesBatchDenorm
            del predictedValuesDenorm, realValuesDenorm
            del loss, denormError
            
            if device.type == "cuda":
                torch.cuda.empty_cache()  
            
        ###########################################################################
        # Handle the losses over Validation epochs
        # Add the mean of the losses of the current epoch to the overall summary
        summaryValidationAllEpochs.AppendToOverallSummaryMeanValuesOfCurrentEpochSummary(summaryCurrentEpochValidation)
        # Plot losses
        summaryValidationAllEpochs.PlotValuesInSummaryAcrossTime(outputFolder = outputFolder,filePrefix = 'VAL_PLOT_')
        # Save losses to matlab
        summaryValidationAllEpochs.BringValuesToMatlab(outputFolder = outputFolder, filePrefix = 'VAL_')
        
        ###########################################################################
        # Save to matlab the values over the current epoch
        summaryCurrentEpochValidation.BringValuesToMatlab(outputFolder = outputFolder, 
                                                          dataStructure = 2, batchSize = configHolder.config['batch_size'],
                                                          filePrefix = 'VAL_currEpoch')

min_val, min_index = summaryValidationAllEpochs.FindBestEpochValidation('denorm_err')
print("Best checkpoint value: {}".format(min_val))
print("Best checkpoint index: {}".format(min_index))