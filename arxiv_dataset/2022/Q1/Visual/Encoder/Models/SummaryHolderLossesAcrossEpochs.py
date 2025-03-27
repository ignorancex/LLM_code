
# This is a Summary Holder object that is specific for losses across epochs, 
# i.e., storing values that are the average of one epoch loss, considered
# across different epochs.

###############################################################################

import numpy as np
import scipy.io as sio

# from Models import PlotGraphs_utils as PG
# from Models import SummaryHolder    as SH
from Encoder.Models import PlotGraphs_utils as PG
from Encoder.Models import SummaryHolder    as SH
###############################################################################

class SummaryHolderLossesAcrossEpochs(SH.SummaryHolder):
    """ This class holds the summary data from a training phase, and updates it.
    """
    
    # Initialization of the summary data.
    def __init__(self, summaryNames):
        
        super(self.__class__, self).__init__(summaryNames)
        
        return
    
    # To use when, arrived at the end of an epoch, to add the value to the
    # summary of losses.
    # INPUTS:
    # - currentEpochSummary: summary object over the current epoch.
    #   This is not 'SummaryHolderLossesAcrossEpochs' object, but a base
    #   'SummaryHolder'.
    def AppendToOverallSummaryMeanValuesOfCurrentEpochSummary(self, currentEpochSummary):
        
        # All the keys of the overall summary
        key_list = list(self.summary.keys())
        
        # Looping over all the keys of the overall summary
        for i in range(len(self.summary)):
            # Current key name
            current_key = key_list[i]
            # If the key is also in the current epoch summary
            currentKeyIsInCurrentEpochSummary       = current_key in currentEpochSummary.summary.keys()
            # If the corresponding value is not empty in the epoch summary
            currentKeyValueInEpochSummaryIsNotEmpty = len(currentEpochSummary.summary[current_key]) != 0 
            # If previous statements are true ...
            if currentKeyIsInCurrentEpochSummary and currentKeyValueInEpochSummaryIsNotEmpty:
                # ... average the values for the current epoch
                currentEpochAverage = np.mean(currentEpochSummary.summary[current_key], axis=0)
                self.AppendValueInSummary(current_key, currentEpochAverage)
            
        return

    # Plot one dimensional elements in the summary over their temporal values
    # This is good to plot losses, etc
    # INPUTS:
    # - outputFolder: folder where to print the values,
    # - filePrefix: a string to add at the beginning of the file's name.
    #   For example we could want to add a string 'TRAINING' or 'TESTING'
    #   to distinguish the two.
    def PlotValuesInSummaryAcrossTime(self, outputFolder, filePrefix = ''):
        
        # All the keys of the overall summary
        key_list = list(self.summary.keys())
        
        # Looping over the elements in the dictionary
        for i in range(len(self.summary)):
            # Name of folder + file
            path_name = outputFolder + '/' + filePrefix + key_list[i] + '.png'
            # Plot the loss over all epochs
            PG.plot_loss(self.summary[key_list[i]], path_name)
        
        return
    
    # Save the summary to matlab.
    # INPUTS:
    # - outputFolder : folder where to save the summary object.
    # - filePrefix: a string to add at the beginning of the file's name.
    #   For example we could want to add a string 'TRAINING' or 'TESTING'
    #   to distinguish the two.
    def BringValuesToMatlab(self, outputFolder, filePrefix = ''):
        
        # All the keys of the overall summary
        key_list = list(self.summary.keys())
        
        # Looping over the elements in the dictionary
        for i in range(len(self.summary)):
            # Current key
            currentKey = key_list[i]
            # Values
            currentKeyValues = self.summary[currentKey]
            # Perform the reshaping of data
            # Name of folder + file
            path_name  = outputFolder + '/' + filePrefix + currentKey + '.mat'
            # Plot the loss over all epochs
            sio.savemat(path_name, {key_list[i]: currentKeyValues})
        
        return
    
    # Function to perform all the operations that are done at the end of a 
    # training batch: 
    # - calculating and adding the new loss value to the summary;
    # - plotting the losses,
    # - saving the losses in MATLAB.
    # INPUTS:
    # - currentEpochSummary: summary object over the current epoch.
    #   This is not 'SummaryHolderLossesAcrossEpochs' object, but a base
    #   'SummaryHolder'.
    # - outputFolder: folder where to print the values,
    # - filePrefix: a string to add at the beginning of the file's name.
    #   For example we could want to add a string 'TRAINING' or 'TESTING'
    #   to distinguish the two.
    def PerformFinalBatchOperations(self, currentEpochSummary, outputFolder, filePrefix = ''):
        
        # Add the mean of the losses of the current epoch to the overall summary
        self.AppendToOverallSummaryMeanValuesOfCurrentEpochSummary(currentEpochSummary)
        # Plot losses
        self.PlotValuesInSummaryAcrossTime(outputFolder, filePrefix)
        # Save losses to matlab
        self.BringValuesToMatlab(outputFolder, filePrefix)
        
        return

    def FindBestEpochValidation(self, key):
        values_list = self.summary[key]
        min_value = min(values_list)

        min_index = values_list.index(min_value)

        return min_value, min_index