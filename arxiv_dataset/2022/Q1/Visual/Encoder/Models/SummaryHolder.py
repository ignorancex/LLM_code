
# This class holds the summary data from a training phase, and updates it.

###############################################################################

import numpy as np
import scipy.io as sio

###############################################################################

class SummaryHolder(object):
    """ This class holds the summary data from a training phase, and updates it.
    """
    
    ###########################################################################
    # Functions to initialize the summary holder.
    
    # Initialization of the summary data.
    def __init__(self, summaryNames):
        
        self.InitializeSummaryKeys(summaryNames)
        
        return
        
    # Initialize the names of the dictionary
    # INPUTS:
    # - summaryNames: names of the variables to save in the summary holder
    def InitializeSummaryKeys(self, summaryNames):
        
        self.summary = {}
        
        # Looping over the elements in the dictionary
        for i in range(len(summaryNames)):
            # Creates the key and gives as value an empty list
            self.summary[summaryNames[i]] = []
        return
    
    # Append a value in one of the summary values
    # INPUTS:
    # - valueToAppend
    # - key: name of the summary variable where to append the value.
    def AppendValueInSummary(self, key, valueToAppend):
        
        self.summary[key].append(valueToAppend)
        
        return
    
    ###########################################################################
    # Functions to save the summary holder to MATLAB
    
    def BringValueToMatlabGivenKey(self, key, outputFolder, dataStructure, batchSize, 
                                   filePrefix = '', inverseIndexing = None):
        
        # Values
        currentKeyValues = self.summary[key]
        # Perform the reshaping of data
        currentKeyValues = SummaryHolder.PerformDataReshaping(currentKeyValues, dataStructure, 
                                                              batchSize, inverseIndexing = inverseIndexing)
        
        # Name of folder + file
        path_name  = outputFolder + '/' + filePrefix + key + '.mat'
        
        # Plot the loss over all epochs
        sio.savemat(path_name, {key: currentKeyValues})
        
        return
    
    # Save the values to matlab
    def BringValuesToMatlab(self, outputFolder, dataStructure, batchSize, filePrefix = '', inverseIndexing = None):
        
        # All the keys of the overall summary
        key_list = list(self.summary.keys())
        
        # Looping over the elements in the dictionary
        for i in range(len(self.summary)):
            # Current key
            currentKey = key_list[i]
            # Bring the value to matlab
            self.BringValueToMatlabGivenKey(currentKey, outputFolder, dataStructure, batchSize, 
                                       filePrefix, inverseIndexing)
            
        return
    
    ###########################################################################
    # Functions to perform reshaping of data.
    
    @staticmethod
    def PerformDataReshaping(data, dataStructure, batchSize, inverseIndexing = None):
        
        if dataStructure == 0:   # KVAE with only one sequence case
            
            return SummaryHolder.PerformDataReshaping4D(data, batchSize, inverseIndexing)

        elif dataStructure == 1: # KVAE with many sequences case
            
            return SummaryHolder.PerformDataReshaping5D(data, batchSize)
        
        elif dataStructure == 2: # VAE case
            
            return SummaryHolder.PerformDataReshaping3D(data, batchSize)
    
    @staticmethod
    def PerformDataReshaping5D(data, batchSize):
        
        data = np.asarray(data)
        data = np.squeeze(data)
        
        # Case for example of learning rate: there is a single value per epoch
        # and after using the squeeze function, data will have dimension ()
        if data.ndim == 0:
            
            return data
        
        # Case for example of losses: there is only one loss per batch, so
        # the data is structured as: [batch_size]
        if data.ndim == 1:
            
            return data
        
        # Case for example of states when batch size = 1 (as in testing), so
        # the data is structured as:
        # [number of batches, sequence length, state dimension]
        elif data.ndim == 3 and batchSize == 1:
            
            return np.reshape(data, (data.shape[0]*data.shape[1], data.shape[2]))
        
        # Case for example of sigma when batch size = 1 (as in testing), so
        # the data is structured as:
        # [number of batches, sequence length, state dimension x, state dimension y]
        elif data.ndim == 4 and batchSize == 1:
            
            return np.reshape(data, (data.shape[0]*data.shape[1], data.shape[2], data.shape[3]))
        
        # Case for example of states when batch size != 1 (as in training), so
        # the data is structured as:
        # [number of batches, batch size, sequence length, state dimension]
        elif data.ndim == 4 and batchSize != 1:
            
            # Bring axis of trajectories more inside
            data = np.swapaxes(data, 1,2)
            
            return np.reshape(data, (data.shape[0]*data.shape[1], data.shape[2], data.shape[3]))
        
        # Case for example of sigmas when batch size != 1 (as in training), so
        # the data is structured as:
        # [number of batches, batch size, sequence length, state dimension]
        elif data.ndim == 5 and batchSize != 1:            
            # Bring axis of trajectories more inside
            data = np.swapaxes(data, 1,2)
            
            return np.reshape(data, (data.shape[0]*data.shape[1], data.shape[2], data.shape[3], data.shape[4]))
        
    @staticmethod
    def PerformDataReshaping4D(data, batchSize, inverseIndexing = None):
        
        data = np.asarray(data)
        data = np.squeeze(data)
        
        # Case for example of learning rate: there is a single value per epoch
        # and after using the squeeze function, data will have dimension ()
        if data.ndim == 0:
            
            return data
        
        # Case for example of losses: there is only one loss per batch, so
        # the data is structured as: [batch_size]
        if data.ndim == 1:
            
            return data
        
        # Case for example of states when batch size = 1 (as in testing), so
        # the data is structured as:
        # [number of batches, sequence length, state dimension]
        elif data.ndim == 3 and batchSize == 1:
            
            if inverseIndexing is not None:
                data = data[inverseIndexing, :, :]
            return np.reshape(data, (data.shape[0]*data.shape[1], data.shape[2]))
        
        # Case for example of sigma when batch size = 1 (as in testing), so
        # the data is structured as:
        # [number of batches, sequence length, state dimension x, state dimension y]
        elif data.ndim == 4 and batchSize == 1:
            
            if inverseIndexing is not None:
                data = data[inverseIndexing, :, :]
            return np.reshape(data, (data.shape[0]*data.shape[1], data.shape[2], data.shape[3]))
        
        # Case for example of states when batch size != 1 (as in training), so
        # the data is structured as:
        # [number of batches, batch size, sequence length, state dimension]
        elif data.ndim == 4 and batchSize != 1:
            
            if inverseIndexing is not None:
                data = np.reshape(data, (data.shape[0]*data.shape[1], data.shape[2], data.shape[3]))
                data = data[inverseIndexing, :, :]
                data = np.reshape(data, (data.shape[0]*data.shape[1], data.shape[2]))
                return data
            else:
                data = np.reshape(data, (data.shape[0]*data.shape[1]*data.shape[2], data.shape[3]))
                return data
        
        # Case for example of sigmas when batch size != 1 (as in training), so
        # the data is structured as:
        # [number of batches, batch size, sequence length, state dimension]
        elif data.ndim == 5 and batchSize != 1:
            
            if inverseIndexing is not None:
                data = np.reshape(data, (data.shape[0]*data.shape[1], data.shape[2], data.shape[3], data.shape[4]))
                data = data[inverseIndexing, :, :, :]
                data = np.reshape(data, (data.shape[0]*data.shape[1], data.shape[2], data.shape[3]))
                return data
            else:
                data = np.reshape(data, (data.shape[0]*data.shape[1]*data.shape[2], data.shape[3], data.shape[4]))
                return data
            
    @staticmethod
    def PerformDataReshaping3D(data, batchSize):
        
        data = np.asarray(data)
        # data = np.squeeze(data)
        
        return data
        
        '''
        # Case for example of learning rate: there is a single value per epoch
        # and after using the squeeze function, data will have dimension ()
        if data.ndim == 0:
            
            return data
        
        # Case for example of losses: there is only one loss per batch, so
        # the data is structured as: [batch_size]
        if data.ndim == 1:
            
            return data
        
        # Case for example of states when batch size != 1 (as in training), so
        # the data is structured as:
        # [number of batches, elements in batch, state dimension]
        elif data.ndim == 3 and batchSize != 1:
            return np.reshape(data, (data.shape[0]*data.shape[1], data.shape[2]))
        
        # Case for example of sigmas when batch size != 1 (as in training), so
        # the data is structured as:
        # [number of batches, elements in batch, state dimension, state dimension]
        elif data.ndim == 4 and batchSize != 1:
            return np.reshape(data, (data.shape[0]*data.shape[1], data.shape[2], data.shape[3]))
        '''
    
        
    
    