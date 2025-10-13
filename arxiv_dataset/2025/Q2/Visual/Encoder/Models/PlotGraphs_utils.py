
# Just some functions for plotting.

import matplotlib.pyplot as plt

import numpy as np


from PIL import Image,ImageDraw

from torchvision import transforms

colors_array = np.array(['lightpink', 'grey', 'blue', 'cyan', 'lime', 'green', 'yellow', 'gold', 'red', 'maroon',
                   'rosybrown', 'salmon', 'sienna', 'palegreen', 'sandybrown', 'deepskyblue', 
                   'fuchsia', 'purple', 'crimson', 'cornflowerblue', 
                   'midnightblue', 'mediumturquoise', 'bisque', 'gainsboro', 'indigo',
                   'white', 'coral', 'powderblue', 'cadetblue', 'orchid', 'burlywood', 'olive', 'lavender', 
                   'olivedrab', 'seashell', 'mistyrose', 'firebrick', 'dimgrey', 'tan', 'darkorange',
                   'tomato', 'dodgerblue', 'slateblue', 'rebeccapurple', 'moccasin'])

# states: these could be both the states or the derivatives:
# n_states_to_display: how many of the states given should be displayed
# file : where to save them
def scatter_states(states, n_states_to_display, file):
    
    x_axis = np.arange(len(states))
    fig, axes = plt.subplots(n_states_to_display)
    for i in range(0,n_states_to_display):
        axes[i].scatter(x_axis, states[:, i], s=1) 
    plt.savefig(file)
    # plt.show()
    
    return

def plot_states(states, n_states_to_display, file):
    
    x_axis = np.arange(len(states))
    fig, axes = plt.subplots(n_states_to_display)
    for i in range(0,n_states_to_display):
        axes[i].plot(x_axis, states[:, i]) 
    plt.savefig(file)
    # plt.show()
    return

def plot_predicted_vs_real_states(predicted_states, real_states, file):
    
    fig, ax = plt.subplots(1, 1, figsize=(9, 4.5))
    fig.tight_layout(pad=0, h_pad=0, w_pad=0)
    ax.plot(real_states[:, 0], real_states[:, 1], 'red')
    ax.plot(predicted_states[:, 0], predicted_states[:, 1], 'black')
    plt.savefig(file)
    
    return

def plot_predicted_vs_real_states_on1D(predicted_states, real_states, n_states_to_display, file):
    
    x_axis = np.arange(len(real_states))
    fig, axes = plt.subplots(n_states_to_display)
    for i in range(0,n_states_to_display):
        axes[i].plot(x_axis, real_states[:, i], 'red') 
        axes[i].plot(x_axis, predicted_states[:, i], 'black') 
    plt.savefig(file)
    # plt.show()
    
    return

def plot_predicted_vs_real_states_onScatterPlotWithQuivers(predicted_states, real_states, file):
    
    fig, ax = plt.subplots(1, 1, figsize=(9, 4.5))
    fig.tight_layout(pad=0, h_pad=0, w_pad=0)

    plt.scatter(predicted_states[:, 0], predicted_states[:, 1], c = 'red', cmap="RdYlGn", s=45)
    plt.scatter(real_states[:, 0],      real_states[:, 1], c = 'blue', cmap="RdYlGn", s=45)
    plt.plot([real_states[:, 0], predicted_states[:, 0]],
             [real_states[:, 1], predicted_states[:, 1]])

    plt.savefig(file)

    return

def plot_clusters(cluster_assignment, odometry, file):
    
    fig, ax = plt.subplots(1, 1, figsize=(9, 4.5))
    fig.tight_layout(pad=0, h_pad=0, w_pad=0)
    
    c = colors_array[cluster_assignment[:]]
    plt.scatter(odometry[:, 0], odometry[:, 1], c = c, cmap="RdYlGn", s=45)
    plt.savefig(file)

# Functions for plotting the evolution of the losses
def plot_loss(loss, file):
    
    plt.close('all') # first close all other plots, or they will appear one over the others
    plt.plot(loss) 
    plt.savefig(file)
    # plt.show()
    
    return

def plot_alpha_values(alpha_values_to_plot,file_name):    
    
    fig = plt.figure(figsize=[6, 6])
    ax = fig.gca()

    for i in range(alpha_values_to_plot.shape[1]):
        ax.plot(alpha_values_to_plot[:,i], linestyle='-')
    plt.title('alpha values')
    plt.savefig(file_name)
    # plt.show()

def HandleLossOverAllEpochs(averageLossesOverAllEpochs, lossesOverCurrentEpoch, folderFileName):
    
    # Average the loss for current epoch
    lossesOverCurrentEpochAverage   = np.mean(lossesOverCurrentEpoch, axis=0)
    
    # Insert average of current epoch in array containing the averages of all epochs
    averageLossesOverAllEpochs.append(lossesOverCurrentEpochAverage)
    
    # From torch to numpy
    #averageLossesOverAllEpochsArray = np.asarray(averageLossesOverAllEpochs)
    
    # Plot the loss over all epochs
    plot_loss(averageLossesOverAllEpochs, folderFileName)
    

    return averageLossesOverAllEpochs

def PrintImagesWithInputsAndPredictions(currentInputImagesBatch, currentInputSingleValuesBatch, 
                                        realValuesDenorm, predictedValuesDenorm, 
                                        currentOutputSingleValuesImagesBatch, predictedValuesBatch,
                                        filePath):
    
    for imgIndex in range(currentInputImagesBatch.shape[0]):
        
        # Path of the current image
        currentImageFilePath = filePath + "_image_" + str(imgIndex) + ".png"
        # Take out current image
        currentImage = currentInputImagesBatch[imgIndex,:,:,:]
        currentImage_IM = transforms.ToPILImage()(currentImage).convert("RGB")
        
        # Draw image and add text
        draw = ImageDraw.Draw(currentImage_IM)           
        textOnImage = ' Real mass: {} \n Predicted mass: {} \n Real mass (norm): {} \n Predicted mass (norm): {} \n Image ratio x (norm): {} \n Image ration y (norm): {} \n Depth (norm): {}'.format(
            realValuesDenorm[imgIndex].item(), predictedValuesDenorm[imgIndex].item(), 
            currentOutputSingleValuesImagesBatch[imgIndex].item(), predictedValuesBatch[imgIndex].item(), 
            currentInputSingleValuesBatch[imgIndex,0].item(), 
            currentInputSingleValuesBatch[imgIndex,1].item(), currentInputSingleValuesBatch[imgIndex,2].item())
        draw.text( (0,20), textOnImage )   
        # Save to path
        currentImage_IM.save(currentImageFilePath, "PNG")
    
    return