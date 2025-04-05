"""Convenient plot functions

The functions in this module allow to effectively display MNIST images and training plots

Author: Sergio P. Perez
"""

import matplotlib.pyplot as plt
import numpy as np

#####################################################################################
#
# FUNCTION: plot_image
#
#####################################################################################

def plot_image(phi):
    """Plot one image from MNIST

    Args:
        phi: MNIST image
    Returns: 
        plot of the image
    """

    if phi.shape == (28,28):

        plt.imshow(np.reshape(phi,(28,28)), cmap='Greys', vmin=-1, vmax=1, extent=[0,100,0,1], aspect=100)
        plt.axis('off')

    elif phi.shape == (784,):

        plt.imshow(np.reshape(phi,(28,28)), cmap='Greys', vmin=-1, vmax=1, extent=[0,100,0,1], aspect=100)
        plt.axis('off')

#####################################################################################
#
# FUNCTION: plot_2images
#
#####################################################################################
        
def plot_2images(phi1, phi2):
    """Plot two images from MNIST

    Args:
        phi1: MNIST image
        phi2: MNIST image
    Returns: 
        plot of the images
    """

    if phi1.shape == (28,28):
        
        f, axarr = plt.subplots(2)

        axarr[0].imshow(np.reshape(phi1,(28,28)), cmap='Greys', vmin=-1, vmax=1, extent=[0,100,0,1], aspect=100)
        axarr[0].set_xticks([])
        axarr[0].set_yticks([])
        axarr[1].imshow(np.reshape(phi2,(28,28)), cmap='Greys', vmin=-1, vmax=1, extent=[0,100,0,1], aspect=100)
        axarr[1].set_xticks([])
        axarr[1].set_yticks([])

    elif phi1.shape == (784,):
        
        f, axarr = plt.subplots(1,2)
        axarr[0].imshow(np.reshape(phi1,(28,28)), cmap='Greys', vmin=-1, vmax=1, extent=[0,100,0,1], aspect=100)
        axarr[0].set_xticks([])
        axarr[0].set_yticks([])
        axarr[1].imshow(np.reshape(phi2,(28,28)), cmap='Greys', vmin=-1, vmax=1, extent=[0,100,0,1], aspect=100)
        axarr[1].set_xticks([])
        axarr[1].set_yticks([])

#####################################################################################
#
# FUNCTION: plot_3images
#
#####################################################################################
        
def plot_3images(phi1, phi2, phi3):
    """Plot three images from MNIST

    Args:
        phi1: MNIST image
        phi2: MNIST image
        phi3: MNIST image
    Returns: 
        plot of the images
    """

    if phi1.shape == (28,28):
        
        f, axarr = plt.subplots(3)

        axarr[0].imshow(np.reshape(phi1,(28,28)), cmap='Greys', vmin=-1, vmax=1, extent=[0,100,0,1], aspect=100)
        axarr[0].set_xticks([])
        axarr[0].set_yticks([])
        axarr[0].set_title("Original", fontsize=25)
        axarr[1].imshow(np.reshape(phi2,(28,28)), cmap='Greys', vmin=-1, vmax=1, extent=[0,100,0,1], aspect=100)
        axarr[1].set_xticks([])
        axarr[1].set_yticks([])
        axarr[1].set_title("Damaged", fontsize=25)
        axarr[2].imshow(np.reshape(phi3,(28,28)), cmap='Greys', vmin=-1, vmax=1, extent=[0,100,0,1], aspect=100)
        axarr[2].set_xticks([])
        axarr[2].set_yticks([])
        axarr[2].set_title("Restored", fontsize=25)

    elif phi1.shape == (784,):
        
        f, axarr = plt.subplots(1,3, figsize=(16,16))
        axarr[0].imshow(np.reshape(phi1,(28,28)), cmap='Greys', vmin=-1, vmax=1, extent=[0,100,0,1], aspect=100)
        axarr[0].set_xticks([])
        axarr[0].set_yticks([])
        axarr[0].set_title("Original", fontsize=25)
        axarr[1].imshow(np.reshape(phi2,(28,28)), cmap='Greys', vmin=-1, vmax=1, extent=[0,100,0,1], aspect=100)
        axarr[1].set_xticks([])
        axarr[1].set_yticks([])
        axarr[1].set_title("Damaged", fontsize=25)
        axarr[2].imshow(np.reshape(phi3,(28,28)), cmap='Greys', vmin=-1, vmax=1, extent=[0,100,0,1], aspect=100)
        axarr[2].set_xticks([])
        axarr[2].set_yticks([])
        axarr[2].set_title("Restored", fontsize=25)
        
#####################################################################################
#
# FUNCTION: loss_acc_plots
#
#####################################################################################       
        
def loss_acc_plots(history):
    """Plot the history of a tensorflow model

    Args:
        history: tensorflow training history
    Returns: 
        plots of loss function and accuracy across epochs
    """
    
    #Loss plot
    plt.style.use('ggplot')
    loss_plot = history.plot(y="loss", legend=False, fontsize=15, linewidth=3)
    loss_plot.set_xlabel('Epochs',fontsize = 20)
    loss_plot.set_ylabel("Loss",fontsize = 20)
    loss_plot.set_title("Loss vs. Epochs",fontsize = 20)
    
    #Accuracy plot
    plt.style.use('ggplot')
    acc_plot = history.plot(y="accuracy", legend=False, fontsize=15, linewidth=3)
    acc_plot.set_xlabel('Epochs',fontsize = 20)
    acc_plot.set_ylabel("Accuracy",fontsize = 20)
    acc_plot.set_title("Accuracy vs. Epochs",fontsize = 20)