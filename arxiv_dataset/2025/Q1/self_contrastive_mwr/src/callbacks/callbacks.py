import os

import numpy as np

import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt



class PerformancePlotCallback(keras.callbacks.Callback):
    def __init__(self, x_test, y_test, model_name):
        """
        Initialize the Callbacks class.

        Args:
            x_test (array-like): The input data for testing.
            y_test (array-like): The target data for testing.
            model_name (str): The name of the model.
        """
        self.x_test = x_test
        self.y_test = y_test
        self.model_name = model_name
        
    def on_epoch_end(self, epoch, logs={}):
        """
        Callback function called at the end of each epoch.
        Plots the predictions of the model on the test data.

        Args:
            epoch (int): The current epoch number.
            logs (dict, optional): Dictionary containing the training metrics for the current epoch. Defaults to {}.
        """
        y_pred = self.model.predict(self.x_test)
        if len(self.model.output_names) > 1:
            index = self.model.output_names.index('output')
            y_pred = y_pred[index]
        y_pred = np.asarray(y_pred, dtype=np.float32)
        fig, ax = plt.subplots(figsize=(8,4))
        plt.scatter(self.y_test, y_pred, alpha=0.6, 
            color='#FF0000', lw=1, ec='black')
        
        lims = [-0.1, 1.1]

        plt.plot(lims, lims, lw=1, color='#0000FF')
        plt.ticklabel_format(useOffset=False, style='plain')
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.xlim(lims)
        plt.ylim(lims)

        plt.tight_layout()
        plt.title(f'Predictions - Epoch: {epoch}')
        folder = os.path.join('..', '_performance_plot_callback', self.model_name)
        os.makedirs(folder, exist_ok=True)
        plt.savefig(os.path.join(folder, f'{epoch:04d}.png'))
        plt.close()