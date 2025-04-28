import sys
sys.path.append('../')

from src.Modules.Utils.Imports import *
from src.Modules.Utils.ModelWrapper import ModelWrapper
from src.Modules.Models.BuildBINNs import BINN
import src.Modules.Loaders.DataFormatter as DF
from src.custom_functions import to_torch, to_numpy

def BINN_training(inputs, outputs, binn, binn_save_name):
    
    """
    Train a BINN model on provided inputs and outputs.

    This function performs two stages of training: first training only the surface fitter, and then jointly training the surface fitter and the PDE loss.

    Parameters:
        inputs (np.ndarray): Input data for training the BINN model.
        outputs (np.ndarray): Output data for training the BINN model.
        binn (BINN): The BINN model to be trained.
        binn_save_name (str): The name for saving the trained BINN model.
        
    Returns:
        BINN: The trained BINN model after both stages of training.
        ModelWrapper: The model wrapper containing the trained BINN model.
    """

    N = len(inputs)
    split = int(0.8*N)
    p = np.random.permutation(N)
    x_train = to_torch(inputs[p[:split]])
    y_train = to_torch(outputs[p[:split]])
    x_val = to_torch(inputs[p[split:]])
    y_val = to_torch(outputs[p[split:]])
    
    # compile 
    parameters = binn.parameters()
    opt = torch.optim.Adam(parameters, lr=1e-3)
    model_only_SF_loss = ModelWrapper(
        model=binn,
        optimizer=opt,
        loss=binn.gls_loss,
        augmentation=None,
        save_name=f"../../results/weights/SF_only_{binn_save_name}")

    epochs = int(1e4)
    batch_size = 1000
    rel_save_thresh = 0.01

    # first only train the surface fitter
    model_only_SF_loss.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=None,
        verbose=0,
        validation_data=[x_val, y_val],
        early_stopping=1e3,
        rel_save_thresh=rel_save_thresh)

    # Now create a second model that will also train the surface fitter and PDE loss
    model = ModelWrapper(
        model=binn,
        optimizer=opt,
        loss=binn.loss,
        augmentation=None,
        save_name=f"../../results/weights/BINN_training_{binn_save_name}")

    epochs = int(1e6)
    batch_size = 1000
    rel_save_thresh = 0.01

    # train jointly
    model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=None,
        verbose=0,
        validation_data=[x_val, y_val],
        early_stopping=1e5,
        rel_save_thresh=rel_save_thresh)
    
    return binn,model
    
