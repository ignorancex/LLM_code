import sys, importlib, time
sys.path.append('../../')
from src.Modules.Utils.Imports import *
from src.Modules.Models.BuildBINNs import BINN
from src.Modules.Utils.ModelWrapper import ModelWrapper
from src.Modules.Utils.Gradient import Gradient
device = torch.device(GetLowestGPU(pick_from=[0]))

def unique_inputs(inputs):
    """
    Returns unique values of input data along the first and second columns.

    Args:
        inputs (numpy.ndarray): The input data with shape (n_samples, 2).

    Returns:
        numpy.ndarray: Unique values of the first column.
        numpy.ndarray: Unique values of the second column.
    """
    x = np.unique(inputs[:,0])
    t = np.unique(inputs[:,1])
    return x,t

def MSE(a,b):
    """
    Calculate the mean squared error between two arrays.

    Args:
        a (numpy.ndarray): The first array.
        b (numpy.ndarray): The second array.

    Returns:
        float: The mean squared error between `a` and `b`.
    """
    assert a.shape == b.shape
    return ((a - b)**2).mean()

# helper functions
def to_torch(x):
    """
    Convert a NumPy array to a PyTorch tensor and move it to the specified device.

    Args:
        x (numpy.ndarray): The input array.

    Returns:
        torch.Tensor: A PyTorch tensor on the specified device.
    """
    return torch.from_numpy(x).float().to(device)

def to_numpy(x):
    """
    Convert a PyTorch tensor to a NumPy array.

    Args:
        x (torch.Tensor): The input PyTorch tensor.

    Returns:
        numpy.ndarray: A NumPy array containing the data from the input tensor.
    """
    return x.detach().cpu().numpy()

def load_model(binn_name, x, t, save_name):
    
    """
    Load a BINN model from a saved checkpoint.

    Args:
        binn_name (str): The name of the BINN model.
        x (numpy.ndarray): The unique values of the x column.
        t (numpy.ndarray): The unique values of the t column.
        save_name (str): The name of the saved checkpoint.
        device (torch.device): The target device for model loading.

    Returns:
        ModelWrapper: A wrapper for the loaded BINN model.
        BINN: The loaded BINN model.
    """
    
    weight = '_best_val'
    
    # instantiate BINN
    binn = BINN(binn_name, x, t).to(device)

    # wrap model and load weights
    parameters = binn.parameters()
    model = ModelWrapper(
        model=binn,
        optimizer=None,
        loss=None,
        save_name=save_name)
    
    #model.save_name += '_' + binn.name + weight
    model.save_name += weight
    model.load(model.save_name + '_model', device=device)

    return model, binn


def recover_binn_params(binn):
    
    """
    Recover the learned parameters (diffusion and growth terms) from a BINN model.

    Args:
        binn (BINN): The BINN model.

    Returns:
        callable: A function for calculating the learned diffusion term.
        callable: A function for calculating the learned growth term.
    """
    
    # learned diffusion term
    def D(u):
        D = binn.diffusion(to_torch(u)[:, None])
        return to_numpy(D).reshape(-1)

    # learned growth term
    def G(u):
        r = binn.growth(to_torch(u)[:, None])
        return to_numpy(r).reshape(-1)  

    return D, G
