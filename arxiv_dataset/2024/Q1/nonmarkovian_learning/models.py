from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D,GRU,Flatten,Dense

def gen_model_AR(w,n_steps_in):
    """Generate the basic auto-regressive model

    Args:
        w (int): the window size
        n_steps_in (int): the dimension of the input

    Returns:
        object: the AR(w) model
    """
    modelAR = Sequential()
    modelAR.add(Conv1D(filters=1, kernel_size=int(w), activation='sigmoid', input_shape=(n_steps_in, 1)))
    modelAR.add(Flatten())    
    return modelAR

def gen_model_RNN(w,n_steps_in):
    """Generate the GRU model

    Args:
        w (int): the number of units in the GRU layer
        n_steps_in (int): the dimension of the input

    Returns:
        object: the GRU(w) model
    """    
    model = Sequential()
    model.add(GRU(w,return_sequences=True,input_shape=(n_steps_in,1)))
    model.add(Dense(1,activation='sigmoid'))
    model.add(Flatten())
    return model 

def gen_model_CNN(w,n_steps_in,n_filters=10):
    """Generate the CNN model

    Args:
        w (int): the number of units in the GRU layer
        n_steps_in (int): the dimension of the input
        n_filters (int, optional): Number of filters in the first convolutional layer. Defaults to 10.

    Returns:
        object: the CNN(w) model
    """
    modelCNN = Sequential()
    modelCNN.add(Conv1D(filters=n_filters, kernel_size=int(w), activation='relu', input_shape=(n_steps_in, 1)))
    modelCNN.add(Conv1D(filters=1, kernel_size=int(1), activation='sigmoid'))
    modelCNN.add(Flatten())    
    return modelCNN