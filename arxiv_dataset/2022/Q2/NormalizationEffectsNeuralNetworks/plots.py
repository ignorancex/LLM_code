import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from functions import generate_file_name, validate_directory


def load_accuracy_for_single_gamma(directory, dataset_name, model_name,
                                   criterion_name, gamma, hidden_units, epochs,
                                   batch_size):
    """Returns a DataFrame with test and train accuracy by epoch for a single 
    gamma value
    
    Parameters
    ----------
    directory: str
        location of the data file
    dataset_name: str
        'mnist' or 'cifar10'
    model_name: str
        'mlp' or 'cnn'
    criterion_name: str
        'ce' (for Cross Entropy loss) or 'mse' (for Mean Squared Error loss)
    gamma: float
        the mean-field scaling parameter
    hidden_units: int
        the number of nodes in the hidden layer
    epochs: int
        number of times to iterate through the data set for training the model 
        and calculating accuracy
    batch_size: int
        the number of images per batch
    """

    # Determine data file name
    fname = generate_file_name(
        dataset_name=dataset_name,
        model_name=model_name,
        criterion_name=criterion_name,
        gamma=gamma,
        hidden_units=hidden_units,
        epochs=epochs,
        batch_size=batch_size)
    
    # Create full path to data file, including extension
    results_folder = 'results/'
    path = os.path.join(directory, results_folder)
    path = os.path.join(path, fname) + '.csv'
    
    # Load data file
    data = pd.read_csv(path, index_col=0)

    return data


def load_accuracy_for_multiple_gamma(directory, dataset_name, model_name, 
                                     criterion_name, gamma_list, hidden_units,
                                     epochs, batch_size, is_test_accuracy):
    """Returns a DataFrame with either test or train accuracy by epoch for a 
    list of gamma values
    
    Parameters
    ----------
    directory: str
        location of the data file
    dataset_name: str
        'mnist' or 'cifar10'
    model_name: str
        'mlp' or 'cnn'
    criterion_name: str
        'ce' (for Cross Entropy loss) or 'mse' (for Mean Squared Error loss)
    gamma_list: list of floats
        the mean-field scaling parameters
    hidden_units: int
        the number of nodes in the hidden layer
    epochs: int
        number of times to iterate through the data set for training the model 
        and calculating accuracy
    batch_size: int
        the number of images per batch
    is_test_accuracy: bool
        True for test accuracy or False for train accuracy
    """

    column = 'Test' if is_test_accuracy else 'Train'
    
    # Dictionary to store data by gamma
    dict_data = dict()

    # Iterate over list of gamma values
    for gamma in gamma_list:
        
        # Load accuracy data
        data = load_accuracy_for_single_gamma(
            directory=directory,
            dataset_name=dataset_name,
            model_name=model_name,
            criterion_name=criterion_name,
            gamma=gamma,
            hidden_units=hidden_units,
            epochs=epochs,
            batch_size=batch_size)
        dict_data[gamma] = data[column]
        
    # Concatenate accuracy data over gamma values
    results = pd.concat(dict_data, axis=1)[gamma_list]
    
    return results


def plot_accuracy_for_multiple_gamma(data, is_test_accuracy):
    """Returns axes for a new figure plotting test or train accuracy vs epochs 
    for multiple values of gamma
    
    Parameters
    ----------
    data: DataFrame
        index = epochs, columns = gammas, data = test or train accuracy
    is_test_accuracy: bool
        True for test accuracy or False for train accuracy
    """

    # Create a new figure and plot accuracy data
    plt.figure()
    ax = data.plot()
    ax.set_ylim([0,1])

    # Label legend and x- and y-axes
    plt.legend(title='gamma', loc='lower right')
    plt.xlabel('Number of Epochs')
    y_label = 'Test Accuracy' if is_test_accuracy else 'Train Accuracy'
    plt.ylabel(y_label)
    
    return ax


def plot_test_and_train_accuracy_single_gamma(data):
    """Returns axes for a new figure plotting test and train accuracy vs epochs 
    for a single value of gamma
    
    Parameters
    ----------
    data: DataFrame
        index = epochs, columns = ['Train', 'Test'], data = accuracy
    """

    # Create a new figure and plot accuracy data
    plt.figure()
    ax = data.plot()

    # Label legend and x- and y-axes
    plt.legend(title='Data', loc='lower right')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Accuracy')
    
    return ax


def run_test_and_train_accuracy_plots(dataset_name, model_name, criterion_name,
                                      gamma, hidden_units, epochs, batch_size, 
                                      directory):
    """Plots and saves a figure comparing test vs train accuracy for a single 
    value of gamma
    
    Parameters
    ----------
    dataset_name: str
        'mnist' or 'cifar10'
    model_name: str
        'mlp' or 'cnn'
    criterion_name: str
        'ce' (for Cross Entropy loss) or 'mse' (for Mean Squared Error loss)
    gamma: float
        the mean-field scaling parameter
    hidden_units: int
        the number of nodes in the hidden layer
    epochs: int
        number of times to iterate through the data set for training the model 
        and calculating accuracy
    batch_size: int
        the number of images per batch
    directory: str
        location of the data files and figures
    """

    # Load accuracy data
    data = load_accuracy_for_single_gamma(
        directory=directory,
        dataset_name=dataset_name,
        model_name=model_name,
        criterion_name=criterion_name,
        gamma=gamma,
        hidden_units=hidden_units,
        epochs=epochs,
        batch_size=batch_size)
    
    # Create figure and plot data
    ax = plot_test_and_train_accuracy_single_gamma(data=data)
    
    # Generate file name
    fname = 'plot_test_train_{dataset_name}_{model_name}_g{gamma}_h{hidden}_e{epochs}_b{batch_size}'.format(
        dataset_name=dataset_name,
        model_name=model_name,
        gamma=gamma,
        hidden=hidden_units,
        epochs=epochs,
        batch_size=batch_size)

    # Generate full path
    figures_directory = os.path.join(directory, 'figures/')
    validate_directory(directory=figures_directory)
    path = os.path.join(figures_directory, fname)
    
    # Save figure
    ax.figure.savefig(path + '.png', dpi=300, bbox_inches='tight')
    ax.figure.savefig(path + '.pdf', dpi=300, bbox_inches='tight')
    return


def run_multiple_gamma_accuracy_plots(dataset_name, model_name, criterion_name,
                                      gamma_list, hidden_units, epochs,
                                      batch_size, is_test_accuracy, directory):
    """Plots and saves a figure of test or train accuracy for a multiple 
    values of gamma
    
    Parameters
    ----------
    dataset_name: str
        'mnist' or 'cifar10'
    model_name: str
        'mlp' or 'cnn'
    criterion_name: str
        'ce' (for Cross Entropy loss) or 'mse' (for Mean Squared Error loss)
    gamma_list: list of floats
        the mean-field scaling parameters
    hidden_units: int
        the number of nodes in the hidden layer
    epochs: int
        number of times to iterate through the data set for training the model 
        and calculating accuracy
    batch_size: int
        the number of images per batch
    is_test_accuracy: bool
        True for test accuracy or False for train accuracy
    directory: str
        location of the data files and figures
    """

    # Load accuracy data
    data = load_accuracy_for_multiple_gamma(
        directory=directory,
        dataset_name=dataset_name,
        model_name=model_name,
        criterion_name=criterion_name,
        gamma_list=gamma_list,
        hidden_units=hidden_units,
        epochs=epochs,
        batch_size=batch_size,
        is_test_accuracy=is_test_accuracy)
    
    # Create figure and plot data
    ax = plot_accuracy_for_multiple_gamma(
        data=data,
        is_test_accuracy=is_test_accuracy)
    
    # Generate file name
    train_test = 'test' if is_test_accuracy else 'train'
    fname = 'plot_{dataset_name}_{model_name}_h{hidden}_e{epochs}_b{batch_size}_{train_test}'.format(
        dataset_name=dataset_name,
        model_name=model_name,
        hidden=hidden_units,
        epochs=epochs,
        batch_size=batch_size,
        train_test=train_test)

    # Generate full path
    figures_directory = os.path.join(directory, 'figures/')
    validate_directory(directory=figures_directory)
    path = os.path.join(figures_directory, fname)
    
    # Save figure
    ax.figure.savefig(path + '.png', dpi=300, bbox_inches='tight')
    ax.figure.savefig(path + '.pdf', dpi=300, bbox_inches='tight')
    msg = "Successfully saved to {fname}".format(fname=fname)
    print(msg)
    return


if __name__ == "__main__":
    
    # PARAMETERS TO RUN FROM COMMAND LINE
    # dataset_name = str(sys.argv[1])
    # model_name = str(sys.argv[2])
    # criterion_name = str(sys.argv[3])
    # gamma_list = float(sys.argv[4])
    # hidden_units = int(sys.argv[5])
    # epochs = int(sys.argv[6])
    # batch_size = int(sys.argv[7])
    # is_test_accuracy = bool(sys.argv[7])
    # directory = str(sys.argv[8])
    
    # PARAMETERS TO RUN LOCALLY
    dataset_name = 'cifar10'
    model_name = 'cnn'
    criterion_name = 'ce'
    gamma_list = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    hidden_units = 3000
    epochs = 1000
    batch_size = 500
    is_test_accuracy = False
    directory = '/project/scalingnn/test/'
        
    run_multiple_gamma_accuracy_plots(
        dataset_name=dataset_name,
        model_name=model_name,
        criterion_name=criterion_name,
        gamma_list=gamma_list,
        hidden_units=hidden_units,
        epochs=epochs,
        batch_size=batch_size,
        is_test_accuracy=is_test_accuracy,
        directory=directory)
