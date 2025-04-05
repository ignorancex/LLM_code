import os
import sys
import torch.nn as nn
import torch.optim as optim
from functions import (
    load_mnist_data, load_cifar_data, determine_device, run_model,
    generate_file_name, save_results, save_state)
from models import MLP, CNN


def process(dataset_name, model_name, criterion_name, gamma, hidden_units,
            epochs, batch_size, directory):
    """Trains a neural network model on a dataset and saves the resulting 
    model accuracy and model parameters to files
    
    Parameters
    ----------
    dataset_name: str
        'mnist' or 'cifar10'
    model_name: str
        'mlp' (multi-layer perceptron) or 'cnn' (convolutional neural network)
    criterion_name: str
        'ce' (Cross Entropy loss) or 'mse' (Mean Squared Error loss)
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
        the local where accuracy results and model parameters are saved
        (requires folders 'results' and 'models')
    """

    # Information
    print("Dataset:    {}".format(dataset_name.upper()))
    print("Model:      {}".format(model_name.upper()))
    print("Criterion:  {}".format(criterion_name.upper()))
    print("Parameters: g={g}, h={h}, e={e}, b={b}".format(
        g=gamma, h=hidden_units, e=epochs, b=batch_size))

    # Determine device
    device = determine_device(do_print=True)
    
    # Load data
    if dataset_name.upper() == 'MNIST':
        train_loader, test_loader = load_mnist_data(batch_size=batch_size)
    elif dataset_name.upper() == 'CIFAR10':
        train_loader, test_loader = load_cifar_data(batch_size=batch_size)
    else:
        raise ValueError("Dataset '{0}' unknown".format(dataset_name))
        
    # Neural network model
    if model_name.upper() == 'MLP':
        beta = 2 - 2 * gamma
        learning_rate = 1.0 / (hidden_units ** beta)
        model = MLP(hidden_units=hidden_units, gamma=gamma)
    elif model_name.upper() == 'CNN':
        learning_rate = 1.0
        model = CNN(hidden_units=hidden_units, gamma=gamma)
        model.initialize_parameters()
    else:
        raise ValueError("Model '{0}' unknown".format(dataset_name))
    model.to(device)
    
    # Criterion (loss function)
    if criterion_name.upper() == 'CE':
        criterion = nn.CrossEntropyLoss()
        do_encoding = False
    elif criterion_name.upper() == 'MSE':
        criterion = nn.MSELoss()
        do_encoding = True
    else:
        raise ValueError("Criterion '{0}' unknown".format(criterion_name))
    
    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Run model
    results = run_model(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        do_encoding=do_encoding,
        epochs=epochs)

    # File name
    file_name = generate_file_name(
        dataset_name=dataset_name,
        model_name=model_name,
        criterion_name=criterion_name,
        gamma=gamma,
        hidden_units=hidden_units,
        epochs=epochs,
        batch_size=batch_size)
    
    # Save accuracy results
    results_directory = os.path.join(directory, 'results/')
    save_results(
        results=results,
        directory=results_directory,
        file_name=file_name)

    # Save model state
    models_directory = os.path.join(directory, 'models/')
    save_state(
        model=model,
        directory=models_directory,
        file_name=file_name)
    
    return


if __name__ == "__main__":
    
    # PARAMETERS TO RUN FROM COMMAND LINE
    # dataset_name = str(sys.argv[1])
    # model_name = str(sys.argv[2])
    # criterion_name = str(sys.argv[3])
    # gamma = float(sys.argv[4])
    # hidden_units = int(sys.argv[5])
    # epochs = int(sys.argv[6])
    # batch_size = int(sys.argv[7])
    # directory = str(sys.argv[8])

    # PARAMETERS TO RUN LOCALLY
    dataset_name = 'mnist'
    model_name = 'mlp'
    criterion_name = 'mse'
    gamma = 0.6
    hidden_units = 100
    epochs = 500
    batch_size = 20
    directory = '/project/scalingnn/test/'

    process(
        dataset_name=dataset_name,
        model_name=model_name,
        criterion_name=criterion_name,
        gamma=gamma,
        hidden_units=hidden_units,
        epochs=epochs,
        batch_size=batch_size,
        directory=directory)
