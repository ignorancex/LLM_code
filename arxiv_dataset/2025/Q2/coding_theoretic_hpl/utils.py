import argparse
import yaml
import os
from collections import ChainMap
import shutil

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2 as transforms

from models import SmallNetwork, RandomCode, \
    ReedMullerCode, ResNetD, \
    BCHCode, LogSumExpCode, MettesCode, KasarlaCode, PrototypicalLoss
    
def parse_arguments(verbose=True):
    """Parses the training parameters specified by the yaml file at the config file path"""

    # Parse arguments
    parser = argparse.ArgumentParser(prog="Coding-Based Learning", description="A coding-based learning approach to classification tasks in ML")
    parser.add_argument("-i", "--identifier", help="Unique run identifier")
    parser.add_argument("-r", "--root", help="Full path where the 'root/identifier/' subdirectory will be created, to store all program outputs")
    parser.add_argument("-c", "--config", help="Full path to the config file (including .yaml)")
    parser.add_argument("-d", "--dataset", help="Full path to dataset")
    args = parser.parse_args()

    # the file /path/to/config.yaml exists
    if not os.path.isfile(args.config):
        raise FileNotFoundError(f"Could not find the config file {args.config}. Is this a path?")
    # /path/to/root/ exists, else it is created
    if not os.path.isdir(args.root):
        print(f"Could not find the directory {args.root}; attempting to create...", end=" ")
        try:
            os.mkdir(args.root)
            print("directory created successfully")
        except IOError as e:
            print("could not create directory")
            raise
    #create /path/to/root/identifier/
    try:
        print(f"Creating directory {os.path.join(args.root, args.identifier)}...", end=" ")
        os.mkdir(os.path.join(args.root, args.identifier))
        print("success")
    except IOError as e:
        print("could not create directory")
        raise
    # check that /path/to/dataset exists
    if not os.path.isdir(args.dataset):
        raise NotADirectoryError(f"Could not find the dataset directory {args.dataset}.")

    if verbose:
        print(f'''
=============================================
Root dir for run:         {args.root}
Dataset root:             {args.dataset}
Config file path:         {args.config}
Unique run identifier:    {args.identifier}
=============================================
        ''')

    return args.root, args.dataset, args.config, args.identifier

def get_parameters(config_file_path):
    """Transfer arguments to dictionary format."""
    with open(config_file_path, 'r') as yaml_file:
        args = yaml.load(yaml_file, Loader = yaml.FullLoader)
    
    # Load into separate dictionaries
    dataset_params = dict(ChainMap(*args["dataset"]))
    optimiser_params = dict(ChainMap(*args["optimiser"]))
    scheduler_params = dict(ChainMap(*args["scheduler"]))
    model_params = dict(ChainMap(*args["model"]))
    loss_params = dict(ChainMap(*args["loss"]))
    train_params = dict(ChainMap(*args["train_params"]))

    if model_params["prototypes"] == "one_hot":
        assert model_params["latent_dim"] == dataset_params["n_classes"], "In one-hot encoding the latent dimension is the same as the number of classes! Double check the config file."
    elif model_params["prototypes"] =="kasarla":
        assert model_params["latent_dim"] == dataset_params["n_classes"]-1, "The Kasarla prototypes requires the latent dimension to be one smaller than the number of classes! Double check the config file."


    return dataset_params, optimiser_params, scheduler_params, model_params, loss_params, train_params

def train_epoch_encoded(trainloader, model, codebook, loss_fcn, optimiser, device):
    """Performs 1 epoch training with prototypes."""
    model.train()
    for batch_nr, (x,y) in enumerate(trainloader):
        # Load data onto device
        x = x.to(device)
        y = y.to(device)
        # Training code; here is the difference to training with one-hot encoding
        optimiser.zero_grad()
        z = model(x)
        loss = loss_fcn(z, y, codebook)
        loss.backward()
        optimiser.step()

def train_epoch_onehot(trainloader, model, loss_fcn, optimiser, device):
    """Performs 1 epoch training with one-hot encoding."""
    model.train()
    for batch_nr, (x,y) in enumerate(trainloader):
        # Load data onto device
        x = x.to(device)
        y = y.to(device)
        # Training code
        optimiser.zero_grad()
        out = model(x)
        loss = loss_fcn(out, y)
        loss.backward()
        optimiser.step()

def evaluate_onehot(dataloader, model, loss_fcn, device):
    """Calculates top-1 accuracy for the one-hot encoding scheme."""
    # Setup
    model.eval()
    loss = 0.0
    correct = 0.0

    # Loop through and evaluate
    with torch.no_grad():
        for batch_nr, (x,y) in enumerate(dataloader):
            # Load data onto device
            x = x.to(device)
            y = y.to(device)
            out = model(x)
            loss += loss_fcn(out, y)
            # Decode
            correct += batch_correct_onehot(out, y)
    
    # Normalize by number of examples
    accuracy = correct/len(dataloader.dataset)
    loss = loss/len(dataloader.dataset)
    
    return loss, accuracy

def evaluate_encoded(dataloader, model, codebook, loss_fcn, device):
    """Calculates top-1 accuracy with prototypes, using a nearest-neighbour classification rule."""
    # Setup
    model.eval()
    loss = 0.0
    correct = 0.0

    # Loop through and evaluate
    with torch.no_grad():
        for batch_nr, (x,y) in enumerate(dataloader):
            # Load data onto device
            x = x.to(device)
            y = y.to(device)
            z = model(x)
            loss += loss_fcn(z, y, codebook)
            # Decode
            yhat = loss_fcn.decode(z, codebook)
            correct += y.eq(yhat.squeeze()).sum()
        
    # Normalize by number of examples
    accuracy = correct/len(dataloader.dataset)
    loss = loss/len(dataloader.dataset)
    
    return loss, accuracy


def batch_correct_onehot(output, target):
    """Computes top-1 correct predictions over the given (batched) outputs."""
    with torch.no_grad():
        _,pred = output.topk(1,1,True,True) #top-1 over dim 1 (keep batch), sorted
        correct = target.eq(pred.squeeze()).sum()
        return correct

def save_checkpoint(to_save, is_best, savedir, filename="checkpoint.pth"):
    """Saves the `to_save` data in `/path/to/savedir/filename`.
    Overwrites the previous best model if the `is_best` flag is True. """
    # Save checkpoint
    checkpoint_filename = os.path.join(savedir, filename)
    torch.save(to_save, checkpoint_filename)
    # Store separately if best so far
    if is_best:
        best_filename = os.path.join(savedir, "best_model.pth")
        shutil.copyfile(checkpoint_filename, best_filename)

def get_dataloaders(dataset_params, dataset_root_path):
    """
    Load train and test sets from directory
    Known to break for older versions of Torchvision (i.e., that do not use transforms.v2)
    Applies the data augmentation described in the paper.
    """
    if dataset_params["type"] == "mnist":
        mnist_augmentation = transforms.Compose([
            transforms.RandomCrop(size=28, padding=4),
            transforms.RandomRotation(30),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
        ])
        train_dataset = datasets.MNIST(dataset_root_path, train=True, download=False, transform=mnist_augmentation)
        test_dataset = datasets.MNIST(dataset_root_path, train=False, download=False, transform=transforms.Compose([transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)]))
    if dataset_params["type"] == "cifar100":
        # CIFAR-100 means: (0.5071, 0.4865, 0.4409)
        # CIFAR-100 std: (0.2673, 0.2564, 0.2762)
        train_augmentation = transforms.Compose([
            transforms.RandomCrop(size=32, padding=4),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(15),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize((0.5071, 0.4865, 0.4409),(0.2673, 0.2564, 0.2762))
        ])
        test_augmentation = transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize((0.5071, 0.4865, 0.4409),(0.2673, 0.2564, 0.2762))
        ])
        train_dataset = datasets.CIFAR100(dataset_root_path, train=True, download=False, transform=train_augmentation)
        test_dataset = datasets.CIFAR100(dataset_root_path, train=False, download=False, transform=test_augmentation)

    #####
    ## Create val set, loaders, and samplers
    #####
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [0.8, 0.2]) #80% to train set, rest to val set
    
    num_workers = 16
    train_loader = DataLoader(train_dataset, batch_size = dataset_params["batch_size"], shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size = dataset_params["batch_size"], shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size= dataset_params["batch_size"], shuffle=True, num_workers=num_workers)
    return train_loader, val_loader, test_loader

def get_model(model_params):
    """Returns the model specified by the model parameters."""
    if model_params["type"] == "toy_model":
        network = SmallNetwork(model_params["latent_dim"])
    elif model_params["type"] == "resnet34":
        network = ResNetD(depth=34, out_dim=model_params["latent_dim"])
    
    return network
    """return nn.SyncBatchNorm.convert_sync_batchnorm(network)"""
    
def get_optimiser_scheduler(model, optimiser_params, scheduler_params, train_params):
    """Returns the optimiser and learning rate scheduler, 
    as specified by the optimiser and sheduler parameters, respectively."""
    #####
    ## Get optimiser
    #####
    if optimiser_params["type"] == "sgd":
        optimiser = torch.optim.SGD(
            model.parameters(), 
            lr=optimiser_params["lr"],
            momentum=optimiser_params["momentum"],
            weight_decay=optimiser_params["decay"],
            nesterov=optimiser_params["nesterov"]
        )
    elif optimiser_params["type"] == "adam":
        optimiser = torch.optim.Adam(
            model.parameters(),
            lr = optimiser_params["lr"],
            weight_decay=optimiser_params["decay"]
        )
    #####
    ## Get learning rate scheduler
    #####
    if scheduler_params["type"] == "cosine_no_restart":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimiser, T_max=train_params["epochs"])
    elif scheduler_params["type"] == "multi_step":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimiser, milestones=scheduler_params["multi_step_epochs"], gamma=scheduler_params["factor"])

    return optimiser, scheduler
    
def get_loss(loss_params):
    """Returns the loss functions specified by the loss parameters."""
    if loss_params["type"] == "crossentropy":
        loss_fcn = nn.CrossEntropyLoss()
    elif loss_params["type"] == "prototypical":
        loss_fcn = PrototypicalLoss()
    elif loss_params["type"] == "nll":
        raise NotImplementedError("Please use CrossEntropyLoss instead. Models do not have log-softmax layers, so to avoid unwanted behaviour, avoid NLLLoss.")

    return loss_fcn

def get_code(model_params, dataset_params):
    """Returns a codebook of hyperspherical prototypes as specified by the parameters."""
    if model_params["prototypes"] == "random_code":
        code = RandomCode(latent_dim=model_params["latent_dim"], num_classes=dataset_params["n_classes"])
    elif model_params["prototypes"] == "rm_code":
        code = ReedMullerCode(latent_dim=model_params["latent_dim"], num_classes=dataset_params["n_classes"])
    elif model_params["prototypes"] == "bch_code":
        code = BCHCode(latent_dim=model_params["latent_dim"], num_classes=dataset_params["n_classes"])
    elif model_params["prototypes"] == "mettes":
        code = MettesCode(latent_dim=model_params["latent_dim"], num_classes=dataset_params["n_classes"])
    elif model_params["prototypes"] == "logsumexp":
        code = LogSumExpCode(latent_dim=model_params["latent_dim"], num_classes=dataset_params["n_classes"])
    elif model_params["prototypes"] == "kasarla":
        code = KasarlaCode(num_classes=dataset_params["n_classes"])
    return code


if __name__ == "__main__":
    pass
