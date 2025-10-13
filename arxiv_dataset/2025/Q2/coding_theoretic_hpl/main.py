import torch 
from torch.utils.tensorboard import SummaryWriter

import os
import numpy as np
from datetime import datetime
import shutil

from utils import parse_arguments, get_parameters, get_dataloaders, \
    get_model, get_optimiser_scheduler, get_loss, train_epoch_onehot, train_epoch_encoded, \
        evaluate_onehot, evaluate_encoded, save_checkpoint, get_code

def main():
    """Main file."""

    #####
    ## Parse instructions
    #####

    # Parse arguments
    save_root_path, dataset_root_path, config_path, run_identifier = parse_arguments()
    # Parse config file
    dataset_params, optimiser_params, scheduler_params, model_params, loss_params, train_params = get_parameters(config_path)
    # Set up the logger in /path/to/root_path/identifier, with file suffix '.logfile'
    logger = SummaryWriter(os.path.join(save_root_path, run_identifier), filename_suffix=".logfile")
    logger.add_text("Dataset Parameters", " ---- ".join(f"{k}:{v}" for k,v in dataset_params.items()))
    logger.add_text("Optimiser Parameters", " ---- ".join(f"{k}:{v}" for k,v in optimiser_params.items()))
    logger.add_text("Scheduler Parameters", " ---- ".join(f"{k}:{v}" for k,v in scheduler_params.items()))
    logger.add_text("Model Parameters", " ---- ".join(f"{k}:{v}" for k,v in model_params.items()))
    logger.add_text("Loss Parameters", " ---- ".join(f"{k}:{v}" for k,v in loss_params.items()))

    #####
    ## Training setup 
    #####

    # Set device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Get dataloaders
    trainloader, valloader, testloader = get_dataloaders(dataset_params, dataset_root_path)

    # Initialise model
    model = get_model(model_params).to(device)
    
    # Initialise a codebook if desired
    if model_params["prototypes"] != "one_hot":
        code = get_code(model_params, dataset_params).to(device)
        code.verify_quality()
        codebook = code.codebook

    # Instantiate optimiser and learning rate scheduler
    optimiser, scheduler = get_optimiser_scheduler(model, optimiser_params, scheduler_params, train_params)

    #Get loss function
    loss_fcn = get_loss(loss_params)

    #####
    ## Training loop
    #####
    print(f"=============================================\nStarting training\n=============================================")
    best_acc = torch.tensor(0)

    for epoch in range(train_params["epochs"]):
        print(f"Epoch {epoch+1}/{train_params['epochs']} -- ", end="")
        epoch_start_time = datetime.now()

        # Train loop
        if model_params["prototypes"] == "one_hot":
            train_epoch_onehot(trainloader, model, loss_fcn, optimiser, device)
        else:
            train_epoch_encoded(trainloader, model, codebook, loss_fcn, optimiser, device)
        scheduler.step()

        # Evaluation/logging/...
        print(str(datetime.now()-epoch_start_time).split(".")[0], "s") 

	    # Validate and checkpoint
        if (epoch%train_params["val_freq"] == 0):
            # Calculate loss and accuracy for all workers
            if model_params["prototypes"] == "one_hot":            
                train_loss, train_acc = evaluate_onehot(trainloader, model, loss_fcn, device)
                val_loss, val_acc = evaluate_onehot(valloader, model, loss_fcn, device)
            else:
                train_loss, train_acc = evaluate_encoded(trainloader, model, codebook, loss_fcn, device)
                val_loss, val_acc = evaluate_encoded(valloader, model, codebook, loss_fcn, device)

            # Print and log
            print(f"\t -- Train loss: {train_loss:.2e} -- Train acc: {train_acc*100:.2f}% \n\t -- Val loss: {val_loss:.2e} -- Val acc: {val_acc*100:.2f}%")
            logger.add_scalar("Train/Loss", train_loss, epoch+1)
            logger.add_scalar("Train/Acc", train_acc, epoch+1)
            logger.add_scalar("Val/Loss", val_loss, epoch+1)
            logger.add_scalar("Val/Acc", val_acc, epoch+1)

            # Checkpoint
            is_best = val_acc > best_acc #save separately if this is the best
            best_acc = torch.maximum(val_acc, best_acc)
            if model_params["prototypes"] == "one_hot":
                save_checkpoint({
                    'epoch': epoch+1,
                    'parameters': model.state_dict(),
                    'optimiser': optimiser.state_dict(),
                    'scheduler': scheduler.state_dict()
                }, is_best, os.path.join(save_root_path,run_identifier))
            else:
                save_checkpoint({
                    'epoch': epoch+1,
                    'parameters': model.state_dict(),
                    'codebook' : code.state_dict(),
                    'optimiser': optimiser.state_dict(),
                    'scheduler': scheduler.state_dict()
                }, is_best, os.path.join(save_root_path,run_identifier))

    #####
    ## Test and save final model
    #####
    if model_params["prototypes"] == "one_hot":            
        torch.save({"parameters" : model.state_dict()}, os.path.join(*(save_root_path, run_identifier, "final_model.pth")))
        test_loss, test_acc = evaluate_onehot(testloader, model, loss_fcn, device)
    else:
        torch.save({"parameters" : model.state_dict(), "codebook" : code.state_dict()}, os.path.join(*(save_root_path, run_identifier, "final_model.pth")))
        test_loss, test_acc = evaluate_encoded(testloader, model, codebook, loss_fcn, device)

    print("---------\nTest performance")
    print(f"\t -- Test loss: {test_loss:.2e} -- Test acc: {test_acc*100:.2f}%")
    logger.add_scalar("Test/Loss", test_loss, train_params["epochs"])
    logger.add_scalar("Test/Acc", test_acc, train_params["epochs"])

    # Cleanup
    logger.close()


if __name__ == "__main__":
    main()
