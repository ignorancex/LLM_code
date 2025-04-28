"""Function used to finetune base models
"""
import copy
import time
import torch
import logging
from src.data_zsre import create_dataloader
from src.loss_functions import multi_loss_fn
from src.evaluate import evaluate_models_fn


def training(args, model_original, tokenizer, train_data, test_data, base_dir, lr=1e-4):
    """ Performs SGD on the model_base to finetune on a subset of the training data
    :param args: Arguments of model_base.
    :param model_original: Original model (pretrained, no finetuning).
    :param tokenizer: Tokienzer for given model.
    :param train_data: Subset of training data to finetune model.
    :param test_data: Test dataset, 
    :param lr: Learning rate used in SGD.
    :return logs: Finetuned model, list of results (loss over train_data and test_data).
    """
    # Set device and model
    device = args.device
    model_ft = copy.deepcopy(model_original).to(device)

    # Evaluate original model
    logs = [evaluate_models_fn(
        args, model_ft, tokenizer, train_data, test_data)]
    logging.info(f"{logs}")

    # Create optimizer/ create finetune dataset
    optimizer = torch.optim.Adam(model_ft.parameters(), lr=lr)
    dataloader_finetune = create_dataloader(train_data, args.batch_size)

    # SGD to finetune
    logging.info("Start Finetuning")
    for epoch in range(args.iterations):
        t1 = time.time()
        model_ft.train()
        for batch_to_finetune in dataloader_finetune:
            model_ft.zero_grad()
            loss = multi_loss_fn(model_ft, batch_to_finetune, args.task)
            loss.backward()
            optimizer.step()
            model_ft.train()
        logs.append(evaluate_models_fn(args, model_ft,
                    tokenizer, train_data, test_data))
        if args.task == "zsre":
            print(
                f"Epoch {epoch+1}\t{logs[-1]['loss_train']:.2f}\t{logs[-1]['loss_test']:.2f}\t{time.time() - t1:.2f}")
        if args.task == "wiki":
            print(
                f"Epoch {epoch+1}\t{logs[-1]['loss_train']:.2f}\t{logs[-1]['loss_test']:.2f}\t{time.time() - t1:.2f}")

    # Save Final Model
    torch.save(model_ft.state_dict(), base_dir+args.results_dir+"/model_" +
               args.task+"_"+str(args.data_seed)+"_"+str(args.n)+"_"+str(args.n_test))
    return (model_ft, logs)
