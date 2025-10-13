import torch
import matplotlib.pyplot as plt
import os


def extract_embeddings(model, dataloader, device, task, test_mode:bool=False):
    """
    Extract embeddings from the model and plot the first 3 outputs and targets to see how well the model is doing

    Args:
    - model: the model to extract embeddings from
    - dataloader: the dataloader to extract embeddings from
    - device: the device to run the model on
    - task: the task to extract embeddings for
    - result_path: the path to save the plots
    
    Returns:
    - embeddings: the embeddings extracted from the model
    """
    embeddings = []
    test_outputs = []
    test_targets = []

    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):

            if test_mode and i>10:
                break
            
            # change images dtype to float16
            images = images.type(torch.float32).to(device)
            labels[task] = labels[task].type(torch.float32).to(device)

            # with torch.no_grad():
            #     image_features = sample_clip_model.encode_image(images)
            # _, outputs = decoder_under_train(image_features)
            
            embedding, outputs = model(images)
            
            embeddings.append(embedding.detach().cpu())
            if (task != 'class_scene' and i<3) or task == 'class_scene':
                test_outputs.append(outputs.detach().cpu())
                test_targets.append(labels[task].detach().cpu())
            else:
                continue

    embeddings = torch.cat(embeddings, dim=0)

    # Plot the first 3 outputs and targets to see how well the model is doing
    test_outputs = torch.cat(test_outputs, dim=0)
    test_targets = torch.cat(test_targets, dim=0)

    return embeddings, test_outputs, test_targets