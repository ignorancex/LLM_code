from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
import torch

OUT_CHANNELS = {
    "keypoints3d": 1,
    "reshading": 1,
    "edge_texture": 1,
    "normal": 3,
    "segment_semantic": 18,
    "class_scene": 0,
    "depth_euclidean": 1,
    "principal_curvature": 2,
    "principal_curvature_old": 3
}

def plot_data_distribution(network, num_classes, dirichlet_distribution):
    left = np.zeros(network.n_clients)
    plt.figure()
    plt.rcParams.update({'font.size': 13})
    colors = plt.cm.get_cmap('tab20', num_classes)
    for i in range(num_classes):
        # print(f"client {i} - {dirichlet_distribution[i]}")
        plt.barh(range(network.n_clients), dirichlet_distribution[:,i], left=left, color= colors(i))
        left += dirichlet_distribution[:,i]
    
    plt.xlabel('Class Ratio')
    plt.ylabel('Client')
    plt.yticks(range(network.n_clients), [f'{i+1}' for i in range(network.n_clients)])
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.title(rf'$\alpha = 10^{int(np.log10(network.alpha))}$')
    plt.tight_layout()
    plt.savefig(os.path.join(network.save_dir, f"{network.task}_data_distribution_plot_alpha_{network.alpha}.png"))

def plot_tsne(embeddings, assignments, task, result_path, labels, number=None):

    tsne = TSNE(n_components=2, perplexity=10, n_iter=300)
    tsne_embeddings = tsne.fit_transform(embeddings)

    # Plot the TSNE embeddings
    plt.figure(figsize=(10, 10))
    plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], c=assignments, cmap='tab20')
    plt.xticks([])
    plt.yticks([])
    plt.title('TSNE of embeddings - Clusterwise')
    plt.savefig(os.path.join(result_path, f"{task}_cluster_tsne_embeddings.png"))

    if task=="class_scene":

        plt.figure(figsize=(10, 10))
        colors = sns.color_palette("tab10", len(labels.unique()))
        print(colors)
        for i in labels.unique():
            idx = labels == i
            idx = idx.cpu().numpy()
            idx = np.where(idx)[0]
            print(type(idx))
            print(type(tsne_embeddings))
            print(tsne_embeddings.shape)
            print(idx[:5])
            print(f"i: {i}")
            print(f"colors i: {colors[int(i)]}")
            plt.scatter(tsne_embeddings[idx, 0], tsne_embeddings[idx, 1], color=colors[int(i)], label='Class {}'.format(i))
        plt.xticks([])
        plt.yticks([])
        plt.title('TSNE of the embeddings - Classwise')
        plt.legend()
        plt.savefig(os.path.join(result_path, f"{task}_class_tsne_embeddings.png"))


def plot_loss(loss_hist, result_path, task):
    #plot the losses
    plt.plot(loss_hist)
    plt.title("Loss history")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(result_path, f"{task}_clip_encoder_loss.png"))

def plot_outputs(test_outputs, test_targets, task, result_path):

    # Suppress the clipping warning
    warnings.filterwarnings("ignore", category=UserWarning, message="Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).")

    # pick color map based on task
    if OUT_CHANNELS[task] == 1:
        cmap = 'gray'
    else:
        cmap = None

    print(f"cmap chosen: {cmap}")
    # Plot the first 3 outputs and targets to see how well the model is doing
    plt.figure(figsize=(10, 10))
    # Create a flag to show if .permute() is needed
    need_permute = test_outputs.size()[1] in [2, 3]
    augment_channels = OUT_CHANNELS[task] == 2

    for i in range(3):
        
        # If the output channels is 2, add a third zero channel and permute the tensor
        if augment_channels:
            plotted_output = torch.cat((test_outputs[i], torch.zeros_like(test_outputs[i][:1])), dim=0)
            plotted_target = torch.cat((test_targets[i], torch.zeros_like(test_targets[i][:1])), dim=0)
        else:
            plotted_output = test_outputs[i]
            plotted_target = test_targets[i]

        plt.subplot(3, 2, 2*i+1)
        if need_permute:
            plt.imshow((plotted_output.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8), cmap=cmap)
        else:
            plt.imshow(plotted_output.squeeze().cpu().numpy(), cmap=cmap)
        plt.title('Output')
        plt.axis('off')
        
        plt.subplot(3, 2, 2*i+2)
        if need_permute:
            plt.imshow((plotted_target.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8), cmap=cmap)
        else:
            plt.imshow(plotted_target.squeeze().cpu().numpy(), cmap=cmap)
        plt.title('Target')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(result_path, f"{task}_clip_encoder_outputs.png"))


# Plot the outputs and targets for the segmentation task
def plot_outputs_segment(test_outputs, test_targets, task, result_path):

    # Suppress the clipping warning
    warnings.filterwarnings("ignore", category=UserWarning, message="Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).")

    # Plot the first 3 outputs and targets to see how well the model is doing
    plt.figure(figsize=(10, 10))

    test_outputs = torch.argmax(test_outputs, dim=1)
    print(test_outputs.shape)
    print(test_targets.shape)

    for i in range(3):
        plt.subplot(3, 2, 2*i+1)
        plt.imshow(test_outputs[i].squeeze().cpu().numpy())
        plt.title('Output')
        plt.axis('off')
        plt.subplot(3, 2, 2*i+2)
        plt.imshow(test_targets[i].squeeze().cpu().numpy())
        plt.title('Target')
        plt.axis('off')
    plt.savefig(os.path.join(result_path, f"{task}_clip_encoder_outputs.png"))


# Plot the confusion matrix for class_scene task
def plot_confusion_matrix(test_outputs, test_targets, task, result_path):
    """
    Plots and saves a confusion matrix.
    Parameters:
    cm (array-like): Confusion matrix to be plotted.
    classes (list): List of class names corresponding to the labels.
    result_path (str): Path where the confusion matrix image will be saved.
    task (str): Name of the task, used to name the saved image file.
    Returns:
    None
    """
    
    # Get the predicted labels
    _, predicted = torch.max(test_outputs, 1)
    predicted = predicted.cpu().numpy()
    classes = test_targets.unique().cpu().numpy()
    test_targets = test_targets.cpu().numpy()
    # Calculate the confusion matrix
    cm = confusion_matrix(test_targets, predicted)

    # Plot the confusion matrix
    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(result_path, f"{task}_confusion_matrix.png"))