import torch
import torch.nn.functional as F


def rand_projections(dim, num_projections=1000):
    projections = torch.randn((num_projections, dim))
    projections = projections / torch.sqrt(torch.sum(projections ** 2, dim=1, keepdim=True))
    return projections


def calculate_batch_certainty(images_batch):
    """
    Calculate the average pixel-wise certainty of each image in a batch and sum it over the batch using max(t, 1-t),
    where t is the estimated probability of each pixel being one of the values (considering a binary scenario).

    Args:
        images_batch (torch.Tensor): A batch of images, with shape (batch_size, channels, height, width).

    Returns:
        torch.Tensor: The total sum of average pixel-wise certainty over the batch.
    """
    total_certainty = 0

    for image in images_batch:
        # Flatten the image to shape (channels, height * width)
        image_flat = image.view(image.size(0), -1)  # (channels, height * width)

        # Calculate certainty using max(t, 1-t)
        certainty = torch.max(image_flat, 1 - image_flat)
        average_certainty = torch.mean(certainty)

        # Add the average certainty to the total certainty
        total_certainty += average_certainty

    # Divide by the number of images to get the average over the batch
    return total_certainty
