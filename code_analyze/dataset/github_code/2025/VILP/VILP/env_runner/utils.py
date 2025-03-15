import numpy as np
import cv2

def resize_images(batch_images, new_height, new_width):
    """
    Resize a batch of images to a new height and width.

    Parameters:
    - batch_images: numpy array of shape (batch, steps, 3, H, W)
    - new_height: int, the new height of the images
    - new_width: int, the new width of the images

    Returns:
    - resized_images: numpy array of shape (batch, steps, 3, H2, W2)
    """
    batch, steps, channels, _, _ = batch_images.shape
    resized_images = np.empty((batch, steps, channels, new_height, new_width), dtype=batch_images.dtype)
    
    for i in range(batch):
        for j in range(steps):
            image = batch_images[i, j, :, :, :].transpose(1, 2, 0)  # Reshape to (H, W, C)
            resized_image = cv2.resize(image, (new_width, new_height))
            resized_images[i, j, :, :, :] = resized_image.transpose(2, 0, 1)  # Back to (C, H, W)
    
    return resized_images