import os
import cv2

def resize_images_in_place(folder, target_size=(320, 240)):
    """
    Resize all images in a folder to the specified target size and overwrite the originals.

    Args:
        folder (str): Path to the folder containing image files.
        target_size (tuple): Target image size as (width, height).
    """
    # Get all files in the folder
    files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]

    for file in files:
        # Full file path
        file_path = os.path.join(folder, file)

        # Read the image
        image = cv2.imread(file_path)
        if image is None:
            print(f"Failed to read {file_path}")
            continue

        # Resize image to target size
        resized_image = cv2.resize(image, target_size)

        # Overwrite the original image with resized image
        cv2.imwrite(file_path, resized_image)

        print(f"Resized image saved to {file_path}")

# Replace with your actual folder paths:
folder = 'C:/Users/86138/Desktop/target_result/SRPCA/overpass/'

resize_images_in_place(folder, target_size=(320, 240))
