import numpy as np
import math
import pandas as pd
from PIL import Image, ImageDraw
import os
import re
from combine_fornoise import combine_snapshots

# Define the directory containing the input images
input_images_dir = "./generated_images/"
# input_images_dir = "./test_img/"
image_files = [f for f in os.listdir(input_images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]


# Sort the image files based on the number in their filenames (assuming filenames like "pool1.png", "pool2.jpg", etc.)
def extract_number(filename):
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else 0


# Function to add noise to X, Y, and Theta values
def add_noise_and_save(snapshots_df, noise_range_xy, noise_range_theta, output_dir, base_filename, num_files=10):
    for i in range(1, num_files + 1):
        # Add noise to the X and Y coordinates
        # snapshots_df['X'] += np.random.randint(-noise_range_xy, noise_range_xy + 1, size=len(snapshots_df))
        # snapshots_df['Y'] += np.random.randint(-noise_range_xy, noise_range_xy + 1, size=len(snapshots_df))

        snapshots_df['X'] += np.random.normal(0, 1, size=len(snapshots_df)).astype(int)
        snapshots_df['Y'] += np.random.normal(0, 1, size=len(snapshots_df)).astype(int)

        # # Add noise to the Theta values
        # ### change this part to make it from a gaussian distrition
        # snapshots_df['Theta'] += np.random.random(-noise_range_theta, noise_range_theta, size=len(snapshots_df))

        # Add Gaussian noise to the Theta values
        noise_std_theta = 0.5
        snapshots_df['Theta'] += np.random.normal(0, noise_std_theta, size=len(snapshots_df))

        # Create the noise folder if it doesn't exist
        noise_folder = os.path.join(output_dir, 'noise')
        os.makedirs(noise_folder, exist_ok=True)

        # Save the noisy DataFrame to an Excel file
        noisy_file_path = os.path.join(noise_folder, f"{base_filename}_noise{i}.xlsx")
        snapshots_df.to_excel(noisy_file_path, index=False)
        print(f"Saved noisy file: {noisy_file_path}")


image_files.sort(key=extract_number)

# Define the window and snapshot sizes
window_size = (200, 200)
snapshot_size = (160, 110)
gap = 10

# Loop over each input image
for i, image_file in enumerate(image_files, start=1):
    # Create a directory for each input image named after the image (excluding extension)
    image_name = os.path.splitext(image_file)[0]
    output_dir = os.path.join("./data_100/", image_name)
    # output_dir = os.path.join("./test_1000/", image_name)
    os.makedirs(output_dir, exist_ok=True)

    # Save the input image in the directory
    input_image_save_path = os.path.join(output_dir, image_file)
    image_path = os.path.join(input_images_dir, image_file)
    image = Image.open(image_path)
    image.save(input_image_save_path)

    # Create a subdirectory for the snapshots named after the image (excluding extension)
    snapshots_dir = os.path.join(output_dir, image_name)
    os.makedirs(snapshots_dir, exist_ok=True)

    # Get image dimensions
    image_width, image_height = image.size

    # Calculate the available area for the central points
    x_min = gap
    y_min = gap
    x_max = image_width - gap
    y_max = image_height - gap

    # Generate uniformly distributed central points within the available area
    np.random.seed(i)
    central_points_x = np.random.uniform(x_min, x_max, 221).astype(int)
    central_points_y = np.random.uniform(y_min, y_max, 221).astype(int)
    central_points = list(zip(central_points_x, central_points_y))

    # Initialize a list to hold the information about each snapshot
    snapshots_info = []

    # Process each central point
    for n, point in enumerate(central_points, start=1):
        # Crop the initial [200, 200] window
        top_left = (point[0] - window_size[0] // 2, point[1] - window_size[1] // 2)
        cropped_window = image.crop(
            (top_left[0], top_left[1], top_left[0] + window_size[0], top_left[1] + window_size[1]))

        # Generate a random rotation angle between 0 and 2 radians
        rotation_angle_radians = np.random.uniform(0, 2)
        rotation_angle_degrees = math.degrees(rotation_angle_radians)

        # Rotate the cropped window
        rotated_window = cropped_window.rotate(rotation_angle_degrees, expand=True)

        # Crop the [160, 110] snapshot from the center of the rotated image
        rotated_center = (rotated_window.width // 2, rotated_window.height // 2)
        final_top_left = (rotated_center[0] - snapshot_size[0] // 2, rotated_center[1] - snapshot_size[1] // 2)
        final_cropped_snapshot = rotated_window.crop((final_top_left[0], final_top_left[1],
                                                      final_top_left[0] + snapshot_size[0],
                                                      final_top_left[1] + snapshot_size[1]))

        # Save the snapshot as an image file in the snapshots directory
        snapshot_filename = f"{image_name}_{n}.png"
        final_cropped_snapshot.save(os.path.join(snapshots_dir, snapshot_filename))

        # Save the snapshot coordinates and corresponding angle in radians
        snapshot_info = {
            "Filename": snapshot_filename,
            "Input Image": image_file,
            "X": point[0],
            "Y": point[1],
            "Theta": rotation_angle_radians
        }
        snapshots_info.append(snapshot_info)

    # Convert the list of dictionaries to a DataFrame
    snapshots_df = pd.DataFrame(snapshots_info)

    # Save the DataFrame to an Excel file in the output directory
    output_snapshots_path = os.path.join(output_dir, f"{image_name}_snapshots_info.xlsx")
    snapshots_df.to_excel(output_snapshots_path, index=False)

    print(f"Processed {image_file}: snapshots and information saved to {output_dir}")




#########################------------------------------------------add noise

# Parameters for noise
noise_range_xy = 100  # Noise range for X and Y is [-100, 100]
noise_range_theta = 0.5  # Noise range for Theta is [-0.5, 0.5]
num_noisy_files = 100  # Number of noisy files to generate for each .xlsx file

# Directory containing the pool directories (e.g., pool1, pool2, ...)
# root_dir = './data_100/'
root_dir = './data_100/'
# root_dir = './test_1000/'
# out_dir = './test'

# Iterate over each pool directory
for pool_dir in os.listdir(root_dir):
    pool_path = os.path.join(root_dir, pool_dir)
    if os.path.isdir(pool_path):
        # Find the .xlsx file in the pool directory
        for file in os.listdir(pool_path):
            if file.endswith('.xlsx'):
                xlsx_path = os.path.join(pool_path, file)
                snapshots_df = pd.read_excel(xlsx_path)

                # Generate noisy versions of the .xlsx file
                base_filename = os.path.splitext(file)[0]
                add_noise_and_save(snapshots_df, noise_range_xy, noise_range_theta, pool_path, base_filename,
                                   num_noisy_files)




combine_snapshots(root_dir)