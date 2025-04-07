import numpy as np
import math
import pandas as pd
from PIL import Image, ImageDraw, ImageOps
import os
import matplotlib.pyplot as plt


# Function to convert meters to pixels
def meters_to_pixels(meters, dpi):
    inch_to_mm = 25.4
    pixels = int((meters * 1000) / inch_to_mm * dpi)
    return pixels


# Directory where the pool folders are located
root_dir = './data'

def combine_snapshots(root_dir):

    # Iterate over each pool directory
    for pool_dir in os.listdir(root_dir):
        pool_path = os.path.join(root_dir, pool_dir)
        if os.path.isdir(pool_path):
            noise_folder = os.path.join(pool_path, 'noise')

            # Iterate over each noisy .xlsx file in the noise folder
            for noise_file in os.listdir(noise_folder):
                if noise_file.endswith('.xlsx'):
                    # Load the noisy Excel file
                    noise_xlsx_path = os.path.join(noise_folder, noise_file)
                    snapshots_df = pd.read_excel(noise_xlsx_path)

                    # Create a blank canvas and mask with the specified size
                    canvas_size = (2245, 1587)
                    canvas = Image.new("RGB", canvas_size)
                    mask = Image.new("L", canvas_size, 0)  # "L" mode for grayscale, initialized to 0 (black)
                    dpi = 75

                    # Directory where the snapshots are saved (e.g., pooln/pooln)
                    snapshots_dir = os.path.join(pool_path, pool_dir)

                    # Iterate over the DataFrame and place each rotated snapshot on the canvas and mask
                    for index, row in snapshots_df.iterrows():
                        snapshot_image_path = os.path.join(snapshots_dir, row['Filename'])
                        snapshot_image = Image.open(snapshot_image_path)

                        # Convert the image to RGBA mode if it has a palette and transparency
                        if snapshot_image.mode == "P":
                            snapshot_image = snapshot_image.convert("RGBA")

                        # Apply the negative rotation from the Excel file
                        rotated_snapshot = snapshot_image.rotate(-math.degrees(row['Theta']), expand=True)

                        # Determine the position in pixels
                        if isinstance(row['X'], float):
                            x_pos = meters_to_pixels(row['X'], dpi)
                        else:
                            x_pos = row['X'] - rotated_snapshot.width // 2

                        if isinstance(row['Y'], float):
                            y_pos = meters_to_pixels(row['Y'], dpi)
                        else:
                            y_pos = row['Y'] - rotated_snapshot.height // 2

                        # Paste the rotated snapshot onto the canvas
                        canvas.paste(rotated_snapshot, (x_pos, y_pos), rotated_snapshot.convert("RGBA"))

                        # Create a mask from the rotated snapshot where the non-transparent pixels are marked
                        snapshot_mask = Image.new("L", rotated_snapshot.size, 0)
                        snapshot_mask.paste(255, (0, 0),
                                            rotated_snapshot.split()[3])  # Use the alpha channel to create the mask

                        # Paste the mask onto the main mask canvas
                        mask.paste(snapshot_mask, (x_pos, y_pos), snapshot_mask)

                    # Save the final stitched image in the noise/img folder
                    img_output_dir = os.path.join(noise_folder, 'img')
                    os.makedirs(img_output_dir, exist_ok=True)
                    stitched_image_path = os.path.join(img_output_dir, f"{os.path.splitext(noise_file)[0]}.png")
                    canvas.save(stitched_image_path)

                    # Save the corresponding mask in the noise/mask folder
                    mask_output_dir = os.path.join(noise_folder, 'mask')
                    os.makedirs(mask_output_dir, exist_ok=True)
                    mask_image_path = os.path.join(mask_output_dir, f"{os.path.splitext(noise_file)[0]}_mask.png")
                    mask.save(mask_image_path)

                    # Display the final stitched image and the mask
                    # plt.figure(figsize=(10, 8))
                    # plt.imshow(canvas)
                    # plt.title(f"Stitched Image for {os.path.splitext(noise_file)[0]}")
                    # plt.axis("off")
                    # plt.show()
                    #
                    # plt.figure(figsize=(10, 8))
                    # plt.imshow(mask, cmap='gray')
                    # plt.title(f"Mask for {os.path.splitext(noise_file)[0]}")
                    # plt.axis("off")
                    # plt.show()

                    print(f"Stitched image saved as {stitched_image_path}")
                    print(f"Mask image saved as {mask_image_path}")
