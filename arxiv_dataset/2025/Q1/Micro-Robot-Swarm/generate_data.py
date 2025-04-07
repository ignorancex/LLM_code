from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import os

# Load the coordinates from the uploaded Excel file
file_path = './filtered_midpoints_coordinates.xlsx'
coordinates_df = pd.read_excel(file_path)

# Load the image
image_path = './modified_image.png'
original_image = Image.open(image_path)

# Define the radius of the white circles
radius = 17  # This can be adjusted according to your needs

# Specify the number of images to generate
n = 10  # Adjust this value as needed

# Create output directory if it doesn't exist
output_dir = './test_img'
os.makedirs(output_dir, exist_ok=True)

for i in range(n):
    # Create a copy of the original image for each iteration
    image = original_image.copy()
    draw = ImageDraw.Draw(image)

    # Draw white circles on the image at each midpoint from the Excel file
    for _, row in coordinates_df.iterrows():
        x, y = row['X Coordinate'], row['Y Coordinate']
        if random.random() < 0.5:
            # Randomize the size of the black circle
            draw.ellipse([(x - radius, y - radius),
                          (x + radius, y + radius)], fill="black")

    # Convert the image to a NumPy array
    # img_array = np.array(image)

    # Plot and save the image without axis and black boundary
    # plt.figure(figsize=(10, 8))
    # plt.imshow(image, cmap='gray')
    # plt.axis('off')
    # # plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    #
    # plt.show()

    # Save the image
    output_path = os.path.join(output_dir, f'pool{i+1}.png')
    # plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    # plt.close()
    image.save(output_path)

print(f'{n} images have been generated and saved in {output_dir}.')
