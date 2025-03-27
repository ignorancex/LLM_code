from PIL import Image, ImageDraw, ImageFont

# File paths
t2t_image_path = '../figs/visualizations/t2t/sentiment_distribution_t2t.png'
i2t_image_path = '../figs/visualizations/i2t/sentiment_distribution_i2t.png'
output_path = '../figs/visualizations/sentiment_distribution_pval.png'

# Open the images
t2t_image = Image.open(t2t_image_path)
i2t_image = Image.open(i2t_image_path)

# Get dimensions
width_t2t, height_t2t = t2t_image.size
width_i2t, height_i2t = i2t_image.size

# Calculate new width (max of both) and total height
new_width = max(width_t2t, width_i2t)
total_height = height_t2t + height_i2t

# Create a new image with a white background
merged_image = Image.new('RGB', (new_width, total_height), (255, 255, 255))

# Paste the images into the new image
merged_image.paste(t2t_image, (0, 0))
merged_image.paste(i2t_image, (0, height_t2t - 190))  # Adjust this value to control overlap

# Add sidebar
sidebar_width = 100  # Adjust sidebar thickness as needed
final_width = new_width + sidebar_width

# Create a new image to include the sidebar
final_image = Image.new('RGB', (final_width, total_height), (255, 255, 255))

# Paste the merged image into the new image with space for the sidebar
final_image.paste(merged_image, (sidebar_width, 0))

# Draw the sidebar annotations
draw = ImageDraw.Draw(final_image)

# Font settings
font_size = 100
try:
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"  # Common font path
    font = ImageFont.truetype(font_path, font_size)
except IOError:
    font = ImageFont.load_default()

# Add text to the sidebar
annotations = [("(a)", 0.05), ("(b)", 0.275), ("(c)", 0.52), ("(d)", 0.74)]

for text, y_pos in annotations:
    draw.text((10, int(y_pos * total_height)), text, fill="black", font=font)

# Save the new image with the sidebar
final_image.save(output_path)

print(f"Merged image with sidebar saved at {output_path}")
