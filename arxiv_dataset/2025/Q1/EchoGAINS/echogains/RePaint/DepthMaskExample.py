import os
from PIL import Image

def duplicate_image(directory, output_directory, num_copies=10):
    for filename in os.listdir(directory):
        if filename.lower().endswith('.png'):   
            image_path = os.path.join(directory, filename)
            os.makedirs(os.path.dirname(output_directory), exist_ok=True)
            img = Image.open(image_path)
            for i in range(1, num_copies + 1):
                new_filename = f"{i:06d}.png"
                output_path = os.path.join(output_directory, new_filename)
                img.save(output_path)
                print(f'Copy {i} saved as {output_path}')


def create_mask(image_path, output_path):
    img = Image.open(image_path).convert("L") #if not
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    img = img.point(lambda p: 255 if p == 0 else 0)  
    pixels = img.load()
    width, height = img.size

    for x in range(width):
        lowest_black_pixel = -1
        for y in range(height - 1, -1, -1):  
            if pixels[x, y] == 0:
                lowest_black_pixel = y
                break

        if lowest_black_pixel != -1:
            height_to_change = int(0.65 * lowest_black_pixel) #Lower part of image becomes masked
            for y in range(height_to_change):
                pixels[x, y] = 255  

    img.save(output_path)

if __name__ == '__main__':
    image_datapath = "/home/gillesv/PycharmProjects/repaint_aksel_lenz/RePaint/temp/image_dir/000000.png"

    create_mask(image_datapath, "/home/gillesv/PycharmProjects/repaint_aksel_lenz/RePaint/temp/mask_dir/000000.png")
    duplicate_image("/home/gillesv/PycharmProjects/repaint_aksel_lenz/RePaint/temp/image_dir", "/home/gillesv/PycharmProjects/repaint_aksel_lenz/RePaint/temp/out_dir")
    #duplicate_image("/home/gillesv/PycharmProjects/repaint_aksel_lenz/RePaint/temp/mask_dir", "/home/gillesv/PycharmProjects/repaint_aksel_lenz/RePaint/temp/out_dir")
