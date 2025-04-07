import os, glob, json
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

def render_text_to_image(text, font_path, image_resolution):
    font_size = 100  # You can adjust this based on your preference
    font = ImageFont.truetype(font_path, font_size)
    
    res = []
    for char in text:
        image = Image.new('RGB', (image_resolution, image_resolution), color='white')
        draw = ImageDraw.Draw(image)
        
        # Calculate the position to center the character in the image
        text_width, text_height = draw.textsize(char, font)
        x = (image_resolution - text_width) // 2
        y = (image_resolution - text_height) // 2
        
        draw.text((x, y), char, font=font, fill='black')
        res.append([char, image])
    return res

if __name__ == "__main__":
    tar = "/DATA/bvac/personal/reserach/font_gen/mxfont/data/images/test"  # Specify the output folder
    input_text = "留自落街"
    image_resolution = 128
    
    src = '/DATA/bvac/personal/reserach/font_gen/use_fonts/ch'
    font_names = os.listdir(src)
    font_names = set([os.path.basename(each) for each in font_names if each.endswith('.ttf')])

    for font_name in tqdm(font_names):
        ttf_file_path = os.path.join(src, font_name)
        output_folder = os.path.join(tar, font_name.split('.')[0])
        os.makedirs(output_folder, exist_ok=True)
        
        images = render_text_to_image(input_text, ttf_file_path, image_resolution)
        
        for each in images:
            each[1].convert("L").save(os.path.join(output_folder, each[0]+'.png'))
    