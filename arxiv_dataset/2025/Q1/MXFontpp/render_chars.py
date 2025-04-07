from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


def render_text_to_image(text, font_path, output_folder, image_resolution):
    font_size = 100  # You can adjust this based on your preference
    font = ImageFont.truetype(font_path, font_size)
    
    for char in tqdm(text):
        image = Image.new('RGB', (image_resolution, image_resolution), color='white')
        draw = ImageDraw.Draw(image)
        
        # Calculate the position to center the character in the image
        text_width, text_height = draw.textsize(char, font)
        x = (image_resolution - text_width) // 2 - 10
        y = (image_resolution - text_height) // 2 - 10
        
        draw.text((x, y), char, font=font, fill='black')
        
        # Save the image with the character as the filename
        filename = f"{output_folder}/{ord(char)}.png"
        image.convert("L").save(filename)
        print(filename, (x,y,text_width,text_height))

if __name__ == "__main__":
    import os, json, glob
    from tqdm import tqdm
    
    for idx, ttf_file_path in enumerate(tqdm(glob.glob('/DATA/bvac/personal/reserach/font_gen/mxfont/weizu_fonts/Arial-Unicode*.ttf'))):
        # ttf_file_path = "/DATA/bvac/personal/reserach/font_gen/mxfont/weizu_fonts/Arial-Unicode.ttf"  # Replace with the path to your TTF file
        # output_folder = "/DATA/bvac/personal/reserach/font_gen/mxfont/russua_source2"  # Specify the output folder
        output_folder = "/DATA/bvac/personal/reserach/font_gen/mxfont/ECAI/UFSC-vis/{}".format(os.path.basename(ttf_file_path.split('.')[0]))
        os.makedirs(output_folder, exist_ok=True)
        # input_text = open('/home/data/fontgen/ch_chars_train.txt').read()
        input_text = "韧鞭靴鞍"
        # print(input_text)
        image_resolution = 128
    
        render_text_to_image(input_text, ttf_file_path, output_folder, image_resolution)
