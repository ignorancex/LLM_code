import os 
from tqdm import tqdm 
import argparse 
from PIL import Image
from lib.dataset.dataset_fairface import FairFace
from lib.dataset.dataset_country211 import Country211
from lib.dataset.dataset_aircraft import Aircraft
from lib.dataset.dataset_celeba import CelebA

from PIL import Image, ImageDraw, ImageFont

def get_text_image(text, out):
    image_size = 500
    text_color=(255, 0, 0)
    img_fraction = 0.9
    fontsize = 1  #

    font = ImageFont.truetype("./Arial.ttf", fontsize)
    count_times = 0 
    while font.getsize(text)[0] < img_fraction*image_size:
        # iterate until the text size is just larger than the criteria
        fontsize += 1
        font = ImageFont.truetype("./Arial.ttf", fontsize)
        count_times += 1 

        if count_times > 100:
            break

    font = ImageFont.truetype("./Arial.ttf", fontsize)

    image = Image.new("RGBA", (image_size, image_size), color=(0, 0, 0, 0))  # Transparent background
    draw = ImageDraw.Draw(image)

    # Calculate text size and position
    text_width, text_height = draw.textbbox((0, 0), text, font=font)[2:4]
    text_x = (image_size - text_width) // 2
    text_y = (image_size - text_height) // 2

    # Add text to image
    draw.text((text_x, text_y), text, fill=text_color, font=font)
    # Save the image
    image.save(f"{out}/0_attack.png", "PNG")    


parser = argparse.ArgumentParser(description='Get logo scores')
args = parser.parse_args()

for dataset in tqdm(["aircraft", "country211", "celeba_smiling", "fairface_age", "fairface_gender"]):
    args.dataset = dataset
    args.transparency = 1.0
    args.factor_shrink = 4
    args.owlv2 = False

    if args.dataset == "fairface_age":
        args.concept = "age"
        dataset = FairFace(args, split="train")
        _, pairs = dataset.get_prompts()

    elif args.dataset == "fairface_gender":
        args.concept = "gender"
        dataset = FairFace(args, split="train")
        _, pairs = dataset.get_prompts()

    elif args.dataset == "celeba_smiling":
        dataset = CelebA(args, split="train", concept="Smiling")
        _, pairs = dataset.get_prompts()

    elif args.dataset == "country211":
        dataset = Country211(args, split="train")
        _, pairs = dataset.get_prompts()
    
    elif args.dataset == "aircraft":
        dataset = Aircraft(args, split="train")
        _, pairs = dataset.get_prompts()


    for pair in pairs: 

        to_save_dir = f"output/artifacts_typo/{args.dataset}/{pair}"
        os.makedirs(to_save_dir, exist_ok=True)
        get_text_image(pair, to_save_dir)