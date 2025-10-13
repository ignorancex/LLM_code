import easyocr
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import os
import pickle

def process_images_and_generate_text_images(dataset_path, dest_dir, image_size=500, text_color=(255, 0, 0)):
    """
    Extracts text from images, saves each unique text as an image, and logs mappings.

    Args:
        dataset_path (str): Path to directory with input images.
        dest_dir (str): Directory to save outputs.
        image_size (int): Size of square text image.
        text_color (tuple): RGB color for text.
    """

    os.makedirs(dest_dir, exist_ok=True)
    os.makedirs(f"{dest_dir}/texts", exist_ok=True)

    text_data = {}         # text → list of image paths
    index_to_text = {}     # index → text
    seen_texts = {}        # text → index

    reader = easyocr.Reader(["en"], gpu=True)

    for file in tqdm(os.listdir(dataset_path), desc="Processing images"):
        file_path = os.path.join(dataset_path, file)

        try:
            extracted_texts = reader.readtext(file_path, detail=0)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            os.remove(file_path)
            continue

        for text in extracted_texts:
            if text not in text_data:
                text_data[text] = []

                # Generate and save image for new text
                fontsize = 1
                font_path = "./Arial.ttf"
                font = ImageFont.truetype(font_path, fontsize)
                img_fraction = 0.9
                while font.getsize(text)[0] < img_fraction * image_size and fontsize < 100:
                    fontsize += 1
                    font = ImageFont.truetype(font_path, fontsize)

                image = Image.new("RGBA", (image_size, image_size), color=(0, 0, 0, 0))
                draw = ImageDraw.Draw(image)

                text_width, text_height = draw.textbbox((0, 0), text, font=font)[2:4]
                text_x = (image_size - text_width) // 2
                text_y = (image_size - text_height) // 2

                draw.text((text_x, text_y), text, fill=text_color, font=font)

                index = len(index_to_text)
                image.save(f"{dest_dir}/texts/{index}.png", "PNG")
                index_to_text[index] = text
                seen_texts[text] = index

            text_data[text].append(file_path)

    # Save mappings
    with open(os.path.join(dest_dir, "text_data.pkl"), "wb") as f:
        pickle.dump(text_data, f)

    with open(os.path.join(dest_dir, "index_to_text.pkl"), "wb") as f:
        pickle.dump(index_to_text, f)

    print(f"Saved {len(index_to_text)} text images and data.")



def main(): 
    dataset_path = "cc12m_artifacts_dataset/logos"  # Replace with your dataset path
    dest_dir = "cc12m_artifacts_dataset"        # Replace with your desired output path
    process_images_and_generate_text_images(dataset_path, dest_dir)


if __name__ == "__main__":
    main()