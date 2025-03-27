from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import requests
from datasets import load_dataset
from PIL import Image
import io
import argparse
from tqdm import tqdm
import os
from openai import AzureOpenAI
import json
import pandas as pd
import re
import math
import base64
from mimetypes import guess_type

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--split", type=str, default="plate_of_food")
args = arg_parser.parse_args()

SPLIT = args.split
DATASET_CACHE_DIR = "/projects/dollarstreet/"
dataset = load_dataset("dollarstreet", split=SPLIT, cache_dir=DATASET_CACHE_DIR)

with open("secrets.json", "r") as f:
    secrets = json.load(f)

api_base = secrets["GPT4V_OPENAI_ENDPOINT"]
api_key=secrets["GPT4V_OPENAI_API_KEY"]
deployment_name = "gpt4v"
api_version = "2024-02-15-preview"

# Function to encode the image
def encode_image(image):
    # Encoding image to base64
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str

client = AzureOpenAI(
    api_key=api_key,
    api_version=api_version,
    base_url=api_base,
)

results = []
# Loop through each example in the dataset
for example in tqdm(dataset):
    id = example["id"]
    image = example["image"]
    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")
    income = example["income"]
    user_image_path = "/tmp/test.jpg"
    with open(user_image_path, "wb") as f:
        image.save(f)
    image = Image.open(user_image_path)

    # resize the image to 512x512
    image = image.resize((512, 512))

    # Encode the image
    base64_image = encode_image(image)
    try:
        response = client.chat.completions.create(
            model=deployment_name,
            messages=[
                { "role": "system", "content": "You are a helpful assistant. Your answer should include strictly only a valid geographical subregion according to the United Nations geoscheme developed by UNSD." },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Strictly follow the United Nations geoscheme for subregions. Which geographical subregion of the United Nations geoscheme is this image from? Make an educated guess. Answer in one to three words."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "low"
                            }
                        }
                    ]
                }
            ],
            max_tokens=300
        )
        answer = response.json()
        answer = json.loads(answer)
        results.append({
            "id": id,
            "model": "gpt-4-turbo-vision-preview",
            "split": SPLIT,
            "response": answer["choices"][0]["message"]["content"],
            "true_country": example["country_name"],
            "true_region": example["region_id"],
            "true_place": example["place"],
            "income": income
        })
    except Exception as e:
        results.append({
            "id": id,
            "model": "gpt-4-turbo-vision-preview",
            "split": SPLIT,
            "response": "ResponsibleAIPolicyViolation",
            "true_country": example["country_name"],
            "true_region": example["region_id"],
            "true_place": example["place"],
            "income": income
        })
    image.close()

df = pd.DataFrame(results)
print(df.head())
df.to_csv(f"results/gpt/azure/{SPLIT}.csv", index=False)
