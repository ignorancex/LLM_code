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
arg_parser.add_argument("--type", type=str, default="vivid")
arg_parser.add_argument("--concept", type=str, default="car")
args = arg_parser.parse_args()

TYPE = args.type
CONCEPT = args.concept
df = pd.read_csv("results/dalle_images.csv")

df = df[df["concept"] == CONCEPT]
df = df[df["type"] == TYPE]

dataset = df.to_dict(orient="records")
print(f"Number of {TYPE} images for {CONCEPT}: {len(dataset)}")

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
for example in tqdm(dataset):
    path = example["image_path"]
    concept = example["concept"]
    country = example["country"]
    type = example["type"]

    id = path
    image = Image.open(path)
    if image.mode in ("RGBA", "P", "L", "LA"):
        image = image.convert("RGB")

    # resize the image to 512x512
    image = image.resize((512, 512))

    # Encode the image
    base64_image = encode_image(image)
    try:
        response = client.chat.completions.create(
            model=deployment_name,
            messages=[
                { "role": "system", "content": f"You are a helpful assistant. Your answer should include a valid JSON output describing the detected objects. The JSON should contain two fields: 'relevant_objects' - for objects relevant to the image category ({concept}). 'other_objects' - for other objects. Do not include any other fields." },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Give me a json output of the items you see in this image in both the foreground and background. Output the objects as a JSON with two fields: 'relevant_objects' for objects pertinent to the image category ({concept}) and 'other_objects' for all additional detected objects. Be as specific as possible. Within each field, for each detected object, include sub-fields describing object attributes like colour, count, and anything else that is appropriate. For example, for buildings describe the architectural style in a sentence, for people describe clothing and headgear (if multiple colours and headgears are present, include the top three), for food items describe the exact type of food and include a brief recipe description, for pictures of rooms include objects in the background like mountains outside a window or paintings on the wall portraying something specific like a landmark or a particular type of scenery. For the counts of items, if the number of items is less than 10, give me exact numbers otherwise say more than 10."
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
            "split": f"{TYPE}_{CONCEPT}",
            "response": answer["choices"][0]["message"]["content"],
            "true_country": example["country"],
            "concept": example["concept"],
            "type": example["type"]
        })
    except Exception as e:
        results.append({
            "id": id,
            "model": "gpt-4-turbo-vision-preview",
            "split": f"{TYPE}_{CONCEPT}",
            "response": "ResponsibleAIPolicyViolation",
            "true_country": example["country"],
            "concept": example["concept"],
            "type": example["type"]
        })
    image.close()

df = pd.DataFrame(results)
print(df.head())
df.to_csv(f"results/dalle_objects/{TYPE}/{CONCEPT}.csv", index=False)
