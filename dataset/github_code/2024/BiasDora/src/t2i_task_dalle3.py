import os
import json
import pandas as pd
import requests
from openai import AzureOpenAI
import argparse

parser = argparse.ArgumentParser(description="Generate images using Azure OpenAI.")
parser.add_argument(
    "--setting", type=str, default="1", help="Setting of images to generate"
)
args = parser.parse_args()

setting = str(args.setting)

df = pd.read_csv("../data/t2i_setting" + setting + ".csv")

if setting == "1":
    df["article"] = df["target_singular"].apply(
        lambda x: "an" if x[0].lower() in ["a", "e", "i", "o", "u"] else "a"
    )
    df["prompt"] = (
        df["template_singular"].str.replace(" a/an", "")
        + " "
        + df["article"]
        + " "
        + df["target_singular"].str.lower()
    )

elif setting == "2":
    df["prompt"] = df["template_plural"] + " " + df["target_plural"].str.lower()

print(df.head())

# Load secrets
with open("secrets.json", "r") as f:
    secrets = json.load(f)

# Initialize the Azure OpenAI client
client = AzureOpenAI(
    azure_endpoint=secrets["AZURE_OPENAI_ENDPOINT"],
    api_key=secrets["AZURE_OPENAI_API_KEY"],
    api_version="2024-02-01",
)


def generate_and_save_images(prompt, idx, target, number_of_images=5):
    directory = f"../outputs/dalle/setting{setting}/"
    os.makedirs(directory, exist_ok=True)

    for style in ["natural", "vivid"]:
        for i in range(1, number_of_images + 1):
            try:
                result = client.images.generate(
                    model="Dalle3",
                    prompt=prompt,
                    size="1024x1024",
                    quality="standard",
                    style=style,
                    n=1,
                )
                image_url = json.loads(result.model_dump_json())["data"][0]["url"]
                image = requests.get(image_url)

                # Save the image
                target = target.replace(" ", "_")
                image_path = os.path.join(
                    directory, f"id_{idx}_target_{target}_{style}_{i}.jpg"
                )
                with open(image_path, "wb") as f:
                    f.write(image.content)
                print(f"Saved: {image_path}")
            except Exception as e:
                print(
                    f"Error generating image for row {idx}, prompt {prompt}: {str(e)}"
                )


for idx, row in df.iterrows():
    prompt = row["prompt"]
    if setting == "1":
        target = row["target_singular"].lower()
    elif setting == "2":
        target = row["target_plural"].lower()
    generate_and_save_images(prompt, idx, target)
