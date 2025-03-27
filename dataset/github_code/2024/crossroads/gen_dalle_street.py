import os
import json
import requests
from openai import AzureOpenAI
import argparse

parser = argparse.ArgumentParser(description="Generate images using Azure OpenAI.")
parser.add_argument("--category", type=str, default="car", help="Category of images to generate")
args = parser.parse_args()


# Load secrets
with open("secrets.json", "r") as f:
    secrets = json.load(f)

# Initialize the Azure OpenAI client
client = AzureOpenAI(
    azure_endpoint=secrets["AZURE_OPENAI_ENDPOINT"],
    api_key=secrets["AZURE_OPENAI_API_KEY"],
    api_version="2024-02-01",
)

# Load country and category data
with open("countries.json", "r") as f:
    countries = json.load(f)

categories = [
    str(args.category)
]

categories = [
    "plate_of_food",
    "front_door",
    "kitchen",
    "wall_decoration",
    "home",
    "social_drink",
    "family_snapshots",
    "home",
    "car",
    "cups_mugs_glasses"
]

# Function to generate and save images
def generate_and_save_images(country, category, number_of_images):
    # Format category string
    formatted_category = category.replace('_', ' ')
    prompt = f"A typical scene of {formatted_category} in {country}, culturally accurate and detailed."
    if category == "family":
        prompt = f"A typical family gathering in {country}, culturally accurate and detailed."
    elif category == "car":
        prompt = f"A typical scene of {country}'s iconic vehicles in everyday use, culturally accurate and detailed."

    # Directory path
    directory = f"results/dalle_natural/{category}/{country}"
    os.makedirs(directory, exist_ok=True)

    # Generate images
    for i in range(1, number_of_images + 1):
        try:
            result = client.images.generate(
                model="Dalle3",
                prompt=prompt,
                size="1024x1024",
                quality="standard",
                style="natural", # default is "vivid"
                n=1
            )
            image_url = json.loads(result.model_dump_json())['data'][0]['url']
            image = requests.get(image_url)

            # Save the image
            image_path = os.path.join(directory, f"{country}_{i}.jpg")
            with open(image_path, "wb") as f:
                f.write(image.content)
            print(f"Saved: {image_path}")

        except Exception as e:
            print(f"Error generating image for {prompt}: {str(e)}")

# Generate images for each country and category
for region, country_list in countries.items():
    for country in country_list:
        for category in categories:
            generate_and_save_images(country, category, 10)
