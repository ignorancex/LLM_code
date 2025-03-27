import os
import json
import requests
import random
from tqdm import tqdm

# Directory paths
json_dir = './midjourney'
output_dir = 'mjimages'

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Create a session for requests, allowing reuse of cookies
session = requests.Session()

# List of user agents to randomize
user_agents = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15',
    'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:89.0) Gecko/20100101 Firefox/89.0',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:85.0) Gecko/20100101 Firefox/85.0',
    'Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15A372 Safari/604.1'
]

# Add random user agent to the session headers
session.headers.update({
    'User-Agent': random.choice(user_agents),
    'Referer': 'https://www.midjourney.com/',
})


# Function to download images
def download_image(url, filename):
    if os.path.exists(filename):
        # print(f"Skipping {filename}, already downloaded.")
        return

    try:
        response = session.get(url, stream=True)
        if response.status_code == 200:
            with open(filename, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            # print(f"Downloaded {url} to {filename}")
        # else:
        # print(f"Failed to download {url}, status code: {response.status_code}")
    except Exception as e:
        print(f"Error downloading {url}: {e}")


# Function to transform the Discord URL to MidJourney URL format
def transform_url(discord_url):
    # Extract the file identifier part of the URL
    filename = discord_url.split('/')[-1]
    file_id = filename.split('_')[0]  # Use the first part of the filename
    return f"https://cdn.midjourney.com/{file_id}/0_0.png"


# Iterate over all JSON files in the specified directory
for json_file in tqdm(os.listdir(json_dir)[::-1]):
    if json_file.endswith('.json'):
        json_file_path = os.path.join(json_dir, json_file)
        #print(f"Processing {json_file_path}...")

        # Load each JSON file
        with open(json_file_path, 'r') as file:
            data = json.load(file)

        # Loop through messages in each JSON file and download images
        for message_group in data['messages']:
            for message in message_group:
                attachments = message.get('attachments', [])
                for attachment in attachments:
                    img_url = attachment['url']
                    # Transform the URL to the desired format
                    midjourney_url = transform_url(img_url)
                    img_filename = os.path.join(output_dir, attachment['filename'])
                    download_image(midjourney_url, img_filename)

print("All images downloaded successfully!")
