import os
import pickle
import requests
from tqdm import tqdm

# Paths
save_dir = "cc12m_artifacts_dataset/logos"
mapping_file = "cc12m_artifacts_dataset/logo_filename_to_url.pkl"

# Load mapping {filename → url}
with open(mapping_file, 'rb') as f:
    filename_to_url = pickle.load(f)

# Create directory if it doesn't exist
os.makedirs(save_dir, exist_ok=True)

# Download loop
headers = {"User-Agent": "Mozilla/5.0"}
for filename, url in tqdm(filename_to_url.items(), desc="Downloading images"):
    save_path = os.path.join(save_dir, filename)

    # Skip if already downloaded
    if os.path.exists(save_path):
        continue

    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200 and response.content:
            with open(save_path, 'wb') as f:
                f.write(response.content)
        else:
            print(f"Failed to download {filename}: status {response.status_code}")
    except Exception as e:
        print(f"Error downloading {filename}: {e}")