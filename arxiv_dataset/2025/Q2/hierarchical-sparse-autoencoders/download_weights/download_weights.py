#!/usr/bin/env python3
"""
Download all files from a public S3 bucket into a local directory,
then write the absolute path of that directory to a file.
"""
import os
import sys
import urllib.request
import xml.etree.ElementTree as ET

# Configuration
BUCKET_NAME = "hierarchical-sparse-autoencoder"
DEST_DIR = "../hsae_models"
PATH_FILE = "../hsae_models_path.txt"

# Ensure destination directory exists
os.makedirs(DEST_DIR, exist_ok=True)

# S3 bucket listing URL (public)
listing_url = f"http://{BUCKET_NAME}.s3.amazonaws.com/"

try:
    with urllib.request.urlopen(listing_url) as response:
        xml_content = response.read()
except Exception as e:
    print(f"Error fetching bucket listing: {e}", file=sys.stderr)
    sys.exit(1)

# Parse XML and extract object keys
root = ET.fromstring(xml_content)
namespace = {'s3': 'http://s3.amazonaws.com/doc/2006-03-01/'}
keys = [elem.text for elem in root.findall('s3:Contents/s3:Key', namespace)]

if not keys:
    print("No objects found in bucket.")
    sys.exit(0)

# Download each object
total = len(keys)
for idx, key in enumerate(keys, start=1):
    url = f"http://{BUCKET_NAME}.s3.amazonaws.com/{key}"
    dest_path = os.path.join(DEST_DIR, os.path.basename(key))
    try:
        print(f"[{idx}/{total}] Downloading {key} -> {dest_path}")
        urllib.request.urlretrieve(url, dest_path)
    except Exception as e:
        print(f"Failed to download {key}: {e}", file=sys.stderr)

# Write the absolute path of DEST_DIR to PATH_FILE
abs_path = os.path.abspath(DEST_DIR)
with open(PATH_FILE, 'w') as f:
    f.write(abs_path + os.linesep)

print(f"Downloaded {total} files to {abs_path}")
print(f"Directory path written to {PATH_FILE}")
