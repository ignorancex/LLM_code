import os
import pytest
import requests
from pathlib import Path
import gzip
import shutil
import zipfile


@pytest.fixture(scope="session", autouse=True)
def ensure_downloaded_and_extracted():
    urls = {
        "gen-ip054.mps.gz": "https://miplib.zib.de/WebData/instances/gen-ip054.mps.gz",
        "flugpl.mps.gz": "https://miplib.zib.de/WebData/instances/flugpl.mps.gz",
        "maros_meszaros_dataset1.zip": "http://www.doc.ic.ac.uk/~im/QPDATA1.ZIP",
    }
    project_dir = Path(__file__).parent.parent
    cache_dir = project_dir / ".pytest_cache"
    cache_dir.mkdir(exist_ok=True)

    for filename, url in urls.items():
        file_path = cache_dir / filename
        extracted_path = cache_dir / filename.replace(".gz", "").replace(".zip", "")

        if not file_path.exists():
            print(f"Downloading '{filename}'...")
            try:
                with requests.get(url, stream=True) as response:
                    response.raise_for_status()
                    with open(file_path, "wb") as file:
                        for chunk in response.iter_content(chunk_size=8192):
                            file.write(chunk)
                print(f"File '{filename}' downloaded successfully.")
            except requests.exceptions.RequestException as e:
                pytest.exit(f"Failed to download '{filename}': {e}")
        else:
            print(f"File '{filename}' already exists. Skipping download.")

        if filename.endswith(".gz"):
            if not extracted_path.exists():
                print(f"Extracting '{filename}'...")
                try:
                    with gzip.open(file_path, "rb") as gz_file:
                        with open(extracted_path, "wb") as extracted_file:
                            shutil.copyfileobj(gz_file, extracted_file)
                    print(f"File '{extracted_path.name}' extracted successfully.")
                except Exception as e:
                    pytest.exit(f"Failed to extract '{filename}': {e}")
            else:
                print(
                    f"File '{extracted_path.name}' already extracted. Skipping extraction."
                )

        elif filename.endswith(".zip"):
            if not extracted_path.exists():
                print(f"Extracting '{filename}'...")
                os.mkdir(extracted_path)
                try:
                    with zipfile.ZipFile(file_path, "r") as zip_ref:
                        zip_ref.extractall(extracted_path)
                    print(
                        f"File '{filename}' extracted successfully to '{extracted_path}'."
                    )
                except Exception as e:
                    pytest.exit(f"Failed to extract '{filename}': {e}")
            else:
                print(f"File '{filename}' already extracted. Skipping extraction.")

    print("All required files are ready.")
    return cache_dir
